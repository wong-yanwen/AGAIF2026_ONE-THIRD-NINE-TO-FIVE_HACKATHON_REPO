import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import duckdb
from sklearn.cluster import DBSCAN
from data_pipeline.config import ASEAN_BOUNDS

def get_utm_crs(longitude):
    """
    Dynamically calculates the correct UTM EPSG code for accurate metric 
    distance calculations based on the centroid longitude of the data.
    """
    utm_zone = int((longitude + 180) / 6) + 1
    return f"EPSG:326{utm_zone}"

def process_candidate_site_clusters(df_raw, eps_meters=500, min_samples=1):
    """
    Clusters crowdsourced OpenCelliD antenna records into consolidated site locations 
    using DBSCAN to group antennas on the same physical tower.
    """
    print("🗼 Clustering raw OpenCelliD antennas into unified physical sites...")
    
    # One-Hot Encode the radio technologies
    if 'radio' in df_raw.columns:
        radio_dummies = pd.get_dummies(df_raw['radio'], prefix='radio', dummy_na=False)
        df_raw = pd.concat([df_raw, radio_dummies], axis=1)
        
    # Ensure standard columns exist so downstream ML models don't crash 
    # if a specific region lacks 5G (NR) or older techs.
    expected_radios = ['radio_LTE', 'radio_UMTS', 'radio_GSM', 'radio_NR']
    for r in expected_radios:
        if r not in df_raw.columns:
            df_raw[r] = 0
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_raw, 
        geometry=gpd.points_from_xy(df_raw['longitude'], df_raw['latitude']),
        crs="EPSG:4326"
    )
    
    # Project to metric CRS for accurate distance-based clustering
    avg_lon = gdf['geometry'].x.mean()
    local_crs = get_utm_crs(avg_lon)
    gdf_metric = gdf.to_crs(local_crs)
    
    # Extract metric coordinates for DBSCAN
    coords = np.array(list(zip(gdf_metric.geometry.x, gdf_metric.geometry.y)))
    
    # Cluster antennas within eps_meters of each other
    clustering = DBSCAN(eps=eps_meters, min_samples=min_samples).fit(coords)
    gdf_metric['cluster_id'] = clustering.labels_
    
    # Dynamic aggregation dictionary to sum up all radio types
    agg_dict = {
        'radio': 'count', 
        'range': 'mean',
        'radio_LTE': 'sum',
        'radio_UMTS': 'sum',
        'radio_GSM': 'sum',
        'radio_NR': 'sum'
    }
    
    # Aggregate back to single site points (centroids of clusters)
    clustered_sites = gdf_metric.dissolve(by='cluster_id', aggfunc=agg_dict).reset_index()

    # Convert the dissolved MultiPoint clusters into single Point centroids
    clustered_sites['geometry'] = clustered_sites['geometry'].centroid

    clustered_sites = clustered_sites.rename(columns={
        'radio': 'antenna_count',
        'radio_LTE': 'antennas_4G',
        'radio_UMTS': 'antennas_3G',
        'radio_GSM': 'antennas_2G',
        'radio_NR': 'antennas_5G'
    })
    
    # Revert back to standard GPS coordinates for downstream GEE merging
    return clustered_sites.to_crs("EPSG:4326")


def fetch_overture_layer(gdf_sites, region_name, layer_name):
    """
    Streams infrastructure data directly from Overture Maps' AWS S3 buckets 
    using DuckDB, dynamically fetching the latest release to prevent broken paths.
    """
    bounds = ASEAN_BOUNDS.get(region_name)
    if not bounds:
        raise ValueError(f"Region {region_name} not found in bounds mapping.")
        
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # 1. Dynamically fetch the latest active release version to prevent IO Errors
    print("📡 Fetching latest Overture Maps release version from STAC catalog...")
    try:
        catalog = requests.get('https://stac.overturemaps.org/catalog.json').json()
        release_version = catalog.get('latest')
    except Exception:
        # Fallback to the current stable release as of August 2026
        release_version = '2026-07-22.0'
        
    release_path = f"s3://overturemaps-us-west-2/release/{release_version}"
    print(f"☁️ Querying Overture Maps (AWS S3) at {release_version} for {layer_name} in {region_name}...")
    
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET http_keep_alive=false;") # prevent infinite hanging
    
    # 2. Extract geometry AS WKB (Well-Known Binary) for GeoPandas compatibility
    if layer_name == "power":
        query = f"""
            SELECT ST_AsWKB(geometry) as geom_wkb, class as osm_type 
            FROM read_parquet('{release_path}/theme=base/type=infrastructure/*', filename=true, hive_partitioning=1)
            WHERE bbox.xmax >= {min_lon} AND bbox.xmin <= {max_lon}
            AND bbox.ymax >= {min_lat} AND bbox.ymin <= {max_lat}
            AND class = 'power_line'
        """
    elif layer_name == "roads":
        query = f"""
            SELECT ST_AsWKB(geometry) as geom_wkb, class as osm_type 
            FROM read_parquet('{release_path}/theme=transportation/type=segment/*', filename=true, hive_partitioning=1)
            WHERE bbox.xmax >= {min_lon} AND bbox.xmin <= {max_lon}
            AND bbox.ymax >= {min_lat} AND bbox.ymin <= {max_lat}
            AND class IN ('primary', 'secondary', 'tertiary', 'trunk')
        """
    elif layer_name == "amenities":
        query = f"""
            SELECT ST_AsWKB(geometry) as geom_wkb, categories.primary as osm_type 
            FROM read_parquet('{release_path}/theme=places/type=place/*', filename=true, hive_partitioning=1)
            WHERE bbox.xmax >= {min_lon} AND bbox.xmin <= {max_lon}
            AND bbox.ymax >= {min_lat} AND bbox.ymin <= {max_lat}
            AND categories.primary IN ('school', 'hospital', 'clinic')
        """
    elif layer_name == "tier1_hubs":
        query = f"""
            SELECT ST_AsWKB(geometry) as geom_wkb, subtype as osm_type
            FROM read_parquet('{release_path}/theme=divisions/type=division/*', filename=true, hive_partitioning=1)
            WHERE bbox.xmax >= {min_lon} AND bbox.xmin <= {max_lon}
            AND bbox.ymax >= {min_lat} AND bbox.ymin <= {max_lat}
            AND subtype = 'locality'
        """
    else:
        raise ValueError("Invalid layer name requested.")

    # Execute and convert directly to a Pandas DataFrame
    df = con.execute(query).df()
    
    if df.empty:
        print(f"⚠️ No {layer_name} found in this bounding box.")
        return gpd.GeoDataFrame(columns=['geometry', 'osm_type'], crs="EPSG:4326")

    # Cast DuckDB's bytearray type to standard Python bytes for Shapely compatibility
    df['geom_wkb'] = df['geom_wkb'].apply(bytes)

    # 3. Convert the WKB binary back into shapely geometry objects
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.GeoSeries.from_wkb(df['geom_wkb']), 
        crs="EPSG:4326"
    )
    
    # Clean up the temporary binary column
    gdf = gdf.drop(columns=['geom_wkb'])
    
    return gdf

def engineering_osm_proximity_features(gdf_sites, region_name):
    """
    Calculates proximity distances against the chunk-harvested infrastructure nodes.
    """
    print(f"🌐 Running vectorized distance calculations for {region_name}...")
   
    avg_lon = gdf_sites['geometry'].x.mean()
    dynamic_epsg = get_utm_crs(avg_lon)
    sites_metric = gdf_sites.to_crs(dynamic_epsg)
   
    for layer_name in ["power", "roads", "amenities", "tier1_hubs"]:
        print(f"📥 Extracting chunked vectors for layer: {layer_name}")

        raw_osm = fetch_overture_layer(gdf_sites, region_name, layer_name)

        # MAP the layer to its column name
        name_map = {
            "power": "power", 
            "roads": "road", 
            "amenities": "amenity", 
            "tier1_hubs": "tier1_hub"
        }
        target_col = f"distance_to_{name_map[layer_name]}_m"

        if raw_osm.empty:
            gdf_sites[target_col] = np.nan
            continue
            
        osm_metric = raw_osm.to_crs(dynamic_epsg)
        joined = gpd.sjoin_nearest(sites_metric, osm_metric, distance_col="distance_metrics", how="left")
        joined = joined[~joined.index.duplicated(keep='first')]
        
        gdf_sites[target_col] = joined["distance_metrics"]
       
    return gdf_sites
