import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import duckdb
from sklearn.cluster import DBSCAN
from data_pipeline.config import ASEAN_BOUNDS

# Dynamic EPSG logic for N/S hemispheres
def get_utm_epsg(longitude, latitude):
    zone = int(np.floor((longitude + 180) / 6) + 1)
    if latitude >= 0:
        return 32600 + zone
    return 32700 + zone

def process_candidate_site_clusters(df_raw, eps_meters=50, min_samples=1):
    print("🗼 Clustering raw OpenCelliD antennas into unified physical sites using Haversine...")
    
    if 'radio' in df_raw.columns:
        radio_dummies = pd.get_dummies(df_raw['radio'], prefix='radio', dummy_na=False)
        df_raw = pd.concat([df_raw, radio_dummies], axis=1)
        
    expected_radios = ['radio_LTE', 'radio_UMTS', 'radio_GSM', 'radio_NR']
    for r in expected_radios:
        if r not in df_raw.columns:
            df_raw[r] = 0
    
    gdf = gpd.GeoDataFrame(
        df_raw, 
        geometry=gpd.points_from_xy(df_raw['longitude'], df_raw['latitude']),
        crs="EPSG:4326"
    )
    
    # Spherical clustering, immune to UTM zone distortion
    EARTH_RADIUS_M = 6_371_008.8
    coords_rad = np.radians(np.column_stack([gdf.geometry.y, gdf.geometry.x]))
    
    clustering = DBSCAN(
        eps=eps_meters / EARTH_RADIUS_M, 
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree"
    ).fit(coords_rad)
    
    gdf['cluster_id'] = clustering.labels_
    
    agg_dict = {
        'radio': 'count', 
        'range': 'mean',
        'radio_LTE': 'sum',
        'radio_UMTS': 'sum',
        'radio_GSM': 'sum',
        'radio_NR': 'sum'
    }
    
    # Explicit mean lat/lon cluster centers, not .centroid() on a geographic
    # (lat/lon degrees) CRS. gpd's .centroid computes a PLANAR centroid on
    # angular coordinates, which geopandas correctly warns is inaccurate —
    # negligible distortion at ~50m cluster scale, but the warning is
    # legitimate and this removes it entirely rather than suppressing it.
    cluster_centers = (
        gdf.groupby("cluster_id")[["longitude", "latitude"]]
           .mean()
           .reset_index()
    )
    clustered_sites = (
        gdf.drop(columns="geometry")
           .groupby("cluster_id", as_index=False)
           .agg(agg_dict)
    )
    clustered_sites = clustered_sites.merge(cluster_centers, on="cluster_id", how="left")
    clustered_sites = gpd.GeoDataFrame(
        clustered_sites,
        geometry=gpd.points_from_xy(clustered_sites["longitude"], clustered_sites["latitude"]),
        crs="EPSG:4326"
    )
    
    clustered_sites = clustered_sites.rename(columns={
        'radio': 'antenna_count',
        'radio_LTE': 'antennas_4G',
        'radio_UMTS': 'antennas_3G',
        'radio_GSM': 'antennas_2G',
        'radio_NR': 'antennas_5G'
    })
    
    # Diagnostic only — does not change clustering behavior. min_samples=1
    # DBSCAN can chain-link distant points through intermediate neighbors
    # (A-B close, B-C close, A-C far, all merged into one cluster). Flagging
    # unusually large clusters here makes that visible without assuming it's
    # a bug — could be genuine dense concentration (e.g. central Jakarta) or
    # genuine chaining; worth a manual look either way before trusting it.
    if len(clustered_sites) > 0:
        p99 = clustered_sites['antenna_count'].quantile(0.99)
        giant_clusters = clustered_sites[clustered_sites['antenna_count'] > max(p99, 100)]
        if len(giant_clusters) > 0:
            print(f"⚠️ {len(giant_clusters)} unusually large antenna cluster(s) found "
                  f"(antenna_count > {max(p99, 100):.0f}) — verify these aren't DBSCAN "
                  f"chain artifacts before trusting them: "
                  f"{sorted(giant_clusters['antenna_count'].tolist(), reverse=True)[:5]}")
    
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
    else:
        raise ValueError("Invalid layer name requested.")

    # 3. Execute and convert directly to a Pandas DataFrame
    df = con.execute(query).df()
    
    if df.empty:
        print(f"⚠️ No {layer_name} found in this bounding box.")
        return gpd.GeoDataFrame(columns=['geometry', 'osm_type'], crs="EPSG:4326")

    # 4. Cast DuckDB's bytearray type to standard Python bytes for Shapely compatibility
    df['geom_wkb'] = df['geom_wkb'].apply(bytes)

    # 5. Convert the WKB binary back into shapely geometry objects
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.GeoSeries.from_wkb(df['geom_wkb']), 
        crs="EPSG:4326"
    )
    
    # 6. Clean up the temporary binary column
    gdf = gdf.drop(columns=['geom_wkb'])
    
    return gdf

def engineering_osm_proximity_features(gdf_sites, region_name):
    print(f"🌐 Running vectorized distance calculations for {region_name} using UTM Chunks...")
    
    gdf_sites = gdf_sites.copy()
    
    # Assign each tower to its proper UTM zone + hemisphere
    gdf_sites["_utm_epsg"] = [
        get_utm_epsg(lon, lat) for lon, lat in zip(gdf_sites.geometry.x, gdf_sites.geometry.y)
    ]
   
    for layer_name in ["power", "roads", "amenities"]:
        print(f"📥 Extracting Overture vectors for layer: {layer_name}")
        raw_osm = fetch_overture_layer(gdf_sites, region_name, layer_name)
        
        name_map = {"power": "power", "roads": "road", "amenities": "amenity"}
        target_col = f"distance_to_{name_map[layer_name]}_m"

        if raw_osm.empty:
            gdf_sites[target_col] = np.nan
            continue
            
        # Process distances strictly within matched UTM zones
        for epsg, idx in gdf_sites.groupby("_utm_epsg").groups.items():
            local_sites = gdf_sites.loc[idx].copy()
            sites_metric = local_sites.to_crs(f"EPSG:{epsg}")
            osm_metric = raw_osm.to_crs(f"EPSG:{epsg}")
            
            joined = gpd.sjoin_nearest(sites_metric, osm_metric, distance_col="distance_metrics", how="left")
            joined = joined[~joined.index.duplicated(keep="first")]
            
            gdf_sites.loc[joined.index, target_col] = joined["distance_metrics"]
       
    # Clean up the temporary UTM column
    gdf_sites = gdf_sites.drop(columns=["_utm_epsg"])
    return gdf_sites