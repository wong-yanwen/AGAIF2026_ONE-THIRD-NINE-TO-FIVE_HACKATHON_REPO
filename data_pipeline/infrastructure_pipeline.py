import time
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point, LineString
from sklearn.cluster import DBSCAN
from config import ASEAN_BOUNDS

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
    
    # Aggregate back to single site points (centroids of clusters)
    clustered_sites = gdf_metric.dissolve(by='cluster_id', aggfunc={
        'radio': 'count', 
        'range': 'mean'
    }).reset_index()

    # Convert the dissolved MultiPoint clusters into single Point centroids
    clustered_sites['geometry'] = clustered_sites['geometry'].centroid

    clustered_sites = clustered_sites.rename(columns={'radio': 'antenna_count'})
    
    # Revert back to standard GPS coordinates for downstream GEE merging
    return clustered_sites.to_crs("EPSG:4326")

def fetch_osm_layer_chunked(gdf_sites, region_name, osm_query_type, step_size_deg=2.0):
    """
    Queries the Overpass API using a sliding spatial grid window that filters out 
    empty ocean cells based on tower presence, drastically reducing API load.
    """
    bounds = ASEAN_BOUNDS.get(region_name)
    if not bounds:
        raise ValueError(f"Region {region_name} not found in configuration bounds mapping.")
        
    min_lon, min_lat, max_lon, max_lat = bounds
    
    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter"
    ]
    url_idx = 0
    
    lon_grid = np.arange(min_lon, max_lon + step_size_deg, step_size_deg)
    lat_grid = np.arange(min_lat, max_lat + step_size_deg, step_size_deg)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ESG-Research-Jendela-v2',
        'Accept': 'application/json'
    }
    
    features = []
    
    # Pre-calculate which windows actually contain towers to save API requests
    valid_windows = []
    for i in range(len(lon_grid) - 1):
        for j in range(len(lat_grid) - 1):
            w_min_lon, w_max_lon = lon_grid[i], lon_grid[i+1]
            w_min_lat, w_max_lat = lat_grid[j], lat_grid[j+1]
            
            # Check if any towers fall inside this specific window using spatial indexing
            towers_in_cell = gdf_sites.cx[w_min_lon:w_max_lon, w_min_lat:w_max_lat]
            
            if not towers_in_cell.empty:
                valid_windows.append((w_min_lat, w_min_lon, w_max_lat, w_max_lon))

    total_cells = len(valid_windows)
    print(f"🗺️ Orchestrating optimized grid splitting for {region_name} ({osm_query_type}).")
    print(f"🌊 Filtered out ocean cells. Querying only {total_cells} active land windows containing infrastructure...")

    for idx, (w_min_lat, w_min_lon, w_max_lat, w_max_lon) in enumerate(valid_windows):
        bbox_str = f"{w_min_lat},{w_min_lon},{w_max_lat},{w_max_lon}"
        settings = f"[out:json][timeout:90][bbox:{bbox_str}];"
        
        queries = {
            "power": f"{settings}(way['power'='line'];node['power'='substation'];way['power'='substation'];);out geom;",
            "roads": f"{settings}(way['highway'~'^(primary|secondary|tertiary|trunk)$'];);out geom;",
            "amenities": f"{settings}(node['amenity'~'^(school|clinic|hospital)$'];way['amenity'~'^(school|clinic|hospital)$'];);out geom;"
        }
        
        retries = 4
        backoff_time = 10  # Reduced initial penalty to 10 seconds
        
        while retries > 0:
            current_url = overpass_urls[url_idx % len(overpass_urls)]
            try:
                response = requests.post(
                    current_url, 
                    data={'data': queries[osm_query_type]}, 
                    headers=headers, 
                    timeout=110
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for element in data.get('elements', []):
                        if element['type'] == 'node':
                            geom = Point(element['lon'], element['lat'])
                        elif 'geometry' in element:
                            geom = LineString([(pt['lon'], pt['lat']) for pt in element['geometry']])
                        else:
                            continue
                            
                        features.append({
                            'geometry': geom,
                            'osm_type': element.get('tags', {}).get('amenity') or 
                                        element.get('tags', {}).get('power') or 
                                        element.get('tags', {}).get('highway')
                        })
                    break 
                    
                elif response.status_code == 429:
                    print(f"   ⚠️ HTTP 429 on {current_url.split('//')[1].split('/')[0]}. Waiting {backoff_time}s...")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, 30) # 💥 Capped backoff at 30 seconds max
                    url_idx += 1 
                    retries -= 1
                else:
                    break
                    
            except requests.exceptions.Timeout:
                url_idx += 1
                retries -= 1
        
        # Polite baseline breather
        time.sleep(2.0)

    if len(features) == 0:
        return gpd.GeoDataFrame(columns=['geometry', 'osm_type'], crs="EPSG:4326")

    return gpd.GeoDataFrame(features, crs="EPSG:4326")


def engineering_osm_proximity_features(gdf_sites, region_name):
    """
    Calculates proximity distances against the chunk-harvested infrastructure nodes.
    """
    print(f"🌐 Running vectorized distance calculations for {region_name}...")
    
    avg_lon = gdf_sites['geometry'].x.mean()
    dynamic_epsg = get_utm_crs(avg_lon)
    sites_metric = gdf_sites.to_crs(dynamic_epsg)
    
    for layer_name in ["power", "roads", "amenities"]:
        print(f"📥 Extracting chunked vectors for layer: {layer_name}")
        
        # 💥 PASS gdf_sites into the updated chunked fetcher
        raw_osm = fetch_osm_layer_chunked(gdf_sites, region_name, layer_name)
        
        if raw_osm.empty:
            target_col = f"distance_to_{layer_name[:-1] if layer_name=='amenities' else layer_name}_m"
            gdf_sites[target_col] = np.nan
            continue
            
        osm_metric = raw_osm.to_crs(dynamic_epsg)
        joined = gpd.sjoin_nearest(sites_metric, osm_metric, distance_col="distance_metrics", how="left")
        joined = joined[~joined.index.duplicated(keep='first')]
        
        target_column_label = f"distance_to_{layer_name[:-1] if layer_name=='amenities' else layer_name}_m"
        gdf_sites[target_column_label] = joined["distance_metrics"]
        
    return gdf_sites
