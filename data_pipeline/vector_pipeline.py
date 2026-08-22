import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import tempfile
import uuid

def get_utm_crs(longitude):
    """Calculates the UTM EPSG code based on longitude."""
    zone_number = int(np.floor((longitude + 180) / 6) + 1)
    return f"EPSG:{32600 + zone_number}"


def process_vector_proximity(df_underserved, site_nodes_final, region):
    print("⏳ Executing local spatial query and projection via DuckDB (GeoParquet)...")
    
    # 1. Project both GeoDataFrames to a metric UTM CRS before saving to Parquet
    avg_lon = df_underserved['geometry'].x.mean()
    local_utm = get_utm_crs(avg_lon)
    
    df_underserved_metric = df_underserved.to_crs(local_utm)
    site_nodes_metric = site_nodes_final.to_crs(local_utm)
    
    # 2. Create a secure temporary directory to hold our GeoParquet files
    temp_dir = tempfile.gettempdir()
    underserved_path = os.path.join(temp_dir, f"underserved_{uuid.uuid4().hex}.parquet")
    sites_path = os.path.join(temp_dir, f"sites_{uuid.uuid4().hex}.parquet")
    
    # 3. Write the metric GeoDataFrames to Parquet
    df_underserved_metric.to_parquet(underserved_path)
    site_nodes_metric.to_parquet(sites_path)

    # 4. Initialize DuckDB and load the spatial extension
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # 5. Update the SQL query to read directly from the Parquet files.
    query = f"""
        SELECT 
            u.*,
            MIN(ST_Distance(u.geometry, s.geometry)) AS distance_to_nearest_tower,
            arg_min(s.distance_to_power_m, ST_Distance(u.geometry, s.geometry)) AS distance_to_power_m,
            arg_min(s.distance_to_road_m, ST_Distance(u.geometry, s.geometry)) AS distance_to_road_m,
            arg_min(s.distance_to_amenity_m, ST_Distance(u.geometry, s.geometry)) AS distance_to_amenity_m,
            arg_min(s.distance_to_tier1_hub_m, ST_Distance(u.geometry, s.geometry)) AS distance_to_tier1_hub_m,
            arg_min(s.antenna_count, ST_Distance(u.geometry, s.geometry)) AS antenna_count,
            arg_min(s.antennas_4G, ST_Distance(u.geometry, s.geometry)) AS antennas_4G,
            arg_min(s.antennas_3G, ST_Distance(u.geometry, s.geometry)) AS antennas_3G,
            arg_min(s.antennas_2G, ST_Distance(u.geometry, s.geometry)) AS antennas_2G,
            arg_min(s.antennas_5G, ST_Distance(u.geometry, s.geometry)) AS antennas_5G
        FROM '{underserved_path}' u
        CROSS JOIN '{sites_path}' s
        GROUP BY ALL
    """
    
    result_df = con.execute(query).df()
    
    # 6. Convert DuckDB's raw bytearray into Shapely geometry objects
    result_df['geometry'] = result_df['geometry'].apply(bytes)
    
    # 7. Project BACK to standard GPS coordinates so the rest of the pipeline works
    result_gdf = gpd.GeoDataFrame(
        result_df, 
        geometry=gpd.GeoSeries.from_wkb(result_df['geometry']), 
        crs=local_utm
    )
    result_gdf = result_gdf.to_crs("EPSG:4326")
    
    # 7. Clean up: Delete the temporary GeoParquet files from disk
    try:
        os.remove(underserved_path)
        os.remove(sites_path)
    except OSError as e:
        print(f"⚠️ Warning: Could not delete temporary file: {e}")
        
    return result_gdf 