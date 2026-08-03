import duckdb
import pandas as pd
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
    
    # 1. Create a secure temporary directory to hold our GeoParquet files
    temp_dir = tempfile.gettempdir()
    
    # Generate unique filenames to avoid collisions if you run this concurrently
    underserved_path = os.path.join(temp_dir, f"underserved_{uuid.uuid4().hex}.parquet")
    sites_path = os.path.join(temp_dir, f"sites_{uuid.uuid4().hex}.parquet")
    
    # 2. Write the GeoDataFrames to GeoParquet.
    # Note: GeoPandas (via PyArrow) inherently writes to the GeoParquet standard.
    # It converts the shapely geometry to WKB internally and writes the necessary metadata.
    df_underserved.to_parquet(underserved_path)
    site_nodes_final.to_parquet(sites_path)

    # 3. Initialize DuckDB and load the spatial extension
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # 4. Update the SQL query to read directly from the Parquet files.
    query = f"""
        SELECT 
            u.*,
            MIN(ST_Distance(u.geometry, s.geometry)) AS distance_to_nearest_tower,
            arg_min(s.distance_to_power_m, ST_Distance(u.geometry, s.geometry)) AS distance_to_power_m,
            arg_min(s.distance_to_road_m, ST_Distance(u.geometry, s.geometry)) AS distance_to_road_m,
            arg_min(s.distance_to_amenity_m, ST_Distance(u.geometry, s.geometry)) AS distance_to_amenity_m,
            arg_min(s.antenna_count, ST_Distance(u.geometry, s.geometry)) AS antenna_count
        FROM '{underserved_path}' u
        CROSS JOIN '{sites_path}' s
        GROUP BY ALL
    """
    
    result_df = con.execute(query).df()
    
    # 5. Clean up: Delete the temporary GeoParquet files from disk
    try:
        os.remove(underserved_path)
        os.remove(sites_path)
    except OSError as e:
        print(f"⚠️ Warning: Could not delete temporary file: {e}")
        
    return result_df