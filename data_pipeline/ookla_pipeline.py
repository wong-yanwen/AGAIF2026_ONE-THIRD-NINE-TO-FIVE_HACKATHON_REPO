import duckdb
import geopandas as gpd
import pandas as pd
from data_pipeline.config import ASEAN_BOUNDS, OOKLA_DATA_URL, GEOJSON_PATH

def get_underserved_sites(region_name):
    bounds = ASEAN_BOUNDS.get(region_name)
    if not bounds:
        raise ValueError(f"Region {region_name} not found in config.")
       
    min_lon, min_lat, max_lon, max_lat = bounds
    print(f"🌍 Querying Ookla data from Ookla Parquet for {region_name}...")
   
    query = f"""
        SELECT
            quadkey AS site_id,
            tile_x AS longitude,
            tile_y AS latitude,
            avg_d_kbps AS download_kbps,
            tests,
            devices
        FROM read_parquet('{OOKLA_DATA_URL}')
        WHERE tile_x >= {min_lon} AND tile_x <= {max_lon}
          AND tile_y >= {min_lat} AND tile_y <= {max_lat}
    """
   
    raw_df = duckdb.query(query).df()
   
    if len(raw_df) == 0:
        print("⚠️ No data found for this bounding box.")
        return raw_df
   
    print(f"🗺️ Performing spatial clip using {region_name} boundaries...")


    # 1. Convert Pandas DataFrame into a spatially-aware GeoDataFrame
    gdf_sites = gpd.GeoDataFrame(
        raw_df,
        geometry=gpd.points_from_xy(raw_df.longitude, raw_df.latitude),
        crs="EPSG:4326"
    )


    # 2. Load custom GeoJSON and isolate the target country's shape dynamically
    asean_gdf = gpd.read_file(GEOJSON_PATH)
    
    # Enforce uniform CRS before the spatial join
    if asean_gdf.crs and asean_gdf.crs != "EPSG:4326":
        print(f"🔄 Aligning Shape CRS from {asean_gdf.crs} to EPSG:4326")
        asean_gdf = asean_gdf.to_crs("EPSG:4326")
        
    target_shape = asean_gdf[asean_gdf['Country'] == region_name]

    # 3. Perform spatial join OR bypass if it's a sub-region test
    if target_shape.empty:
        print(f"⚠️ '{region_name}' is a sub-region. Bypassing national shape clip and using strict bounding box.")
        df_clean = gdf_sites # filtered geographically
    else:
        clipped_gdf = gpd.sjoin(gdf_sites, target_shape, predicate='within')
        df_clean = clipped_gdf.drop(
            columns=['index_right', 'OBJECTID', 'Country', 'Flag'],
            errors='ignore'
        )

    print(f"✅ Spatial filter complete! New site count: {len(df_clean)}")
    
    return df_clean

# ==========================================
#  STRATIFIED OOKLA TILE SCREENING
# ==========================================
def apply_stratified_ookla_screening(gdf_ookla):
    """
    Applies within-stratum 20th percentile filters using population density tiers.
    after GEE extraction layers have been fused in main.py.
    """
    print("📉 Stratifying performance thresholds by regional population profile...")
    
    # Enforce strict data validity guardrails
    valid_tiles = gdf_ookla[(gdf_ookla['tests'] >= 15) & (gdf_ookla['devices'] >= 5)].copy()

    if len(valid_tiles) == 0:
        print("⚠️ No tiles met the baseline evidence threshold criteria (>=15 tests, >=5 devices).")
        return valid_tiles

    
    # Establish population boundaries across the datasets using GEE output
    cutoffs = valid_tiles['population_total'].quantile([0.33, 0.66]).values
    
    def assign_tier(val):
        if val <= cutoffs[0]: return 'rural'
        if val <= cutoffs[1]: return 'peri-urban'
        return 'urban'
        
    valid_tiles['demographic_stratum'] = valid_tiles['population_total'].apply(assign_tier)
    
    # Screen tiles falling under their respective stratum's Q1 line
    q1_thresholds = valid_tiles.groupby('demographic_stratum')['download_kbps'].quantile(0.20).to_dict()
    
    underserved_mask = valid_tiles.apply(
        lambda r: r['download_kbps'] < q1_thresholds[r['demographic_stratum']], axis=1
    )
    filtered_results = valid_tiles[underserved_mask].copy()
    print(f"🎯 Target Acquired: {len(filtered_results):,} verified underserved target tiles.")
    
    return filtered_results