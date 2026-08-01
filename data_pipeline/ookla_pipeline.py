import duckdb
import numpy as np
import geopandas as gpd
import pandas as pd
from config import ASEAN_BOUNDS, OOKLA_DATA_URL, GEOJSON_PATH

def get_underserved_sites(region_name):
    print(f"🛠️ INJECTING MOCK OOKLA DATA FOR FAST TESTING...")
    
    # Generate 200 perfectly formatted fake tiles around Central Malaysia
    np.random.seed(42)
    mock_df = pd.DataFrame({
        # Change this line to generate purely numeric string IDs
        'site_id': [str(10000 + i) for i in range(200)], 
        'longitude': np.random.uniform(100.0, 104.0, 200),
        'latitude': np.random.uniform(2.0, 5.0, 200),
        'download_kbps': np.random.uniform(500, 60000, 200),
        'tests': np.random.randint(1, 50, 200),
        'devices': np.random.randint(1, 15, 200)
    })
    
    # Convert standard DataFrame to a GeoDataFrame so DuckDB has a 'geometry' column
    mock_gdf = gpd.GeoDataFrame(
        mock_df, 
        geometry=gpd.points_from_xy(mock_df['longitude'], mock_df['latitude']),
        crs="EPSG:4326"
    )
    
    return mock_gdf

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