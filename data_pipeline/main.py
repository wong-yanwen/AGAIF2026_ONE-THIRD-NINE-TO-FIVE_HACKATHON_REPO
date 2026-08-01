import os
import pandas as pd
import geopandas as gpd

from ookla_pipeline import get_underserved_sites, apply_stratified_ookla_screening
from gee_pipeline import extract_gee_data, clean_and_merge
from vector_pipeline import process_vector_proximity
from infrastructure_pipeline import process_candidate_site_clusters, engineering_osm_proximity_features
from config import OPENCELLID_PATH, OUTPUT_FILE_PATH, ASEAN_MCCS, ASEAN_BOUNDS

# Import from the newly created Algorithm Lead module
from model_pipeline import apply_spatial_blocking_cv, calculate_esg_priority_matrix, apply_governance_confidence_mask

def main():
    region = "Malaysia"
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    bounds = ASEAN_BOUNDS.get(region)

    # ==========================================================
    # DATA ENGINEERING: INFRASTRUCTURE INGESTION & BOUNDING
    # ==========================================================
    print(f"🏗️ Loading raw antenna data from {OPENCELLID_PATH}...")
    df_raw_cell_global = pd.read_csv(OPENCELLID_PATH, compression='gzip')
    df_raw_cell_global = df_raw_cell_global.rename(columns={'lon': 'longitude', 'lat': 'latitude'})
    
    # MEMORY FIX: Strictly bound the dataset before passing to DBSCAN
    df_raw_cell = df_raw_cell_global[
        (df_raw_cell_global['mcc'].isin(ASEAN_MCCS)) &
        (df_raw_cell_global['longitude'] >= bounds[0]) & (df_raw_cell_global['longitude'] <= bounds[2]) &
        (df_raw_cell_global['latitude'] >= bounds[1]) & (df_raw_cell_global['latitude'] <= bounds[3])
    ].copy()
    print(f"✅ Extracted {len(df_raw_cell)} antenna nodes within {region} boundaries.")
    
    site_nodes = process_candidate_site_clusters(df_raw_cell)

    # -----🛑 BYPASS LIVE OSM SCRAPING FOR TESTING 🛑-------------------------------------
    print("🛠️ INJECTING MOCK OSM DISTANCES FOR FAST TESTING...")
    site_nodes_final = site_nodes.copy()
    site_nodes_final['distance_to_power_m'] = 1500.0  # Fake distance to power lines
    site_nodes_final['distance_to_road_m'] = 300.0    # Fake distance to roads
    site_nodes_final['distance_to_amenity_m'] = 4500.0 # Fake distance to schools/clinics
    # --------------------------------------------------------------------------------------
    # ACTUAL 
    #site_nodes_final = engineering_osm_proximity_features(site_nodes, region)

    # ==========================================================
    # DATA ENGINEERING: GEOSPATIAL MERGES & STRATIFICATION
    # ==========================================================
    df_underserved = get_underserved_sites(region)

    if not df_underserved.empty:
        df_proximity = process_vector_proximity(df_underserved, site_nodes_final, region)
        df_env = extract_gee_data(df_proximity)
        master_matrix = clean_and_merge(df_proximity, df_env)  
        
        master_matrix = apply_stratified_ookla_screening(master_matrix)
        
        exogenous_features = [
            'population_total', 'elevation_m', 'slope_degrees', 
            'distance_to_power_m', 'distance_to_road_m', 'antenna_count'
        ]
        
        # Fill missing values to prevent algorithm crash
        master_matrix[exogenous_features] = master_matrix[exogenous_features].fillna(master_matrix[exogenous_features].median())

        # ==========================================================
        # ALGORITHM HANDOFF: MODELING, SCORING & GOVERNANCE
        # ==========================================================
        target_variable = 'download_kbps'
        modeled_matrix = apply_spatial_blocking_cv(master_matrix, exogenous_features, target_variable)
        scored_matrix = calculate_esg_priority_matrix(modeled_matrix)
        final_matrix = apply_governance_confidence_mask(scored_matrix)

        final_matrix['inference_status'] = 'Candidate Site - Validation Required'
        final_matrix['field_survey_triggered'] = final_matrix['confidence_tier'].apply(
            lambda x: True if 'Sufficient' in x else False
        )

        final_gdf = gpd.GeoDataFrame(
            final_matrix, 
            geometry=gpd.points_from_xy(final_matrix['longitude'], final_matrix['latitude']), 
            crs="EPSG:4326"
        )
        final_gdf.to_parquet(OUTPUT_FILE_PATH, compression='snappy', index=False)
        print(f"\n🚀 Complete ESG Pipeline Operationalized! Matrix saved to: {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()