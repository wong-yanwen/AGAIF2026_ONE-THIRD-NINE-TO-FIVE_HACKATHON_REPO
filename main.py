import os
import time
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime

from data_pipeline.ookla_pipeline import get_underserved_sites, apply_stratified_ookla_screening
from data_pipeline.gee_pipeline import extract_gee_data, clean_and_merge
from data_pipeline.vector_pipeline import process_vector_proximity
from data_pipeline.infrastructure_pipeline import process_candidate_site_clusters, engineering_osm_proximity_features
from data_pipeline.config import OPENCELLID_PATH, OUTPUT_FILE_PATH, ASEAN_BOUNDS, DATA_DIR, MODEL_FEATURES, ASEAN_MCC_BY_REGION
from models.model_pipeline import apply_spatial_blocking_cv, calculate_esg_priority_matrix, apply_governance_confidence_mask

def main(region_name="Malaysia"):

    # ==========================================================
    # START TIMER
    # ==========================================================
    start_time = time.time()
    print(f"\n▶️ Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # --- SMART CASING FIX ---
    region = next((k for k in ASEAN_BOUNDS.keys() if k.lower() == region_name.lower()), region_name.title())
    
    bounds = ASEAN_BOUNDS.get(region)
    if bounds is None:
        raise ValueError(f"❌ Region '{region}' not found in ASEAN_BOUNDS! Check your spelling.")

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    bounds = ASEAN_BOUNDS.get(region)

    # ==========================================================
    # DATA ENGINEERING: INFRASTRUCTURE INGESTION & BOUNDING
    # ==========================================================
    print(f"🏗️ Loading raw antenna data from {OPENCELLID_PATH}...")
    df_raw_cell_global = pd.read_csv(OPENCELLID_PATH, compression='gzip')
    df_raw_cell_global = df_raw_cell_global.rename(columns={'lon': 'longitude', 'lat': 'latitude'})

    # MEMORY & LEAKAGE FIX: Strictly bound the dataset by exact Country MCC, not all ASEAN
    region_mcc = ASEAN_MCC_BY_REGION[region]
    
    df_raw_cell = df_raw_cell_global[
        (df_raw_cell_global['mcc'] == region_mcc) &
        (df_raw_cell_global['longitude'] >= bounds[0]) & (df_raw_cell_global['longitude'] <= bounds[2]) &
        (df_raw_cell_global['latitude'] >= bounds[1]) & (df_raw_cell_global['latitude'] <= bounds[3])
    ].copy()
    print(f"✅ Extracted {len(df_raw_cell)} {region} antenna nodes (MCC {region_mcc}).")
    
    site_nodes = process_candidate_site_clusters(df_raw_cell)

    # -----🛑 BYPASS LIVE OSM SCRAPING FOR TESTING 🛑-------------------------------------
    #print("🛠️ INJECTING MOCK OSM DISTANCES FOR FAST TESTING...")
    #site_nodes_final = site_nodes.copy()
    #site_nodes_final['distance_to_power_m'] = 1500.0  # Fake distance to power lines
    #site_nodes_final['distance_to_road_m'] = 300.0    # Fake distance to roads
    #site_nodes_final['distance_to_amenity_m'] = 4500.0 # Fake distance to schools/clinics
    # --------------------------------------------------------------------------------------
    
    # ACTUAL 
    site_nodes_final = engineering_osm_proximity_features(site_nodes, region)

    # ==========================================================
    # DATA ENGINEERING: GEOSPATIAL MERGES & STRATIFICATION
    # ==========================================================
    df_underserved = get_underserved_sites(region)

    if not df_underserved.empty:
        df_proximity = process_vector_proximity(df_underserved, site_nodes_final, region)
        df_env = extract_gee_data(df_proximity, country_name=region)
        master_matrix = clean_and_merge(df_proximity, df_env)  
        
        master_matrix = apply_stratified_ookla_screening(master_matrix)

        # --- GUARDRAIL TO PREVENT CRASH ---
        if master_matrix.empty:
            print(f"\n🛑 Insufficient valid data in {region} to train the model.")
            print("Exiting pipeline gracefully to prevent algorithm crash.\n")
            return
        # -----------------------------------

        # ==========================================================
        # DATA INTEGRITY: Domain-Aware Imputation
        # ==========================================================
        # 1. Missing antennas = 0 (Do not use median for missing towers)
        antenna_cols = ['antenna_count', 'antennas_4G', 'antennas_3G', 'antennas_2G', 'antennas_5G']
        for col in antenna_cols:
            if col in master_matrix.columns:
                master_matrix[col] = master_matrix[col].fillna(0)
                
        # 2. Missing infrastructure = 10,000m (Extremely remote)
        # Preserve original missingness BEFORE imputation — flagging after
        # fillna() always reads False, since the NaN is already gone by then.
        if 'distance_to_power_m' in master_matrix.columns:
            master_matrix['power_distance_missing'] = master_matrix['distance_to_power_m'].isna()
        if 'distance_to_road_m' in master_matrix.columns:
            master_matrix['road_distance_missing'] = master_matrix['distance_to_road_m'].isna()
        if 'distance_to_amenity_m' in master_matrix.columns:
            master_matrix['amenity_distance_missing'] = master_matrix['distance_to_amenity_m'].isna()

        distance_cols = ['distance_to_power_m', 'distance_to_road_m', 'distance_to_amenity_m', 'distance_to_nearest_tower']
        for col in distance_cols:
            if col in master_matrix.columns:
                master_matrix[col] = master_matrix[col].fillna(10000.0)

        # 3. Missing natural geography (safe to use median)
        geo_cols = ['elevation_m', 'slope_degrees', 'terrain_ruggedness', 'night_radiance_nw_cm2_sr', 'solar_radiation_mj', 'rainfall_mm_hr', 'tree_canopy']
        for col in geo_cols:
            if col in master_matrix.columns:
                master_matrix[col] = master_matrix[col].fillna(master_matrix[col].median())

        # 4. Handle population explicitly (NaN in WorldPop means zero/ocean/unpopulated)
        master_matrix['population_total'] = master_matrix['population_total'].fillna(0)

        # 5. Now calculate engineered features safely directly on the clean DataFrame
        master_matrix['congestion_proxy'] = master_matrix['population_total'] / (master_matrix['antenna_count'] + 1)
        master_matrix['pct_4g_5g'] = (master_matrix['antennas_4G'] + master_matrix['antennas_5G']) / (master_matrix['antenna_count'] + 1)

        missing_features = [f for f in MODEL_FEATURES if f not in master_matrix.columns]
        if missing_features:
            print(f"⚠️ Missing model features for this region: {missing_features}")
        exogenous_features = [f for f in MODEL_FEATURES if f in master_matrix.columns]


        # ==========================================================
        # ALGORITHM HANDOFF: MODELING, SCORING & GOVERNANCE
        # ==========================================================
        target_variable = 'download_kbps'
        modeled_matrix = apply_spatial_blocking_cv(master_matrix, exogenous_features, target_variable)

        # 1. Apply governance mask
        governed_matrix = apply_governance_confidence_mask(modeled_matrix)

        # 2. Check for valid targets BEFORE splitting or scoring
        valid_mask = governed_matrix['confidence_tier'] == 'Sufficient Evidence - Ranked Screening Approved'
        
        if not valid_mask.any():
            print("⚠️ ZERO valid target sites found in region. Injecting neutral schema for UI lead.")
            final_matrix = governed_matrix.copy()
            
            # Manually inject the schema your UI lead expects from calculate_esg_priority_matrix
            expected_cols = [
                'off_grid_likelihood', 'solar_viability', 'logistics_difficulty', 
                'indicative_abatement_tco2e_yr', 'indicative_opex_saving_usd',
                'grid_equivalent_tco2e_yr', 'underperformance_residual',
                'essential_service_weight', 'community_impact',
                'off_grid_score_n', 'solar_score_n', 'community_impact_n',
                'access_ease_n', 'diesel_gate', 'service_shortfall_n',
                'prediction_uncertainty_kbps', 'prediction_uncertainty_pct',
                'priority_score', 'people_connected_per_tonne_co2',
                'power_distance_missing', 'road_distance_missing', 'amenity_distance_missing'
            ]
            for col in expected_cols:
                final_matrix[col] = np.nan
                
            final_matrix['inference_status'] = 'No Action Required - Performing Above Baseline'
            final_matrix['field_survey_triggered'] = False
            final_matrix['national_rank'] = 0
            
        else:
            # Standard scoring for countries with actual targets (Malaysia, Philippines, etc.)
            valid_sites = governed_matrix[valid_mask].copy()
            invalid_sites = governed_matrix[~valid_mask].copy()
            
            scored_valid = calculate_esg_priority_matrix(valid_sites, region=region)
            invalid_sites['priority_score'] = 0.0
            invalid_sites['community_impact'] = 0.0
            invalid_sites['people_connected_per_tonne_co2'] = 0.0
            final_matrix = pd.concat([scored_valid, invalid_sites], ignore_index=True)
            final_matrix.loc[~final_matrix['confidence_tier'].str.contains('Sufficient'), ['top_shap_driver', 'top_shap_value']] = None
            final_matrix['inference_status'] = np.select(
                [
                    final_matrix['confidence_tier'] == 'Sufficient Evidence - Ranked Screening Approved',
                    final_matrix['confidence_tier'] == 'Sufficient Evidence - Performing Above Baseline (Excluded)',
                    final_matrix['confidence_tier'] == 'Thin Evidence - Masked from Prioritization'
                ],
                [
                    'Candidate Site - Validation Required',
                    'Excluded - Performing Above Baseline',
                    'Insufficient Evidence - Additional Data Required'
                ],
                default='Excluded - No Data'
            )
            final_matrix['field_survey_triggered'] = final_matrix['confidence_tier'].apply(
                lambda x: True if 'Ranked Screening Approved' in x else False
            )
            final_matrix = final_matrix.sort_values(
                by=['field_survey_triggered', 'priority_score'], ascending=[False, False]
            ).reset_index(drop=True)
            # Only candidate sites get a national rank — see model_pipeline.py
            # for why excluded/thin-evidence sites are set to 0 instead.
            final_matrix['national_rank'] = 0
            rank_mask = final_matrix['field_survey_triggered']
            final_matrix.loc[rank_mask, 'national_rank'] = np.arange(1, rank_mask.sum() + 1)

        # ==========================================================
        # EXPORT FOR UI LEAD
        # ==========================================================
        final_gdf = gpd.GeoDataFrame(
            final_matrix,
            geometry=gpd.points_from_xy(final_matrix['longitude'], final_matrix['latitude']),
            crs="EPSG:4326"
        )

        country_suffix = region.lower().replace(" ", "_")
        dynamic_parquet_path = os.path.join(DATA_DIR, f"jendela_phase2_esg_matrix_{country_suffix}.parquet")
        
        final_gdf.to_parquet(dynamic_parquet_path, compression='snappy', index=False)
        print(f"\n🚀 Complete ESG Pipeline Operationalized! Matrix saved to: {dynamic_parquet_path}")
        
        # ==========================================================
        # LOCAL VISUALIZATION (DATA ENGINEER ONLY)
        # ==========================================================
        dynamic_html_path = os.path.join(DATA_DIR, f"jendela_phase2_esg_matrix_{country_suffix}.html")
        
        if not valid_mask.any():
            print("\n🗺️ Bypassing Folium map generation (0 ranked sites). Pipeline complete.")
        else:
            try:
                print("\n🗺️ Generating interactive priority map...")
                import webbrowser
                m = final_gdf.explore(
                    column="priority_score",
                    cmap="YlOrRd",
                    marker_kwds={"radius": 6},
                    tooltip=["site_id", "priority_score", "confidence_tier", "population_total"],
                    name="ESG Priority Sites"
                )
                m.save(dynamic_html_path)
                webbrowser.open('file://' + os.path.realpath(dynamic_html_path))
                print(f"✅ Map saved and opened in your browser: {dynamic_html_path}")
            except ImportError:
                print("\n⚠️ Note: To generate the interactive map, you need 'folium' and 'mapclassify'.")
        # ==========================================================
        # END TIMER
        # ==========================================================
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        
        # Format elapsed time into HH:MM:SS
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n⏹️ Pipeline finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Total Execution Time: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s\n")
    

if __name__ == "__main__":
    # 1. If you run `python main.py` with no arguments, default to Malaysia
    if len(sys.argv) == 1:
        target_region = "Malaysia"
    
    # 2. If you pass arguments (like `python main.py brunei darussalam`), join them together
    else:
        target_region = " ".join(sys.argv[1:])
        
    main(region_name=target_region)