import os
import time
import pandas as pd
import geopandas as gpd
from datetime import datetime


from data_pipeline.ookla_pipeline import get_underserved_sites, apply_stratified_ookla_screening
from data_pipeline.gee_pipeline import extract_gee_data, clean_and_merge
from data_pipeline.vector_pipeline import process_vector_proximity
from data_pipeline.infrastructure_pipeline import process_candidate_site_clusters, engineering_osm_proximity_features
from data_pipeline.config import OPENCELLID_PATH, OUTPUT_FILE_PATH, ASEAN_MCCS, ASEAN_BOUNDS, DATA_DIR, MODEL_FEATURES
# Import from the newly created Algorithm Lead module
from models.model_pipeline import apply_spatial_blocking_cv, calculate_esg_priority_matrix, apply_governance_confidence_mask

def main():

    # ==========================================================
    # START TIMER
    # ==========================================================
    start_time = time.time()
    print(f"\n▶️ Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
        # --------------------------

        exogenous_features = MODEL_FEATURES
        # Fill missing values to prevent algorithm crash
        master_matrix[exogenous_features] = master_matrix[exogenous_features].fillna(master_matrix[exogenous_features].median())

        # ==========================================================
        # ALGORITHM HANDOFF: MODELING, SCORING & GOVERNANCE
        # ==========================================================
        target_variable = 'download_kbps'
        modeled_matrix = apply_spatial_blocking_cv(master_matrix, exogenous_features, target_variable)

        # 1. Apply governance mask FIRST
        governed_matrix = apply_governance_confidence_mask(modeled_matrix)

        # 2. Split the dataset so outliers don't ruin the Min-Max scale
        valid_mask = governed_matrix['confidence_tier'].str.contains('Sufficient')
        valid_sites = governed_matrix[valid_mask].copy()
        invalid_sites = governed_matrix[~valid_mask].copy()

        # 3. Calculate 0-100 scores ONLY on valid sites
        scored_valid = calculate_esg_priority_matrix(valid_sites)

        # 4. Zero out the junk sites so they don't get prioritized
        invalid_sites['priority_score'] = 0.0
        invalid_sites['people_connected_per_tonne_co2'] = 0.0

        # 5. Recombine the dataset
        final_matrix = pd.concat([scored_valid, invalid_sites], ignore_index=True)
        
        final_matrix.loc[~final_matrix['confidence_tier'].str.contains('Sufficient'), ['top_shap_driver', 'top_shap_value']] = None
        final_matrix['inference_status'] = 'Candidate Site - Validation Required'
        final_matrix['field_survey_triggered'] = final_matrix['confidence_tier'].apply(
            lambda x: True if 'Sufficient' in x else False
        )
        final_matrix = final_matrix.sort_values(
            by=['field_survey_triggered', 'priority_score'], ascending=[False, False]
        ).reset_index(drop=True)
        final_matrix['national_rank'] = final_matrix.index + 1
        final_gdf = gpd.GeoDataFrame(
            final_matrix, 
            geometry=gpd.points_from_xy(final_matrix['longitude'], final_matrix['latitude']), 
            crs="EPSG:4326"
        )

        # CHECK the final geodataframe
        print ("\n")
        print (final_gdf.head())

        country_suffix = region.lower().replace(" ", "_") # handles 'kuala_lumpur' or 'malaysia' safely
        dynamic_parquet_path = os.path.join(DATA_DIR, f"jendela_phase2_esg_matrix_{country_suffix}.parquet")
        dynamic_html_path = os.path.join(DATA_DIR, f"jendela_phase2_esg_matrix_{country_suffix}.html")

        # Remove the accidental duplicate save block from your code and use the new dynamic path:
        final_gdf.to_parquet(dynamic_parquet_path, compression='snappy', index=False)
        print(f"\n🚀 Complete ESG Pipeline Operationalized! Matrix saved to: {dynamic_parquet_path}")
        

        # ==========================================================
        # VISUALIZATION: GENERATE INTERACTIVE HTML MAP
        # ==========================================================
        try:
            print("\n🗺️ Generating interactive priority map...")
            import webbrowser
            
            # Create an interactive map, coloring the points by their Priority Score
            m = final_gdf.explore(
                column="priority_score",
                cmap="YlOrRd",          # Yellow-Orange-Red color scale
                marker_kwds={"radius": 6}, # Size of the dots
                tooltip=["site_id", "priority_score", "confidence_tier", "population_total"],
                name="ESG Priority Sites"
            )
            
            m.save(dynamic_html_path)
            
            # Automatically open the map in your default web browser
            webbrowser.open('file://' + os.path.realpath(dynamic_html_path))
            print(f"✅ Map saved and opened in your browser: {dynamic_html_path}")
            
        except ImportError:
            print("\n⚠️ Note: To generate the interactive map, you need 'folium' and 'mapclassify'.")
            print("Run this in your terminal: pip install folium mapclassify")

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
    main()