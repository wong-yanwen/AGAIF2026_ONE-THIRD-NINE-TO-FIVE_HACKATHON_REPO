import pandas as pd
import geopandas as gpd
import time
import os
import ee
from data_pipeline.config import BUFFER_RADIUS_M, GEE_SCALE_M, GEE_DRIVE_FOLDER, CHUNK_SIZE

def init_gee():
    try:
        ee.Initialize()
        print("✅ Google Earth Engine successfully initialized with existing credentials.")
    except Exception:
        print("🔑 Credentials not found or expired. Requesting authentication...")
        ee.Authenticate()
        ee.Initialize()
        print("✅ Google Earth Engine successfully authenticated and initialized.")


def build_esg_composite():
    viirs = ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V22').filterDate('2022-01-01', '2023-01-01').select('average').median().rename('night_radiance')
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filterDate('2023-01-01', '2024-01-01').select('surface_solar_radiation_downwards_sum').mean().rename('solar_radiation')
    worldpop = ee.ImageCollection('WorldPop/GP/100m/pop').filterDate('2020-01-01', '2021-01-01').select('population').mean().rename('population')
    nasadem = ee.Image('NASA/NASADEM_HGT/001')
    elevation = nasadem.select('elevation').rename('elevation')
    slope = ee.Terrain.slope(elevation).rename('slope')

    # Extract terrain aspect (compass direction for solar panel viability)
    aspect = ee.Terrain.aspect(elevation).rename('aspect')

    # Terrain Ruggedness Index via local elevation standard deviation (3x3 pixel window)
    ruggedness = elevation.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(radius=3, units='pixels')
    ).rename('terrain_ruggedness')

    rainfall = ee.ImageCollection('NASA/GPM_L3/IMERG_V07').filterDate('2023-01-01', '2024-01-01').select('precipitation').mean().rename('rainfall')

    # NEW: Extract 2000-era baseline tree canopy percentage (0-100)
    tree_canopy = ee.Image('UMD/hansen/global_forest_change_2025_v1_13').select('treecover2000').rename('tree_canopy')

    return ee.Image.cat([viirs, era5, worldpop, elevation, slope, aspect, ruggedness, rainfall, tree_canopy])

def extract_gee_data(df_sites, country_name):
    init_gee()
    composite = build_esg_composite()
    
    print(f"📦 Packaging {len(df_sites)} site geometries for server-side processing...")
    
    # 1. Chunk the dataframe to bypass the 10MB payload limit 
    tasks = []
    task_names = []
    
    # Generate a single timestamp so all chunks group nicely in Google Drive
    run_timestamp = int(time.time())
    
    for i in range(0, len(df_sites), CHUNK_SIZE):
        chunk_df = df_sites.iloc[i:i+CHUNK_SIZE]
        part_num = (i // CHUNK_SIZE) + 1
        
        features = [
            ee.Feature(
                ee.Geometry.Point(float(row['longitude']), float(row['latitude'])), 
                {'site_id': str(row['site_id'])}  # use str, Preserves leading zeros 
            )
            for _, row in chunk_df.iterrows()
        ]
        
        fc = ee.FeatureCollection(features).map(lambda f: f.buffer(BUFFER_RADIUS_M))
        combined_reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.sum(), sharedInputs=True)
        
        reduced_fc = composite.reduceRegions(
            collection=fc,
            reducer=combined_reducer,
            scale=GEE_SCALE_M,
            tileScale=4
        )
        
        # Add 'pt1', 'pt2', etc., to avoid filename collisions
        task_name = f"GEE_Extract_{country_name}_{run_timestamp}_pt{part_num}"
        task = ee.batch.Export.table.toDrive(
            collection=reduced_fc,
            description=task_name,
            folder=GEE_DRIVE_FOLDER,
            fileFormat='CSV'
        )
        
        print(f"🚀 Dispatching chunk {part_num} to GEE servers ({len(chunk_df)} sites)...")
        task.start()
        tasks.append(task)
        task_names.append(task_name)
        
    # 2. Automated Polling Loop for ALL chunks
    print(f"⏳ Waiting for {len(tasks)} batch tasks to complete on Google servers. This will take a while...")
    
    completed_tasks = set()
    while len(completed_tasks) < len(tasks):
        for idx, task in enumerate(tasks):
            if idx in completed_tasks:
                continue
                
            state = task.status().get('state', 'UNKNOWN')
            if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
                completed_tasks.add(idx)
                if state != 'COMPLETED':
                    error_msg = task.status().get('error_message', 'Unknown error')
                    raise RuntimeError(f"❌ GEE Task {task_names[idx]} Failed: {error_msg}")
                else:
                    print(f"✅ Task {task_names[idx]} Finished on Server! ({len(completed_tasks)}/{len(tasks)})")
                    
        if len(completed_tasks) < len(tasks):
            print(f"⏳ {len(completed_tasks)}/{len(tasks)} tasks finished. Sleeping for 2 minutes...")
            time.sleep(120) 
            
    # Extra time to sync gdrive
    print("📥 Waiting 15 seconds for Google Drive to sync local files...")
    time.sleep(15)
    
    # 3. Robust Fallback & Merge Loop
    all_dfs = []
    for task_name in task_names:
        expected_path = f"G:/My Drive/{GEE_DRIVE_FOLDER}/{task_name}.csv"
        
        if not os.path.exists(expected_path):
            print(f"\n⚠️ Auto-sync warning: Could not detect file for chunk: {expected_path}")
            while True:
                user_input = input(f"📥 Paste the local path to {task_name}.csv (or press Enter to retry auto-detect): ").strip()
                
                if not user_input:
                    if os.path.exists(expected_path):
                        resolved_path = expected_path
                        break
                    else:
                        print(f"❌ Still not found at default path. Please download {task_name}.csv manually.")
                        continue
                        
                user_input = user_input.replace('"', '').replace("'", "")
                if os.path.exists(user_input):
                    resolved_path = user_input
                    break
                else:
                    print(f"❌ File not found at provided path. Please try again.")
        else:
            resolved_path = expected_path

        print(f"📖 Loading environment metrics from chunk: {resolved_path}")
        # FIX QUADKEY bug: Tell Pandas it is a string before it even loads
        df_chunk = pd.read_csv(resolved_path, dtype={'site_id': str})
        all_dfs.append(df_chunk)
        
    # Combine all chunks into one massive dataframe to pass downstream
    df_env_full = pd.concat(all_dfs, ignore_index=True)
    return df_env_full


def clean_and_merge(df_ookla, df_env):
    # Check if df_env is empty to prevent errors if a chunk failed silently
    if df_env.empty:
        raise ValueError("❌ df_env is completely empty. No features were returned from GEE.")

    #=================================================================================
    # OLD CODE - TO BE DELETED
    #=================================================================================
    
    # Standardize 'site_id' columns to string type to resolve the merge conflict
    #df_ookla['site_id'] = df_ookla['site_id'].astype(str)
    #df_env['site_id'] = df_env['site_id'].astype(float).astype(int).astype(str)
    #=================================================================================
    
    # Both are clean strings now.
    df_ookla['site_id'] = df_ookla['site_id'].astype(str)
    df_env['site_id'] = df_env['site_id'].astype(str)

    master_df = pd.merge(df_ookla, df_env, on='site_id', how='inner')


    # Explicitly rename using the appended suffix mapping generated by GEE's combined reducer
    master_df = master_df.rename(columns={
        'elevation_mean': 'elevation_m',
        'slope_mean': 'slope_degrees',
        'aspect_mean': 'aspect_degrees',
        'terrain_ruggedness_mean': 'terrain_ruggedness',
        'rainfall_mean': 'rainfall_mm_hr',
        'night_radiance_mean': 'night_radiance_nw_cm2_sr', 
        'population_sum': 'population_total',
        'solar_radiation_mean': 'solar_radiation_raw',
        'tree_canopy_mean': 'tree_canopy'  
    })
    
    if 'solar_radiation_raw' in master_df.columns:
        master_df['solar_radiation_mj'] = master_df['solar_radiation_raw'] / 1000000.0
        
    columns_to_drop = [
        'elevation_sum', 'population_mean', 'night_radiance_sum', 
        'rainfall_sum', 'slope_sum', 'aspect_sum',
        'solar_radiation_sum', 'solar_radiation_raw', 'terrain_ruggedness_sum',
        'tree_canopy_sum'
    ]
    
    master_df = master_df.drop(columns=columns_to_drop, errors='ignore')
    
    # Fill Coastal Zeros from ERA5
    master_df['solar_radiation_mj'] = master_df['solar_radiation_mj'].fillna(0)

    # Convert back to a GeoDataFrame using the existing coordinates
    master_df = gpd.GeoDataFrame(
        master_df,
        geometry=gpd.points_from_xy(master_df['longitude'], master_df['latitude']),
        crs='EPSG:4326'
    )
    return master_df