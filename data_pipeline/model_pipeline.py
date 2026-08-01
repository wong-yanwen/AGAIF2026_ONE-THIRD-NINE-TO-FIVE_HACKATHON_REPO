import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

def apply_spatial_blocking_cv(df, features, target, n_splits=5):
    """
    Executes Spatially Blocked Cross-Validation to eliminate spatial autocorrelation data leakage.
    Owned by Algorithm Lead: Model selection, hyperparameters, and feature arrays are managed here.
    """
    print("🎯 Initializing Spatially Blocked Cross-Validation...")
    
    # Generate 0.5-degree spatial blocks
    df['lat_block'] = (df['latitude'] / 0.5).astype(int)
    df['lon_block'] = (df['longitude'] / 0.5).astype(int)
    df['spatial_block'] = df['lat_block'].astype(str) + "_" + df['lon_block'].astype(str)
    
    unique_blocks = df['spatial_block'].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    df['cv_predicted_speed'] = np.nan
    
    for train_idx, val_idx in kf.split(unique_blocks):
        train_blocks = unique_blocks[train_idx]
        val_blocks = unique_blocks[val_idx]
        
        train_data = df[df['spatial_block'].isin(train_blocks)]
        val_data = df[df['spatial_block'].isin(val_blocks)]
        
        if train_data.empty or val_data.empty:
            continue
            
        # Algorithm Lead can swap algorithms or tune hyperparameters here
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(train_data[features], train_data[target])
        
        preds = model.predict(val_data[features])
        df.loc[val_data.index, 'cv_predicted_speed'] = preds
        
    clean_mask = df['cv_predicted_speed'].notna()
    global_r2 = r2_score(df.loc[clean_mask, target], df.loc[clean_mask, 'cv_predicted_speed'])
    print(f"📉 Spatially Blocked CV Complete. Out-of-Block R²: {global_r2:.3f}")
    
    # Train ultimate model on all data to extract production residuals
    final_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    final_model.fit(df[features], df[target])
    df['predicted_download_kbps'] = final_model.predict(df[features])
    
    return df

def calculate_esg_priority_matrix(df):
    """
    Derives Energy Burden Indices and synthesizes the Joint Priority Score.
    """
    print("🔋 Computing Energy Burden and Carbon Abatement Matrix...")
    
    # Fallbacks for missing vector distances
    df['distance_to_power_m'] = df['distance_to_power_m'].fillna(df['distance_to_power_m'].median())
    df['distance_to_road_m'] = df['distance_to_road_m'].fillna(df['distance_to_road_m'].median())
    df['distance_to_amenity_m'] = df['distance_to_amenity_m'].fillna(10000.0)
    
    # Off-Grid Likelihood
    df['off_grid_likelihood'] = (
        (df['distance_to_power_m'] / 5000.0) * 0.5 + 
        (df['distance_to_road_m'] / 2000.0) * 0.3 + 
        (1.0 / (df['night_radiance_nw_cm2_sr'] + 0.1)) * 0.2
    )
    
    # Solar Viability
    df['solar_viability'] = (
        (df['solar_radiation_mj'] / df['solar_radiation_mj'].max()) * 0.6 + 
        (1.0 / (df['slope_degrees'] + 1.0)) * 0.4
    ) - (df['rainfall_mm_hr'] * 0.1)
    df['solar_viability'] = df['solar_viability'].clip(lower=0.1)
    
    # Logistics Difficulty
    df['logistics_difficulty'] = (
        (df['slope_degrees'] / 15.0) * 0.4 + 
        (df['elevation_m'] / 1000.0) * 0.3 + 
        (df['distance_to_road_m'] / 1000.0) * 0.3
    ).clip(lower=0.1)
    
    # GSMA Decarbonization Benchmarks (13,000 L/yr = 34.2 tCO2e/yr, 65% abatement)
    df['indicative_abatement_tco2e_yr'] = 34.2 * 0.65 
    
    # Underperformance Residual
    df['underperformance_residual'] = df['predicted_download_kbps'] - df['download_kbps']
    df['underperformance_residual'] = df['underperformance_residual'].apply(lambda x: max(x, 1.0))
    
    # Essential Service Weight (Boosts score if within 2.5km of a school/clinic)
    df['essential_service_weight'] = df['distance_to_amenity_m'].apply(lambda d: 50.0 if d <= 2500 else 0.0)
    
    # Priority Equation
    numerator = (df['indicative_abatement_tco2e_yr'] * df['off_grid_likelihood'] * df['solar_viability']) * \
                (df['population_total'] + df['essential_service_weight']) * \
                df['underperformance_residual']
                
    df['priority_score'] = numerator / df['logistics_difficulty']
    
    # Core Pitch Metric
    df['people_connected_per_tonne_co2'] = df['population_total'] / df['indicative_abatement_tco2e_yr']
    
    return df

def apply_governance_confidence_mask(df):
    """
    Implements the three-tier data governance visibility mask.
    """
    print("🛡️ Deploying Data Governance Integrity Mask...")
    def assign_mask(row):
        if row['tests'] >= 15 and row['devices'] >= 5:
            return 'Sufficient Evidence - Ranked Screening Approved'
        elif row['tests'] > 0:
            return 'Thin Evidence - Masked from Prioritization'
        else:
            return 'No Data - Excluded'
            
    df['confidence_tier'] = df.apply(assign_mask, axis=1)
    return df