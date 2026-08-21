import shap
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from data_pipeline.config import MODEL_FEATURES

def apply_spatial_blocking_cv(df, features, target, n_splits=5):
  """Executes Spatially Blocked Cross-Validation to eliminate spatial autocorrelation data leakage.

  Trains and validates only on tiles meeting the Ookla evidence threshold —
  Thin/No-Data tiles have too few tests to serve as reliable training labels.
  Every row in df still receives a prediction; only the fitting/scoring pool
  is restricted.
  """
  print("🎯 Initializing Spatially Blocked Cross-Validation...")

  # Generate 0.5-degree spatial blocks first, on the full dataframe,
  # so these columns survive into the exported output.
  df['lat_block'] = (df['latitude'] / 0.5).astype(int)
  df['lon_block'] = (df['longitude'] / 0.5).astype(int)
  df['spatial_block'] = (
      df['lat_block'].astype(str) + "_" + df['lon_block'].astype(str)
  )

  # Only fit/validate on explicitly flagged underserved target tiles.
  # (The 'is_underserved_target' flag already checks for tests >= 15 and devices >= 5)
  target_mask = df['is_underserved_target'] == True
  train_pool = df[target_mask].copy()
  print(f"📊 Training pool: {len(train_pool):,} / {len(df):,} tiles meet the evidence threshold.")

  unique_blocks = np.array(sorted(train_pool['spatial_block'].unique()))
  num_blocks = len(unique_blocks)

  df['cv_predicted_speed'] = np.nan

  # --- SCALE-ADAPTIVE GUARDRAIL ---
  if num_blocks < 2:
    print(
        f"⚠️ Region too small for spatial CV (only {num_blocks} block found)."
        " Bypassing CV step."
    )
  else:
    actual_splits = min(n_splits, num_blocks)
    print(
        f"🧩 Found {num_blocks} unique spatial blocks. Running"
        f" {actual_splits}-fold CV..."
    )

    kf = KFold(n_splits=actual_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(unique_blocks):
      train_blocks = unique_blocks[train_idx]
      val_blocks = unique_blocks[val_idx]

      train_data = train_pool[train_pool['spatial_block'].isin(train_blocks)]
      val_data = train_pool[train_pool['spatial_block'].isin(val_blocks)]

      if train_data.empty or val_data.empty:
        continue

      model = GradientBoostingRegressor(
          n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
      )
      

      # Fit standard target directly
      model.fit(train_data[features], train_data[target])

      # Predict standard scale
      preds = model.predict(val_data[features])
      df.loc[val_data.index, 'cv_predicted_speed'] = preds

    clean_mask = df['cv_predicted_speed'].notna()
    if clean_mask.sum() > 0:
      global_r2 = r2_score(
          df.loc[clean_mask, target], df.loc[clean_mask, 'cv_predicted_speed']
      )
      print(
          "📉 Spatially Blocked CV Complete. Out-of-Block R²:"
          f" {global_r2:.3f}"
      )

      # Stratified R² — show whether the model generalizes evenly,
      # or whether the pooled number is hiding stratum-specific weakness
      print("\n📊 Out-of-Block R² by demographic stratum:")
      for stratum in df.loc[clean_mask, 'demographic_stratum'].unique():
        stratum_mask = clean_mask & (df['demographic_stratum'] == stratum)
        if stratum_mask.sum() > 1:
          stratum_r2 = r2_score(
              df.loc[stratum_mask, target], df.loc[stratum_mask, 'cv_predicted_speed']
          )
          print(f"   {stratum}: R² = {stratum_r2:.3f}  (n={stratum_mask.sum()})")

  # Train ultimate model ONLY on reliable-evidence tiles, but predict for ALL rows
  print("🧠 Training final production model on evidence-threshold subset...")
  final_model = GradientBoostingRegressor(
      n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
  )
  # Fit standard target directly
  final_model.fit(train_pool[features], train_pool[target])
  
  # Predict standard scale
  df['predicted_download_kbps'] = final_model.predict(df[features])

  # Rows outside the training pool never get a CV-validated prediction;
  # fall back to the production model's prediction so residuals/scoring
  # don't silently propagate NaN into the shortlist.
  missing_cv = df['cv_predicted_speed'].isna()
  df.loc[missing_cv, 'cv_predicted_speed'] = df.loc[missing_cv, 'predicted_download_kbps']

  # ---- SHAP explainability ----
  print("🔍 Computing SHAP feature contributions...")
  explainer = shap.TreeExplainer(final_model)
  shap_values = explainer.shap_values(df[features], check_additivity=False)

  top_feature_idx = np.abs(shap_values).argmax(axis=1)
  df['top_shap_driver'] = [features[i] for i in top_feature_idx]
  df['top_shap_value'] = shap_values[np.arange(len(df)), top_feature_idx]

  return df


def calculate_esg_priority_matrix(df):
  """Derives Energy Burden Indices and synthesizes the Joint Priority Score
  (normalized to 0–100 for human readability).
  """
  print("🔋 Computing Energy Burden and Carbon Abatement Matrix...")

  # Fallbacks for missing vector distances
  # A missing value means the infrastructure is far outside the local bounding box.
  df['distance_to_power_m'] = df['distance_to_power_m'].fillna(10000.0)
  df['distance_to_road_m'] = df['distance_to_road_m'].fillna(10000.0)
  df['distance_to_amenity_m'] = df['distance_to_amenity_m'].fillna(10000.0)

  # Updated Off-Grid Likelihood Formula 
  df['off_grid_likelihood'] = (
      (df['distance_to_power_m'] / 5000.0).clip(upper=1.0) * 0.35
      + (df['distance_to_road_m'] / 2000.0).clip(upper=1.0) * 0.25
      + (1.0 / (df['night_radiance_nw_cm2_sr'] + 0.1)) * 0.25
      + (df['population_total'] / 1000.0).clip(upper=1.0) * 0.15 
  )

  # Dynamic Solar Aspect Logic (Scalable across the equator)
  # Northern Hemisphere (>0) prefers South (180). Southern Hemisphere (<0) prefers North (0).
  optimal_aspect = np.where(df['latitude'] > 0, 180.0, 0.0)
  
  # Calculate shortest angular distance (0 to 180 degrees off from optimal)
  aspect_diff = np.abs(df['aspect_degrees'] - optimal_aspect)
  aspect_diff = np.minimum(aspect_diff, 360.0 - aspect_diff)
  
  # Convert to a 0-1 multiplier (1 = perfect orientation, 0 = completely backward)
  aspect_score = 1.0 - (aspect_diff / 180.0)

  # Solar Viability
  df['solar_viability'] = (
      (df['solar_radiation_mj'] / df['solar_radiation_mj'].max()) * 0.5
      + (1.0 / (df['slope_degrees'] + 1.0)) * 0.3
      + aspect_score * 0.2 
  ) - (df['rainfall_mm_hr'] * 0.1)
  
  df['solar_viability'] = df['solar_viability'].clip(lower=0.1)

  # Logistics Difficulty
  df['logistics_difficulty'] = (
      (df['slope_degrees'] / 15.0) * 0.4
      + (df['elevation_m'] / 1000.0) * 0.3
      + (df['distance_to_road_m'] / 1000.0) * 0.3
      + (df['terrain_ruggedness'] / 50.0) * 0.2  # NEW: Penalize highly rugged terrain
  ).clip(lower=0.1)

  # CO2 Threshold Gate
  # GSMA Decarbonization Benchmarks (13,000 L/yr = 34.2 tCO2e/yr baseline).
  # We multiply by likelihood here so the dashboard displays the EXPECTED actual savings
  expected_abatement = 34.2 * 0.65 * df['off_grid_likelihood'].clip(upper=1.0)
  
  df['indicative_abatement_tco2e_yr'] = np.where(
      df['off_grid_likelihood'] < 0.10,
      0.0,
      expected_abatement
  )

  # OPEX Savings for the Dashboard Lead (US$17,000/yr off-grid baseline)
  opex_credit = 17000 * df['off_grid_likelihood'].clip(upper=1.0)
  df['indicative_opex_saving_usd'] = np.where(
      df['off_grid_likelihood'] < 0.10,
      0.0,
      opex_credit
  )

  # Underperformance Residual
  df['underperformance_residual'] = (
      df['cv_predicted_speed'] - df['download_kbps']
  )
  df['underperformance_residual'] = df['underperformance_residual'].apply(
      lambda x: max(x, 1.0)
  )

  # Essential Service Weight (Boosts score if within 2.5km of a school/clinic)
  df['essential_service_weight'] = df['distance_to_amenity_m'].apply(
      lambda d: 50.0 if d <= 2500 else 0.0
  )

  # Raw Priority Score Equation
  # Note: off_grid_likelihood is removed from this equation because it is already 
  # mathematically baked into indicative_abatement_tco2e_yr
  numerator = (
      (
          df['indicative_abatement_tco2e_yr']
          * df['solar_viability']
      )
      * (df['population_total'] + df['essential_service_weight'])
      * df['underperformance_residual']
  )

  raw_score = numerator / df['logistics_difficulty']

  # 0–100 Min-Max Normalization for human readability on Streamlit
  min_val = raw_score.min()
  max_val = raw_score.max()

  if max_val != min_val:
    df['priority_score'] = (
        ((raw_score - min_val) / (max_val - min_val)) * 100
    ).round(2)
  else:
    df['priority_score'] = 50.0

  # Core Pitch Metric: People connected per tonne of CO2 avoided
  # Handled safely to prevent `inf` corruption in downstream datasets
  df['people_connected_per_tonne_co2'] = np.where(
      df['indicative_abatement_tco2e_yr'] > 0,
      df['population_total'] / df['indicative_abatement_tco2e_yr'],
      0.0
  )

  return df

def apply_governance_confidence_mask(df):
    """Implements the three-tier data governance visibility mask."""
    print("🛡️ Deploying Data Governance Integrity Mask...")

    def assign_mask(row):
        # Added distance check to enforce the "has OpenCelliD records" rule
        has_tower_nearby = row.get('distance_to_nearest_tower', 99999) <= 2750

        if row['tests'] >= 15 and row['devices'] >= 5 and has_tower_nearby:
            if row.get('is_underserved_target', False):
                return 'Sufficient Evidence - Ranked Screening Approved'
            else:
                return 'Sufficient Evidence - Performing Above Baseline (Excluded)'
        elif row['tests'] > 0:
            return 'Thin Evidence - Masked from Prioritization'
        else:
            return 'No Data - Excluded'

    df['confidence_tier'] = df.apply(assign_mask, axis=1)
    return df

def run_pipeline(input_file=None, data_dir="data", region="malaysia"):
  # 1. Resolve input file: explicit path > region-based default > error
  if input_file is None:
    input_file = os.path.join(data_dir, f"jendela_phase2_esg_matrix_{region}.parquet")

  if not os.path.exists(input_file):
    print(f"❌ Input file not found: {input_file}")
    return

  print(f"📥 Loading dataset from {input_file}...")
  df = pd.read_parquet(input_file)
  print(f"Loaded {len(df):,} grid tile records.")

  # ==========================================================
  # DATA INTEGRITY: Domain-Aware Imputation
  # ==========================================================
  # 1. Missing antennas = 0 (Do not use median for missing towers)
  antenna_cols = ['antenna_count', 'antennas_4G', 'antennas_3G', 'antennas_2G', 'antennas_5G']
  for col in antenna_cols:
      if col in df.columns:
          df[col] = df[col].fillna(0)
          
  # 2. Missing infrastructure = 10,000m (Extremely remote)
  distance_cols = ['distance_to_power_m', 'distance_to_road_m', 'distance_to_amenity_m', 'distance_to_nearest_tower']
  for col in distance_cols:
      if col in df.columns:
          df[col] = df[col].fillna(10000.0)

  # 3. Missing natural geography (safe to use median)
  geo_cols = ['elevation_m', 'slope_degrees', 'terrain_ruggedness', 'night_radiance_nw_cm2_sr', 'solar_radiation_mj', 'rainfall_mm_hr']
  for col in geo_cols:
      if col in df.columns:
          df[col] = df[col].fillna(df[col].median())

  # 4. Now calculate engineered features safely directly on the clean DataFrame
  df['congestion_proxy'] = df['population_total'].fillna(0) / (df['antenna_count'] + 1)
  df['pct_4g_5g'] = (df['antennas_4G'] + df['antennas_5G']) / (df['antenna_count'] + 1)

# 2. Define Features & Target
  features = [f for f in MODEL_FEATURES if f in df.columns]
  target = 'download_kbps'

  # Fallback if target column is named differently
  if target not in df.columns:
    for alt in ['download_speed', 'speed_down', 'ookla_download_speed']:
      if alt in df.columns:
        target = alt
        break

  # 3. Execute Pipeline Functions
  df = apply_spatial_blocking_cv(df, features=features, target=target)

  # 3a. Apply governance mask FIRST
  governed_df = apply_governance_confidence_mask(df)

  # 3b. Split the dataset so outliers don't ruin the Min-Max scale
  valid_mask = governed_df['confidence_tier'].str.contains('Ranked Screening Approved')
  valid_sites = governed_df[valid_mask].copy()
  invalid_sites = governed_df[~valid_mask].copy()

  # 3c. Calculate 0-100 scores ONLY on valid sites
  scored_valid = calculate_esg_priority_matrix(valid_sites)

  # ===================================================================
  # Removed for data integrity
  # 3d. Zero out the junk sites so they don't get prioritized
  # invalid_sites['priority_score'] = 0.0
  # invalid_sites['people_connected_per_tonne_co2'] = 0.0
  # ===================================================================

  # 3e. Recombine the dataset
  df = pd.concat([scored_valid, invalid_sites], ignore_index=True)

  df['inference_status'] = 'Candidate Site - Validation Required'
  df['field_survey_triggered'] = df['confidence_tier'].apply(
      lambda x: True if 'Ranked Screening Approved' in x else False
  )

  # 4. Sort by field survey trigger first, then priority score
  df = df.sort_values(
      by=['field_survey_triggered', 'priority_score'],
      ascending=[False, False]
  ).reset_index(drop=True)
  df['national_rank'] = df.index + 1

  # 5. Export Output for Dashboard Lead (Teammate #3)
  output_path = os.path.join(data_dir, "jendela_phase2_esg_scored.parquet")
  df.to_parquet(output_path, index=False)
  print(f"✅ Successfully exported scored priority matrix to '{output_path}'!")

  # Quick Top-5 Summary
  print("\n🏆 Top 5 Priority Sites Preview:")
  cols_to_show = [
      c
      for c in [
          'national_rank',
          'priority_score',
          'people_connected_per_tonne_co2',
          'confidence_tier',
      ]
      if c in df.columns
  ]
  print(df[cols_to_show].head(5).to_string(index=False))


if __name__ == "__main__":
  import sys
  region_arg = sys.argv[1] if len(sys.argv) > 1 else "malaysia"
  run_pipeline(region=region_arg)