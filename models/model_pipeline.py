import shap
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def apply_spatial_blocking_cv(df, features, target, n_splits=5):
  """Executes Spatially Blocked Cross-Validation to eliminate spatial autocorrelation data leakage.

  Trains and validates only on tiles meeting the Ookla evidence threshold —
  Thin/No-Data tiles have too few tests to serve as reliable training labels.
  Every row in df still receives a prediction; only the fitting/scoring pool
  is restricted.
  """
  print("🎯 Initializing Spatially Blocked Cross-Validation...")

  # Only fit/validate on tiles with reliable Ookla labels (evidence threshold).
  evidence_mask = (df['tests'] >= 15) & (df['devices'] >= 5) & (df['is_underserved_target'] == True)
  train_pool = df[evidence_mask].copy()
  print(f"📊 Training pool: {len(train_pool):,} / {len(df):,} tiles meet the evidence threshold.")

  # Generate 0.5-degree spatial blocks (on the reliable subset only)
  train_pool['lat_block'] = (train_pool['latitude'] / 0.5).astype(int)
  train_pool['lon_block'] = (train_pool['longitude'] / 0.5).astype(int)
  train_pool['spatial_block'] = (
      train_pool['lat_block'].astype(str) + "_" + train_pool['lon_block'].astype(str)
  )

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
          n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
      )
      model.fit(train_data[features], train_data[target])

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
      n_estimators=100, max_depth=4, random_state=42
  )
  final_model.fit(train_pool[features], train_pool[target])
  df['predicted_download_kbps'] = final_model.predict(df[features])

  # ---- SHAP explainability ----
  print("🔍 Computing SHAP feature contributions...")
  explainer = shap.TreeExplainer(final_model)
  shap_values = explainer.shap_values(df[features], check_additivity=False)

  top_feature_idx = np.abs(shap_values).argmax(axis=1)
  df['top_shap_driver'] = [features[i] for i in top_feature_idx]
  df['top_shap_value'] = shap_values[np.arange(len(df)), top_feature_idx]

  # Fallback mapping if CV was bypassed
  if df['cv_predicted_speed'].isna().all():
    df['cv_predicted_speed'] = df['predicted_download_kbps']

  return df


def calculate_esg_priority_matrix(df):
  """Derives Energy Burden Indices and synthesizes the Joint Priority Score

  (normalized to 0–100 for human readability).
  """
  print("🔋 Computing Energy Burden and Carbon Abatement Matrix...")

  # Fallbacks for missing vector distances
  df['distance_to_power_m'] = df['distance_to_power_m'].fillna(
      df['distance_to_power_m'].median()
  )
  df['distance_to_road_m'] = df['distance_to_road_m'].fillna(
      df['distance_to_road_m'].median()
  )
  df['distance_to_amenity_m'] = df['distance_to_amenity_m'].fillna(10000.0)

  # Off-Grid Likelihood
  df['off_grid_likelihood'] = (
      (df['distance_to_power_m'] / 5000.0) * 0.5
      + (df['distance_to_road_m'] / 2000.0) * 0.3
      + (1.0 / (df['night_radiance_nw_cm2_sr'] + 0.1)) * 0.2
  )

  # Solar Viability
  df['solar_viability'] = (
      (df['solar_radiation_mj'] / df['solar_radiation_mj'].max()) * 0.6
      + (1.0 / (df['slope_degrees'] + 1.0)) * 0.4
  ) - (df['rainfall_mm_hr'] * 0.1)
  df['solar_viability'] = df['solar_viability'].clip(lower=0.1)

  # Logistics Difficulty
  df['logistics_difficulty'] = (
      (df['slope_degrees'] / 15.0) * 0.4
      + (df['elevation_m'] / 1000.0) * 0.3
      + (df['distance_to_road_m'] / 1000.0) * 0.3
  ).clip(lower=0.1)

 
  # GSMA Decarbonization Benchmarks (13,000 L/yr = 34.2 tCO2e/yr baseline).
  # Scale the abatement credit by off-grid likelihood so sites more likely
  # to actually be diesel-dependent get closer to the full GSMA-cited
  # abatement, rather than crediting every site identically.
  df['indicative_abatement_tco2e_yr'] = 34.2 * 0.65 * df['off_grid_likelihood'].clip(upper=1.0)

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
  numerator = (
      (
          df['indicative_abatement_tco2e_yr']
          * df['off_grid_likelihood']
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
  df['people_connected_per_tonne_co2'] = (
      df['population_total'] / df['indicative_abatement_tco2e_yr']
  )

  return df


def apply_governance_confidence_mask(df):
  """Implements the three-tier data governance visibility mask."""
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

  # 2. Define Features & Target
  features = [
      'population_total',
      'elevation_m',
      'slope_degrees',
      'distance_to_road_m',
      'distance_to_power_m',
      'night_radiance_nw_cm2_sr',
      'solar_radiation_mj',
      'rainfall_mm_hr',
  ]
  features = [f for f in features if f in df.columns]
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
  valid_mask = governed_df['confidence_tier'].str.contains('Sufficient')
  valid_sites = governed_df[valid_mask].copy()
  invalid_sites = governed_df[~valid_mask].copy()

  # 3c. Calculate 0-100 scores ONLY on valid sites
  scored_valid = calculate_esg_priority_matrix(valid_sites)

  # 3d. Zero out the junk sites so they don't get prioritized
  invalid_sites['priority_score'] = 0.0
  invalid_sites['people_connected_per_tonne_co2'] = 0.0

  # 3e. Recombine the dataset
  df = pd.concat([scored_valid, invalid_sites], ignore_index=True)

  df['inference_status'] = 'Candidate Site - Validation Required'
  df['field_survey_triggered'] = df['confidence_tier'].apply(
      lambda x: True if 'Sufficient' in x else False
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


