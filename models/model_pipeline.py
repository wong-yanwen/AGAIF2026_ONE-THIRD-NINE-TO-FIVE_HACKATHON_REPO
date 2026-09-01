import shap
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
  # np.floor(), not .astype(int) directly — Python's int() truncates TOWARD
  # ZERO, not toward negative infinity, so +0.2° and -0.2° both truncate to
  # block 0, incorrectly merging locations on opposite sides of the equator
  # into the same spatial block. Doesn't materially affect Malaysia (mostly
  # north of the equator), but matters for scaling to Indonesia.
  df['lat_block'] = np.floor(df['latitude'] / 0.5).astype(int)
  df['lon_block'] = np.floor(df['longitude'] / 0.5).astype(int)
  df['spatial_block'] = (
      df['lat_block'].astype(str) + "_" + df['lon_block'].astype(str)
  )

  # Only fit/validate on explicitly flagged underserved target tiles.
  # (The 'is_underserved_target' flag already checks for tests >= 15 and devices >= 5)
  target_mask = df['is_underserved_target'] == True
  train_pool = df[target_mask].copy()
  # random_state=42 guarantees reproducibility for a FIXED row order, not
  # order-independence — RandomForest bootstrap sampling selects rows by
  # position, so the same seed on the same data in a different row order can
  # pull different actual rows into each tree. main.py trains on ETL order
  # then sorts before export; standalone model_pipeline.py reloads that
  # sorted parquet — different order, same data. Confirmed on real runs:
  # Malaysia matched exactly (both orders happened to coincide), Indonesia
  # and Philippines drifted by ~0.003-0.005 R² between entry points. Sorting
  # by a stable, unique key before training makes both entry points
  # deterministic against each other, not just against themselves.
  train_pool = train_pool.sort_values("site_id", kind="mergesort").copy()
  print(f"📊 Training pool: {len(train_pool):,} / {len(df):,} tiles meet the evidence threshold.")

  # --- SAFE BYPASS FOR SMALL/WEALTHY REGIONS ---
  if len(train_pool) < 10:
    print(f"⚠️ Insufficient training data ({len(train_pool)} target tiles). Bypassing ML model.")
    # Default predictions to actuals so residual becomes 0
    df['cv_predicted_speed'] = df['download_kbps']
    df['predicted_download_kbps'] = df['download_kbps']
    df['top_shap_driver'] = 'Insufficient Data'
    df['top_shap_value'] = 0.0
    # This branch returns before the model exists to compute tree-variance
    # uncertainty from — placeholder so downstream code never hits a
    # missing-column error on a region too small to train on.
    df['prediction_uncertainty_kbps'] = np.nan
    df['prediction_uncertainty_pct'] = np.nan
    return df
  # --------------------------------------------------


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

      model = RandomForestRegressor(
          n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42
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
  final_model = RandomForestRegressor(
      n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42
  )
  # Fit standard target directly
  final_model.fit(train_pool[features], train_pool[target])
  
  # Predict standard scale
  df['predicted_download_kbps'] = final_model.predict(df[features])

  # Per-site model disagreement — the spread across the Random Forest's 100
  # individual trees' predictions for the same input. This is NOT a
  # calibrated confidence interval (no coverage guarantee has been validated
  # — e.g. "the true value falls within this range X% of the time"); it is
  # raw tree-to-tree disagreement. A site where all 100 trees roughly agree
  # is one the model predicts stably; wide disagreement means treat that
  # prediction (and the residual/priority score built from it) more
  # cautiously. Confirmed on real data to carry a meaningful, non-degenerate
  # range (~5%-40% relative disagreement). Describe as "model disagreement:
  # variation across Random Forest trees" in the UI/pitch, not "confidence."
  print("📏 Computing per-site model disagreement (tree-to-tree variance)...")
  X_all = df[features].values
  tree_preds = np.stack([tree.predict(X_all) for tree in final_model.estimators_])
  df['prediction_uncertainty_kbps'] = tree_preds.std(axis=0)
  df['prediction_uncertainty_pct'] = (
      df['prediction_uncertainty_kbps'] / df['predicted_download_kbps'].replace(0, np.nan)
  ).fillna(0.0)

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


def calculate_esg_priority_matrix(df, region=None):
  """Derives Energy Burden Indices and synthesizes the Joint Priority Score.

  Priority score is built from percentile ranks of each factor, not raw
  multiplicative magnitude. A multiplicative raw-score formula lets a single
  extreme value (an unusually large population, or a site the model badly
  mispredicted) set the ceiling for min-max normalization and compress every
  other site toward zero — confirmed directly against real data, where ~53%
  of ranked sites collapsed to priority_score == 0.0 despite being genuine,
  differentiated candidates. Percentile rank has no such floor-vs-outlier
  problem: every site's score reflects its relative position, not its
  distance from one extreme value. This mirrors the scoring approach already
  validated in the dashboard's priority_v2 (see app.py::score()).
  """
  print("🔋 Computing Energy Burden and Carbon Abatement Matrix...")

  # Off-Grid Likelihood Formula — every component is bounded to [0,1] before
  # its weight is applied, so the whole score is genuinely bounded [0,1].
  # Previously the night-radiance term used a raw 1/(x+0.1) with no ceiling —
  # at radiance=0 that term alone equaled 2.5, more than 3x every other
  # weighted term combined, silently dominating the score. Replaced with a
  # linear darkness score referenced against a typical urban radiance level,
  # which decreases smoothly from 1.0 (fully dark) to 0.0 (urban-bright)
  # instead of spiking unbounded near zero.
  # Off-Grid Likelihood — power remoteness and darkness only. Population and
  # road distance were removed: population isn't evidence a tower runs on
  # diesel (it's a downstream impact question, now handled separately in
  # community_impact below), and road distance is a refuelling/logistics
  # cost, not evidence of grid status. Each input now has exactly one job.
  power_remoteness = (df['distance_to_power_m'] / 5000.0).clip(lower=0.0, upper=1.0)
  darkness_score = (1.0 - (df['night_radiance_nw_cm2_sr'] / 10.0)).clip(lower=0.0, upper=1.0)

  df['off_grid_likelihood'] = (
      0.75 * power_remoteness
      + 0.25 * darkness_score
  )

  # Solar Viability — aspect removed. Panels are mounted at whatever azimuth
  # is chosen; the direction the surrounding natural terrain happens to face
  # doesn't dictate panel orientation, and aspect is circular data (0-360°),
  # which the old logic averaged in a way that breaks near the 0/360 wrap.
  # Tree canopy (already extracted, previously unused) is a more directly
  # relevant shading-risk proxy for a real installation.
  solar_resource = df['solar_radiation_mj'].rank(pct=True)
  canopy_suitability = (1.0 - (df['tree_canopy'] / 100.0)).clip(lower=0.0, upper=1.0)
  slope_suitability = (1.0 - (df['slope_degrees'] / 30.0)).clip(lower=0.0, upper=1.0)
  # Higher rainfall gets a modest penalty
  rainfall_suitability = (1.0 - df['rainfall_mm_hr'].rank(pct=True))

  df['solar_viability'] = (
      0.60 * solar_resource
      + 0.20 * canopy_suitability
      + 0.10 * slope_suitability
      + 0.10 * rainfall_suitability
  )

  df['solar_viability'] = df['solar_viability'].clip(lower=0.1)

  # Logistics Difficulty
  df['logistics_difficulty'] = (
      (df['slope_degrees'] / 15.0) * 0.4
      + (df['elevation_m'] / 1000.0) * 0.3
      + (df['distance_to_road_m'] / 1000.0) * 0.3
      + (df['terrain_ruggedness'] / 50.0) * 0.2  # Penalize highly rugged terrain
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

  # TNB Grid-Connected Comparison (workplan Step 3)
  # Even grid power isn't zero-carbon in Malaysia — this shows the diesel
  # baseline against what the same energy output would cost in grid emissions,
  # so the comparison is honest rather than implying grid = zero emissions.
  # Diesel genset efficiency: ~3.5 kWh generated per litre, a conservative
  # planning figure for small (5-30 kVA) remote telecom gensets, which
  # typically run below their most efficient load band (industry range is
  # 3.5-5 kWh/L for modern diesel gensets at optimal load).
  #
  # TNB is Malaysia's national utility — its 0.574 kg CO2e/kWh emission
  # factor is Malaysia-specific and does not represent Indonesia's or the
  # Philippines' grid mix. Gated to Malaysia only; NaN elsewhere until each
  # country's own grid emission factor is added. This does not feed
  # priority_score or any of the four pillars — it's a display-only
  # comparison metric, so this gap doesn't affect rankings, only what's
  # shown for this one number outside Malaysia.
  if region is not None and region.lower() == "malaysia":
    DIESEL_KWH_PER_LITRE = 3.5
    TNB_GRID_EMISSION_FACTOR_KG_PER_KWH = 0.574  # TNB, 2025

    diesel_energy_equivalent_kwh = 13000 * DIESEL_KWH_PER_LITRE
    grid_equivalent_tco2e_yr_full = (
        diesel_energy_equivalent_kwh * TNB_GRID_EMISSION_FACTOR_KG_PER_KWH
    ) / 1000.0

    df['grid_equivalent_tco2e_yr'] = np.where(
        df['off_grid_likelihood'] < 0.10,
        0.0,
        grid_equivalent_tco2e_yr_full * df['off_grid_likelihood'].clip(upper=1.0)
    )
  else:
    df['grid_equivalent_tco2e_yr'] = np.nan

  # Underperformance Residual — kept in its natural kbps units for display
  # and SHAP; only its percentile RANK feeds the priority score below.
  # Positive values mean measured performance is below what was expected;
  # clipped at 0.0, not 1.0 — the old floor of 1.0 meant overperforming sites
  # (negative raw residual) got folded in alongside genuinely marginal ones,
  # and both ended up tied at the same nonzero value.
  df['underperformance_residual'] = (
      df['cv_predicted_speed'] - df['download_kbps']
  ).clip(lower=0.0)

  # Essential Service Weight — continuous proximity score, not a binary cutoff.
  # The old version (50.0 if within 2.5km, else 0.0) collapsed almost every
  # ranked site into the same flat value, since the median site is only ~565m
  # from an amenity — nearly everyone cleared the 2.5km threshold. That threw
  # away the real variance already present in distance_to_amenity_m. This
  # scales linearly instead: closer sites score higher, smoothly, up to the
  # same 2.5km cutoff distance where it reaches zero.
  df['essential_service_weight'] = (
      1.0 - (df['distance_to_amenity_m'] / 2500.0).clip(upper=1.0)
  ) * 50.0

  # ---- Priority score: 4 pillars, matching the problem statement directly ----
  # Diesel dependence 40%, solar suitability 25%, community impact 30%,
  # implementation feasibility 5%.
  #
  # Community impact (population + essential services + connectivity
  # shortfall) is GATED by off-grid likelihood before it can contribute.
  # Population is real evidence of *impact if converted*, but it is NOT
  # evidence a site is actually diesel-dependent — a dense, well-served urban
  # site should not out-rank a genuine rural diesel candidate just because
  # more people live nearby. The gate ramps smoothly from 0% credit at
  # off_grid_likelihood=0.20 to 100% credit at 0.60 (roughly the 37th to 83rd
  # percentile of this run's ranked population) rather than a hard cutoff,
  # so a site just above the threshold doesn't suddenly get the same
  # population credit as one far above it.
  off_grid_n = df['off_grid_likelihood'].rank(pct=True)
  solar_n = df['solar_viability'].rank(pct=True)
  access_ease_n = 1.0 - df['logistics_difficulty'].rank(pct=True)

  population_n = df['population_total'].rank(pct=True)
  service_n = df['essential_service_weight'].rank(pct=True)
  # distance_to_tier1_hub_m was briefly added here as a rurality signal, then
  # reverted: the query behind it (infrastructure_pipeline.py) filters only
  # on Overture's `subtype = 'locality'`, with no `class` or `population`
  # filter — meaning it measures distance to ANY populated place, from
  # megacities down to tiny hamlets, not distance to a genuinely major hub.
  # A remote village is itself a "locality," so its own distance could read
  # near zero — the opposite of what a rurality signal should show. Left
  # unused until the underlying query is redefined with an actual size/class
  # filter and renamed to something honest like distance_to_major_settlement_m.

  # Rank connectivity shortfall the same way the diesel gate is handled:
  # sites genuinely at or above expected speed (residual == 0) get exactly
  # zero shortfall credit, not the average percentile rank of a mass tie at
  # the old floor value — same tie-ranking issue, same fix pattern.
  residual_n = pd.Series(0.0, index=df.index)
  underperforming_mask = df['underperformance_residual'] > 0
  residual_n.loc[underperforming_mask] = (
      df.loc[underperforming_mask, 'underperformance_residual'].rank(pct=True)
  )

  diesel_gate = ((df['off_grid_likelihood'] - 0.20) / 0.40).clip(lower=0.0, upper=1.0)

  raw_community_impact = (
      0.50 * population_n
      + 0.30 * service_n
      + 0.20 * residual_n
  )
  # Rank community NEED before applying the diesel gate, not after. Applying
  # the gate first and then ranking the product is a bug: ~900 sites tie at
  # exactly 0 after gating, and rank(pct=True) gives tied values their AVERAGE
  # rank position, not 0 — confirmed on real data, this was leaking ~5.7
  # priority points into sites the gate was specifically meant to zero out.
  # Ranking first means gate=0 always means exactly 0 contribution.
  raw_community_n = raw_community_impact.rank(pct=True)
  df['community_impact'] = raw_community_n * diesel_gate
  community_n = df['community_impact']

  # Export the intermediate pillar scores themselves, not just the final
  # blended priority_score — so a site-detail panel can show "Diesel 91,
  # Solar 82, Community 77, Feasibility 54" directly, without needing to
  # recompute any of this from raw columns.
  df['off_grid_score_n'] = off_grid_n
  df['solar_score_n'] = solar_n
  df['community_impact_n'] = community_n
  df['access_ease_n'] = access_ease_n
  df['diesel_gate'] = diesel_gate
  df['service_shortfall_n'] = residual_n

  df['priority_score'] = (
      0.40 * off_grid_n
      + 0.25 * solar_n
      + 0.30 * community_n
      + 0.05 * access_ease_n
  ).mul(100).round(2)

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
    region_slug = region.lower().replace(" ", "_")
    input_file = os.path.join(data_dir, f"jendela_phase2_esg_matrix_{region_slug}.parquet")

  if not os.path.exists(input_file):
    print(f"❌ Input file not found: {input_file}")
    return

  print(f"📥 Loading dataset from {input_file}...")
  df = pd.read_parquet(input_file)
  print(f"Loaded {len(df):,} grid tile records.")

  # ==========================================================
  # DATA INTEGRITY: Domain-Aware Imputation
  # ==========================================================
  # A. Missing antennas = 0 (Do not use median for missing towers)
  antenna_cols = ['antenna_count', 'antennas_4G', 'antennas_3G', 'antennas_2G', 'antennas_5G']
  for col in antenna_cols:
      if col in df.columns:
          df[col] = df[col].fillna(0)
          
  # B. Missing infrastructure
  # Preserve missingness flags already created by main.py.
  # Only generate them if the input file does not already contain them.
  if 'power_distance_missing' not in df.columns:
      df['power_distance_missing'] = df['distance_to_power_m'].isna()
  if 'road_distance_missing' not in df.columns:
      df['road_distance_missing'] = df['distance_to_road_m'].isna()
  if 'amenity_distance_missing' not in df.columns:
      df['amenity_distance_missing'] = df['distance_to_amenity_m'].isna()

  distance_cols = ['distance_to_power_m', 'distance_to_road_m', 'distance_to_amenity_m', 'distance_to_nearest_tower']
  for col in distance_cols:
      if col in df.columns:
          df[col] = df[col].fillna(10000.0)

  # C. Missing natural geography (safe to use median)
  geo_cols = ['elevation_m', 'slope_degrees', 'terrain_ruggedness', 'night_radiance_nw_cm2_sr', 'solar_radiation_mj', 'rainfall_mm_hr', 'tree_canopy']
  for col in geo_cols:
      if col in df.columns:
          df[col] = df[col].fillna(df[col].median())

  # D. Handle population explicitly (NaN in WorldPop means zero/ocean/unpopulated)
  df['population_total'] = df['population_total'].fillna(0)

  # E. Now calculate engineered features safely directly on the clean DataFrame
  df['congestion_proxy'] = df['population_total'] / (df['antenna_count'] + 1)
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

  # 3b. Check for valid targets BEFORE splitting or scoring
  valid_mask = governed_df['confidence_tier'] == 'Sufficient Evidence - Ranked Screening Approved'
  
  if not valid_mask.any():
      print("⚠️ ZERO valid target sites found in region. Injecting neutral schema.")
      df = governed_df.copy()
      
      empty_cols = [
          'top_shap_driver', 'top_shap_value', 'off_grid_likelihood', 
          'solar_viability', 'logistics_difficulty', 'indicative_abatement_tco2e_yr', 
          'indicative_opex_saving_usd', 'grid_equivalent_tco2e_yr',
          'underperformance_residual', 'essential_service_weight', 'community_impact',
          'off_grid_score_n', 'solar_score_n', 'community_impact_n',
          'access_ease_n', 'diesel_gate', 'service_shortfall_n',
          'prediction_uncertainty_kbps', 'prediction_uncertainty_pct',
          'priority_score', 'people_connected_per_tonne_co2',
          'power_distance_missing', 'road_distance_missing', 'amenity_distance_missing'
      ]
      for col in empty_cols:
          df[col] = np.nan
          
      df['inference_status'] = 'No Action Required - Performing Above Baseline'
      df['field_survey_triggered'] = False
      df['national_rank'] = 0
      
  else:
      valid_sites = governed_df[valid_mask].copy()
      invalid_sites = governed_df[~valid_mask].copy()
      
      scored_valid = calculate_esg_priority_matrix(valid_sites, region=region)
      invalid_sites['priority_score'] = 0.0
      invalid_sites['community_impact'] = 0.0
      invalid_sites['people_connected_per_tonne_co2'] = 0.0
      df = pd.concat([scored_valid, invalid_sites], ignore_index=True)
      df.loc[~df['confidence_tier'].str.contains('Sufficient'), ['top_shap_driver', 'top_shap_value']] = None
      df['inference_status'] = np.select(
          [
              df['confidence_tier'] == 'Sufficient Evidence - Ranked Screening Approved',
              df['confidence_tier'] == 'Sufficient Evidence - Performing Above Baseline (Excluded)',
              df['confidence_tier'] == 'Thin Evidence - Masked from Prioritization'
          ],
          [
              'Candidate Site - Validation Required',
              'Excluded - Performing Above Baseline',
              'Insufficient Evidence - Additional Data Required'
          ],
          default='Excluded - No Data'
      )
      df['field_survey_triggered'] = df['confidence_tier'].apply(
          lambda x: True if 'Ranked Screening Approved' in x else False
      )
      df = df.sort_values(
          by=['field_survey_triggered', 'priority_score'], ascending=[False, False]
      ).reset_index(drop=True)
      # Only candidate sites get a national rank — excluded and thin-evidence
      # sites previously received sequential ranks too (e.g. 2,390, 2,391...),
      # which reads as if they'd been meaningfully ranked against candidates.
      df['national_rank'] = 0
      rank_mask = df['field_survey_triggered']
      df.loc[rank_mask, 'national_rank'] = np.arange(1, rank_mask.sum() + 1)
      
  # 5. Export Output for Dashboard Lead (Teammate #3)
  output_path = os.path.join(data_dir, "jendela_phase2_esg_scored.parquet")
  df.to_parquet(output_path, index=False)
  print(f"✅ Successfully exported scored priority matrix to '{output_path}'!")

 # Quick Top-5 Summary
  if not valid_mask.any():
      print("\n🏆 No valid sites to rank for this region.")
  else:
      print("\n🏆 Top 5 Priority Sites Preview:")
      cols_to_show = [
          'national_rank',
          'priority_score',
          'people_connected_per_tonne_co2',
          'confidence_tier',
      ]
      valid_cols = [c for c in cols_to_show if c in df.columns]
      print(df[valid_cols].head(5).to_string(index=False))


if __name__ == "__main__":
  import sys
  region_arg = sys.argv[1] if len(sys.argv) > 1 else "malaysia"
  run_pipeline(region=region_arg)