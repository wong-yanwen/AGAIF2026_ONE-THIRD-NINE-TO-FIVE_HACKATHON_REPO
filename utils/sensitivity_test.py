import pandas as pd

# Load your final produced dataset
df = pd.read_parquet("data/jendela_phase2_esg_matrix_malaysia.parquet")

# Filter only valid sites (the ones eligible for ranking)
df = df[df['confidence_tier'] == 'Sufficient Evidence - Ranked Screening Approved'].copy()

def score_with_weights(data, w_slope, w_elev, w_road, w_rugged):
    # Recalculate Logistics Difficulty with NEW weights
    temp_logistics = (
        (data['slope_degrees'] / 15.0) * w_slope
        + (data['elevation_m'] / 1000.0) * w_elev
        + (data['distance_to_road_m'] / 1000.0) * w_road
        + (data['terrain_ruggedness'] / 50.0) * w_rugged
    ).clip(lower=0.1)
    
    # Recalculate the raw score (likelihood is already baked into abatement)
    numerator = (
        (data['indicative_abatement_tco2e_yr'] * data['solar_viability'])
        * (data['population_total'] + data['essential_service_weight'])
        * data['underperformance_residual']
    )
    
    raw_score = numerator / temp_logistics
    return raw_score

# 1. Baseline Ranking (What is currently in cuurent pipeline: 40% slope, 30% elev, 30% road, 20% ruggedness)
df['baseline_score'] = score_with_weights(df, 0.4, 0.3, 0.3, 0.2)
top_20_baseline = set(df.nlargest(20, 'baseline_score')['site_id'])

# 2. Scenario B (Heavy penalty on Elevation & Ruggedness: 20% slope, 50% elev, 20% road, 30% ruggedness)
df['scenario_b_score'] = score_with_weights(df, 0.2, 0.5, 0.2, 0.3)
top_20_scenario_b = set(df.nlargest(20, 'scenario_b_score')['site_id'])

# 3. Compare them
overlap = top_20_baseline.intersection(top_20_scenario_b)
overlap_percentage = (len(overlap) / 20) * 100

print(f"📊 Sensitivity Analysis Results:")
print(f"Sites remaining in Top 20 despite drastic weight changes: {len(overlap)}/20 ({overlap_percentage}%)")

if overlap_percentage >= 80:
    print("✅ Defense Ready: 'Our model is highly stable. Altering the logistics weights significantly only shifts the ranking by a few sites.'")
else:
    print("⚠️ The ranking is sensitive to your weights. You will need to justify why you picked 0.4/0.3/0.3/0.2.")