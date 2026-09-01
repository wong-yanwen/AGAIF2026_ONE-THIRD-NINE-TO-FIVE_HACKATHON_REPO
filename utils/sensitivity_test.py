import pandas as pd

# Load your final produced dataset
df = pd.read_parquet("data/jendela_phase2_esg_matrix_MALAYSIA.parquet")

# Filter only valid sites (the ones eligible for ranking)
df = df[df['confidence_tier'] == 'Sufficient Evidence - Ranked Screening Approved'].copy()

def score_macro_pillars(data, w_offgrid, w_solar, w_community, w_access):
    # This perfectly mirrors the V2 Additive Percentile math from model_pipeline.py
    score = (
        w_offgrid * data['off_grid_score_n']
        + w_solar * data['solar_score_n']
        + w_community * data['community_impact_n']
        + w_access * data['access_ease_n']
    )
    return score

# 1. Baseline Ranking (What is currently in pipeline: 40% Diesel, 25% Solar, 30% Community, 5% Access)
df['baseline_score'] = score_macro_pillars(df, 0.40, 0.25, 0.30, 0.05)
top_20_baseline = set(df.nlargest(20, 'baseline_score')['site_id'])

# 2. Scenario B: The "Pro-Social" Shift (Community jumps to 50%, Diesel drops to 20%)
df['scenario_social_score'] = score_macro_pillars(df, 0.20, 0.25, 0.50, 0.05)
top_20_social = set(df.nlargest(20, 'scenario_social_score')['site_id'])

# 3. Compare them
overlap = top_20_baseline.intersection(top_20_social)
overlap_percentage = (len(overlap) / 20) * 100

print(f"📊 Macro-Pillar Sensitivity Analysis Results:")
print(f"Sites remaining in Top 20 despite drastic shift from Environmental to Social focus: {len(overlap)}/20 ({overlap_percentage}%)")

if overlap_percentage >= 70:
    print("✅ Defense Ready: 'Our model is highly stable. Even if JENDELA shifts from a decarbonization focus to a pure rural-inclusion focus, the core priority list remains intact.'")
else:
    print("⚠️ The ranking is highly sensitive to policy priorities. Your dashboard slider is the ultimate defense.")