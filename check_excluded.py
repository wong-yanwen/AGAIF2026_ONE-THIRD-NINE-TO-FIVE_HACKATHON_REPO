import pandas as pd
df = pd.read_parquet("data/jendela_phase2_esg_scored.parquet")
ranked = df[df['confidence_tier'] == 'Sufficient Evidence - Ranked Screening Approved']
print("Rows with priority_score exactly 0.0:", (ranked['priority_score'] == 0.0).sum(), "/", len(ranked))
print()
print(ranked['priority_score'].quantile([0.1, 0.25, 0.5, 0.6, 0.75, 0.9]))