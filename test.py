import pandas as pd

df = pd.read_parquet("data/jendela_phase2_esg_matrix_malaysia.parquet")

evidence_mask = (df['tests'] >= 15) & (df['devices'] >= 5)
train_pool = df[evidence_mask].copy()

print("Training pool size:", len(train_pool))
print()
print("download_kbps distribution WITHIN the training pool, by stratum:")
print(train_pool.groupby('demographic_stratum')['download_kbps'].describe())
print()
print("Overall training pool download_kbps spread:")
print(train_pool['download_kbps'].describe())