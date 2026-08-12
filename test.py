import pandas as pd

file_path="data/jendela_phase2_esg_matrix_singapore.parquet"

print(f"📥 Inspecting: {file_path}\n")
df = pd.read_parquet(file_path)

# 1. Print a clean list of every single column name
print("📌 ALL COLUMNS:")
print(df.columns.tolist())
print("-" * 50)

# 2. Print data types and non-null counts (Crucial for UI debugging)
print("📌 DATA TYPES & NULL COUNTS:")
print(df.info())
print("-" * 50)

# 3. Look at the actual data for the first 3 rows
print("📌 DATA PREVIEW (First 3 rows):")
print(df.head(3).to_string())


evidence_mask = (df['tests'] >= 15) & (df['devices'] >= 5)
train_pool = df[evidence_mask].copy()

print("Training pool size:", len(train_pool))
print()
print("download_kbps distribution WITHIN the training pool, by stratum:")
print(train_pool.groupby('demographic_stratum')['download_kbps'].describe())
print()
print("Overall training pool download_kbps spread:")
print(train_pool['download_kbps'].describe())



