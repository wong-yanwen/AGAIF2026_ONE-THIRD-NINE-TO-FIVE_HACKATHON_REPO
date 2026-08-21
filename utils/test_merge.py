import pandas as pd
from data_pipeline.ookla_pipeline import get_underserved_sites

# 1. Load the Ookla data (takes 5 seconds)
print("Loading Ookla...")
df_ookla = get_underserved_sites("Malaysia")
df_ookla['site_id'] = df_ookla['site_id'].astype(str)

# 2. Point this to ONE of your existing GEE chunks in your Google Drive
# Make sure this path matches your actual drive path!
chunk_path = "G:/My Drive/ESG_Hackathon/GEE_Extract_Malaysia_1787053788_pt1.csv"

# 3. Test the fix: Load it strictly as a string
print("Loading GEE Chunk...")
df_env = pd.read_csv(chunk_path, dtype={'site_id': str})

# 4. Attempt the merge
master_df = pd.merge(df_ookla, df_env, on='site_id', how='inner')

print(f"✅ Merge successful! Row count: {len(master_df)}")
print(master_df[['site_id']].head(3))