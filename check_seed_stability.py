"""Multi-seed stability check for stratified out-of-block R².

Does NOT touch model_pipeline.py or main.py. Standalone script, safe to run
and delete — it duplicates just the spatial-block CV loop with a variable
random_state, so we can see how much the per-stratum R² numbers wobble on
their own before deciding whether they're worth chasing further.

Run from the repo root:
    python check_seed_stability.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from data_pipeline.config import MODEL_FEATURES

INPUT_FILE = "data/jendela_phase2_esg_matrix_malaysia.parquet"
TARGET = "download_kbps"
SEEDS = [0, 1, 2, 3, 4]
N_SPLITS = 5


def run_one_seed(train_pool, features, target, seed):
    """Mirrors apply_spatial_blocking_cv's CV loop, but with a variable seed."""
    train_pool = train_pool.copy()
    train_pool["lat_block"] = (train_pool["latitude"] / 0.5).astype(int)
    train_pool["lon_block"] = (train_pool["longitude"] / 0.5).astype(int)
    train_pool["spatial_block"] = (
        train_pool["lat_block"].astype(str) + "_" + train_pool["lon_block"].astype(str)
    )

    unique_blocks = np.array(sorted(train_pool["spatial_block"].unique()))
    num_blocks = len(unique_blocks)

    if num_blocks < 2:
        return None, None  # too small to run CV at all

    actual_splits = min(N_SPLITS, num_blocks)
    kf = KFold(n_splits=actual_splits, shuffle=True, random_state=seed)

    train_pool["cv_pred"] = np.nan
    for train_idx, val_idx in kf.split(unique_blocks):
        train_blocks = unique_blocks[train_idx]
        val_blocks = unique_blocks[val_idx]
        train_data = train_pool[train_pool["spatial_block"].isin(train_blocks)]
        val_data = train_pool[train_pool["spatial_block"].isin(val_blocks)]
        if train_data.empty or val_data.empty:
            continue
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed
        )
        model.fit(train_data[features], train_data[target])
        preds = model.predict(val_data[features])
        train_pool.loc[val_data.index, "cv_pred"] = preds

    clean = train_pool["cv_pred"].notna()
    if clean.sum() < 2:
        return None, train_pool

    pooled_r2 = r2_score(train_pool.loc[clean, target], train_pool.loc[clean, "cv_pred"])
    return pooled_r2, train_pool


def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} rows.")

    features = [f for f in MODEL_FEATURES if f in df.columns]
    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        print(f"⚠️  MODEL_FEATURES not in this file, skipping: {missing}")

    evidence_mask = (df["tests"] >= 15) & (df["devices"] >= 5) & (df["is_underserved_target"] == True)
    train_pool = df[evidence_mask].copy()
    print(f"Training pool: {len(train_pool):,} / {len(df):,} tiles meet the evidence threshold.\n")

    results = []
    for seed in SEEDS:
        pooled_r2, scored_pool = run_one_seed(train_pool, features, TARGET, seed)
        if pooled_r2 is None:
            print(f"seed={seed}: could not compute (too few blocks)")
            continue

        row = {"seed": seed, "pooled": pooled_r2}
        clean = scored_pool["cv_pred"].notna()
        for stratum in scored_pool["demographic_stratum"].unique():
            mask = clean & (scored_pool["demographic_stratum"] == stratum)
            if mask.sum() > 1:
                row[stratum] = r2_score(
                    scored_pool.loc[mask, TARGET], scored_pool.loc[mask, "cv_pred"]
                )
        results.append(row)
        print(
            f"seed={seed}  pooled={row.get('pooled', float('nan')):.3f}  "
            f"rural={row.get('rural', float('nan')):.3f}  "
            f"peri-urban={row.get('peri-urban', float('nan')):.3f}  "
            f"urban={row.get('urban', float('nan')):.3f}"
        )

    print("\n" + "=" * 60)
    res_df = pd.DataFrame(results)
    print("Summary across seeds:")
    for col in ["pooled", "rural", "peri-urban", "urban"]:
        if col in res_df.columns:
            print(f"  {col:<12} mean={res_df[col].mean():+.3f}  "
                  f"std={res_df[col].std():.3f}  "
                  f"range=[{res_df[col].min():+.3f}, {res_df[col].max():+.3f}]")

    print("\nHigh std / wide range relative to the mean means the number is mostly")
    print("noise at this sample size, not a stable signal worth chasing further.")


if __name__ == "__main__":
    main()