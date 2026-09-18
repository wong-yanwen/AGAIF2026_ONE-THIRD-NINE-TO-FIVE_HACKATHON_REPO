"""Sensitivity checks for the JENDELA priority model.

Run from the project root:
    python -m utils.sensitivity_test

The script prefers:
    data/NEW/jendela_phase2_esg_matrix_malaysia.parquet

so staged scoring changes are tested before production promotion. If no staged
file exists, it falls back to:
    data/jendela_phase2_esg_matrix_malaysia.parquet

Tests:
1. Sub-pillar sensitivity:
   mapped power vs VIIRS darkness at 60/40, 50/50, and 40/60.
   This fully rebuilds both the Diesel pillar and the diesel-gated Community
   pillar, matching model_pipeline.py.

2. Macro-pillar sensitivity:
   shifts Diesel from 40% -> 20% and Community from 30% -> 50%.

The script also prints the actual Top-20 candidates for each sub-pillar
scenario so geographic changes can be inspected directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd


APPROVED_TIER = "Sufficient Evidence - Ranked Screening Approved"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve()

# sensitivity_test.py currently lives in utils/.
# If it is ever moved to the project root, this still works.
ROOT = HERE.parent.parent if HERE.parent.name == "utils" else HERE.parent

DATA = ROOT / "data"
STAGED = DATA / "NEW" / "jendela_phase2_esg_matrix_malaysia.parquet"
APPROVED = DATA / "jendela_phase2_esg_matrix_malaysia.parquet"

file_path = STAGED if STAGED.exists() else APPROVED

if not file_path.exists():
    raise FileNotFoundError(
        "Could not find a Malaysia matrix in data/NEW or data/. "
        "Run the standalone model pipeline first:\n"
        "    python -m models.model_pipeline malaysia"
    )

print(f"📥 Loading dataset: {file_path}")

df_raw = pd.read_parquet(file_path)

# Only sites that the backend actually permits to be ranked.
df = df_raw[
    df_raw["confidence_tier"].eq(APPROVED_TIER)
].copy()

if df.empty:
    raise RuntimeError(
        "No approved candidate rows found in the selected Malaysia matrix."
    )


# ---------------------------------------------------------------------
# Required schema
# ---------------------------------------------------------------------

required = {
    "site_id",
    "latitude",
    "longitude",
    "distance_to_power_m",
    "night_radiance_nw_cm2_sr",
    "power_distance_missing",
    "population_total",
    "essential_service_weight",
    "service_shortfall_n",
    "solar_score_n",
    "access_ease_n",
    "off_grid_score_n",
    "community_impact_n",
    "priority_score",
    "demographic_stratum",
}

missing = sorted(required - set(df.columns))

if missing:
    raise RuntimeError(
        f"Matrix is missing required columns: {missing}"
    )


# ---------------------------------------------------------------------
# Regional helper used by app.py
# ---------------------------------------------------------------------

# This is the same practical UI split used by app.py.
# It is NOT a state-level administrative boundary.
df["is_east_malaysia"] = df["longitude"] > 109.0
df["macro_region_sensitivity"] = np.where(
    df["is_east_malaysia"],
    "East Malaysia",
    "Peninsular Malaysia",
)


# ---------------------------------------------------------------------
# Shared components for sub-pillar sensitivity
# ---------------------------------------------------------------------

power_remoteness = (
    df["distance_to_power_m"] / 5000.0
).clip(lower=0.0, upper=1.0)

darkness_score = (
    1.0 - df["night_radiance_nw_cm2_sr"] / 10.0
).clip(lower=0.0, upper=1.0)

power_missing = (
    df["power_distance_missing"]
    .fillna(False)
    .astype(bool)
)

# Community need components that do not depend on the off-grid weighting.
population_n = df["population_total"].rank(pct=True)
service_n = df["essential_service_weight"].rank(pct=True)
residual_n = df["service_shortfall_n"].fillna(0.0)

raw_community = (
    0.50 * population_n
    + 0.30 * service_n
    + 0.20 * residual_n
)

raw_community_n = raw_community.rank(pct=True)


def simulate_subpillar(
    w_power: float,
    w_dark: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Rebuild the affected scoring chain exactly like model_pipeline.py.

    Changing the internal off-grid formula affects:
      1. absolute off_grid_likelihood
      2. percentile-ranked Diesel pillar
      3. diesel_gate
      4. diesel-gated Community pillar
      5. final priority score

    Missing mapped power distance receives no artificial remoteness credit;
    those rows fall back to VIIRS darkness only.
    """

    off_grid = pd.Series(
        np.where(
            power_missing,
            darkness_score,
            (
                w_power * power_remoteness
                + w_dark * darkness_score
            ),
        ),
        index=df.index,
        dtype=float,
    ).clip(lower=0.0, upper=1.0)

    off_grid_score_n = off_grid.rank(pct=True)

    diesel_gate = (
        (off_grid - 0.20) / 0.40
    ).clip(lower=0.0, upper=1.0)

    community_n = raw_community_n * diesel_gate

    priority = (
        0.40 * off_grid_score_n
        + 0.25 * df["solar_score_n"]
        + 0.30 * community_n
        + 0.05 * df["access_ease_n"]
    )

    return (
        off_grid,
        off_grid_score_n,
        community_n,
        priority,
    )


# =====================================================================
# TEST 1 — SUB-PILLAR SENSITIVITY
# =====================================================================

print("\n" + "=" * 86)
print(
    "🔬 TEST 1: SUB-PILLAR SENSITIVITY — "
    "mapped power vs VIIRS darkness"
)
print("=" * 86)

weight_scenarios = [
    ("60 / 40", 0.60, 0.40),
    ("50 / 50 (Baseline)", 0.50, 0.50),
    ("40 / 60", 0.40, 0.60),
]

results = {}

for name, w_power, w_dark in weight_scenarios:
    (
        off_grid,
        off_grid_score_n,
        community_n,
        priority,
    ) = simulate_subpillar(
        w_power=w_power,
        w_dark=w_dark,
    )

    scenario_df = df.copy()

    # Store scenario-specific values explicitly.
    scenario_df["scenario_off_grid_likelihood"] = off_grid
    scenario_df["scenario_off_grid_score_n"] = off_grid_score_n
    scenario_df["scenario_community_impact_n"] = community_n
    scenario_df["scenario_priority_score"] = priority * 100.0

    top_n = min(20, len(scenario_df))

    top_idx = (
        scenario_df["scenario_priority_score"]
        .nlargest(top_n)
        .index
    )

    top = scenario_df.loc[top_idx].copy()

    top = top.sort_values(
        "scenario_priority_score",
        ascending=False,
    ).reset_index(drop=True)

    top["scenario_rank"] = np.arange(
        1,
        len(top) + 1,
    )

    results[name] = {
        "top": top,
        "off_grid": off_grid,
        "off_grid_score_n": off_grid_score_n,
        "community": community_n,
        "priority": priority,
    }


# ---------------------------------------------------------------------
# Baseline reproduction check
# ---------------------------------------------------------------------

baseline = results["50 / 50 (Baseline)"]

baseline_sites = set(
    baseline["top"]["site_id"].astype(str)
)

reproduced = baseline["priority"] * 100.0
actual = df["priority_score"].astype(float)

max_abs_diff = float(
    (reproduced - actual).abs().max()
)

print(
    f"Baseline reproduction max |Δ priority|: "
    f"{max_abs_diff:.6f} points"
)

# priority_score in model_pipeline.py is rounded to 2 decimals.
if max_abs_diff > 0.011:
    raise RuntimeError(
        "Sensitivity baseline does not reproduce model_pipeline.py. "
        "Do not interpret the overlap results until the formulas match."
    )


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

summary_rows = []

for name, _, _ in weight_scenarios:
    top = results[name]["top"]

    top_sites = set(
        top["site_id"].astype(str)
    )

    overlap = len(
        top_sites.intersection(baseline_sites)
    )

    n_top = len(top)

    rural_count = int(
        top["demographic_stratum"]
        .eq("rural")
        .sum()
    )

    east_count = int(
        top["is_east_malaysia"]
        .sum()
    )

    power_missing_count = int(
        top["power_distance_missing"]
        .fillna(False)
        .sum()
    )

    median_radiance = float(
        top["night_radiance_nw_cm2_sr"]
        .median()
    )

    median_power_km = float(
        (
            top["distance_to_power_m"]
            / 1000.0
        ).median()
    )

    summary_rows.append(
        {
            "Power / Darkness": name,
            "Top-20 overlap vs 50/50": (
                f"{overlap}/{n_top} "
                f"({100 * overlap / n_top:.0f}%)"
            ),
            "Rural sites": (
                f"{rural_count}/{n_top}"
            ),
            "East Malaysia sites": (
                f"{east_count}/{n_top}"
            ),
            "Power-missing sites": (
                f"{power_missing_count}/{n_top}"
            ),
            "Median radiance": (
                f"{median_radiance:.2f}"
            ),
            "Median power km": (
                f"{median_power_km:.2f}"
            ),
        }
    )

summary_df = pd.DataFrame(summary_rows)

print("\nSUB-PILLAR SUMMARY")
print(summary_df.to_string(index=False))


# ---------------------------------------------------------------------
# Print actual Top-20 candidates for each scenario
# ---------------------------------------------------------------------

print("\n" + "=" * 86)
print("📍 TOP-20 SITE DETAILS BY POWER / DARKNESS SCENARIO")
print("=" * 86)

detail_cols = [
    "scenario_rank",
    "site_id",
    "longitude",
    "latitude",
    "demographic_stratum",
    "macro_region_sensitivity",
    "distance_to_power_m",
    "night_radiance_nw_cm2_sr",
    "power_distance_missing",
    "scenario_off_grid_likelihood",
    "scenario_off_grid_score_n",
    "scenario_community_impact_n",
    "scenario_priority_score",
]

for scenario_name, _, _ in weight_scenarios:
    top = results[scenario_name]["top"]

    print("\n" + "-" * 86)
    print(f"SCENARIO: {scenario_name}")
    print("-" * 86)

    printable = top[detail_cols].copy()

    # Cleaner terminal display.
    printable["longitude"] = printable["longitude"].round(4)
    printable["latitude"] = printable["latitude"].round(4)

    printable["distance_to_power_m"] = (
        printable["distance_to_power_m"]
        .round(1)
    )

    printable["night_radiance_nw_cm2_sr"] = (
        printable["night_radiance_nw_cm2_sr"]
        .round(3)
    )

    printable["scenario_off_grid_likelihood"] = (
        printable["scenario_off_grid_likelihood"]
        .round(4)
    )

    printable["scenario_off_grid_score_n"] = (
        printable["scenario_off_grid_score_n"]
        .round(4)
    )

    printable["scenario_community_impact_n"] = (
        printable["scenario_community_impact_n"]
        .round(4)
    )

    printable["scenario_priority_score"] = (
        printable["scenario_priority_score"]
        .round(2)
    )

    print(
        printable.to_string(
            index=False,
        )
    )


# ---------------------------------------------------------------------
# Show which sites ENTER / LEAVE relative to the 50/50 baseline
# ---------------------------------------------------------------------

print("\n" + "=" * 86)
print("🔄 TOP-20 MEMBERSHIP CHANGES VS 50/50 BASELINE")
print("=" * 86)

baseline_top = baseline["top"].copy()
baseline_ids = set(
    baseline_top["site_id"].astype(str)
)

for scenario_name in [
    "60 / 40",
    "40 / 60",
]:
    scenario_top = results[scenario_name]["top"].copy()

    scenario_ids = set(
        scenario_top["site_id"].astype(str)
    )

    entered = scenario_ids - baseline_ids
    left = baseline_ids - scenario_ids

    print(f"\n{scenario_name}")

    print(
        f"  Entered Top 20: "
        f"{len(entered)}"
    )

    if entered:
        entered_rows = scenario_top[
            scenario_top["site_id"]
            .astype(str)
            .isin(entered)
        ][
            [
                "scenario_rank",
                "site_id",
                "longitude",
                "latitude",
                "demographic_stratum",
                "macro_region_sensitivity",
                "night_radiance_nw_cm2_sr",
                "distance_to_power_m",
                "scenario_off_grid_likelihood",
                "scenario_priority_score",
            ]
        ].copy()

        print(
            entered_rows.to_string(
                index=False,
            )
        )

    print(
        f"  Left Top 20: "
        f"{len(left)}"
    )

    if left:
        left_rows = baseline_top[
            baseline_top["site_id"]
            .astype(str)
            .isin(left)
        ][
            [
                "scenario_rank",
                "site_id",
                "longitude",
                "latitude",
                "demographic_stratum",
                "macro_region_sensitivity",
                "night_radiance_nw_cm2_sr",
                "distance_to_power_m",
                "scenario_off_grid_likelihood",
                "scenario_priority_score",
            ]
        ].copy()

        print(
            left_rows.to_string(
                index=False,
            )
        )


# =====================================================================
# TEST 2 — MACRO-PILLAR SENSITIVITY
# =====================================================================

print("\n" + "=" * 86)
print(
    "📊 TEST 2: MACRO-PILLAR SENSITIVITY — "
    "environmental vs social policy shift"
)
print("=" * 86)


def score_macro(
    data: pd.DataFrame,
    w_offgrid: float,
    w_solar: float,
    w_community: float,
    w_access: float,
) -> pd.Series:
    """Recombine the already-exported backend-native pillar scores."""
    return (
        w_offgrid * data["off_grid_score_n"]
        + w_solar * data["solar_score_n"]
        + w_community * data["community_impact_n"]
        + w_access * data["access_ease_n"]
    )


base_score = score_macro(
    df,
    0.40,
    0.25,
    0.30,
    0.05,
)

social_score = score_macro(
    df,
    0.20,
    0.25,
    0.50,
    0.05,
)

n = min(20, len(df))

top_base = set(
    df.loc[
        base_score.nlargest(n).index,
        "site_id",
    ].astype(str)
)

top_social = set(
    df.loc[
        social_score.nlargest(n).index,
        "site_id",
    ].astype(str)
)

overlap = len(
    top_base.intersection(top_social)
)

print(
    "Top-20 overlap under substantial policy shift "
    "(Diesel 40→20%, Community 30→50%): "
    f"{overlap}/{n} "
    f"({100 * overlap / n:.0f}%)"
)

print(
    "Defensible pitch line: "
    f"'{overlap} of the top {n} sites remain in the shortlist "
    "under a substantial policy-weight shift.'"
)


print("\n✅ Sensitivity checks complete.")
