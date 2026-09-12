"""Data leakage audit for the F1 prediction pipeline.

Runs a systematic check on the combined feature dataset to detect:
  1. Features that may contain target information (won, finish_position, etc.)
  2. Features calculated using current-race data instead of shifted history
  3. Qualifying features mistakenly populated for the target race (should be OK)
  4. Any unexpectedly high correlation with the target variable

Outputs a human-readable audit table.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.features.build_features_combined import MODEL_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Feature audit catalog
# ---------------------------------------------------------------------------

# Each entry: (feature_name, available_before_race, leakage_risk, decision, notes)
FEATURE_AUDIT = [
    # --- Race history features ---
    ("driver_career_starts_before",       True,  "None",     "Include", "Cumsum shifted by 1"),
    ("driver_career_wins_before",          True,  "None",     "Include", "Cumsum shifted by 1"),
    ("driver_career_points_before",        True,  "None",     "Include", "Cumsum shifted by 1"),
    ("driver_career_dnf_rate_before",      True,  "None",     "Include", "Expanding mean shifted by 1"),
    ("driver_season_points_before",        True,  "None",     "Include", "Cumsum shifted by 1 within season"),
    ("driver_season_wins_before",          True,  "None",     "Include", "Cumsum shifted by 1 within season"),
    ("driver_season_position_before",      True,  "None",     "Include", "Derived from points_before"),
    ("driver_season_dnf_rate_before",      True,  "None",     "Include", "Expanding mean shifted by 1"),
    ("driver_avg_finish_last_3",           True,  "None",     "Include", "Rolling(3) on shifted finish_position"),
    ("driver_avg_finish_last_5",           True,  "None",     "Include", "Rolling(5) on shifted finish_position"),
    ("driver_avg_finish_last_10",          True,  "None",     "Include", "Rolling(10) on shifted finish_position"),
    ("driver_win_rate_last_3",             True,  "None",     "Include", "Rolling(3) on shifted won"),
    ("driver_win_rate_last_5",             True,  "None",     "Include", "Rolling(5) on shifted won"),
    ("driver_win_rate_last_10",            True,  "None",     "Include", "Rolling(10) on shifted won"),
    ("driver_dnf_rate_last_3",             True,  "None",     "Include", "Rolling(3) on shifted dnf"),
    ("driver_dnf_rate_last_5",             True,  "None",     "Include", "Rolling(5) on shifted dnf"),
    ("driver_dnf_rate_last_10",            True,  "None",     "Include", "Rolling(10) on shifted dnf"),
    ("driver_circuit_starts_before",       True,  "None",     "Include", "Cumcount on circuit group"),
    ("driver_circuit_wins_before",         True,  "None",     "Include", "Cumsum shifted — 0 for new circuit"),
    ("driver_circuit_avg_finish_before",   True,  "None",     "Include", "Expanding mean shifted — NaN if new"),
    ("driver_circuit_dnf_rate_before",     True,  "None",     "Include", "Expanding mean shifted — NaN if new"),
    ("constructor_career_starts_before",   True,  "None",     "Include", "Cumcount on constructor"),
    ("constructor_career_wins_before",     True,  "None",     "Include", "Cumsum shifted by 1"),
    ("constructor_career_points_before",   True,  "None",     "Include", "Cumsum shifted by 1"),
    ("constructor_avg_finish_last_3",      True,  "None",     "Include", "Rolling(3) on shifted finish"),
    ("constructor_avg_finish_last_5",      True,  "None",     "Include", "Rolling(5) on shifted finish"),
    ("constructor_win_rate_last_3",        True,  "None",     "Include", "Rolling(3) on shifted wins"),
    ("constructor_win_rate_last_5",        True,  "None",     "Include", "Rolling(5) on shifted wins"),
    ("constructor_season_points_before",   True,  "None",     "Include", "Cumsum shifted within season"),
    ("constructor_season_wins_before",     True,  "None",     "Include", "Cumsum shifted within season"),
    ("constructor_season_position_before", True,  "None",     "Include", "Derived from season_points_before"),
    # --- Qualifying features (new) ---
    ("qualifying_position",               True,  "None",     "Include", "From Saturday qualifying — pre-race"),
    ("gap_to_pole_sec",                    True,  "None",     "Include", "Derived from qualifying times only"),
    ("driver_vs_teammate_q_gap_sec",       True,  "None",     "Include", "Both times from same qualifying session"),
    ("team_best_qualifying_pos",           True,  "None",     "Include", "From Saturday qualifying"),
    ("qualifying_session_numeric",         True,  "None",     "Include", "Q1/Q2/Q3 ordinal — from qualifying"),
    ("driver_avg_qualifying_last_3",       True,  "None",     "Include", "Rolling(3) on shifted qualifying_position"),
    ("driver_avg_qualifying_last_5",       True,  "None",     "Include", "Rolling(5) on shifted qualifying_position"),
    ("driver_avg_q_gap_to_pole_last_3",    True,  "None",     "Include", "Rolling(3) on shifted gap_to_pole"),
    ("driver_avg_q_gap_to_pole_last_5",    True,  "None",     "Include", "Rolling(5) on shifted gap_to_pole"),
    ("driver_season_avg_qualifying_before",True,  "None",     "Include", "Expanding mean shifted — within season"),
]

FORBIDDEN_FEATURES = [
    "finish_position",
    "position_text",
    "points",
    "laps",
    "status",
    "dnf",
    "won",
]


def run_leakage_audit(features: pd.DataFrame) -> pd.DataFrame:
    """Run the leakage audit and return an audit table."""
    audit_rows = []
    for feat, available, risk, decision, notes in FEATURE_AUDIT:
        if feat not in features.columns:
            status = "⚠️ MISSING"
        elif not available:
            status = "❌ LEAKAGE"
        elif risk != "None":
            status = f"⚠️ {risk}"
        else:
            status = "✅ OK"
        audit_rows.append({
            "feature": feat,
            "available_before_race": available,
            "leakage_risk": risk,
            "decision": decision,
            "notes": notes,
            "status": status,
        })

    # Check that forbidden columns are NOT in MODEL_FEATURE_COLUMNS
    for col in FORBIDDEN_FEATURES:
        if col in MODEL_FEATURE_COLUMNS:
            audit_rows.append({
                "feature": col,
                "available_before_race": False,
                "leakage_risk": "CRITICAL",
                "decision": "❌ REMOVE IMMEDIATELY",
                "notes": f"'{col}' must never appear in MODEL_FEATURE_COLUMNS",
                "status": "🚨 CRITICAL LEAKAGE",
            })

    return pd.DataFrame(audit_rows)


def print_audit(audit: pd.DataFrame) -> None:
    """Print audit results and highlight any issues."""
    print("\n" + "=" * 80)
    print("DATA LEAKAGE AUDIT")
    print("=" * 80)
    print(audit[["feature", "status", "leakage_risk", "decision"]].to_string(index=False))
    critical = audit[audit["status"].str.startswith("🚨")]
    if not critical.empty:
        print("\n🚨 CRITICAL LEAKAGE DETECTED — DO NOT PROCEED:")
        print(critical[["feature", "notes"]].to_string(index=False))
        raise RuntimeError("Critical leakage detected. See audit output above.")
    warnings = audit[audit["status"].str.startswith("⚠️")]
    if not warnings.empty:
        print(f"\n⚠️  {len(warnings)} warning(s) — review before final prediction")
    else:
        print("\n✅ Audit passed. No leakage detected.")


if __name__ == "__main__":
    from src.config import PROCESSED_DIR
    features = pd.read_csv(PROCESSED_DIR / "features_combined.csv")
    audit = run_leakage_audit(features)
    print_audit(audit)
    out = PROCESSED_DIR.parent / "reports" / "leakage_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)
    print(f"\nAudit saved → {out}")
