"""Naïve baseline models for benchmarking.

Two baselines are defined:
  1. Qualifying Position Baseline:
     Winner = driver with best qualifying position (P1 always wins).
     No model training needed.
  
  2. Historical Win Rate Baseline:
     Winner = driver with highest career win rate (computed before the race).
     No ML — pure lookup.

These establish the floor: any ML model must beat both to add value.
"""

from __future__ import annotations

import pandas as pd

from src.features.build_features_combined import MODEL_FEATURE_COLUMNS


def evaluate_qualifying_baseline(features: pd.DataFrame) -> dict:
    """Evaluate: does the pole-sitter always win?

    Groups by race and checks if the driver with qualifying_position == 1
    corresponds to the actual winner.
    
    Handles races with missing qualifying data (older seasons) by falling
    back to grid_position if available.
    """
    valid = features.dropna(subset=["won"]).copy()
    valid = valid[valid["qualifying_position"].notna() | valid["won"].notna()]

    results_per_race = []
    for (season, round_), group in valid.groupby(["season", "round"]):
        q_col = "qualifying_position"
        if group[q_col].isna().all():
            continue  # skip races with no qualifying data
        predicted_winner = group.loc[group[q_col].idxmin(), "driver_id"]
        actual_winner_rows = group[group["won"] == 1]
        if actual_winner_rows.empty:
            continue
        actual_winner = actual_winner_rows.iloc[0]["driver_id"]
        results_per_race.append({
            "season": season,
            "round": round_,
            "correct": predicted_winner == actual_winner,
        })

    df = pd.DataFrame(results_per_race)
    if df.empty:
        return {"baseline": "qualifying_position", "winner_accuracy": float("nan"), "n_races": 0}

    return {
        "baseline": "qualifying_position",
        "winner_accuracy": round(df["correct"].mean(), 4),
        "n_races": len(df),
        "correct_predictions": int(df["correct"].sum()),
    }


def evaluate_win_rate_baseline(features: pd.DataFrame) -> dict:
    """Evaluate: does the driver with the highest career win rate win?

    Uses driver_career_wins_before / driver_career_starts_before as the score.
    """
    valid = features.dropna(subset=["won"]).copy()
    valid["career_win_rate"] = (
        valid["driver_career_wins_before"] / valid["driver_career_starts_before"].clip(lower=1)
    )

    results_per_race = []
    for (season, round_), group in valid.groupby(["season", "round"]):
        predicted_winner = group.loc[group["career_win_rate"].idxmax(), "driver_id"]
        actual_winner_rows = group[group["won"] == 1]
        if actual_winner_rows.empty:
            continue
        actual_winner = actual_winner_rows.iloc[0]["driver_id"]
        results_per_race.append({"correct": predicted_winner == actual_winner})

    df = pd.DataFrame(results_per_race)
    if df.empty:
        return {"baseline": "career_win_rate", "winner_accuracy": float("nan"), "n_races": 0}

    return {
        "baseline": "career_win_rate",
        "winner_accuracy": round(df["correct"].mean(), 4),
        "n_races": len(df),
        "correct_predictions": int(df["correct"].sum()),
    }


if __name__ == "__main__":
    from src.config import PROCESSED_DIR
    features = pd.read_csv(PROCESSED_DIR / "features_combined.csv")

    q_baseline = evaluate_qualifying_baseline(features)
    wr_baseline = evaluate_win_rate_baseline(features)

    print("=== Naïve Baselines ===")
    print(f"Qualifying Position Baseline: {q_baseline}")
    print(f"Career Win Rate Baseline:     {wr_baseline}")
    print("\nAny ML model must exceed these to add value.")
