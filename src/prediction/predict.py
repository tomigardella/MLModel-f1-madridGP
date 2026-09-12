"""Final prediction runner for the target Grand Prix.

This module:
  1. Runs the leakage audit (mandatory gate)
  2. Trains the best model on the optimal historical window
  3. Generates win probabilities for each driver in the target GP
  4. Prints and saves the final prediction table
  5. Generates an explanation of top driver predictions

Usage
-----
    python -m src.prediction.predict
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    TARGET, PROCESSED_DIR, MODEL_DIR, FIGURES_DIR, RANDOM_SEED
)
from src.evaluation.leakage_audit import run_leakage_audit, print_audit
from src.evaluation.metrics import predict_race_probabilities
from src.features.build_features_combined import MODEL_FEATURE_COLUMNS


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features_combined.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Combined features not found at {path}.\n"
            "Run: python -m src.features.build_features_combined"
        )
    return pd.read_csv(path)


def load_target_features() -> pd.DataFrame:
    gp_slug = TARGET["circuit_id"]
    path = PROCESSED_DIR / f"{gp_slug}_{TARGET['year']}_prediction_input.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Target features not found at {path}.\n"
            "Run: python -m src.features.build_prediction"
        )
    return pd.read_csv(path)


def train_final_model(
    features: pd.DataFrame,
    best_window_start: int,
    best_model_name: str,
) -> object:
    """Train the selected model on the full optimal window (up to Round 13 of 2026)."""
    from src.modeling.model_registry import get_model_builders

    # Use all data up to (but not including) the target race
    target_year = TARGET["year"]
    target_round = TARGET["round"]

    train = features[
        (features["season"] >= best_window_start)
        & (
            (features["season"] < target_year)
            | (
                (features["season"] == target_year)
                & (features["round"] < target_round)
            )
        )
    ].dropna(subset=["won"]).copy()

    print(f"\nFinal training set: {len(train)} rows, "
          f"{train.groupby(['season', 'round']).ngroups} races")

    models = get_model_builders(MODEL_FEATURE_COLUMNS)
    if best_model_name not in models:
        raise ValueError(
            f"Model '{best_model_name}' not found. "
            f"Available: {list(models.keys())}"
        )

    model = models[best_model_name]
    model.fit(train[MODEL_FEATURE_COLUMNS], train["won"].astype(int))

    # Persist the fitted model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"final_{best_model_name}_{TARGET['circuit_id']}_{TARGET['year']}.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {model_path}")

    return model


def generate_prediction(model, target: pd.DataFrame) -> pd.DataFrame:
    """Produce the final win-probability ranking for the target GP."""
    predictions = predict_race_probabilities(model, target, MODEL_FEATURE_COLUMNS)
    predictions["win_probability_pct"] = (predictions["win_probability"] * 100).round(1)
    predictions["rank"] = range(1, len(predictions) + 1)
    return predictions


def print_prediction(predictions: pd.DataFrame) -> None:
    """Pretty-print the final prediction table."""
    print("\n" + "=" * 60)
    print(f"  {TARGET['gp_name'].upper()} {TARGET['year']}")
    print(f"  Pre-race prediction  |  {TARGET['race_date']}")
    print(f"  Circuit: {TARGET['circuit_id']}")
    print("=" * 60)
    print(f"  {'#':<4} {'Driver':<20} {'Team':<20} {'Win %':>7}")
    print("-" * 60)
    for _, row in predictions.iterrows():
        print(
            f"  {int(row['rank']):<4} "
            f"{row['driver_id']:<20} "
            f"{row['constructor_id']:<20} "
            f"{row['win_probability_pct']:>6.1f}%"
        )
    prob_sum = predictions["win_probability_pct"].sum()
    print(f"\n  Total probability: {prob_sum:.1f}%")
    print("=" * 60)


def plot_prediction(predictions: pd.DataFrame) -> None:
    """Horizontal bar chart of win probabilities."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = plt.cm.RdYlGn(
        [p / predictions["win_probability_pct"].max()
         for p in predictions["win_probability_pct"]]
    )
    bars = ax.barh(
        predictions["driver_id"][::-1],
        predictions["win_probability_pct"][::-1],
        color=colors[::-1],
    )
    ax.set_xlabel("Win probability (%)")
    ax.set_title(
        f"Pre-race win probability\n{TARGET['gp_name']} {TARGET['year']}",
        fontweight="bold",
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / f"prediction_{TARGET['circuit_id']}_{TARGET['year']}.png"
    plt.savefig(out, dpi=150)
    print(f"Prediction plot saved → {out}")
    plt.show()
    plt.close()


def save_prediction(predictions: pd.DataFrame) -> Path:
    """Save the final prediction to CSV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "rank", "driver_id", "driver_code", "constructor_id",
        "qualifying_position", "gap_to_pole_sec",
        "raw_win_probability", "win_probability", "win_probability_pct",
    ]
    available_cols = [c for c in cols if c in predictions.columns]
    out = PROCESSED_DIR / f"final_prediction_{TARGET['circuit_id']}_{TARGET['year']}.csv"
    predictions[available_cols].to_csv(out, index=False)
    return out


def main(
    best_window_start: int = 2021,   # override from window_comparison results
    best_model_name: str = "xgboost",  # override from walk_forward results
) -> pd.DataFrame:
    """Run the full prediction pipeline."""
    # Step 1: Leakage audit (mandatory gate)
    print("\nStep 1: Running leakage audit…")
    features = load_features()
    audit = run_leakage_audit(features)
    print_audit(audit)  # raises if critical leakage detected

    # Step 2: Train final model
    print("\nStep 2: Training final model…")
    model = train_final_model(features, best_window_start, best_model_name)

    # Step 3: Generate predictions
    print("\nStep 3: Generating predictions…")
    target = load_target_features()
    predictions = generate_prediction(model, target)

    # Step 4: Output
    print_prediction(predictions)
    plot_prediction(predictions)
    out = save_prediction(predictions)
    print(f"\nPredictions saved → {out}")

    return predictions


if __name__ == "__main__":
    # These values should be updated after running window_comparison + walk_forward
    predictions = main(
        best_window_start=2021,   # update after comparing windows
        best_model_name="xgboost",  # update after walk-forward comparison
    )

