"""Walk-forward temporal validation across historical seasons.

For each validation season V in [walk_forward_start, walk_forward_end]:
  - Train on all seasons BEFORE V (expanding window)
  - Evaluate on season V

This simulates how the model would have performed if deployed in real time,
with no information from the future leaking into training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import VALIDATION, RANDOM_SEED, PROCESSED_DIR
from src.evaluation.metrics import evaluate
from src.features.build_features_combined import MODEL_FEATURE_COLUMNS
from src.modeling.model_registry import get_model_builders


def run_walk_forward(
    features: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Run expanding-window walk-forward validation.

    Parameters
    ----------
    features:
        Combined feature dataset from build_combined_features().
    feature_columns:
        Which columns to use as model inputs. Defaults to MODEL_FEATURE_COLUMNS.

    Returns
    -------
    pd.DataFrame with columns: model, validation_season, log_loss, brier_score,
                                roc_auc, winner_accuracy, n_races
    """
    if feature_columns is None:
        feature_columns = MODEL_FEATURE_COLUMNS

    start = VALIDATION["walk_forward_start"]
    end = VALIDATION["walk_forward_end"]

    results: list[dict] = []
    for val_season in range(start, end + 1):
        train = features[features["season"] < val_season].copy()
        val = features[features["season"] == val_season].copy()

        if len(train) == 0 or len(val) == 0:
            continue

        models = get_model_builders(feature_columns)
        for name, model in models.items():
            model.fit(train[feature_columns], train["won"].astype(int))
            metrics = evaluate(model, val, feature_columns)
            results.append({
                "model": name,
                "validation_season": val_season,
                "train_seasons": f"{features['season'].min()}–{val_season - 1}",
                "train_races": train.groupby(["season", "round"]).ngroups,
                **metrics,
            })
            print(
                f"  [{val_season}] {name:<25} "
                f"LL={metrics['log_loss']:.4f}  "
                f"BS={metrics['brier_score']:.4f}  "
                f"WA={metrics['winner_accuracy']:.3f}"
            )

    return pd.DataFrame(results)


def summarize_walk_forward(wf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate walk-forward results by model (mean across validation seasons)."""
    return (
        wf.groupby("model")[["log_loss", "brier_score", "roc_auc", "winner_accuracy"]]
        .mean()
        .round(4)
        .sort_values("log_loss")
    )


def save_walk_forward(wf: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "walk_forward_results.csv"
    wf.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    features = pd.read_csv(PROCESSED_DIR / "features_combined.csv")
    print("Running walk-forward validation…")
    wf = run_walk_forward(features)
    path = save_walk_forward(wf)
    print(f"\nSaved → {path}")
    print("\nMean metrics per model:")
    print(summarize_walk_forward(wf).to_string())

