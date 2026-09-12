"""Evaluation metrics for race winner prediction models.

The key evaluation approach:
  - Metrics are computed at the ROW level (one row = one driver in one race).
  - Winner accuracy is computed at the RACE level: did the model correctly
    identify the driver with the highest predicted probability as the winner?
  - Both perspectives are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def evaluate(
    model,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "won",
) -> dict[str, float]:
    """Compute all evaluation metrics for a fitted model on a dataset.

    Parameters
    ----------
    model:
        A fitted sklearn Pipeline with a predict_proba method.
    dataset:
        DataFrame with feature columns and target column.
    feature_columns:
        List of feature column names to pass to the model.
    target_col:
        Name of the binary target column (default: "won").

    Returns
    -------
    dict with keys: log_loss, brier_score, roc_auc, winner_accuracy, n_races
    """
    # Drop rows where target is NaN (target race row)
    valid = dataset.dropna(subset=[target_col])
    if len(valid) == 0:
        return {}

    probas = model.predict_proba(valid[feature_columns])[:, 1]
    labels = valid[target_col].astype(int)

    # Row-level metrics
    ll = log_loss(labels, probas, labels=[0, 1])
    bs = brier_score_loss(labels, probas)

    try:
        auc = roc_auc_score(labels, probas)
    except ValueError:
        auc = float("nan")

    # Race-level winner accuracy
    tmp = valid.assign(_proba=probas)
    grouped = tmp.groupby(["season", "round"], sort=False)
    predicted_winner_idx = grouped["_proba"].idxmax()
    actual_winner_idx = grouped[target_col].idxmax()
    winner_acc = (predicted_winner_idx.values == actual_winner_idx.values).mean()

    n_races = grouped.ngroups

    return {
        "log_loss": round(ll, 6),
        "brier_score": round(bs, 6),
        "roc_auc": round(auc, 6),
        "winner_accuracy": round(float(winner_acc), 4),
        "n_races": n_races,
    }


def predict_race_probabilities(
    model,
    race_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate normalized win probabilities for all drivers in a race.

    The raw probabilities from predict_proba are softmax-normalized across
    all drivers in the race so they sum to 1.0.

    Parameters
    ----------
    model:
        A fitted sklearn Pipeline.
    race_df:
        DataFrame with one row per driver (single race only).
    feature_columns:
        Model feature columns.

    Returns
    -------
    pd.DataFrame with added columns: raw_win_probability, win_probability
    """
    result = race_df.copy()
    raw_probas = model.predict_proba(result[feature_columns])[:, 1]
    result["raw_win_probability"] = raw_probas
    total = raw_probas.sum()
    result["win_probability"] = raw_probas / total if total > 0 else raw_probas
    return result.sort_values("win_probability", ascending=False).reset_index(drop=True)
