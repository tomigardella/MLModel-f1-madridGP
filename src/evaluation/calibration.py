"""Calibration evaluation: do predicted probabilities match observed frequencies?

In a well-calibrated model, when the model says "30% chance of winning",
the driver wins approximately 30% of the time in similar scenarios.

F1 calibration is inherently challenging because:
  - Winners are rare (~5% base rate)
  - Sample sizes per probability bucket are small
  - Race outcomes have high inherent randomness (safety cars, reliability, etc.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.config import FIGURES_DIR, PROCESSED_DIR


def calibration_table(
    labels: pd.Series,
    probas: np.ndarray,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Return calibration statistics per probability bin."""
    data = pd.DataFrame({"label": labels.to_numpy(), "prob": probas})
    data["bin"] = pd.cut(data["prob"], bins=n_bins, labels=False, include_lowest=True)
    table = (
        data.groupby("bin", observed=True)
        .agg(
            n=("label", "size"),
            mean_predicted=("prob", "mean"),
            observed_rate=("label", "mean"),
        )
        .reset_index()
    )
    table["calibration_error"] = (table["mean_predicted"] - table["observed_rate"]).abs()
    table["weighted_error"] = table["n"] * table["calibration_error"]
    return table


def plot_calibration_curves(
    model_probas: dict[str, tuple[pd.Series, np.ndarray]],
    title: str = "Calibration curves",
    save_path: Path | None = None,
) -> None:
    """Plot calibration curves for multiple models.

    Parameters
    ----------
    model_probas:
        Dict mapping model name → (labels Series, probabilities array)
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", lw=1)

    for name, (labels, probas) in model_probas.items():
        try:
            frac_pos, mean_pred = calibration_curve(
                labels, probas, n_bins=5, strategy="uniform"
            )
            ax.plot(mean_pred, frac_pos, marker="o", label=name)
        except ValueError:
            pass

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved calibration plot → {save_path}")
    plt.show()
    plt.close()

