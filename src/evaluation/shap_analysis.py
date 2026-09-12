"""SHAP-based interpretability analysis for the winning model.

Answers: "Why does the model predict this driver has a high chance of winning?"

Generates:
  - SHAP summary plot (feature importance across all predictions)
  - SHAP waterfall plot for individual drivers
  - Feature importance table
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, PROCESSED_DIR


def compute_shap_values(model, X: pd.DataFrame, feature_columns: list[str]):
    """Compute SHAP values for a fitted pipeline.

    Handles sklearn Pipelines by extracting the preprocessed data
    and the underlying classifier.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required: pip install shap")

    # Get preprocessed data
    X_prep = model.named_steps["preprocessor"].transform(X[feature_columns])

    classifier = model.named_steps["classifier"]
    model_name = type(classifier).__name__.lower()

    if any(x in model_name for x in ("xgb", "lgbm", "catboost", "gradientboosting", "randomforest")):
        explainer = shap.TreeExplainer(classifier)
    else:
        explainer = shap.LinearExplainer(classifier, X_prep)

    shap_values = explainer.shap_values(X_prep)

    # For binary classification some models return list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, X_prep


def plot_shap_summary(
    shap_values: np.ndarray,
    X_prep: np.ndarray,
    feature_names: list[str],
    title: str = "SHAP Feature Importance",
    save_path: Path | None = None,
) -> None:
    """Bar-style SHAP summary plot (mean absolute SHAP values)."""
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required: pip install shap")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values, X_prep,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved SHAP summary → {save_path}")
    plt.show()
    plt.close()


def feature_importance_table(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return a DataFrame with mean absolute SHAP importance per feature."""
    importance = np.abs(shap_values).mean(axis=0)
    # Truncate to len(feature_names) in case imputer added indicator columns
    importance = importance[: len(feature_names)]
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def run_shap_analysis(
    model,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    model_name: str = "model",
) -> pd.DataFrame:
    """Full SHAP analysis pipeline — computes values, plots, saves table."""
    print(f"Computing SHAP values for {model_name}…")
    shap_values, X_prep = compute_shap_values(model, test_data, feature_columns)

    plot_shap_summary(
        shap_values, X_prep, feature_columns,
        title=f"SHAP Feature Importance — {model_name}",
        save_path=FIGURES_DIR / f"shap_summary_{model_name}.png",
    )

    importance_df = feature_importance_table(shap_values, feature_columns)
    out = PROCESSED_DIR / f"shap_importance_{model_name}.csv"
    importance_df.to_csv(out, index=False)
    print(f"Saved SHAP importance table → {out}")
    return importance_df

