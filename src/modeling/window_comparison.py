"""Compara el rendimiento de los modelos a través de diferentes ventanas de datos históricos.

Entrena y evalúa cada modelo usando 4 tamaños de ventana histórica:
  - 3yr:  últimas 3 temporadas
  - 5yr:  últimas 5 temporadas
  - 7yr:  últimas 7 temporadas
  - all:  historial completo (2018–presente)

Para cada ventana, se ejecuta validación walk-forward en 2023–2025.
Esto responde a: ¿más datos históricos siempre ayudan?
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import WINDOWS, PROCESSED_DIR
from src.features.build_features_combined import MODEL_FEATURE_COLUMNS
from src.modeling.walk_forward import run_walk_forward, summarize_walk_forward


def run_window_comparison(features: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta walk-forward para cada ventana histórica y retorna los resultados combinados."""
    all_results: list[pd.DataFrame] = []

    for window_label, start_year in WINDOWS.items():
        print(f"\n{'='*60}")
        print(f"Window: {window_label} (starting {start_year})")
        print(f"{'='*60}")

        window_data = features[features["season"] >= start_year].copy()
        wf = run_walk_forward(window_data, MODEL_FEATURE_COLUMNS)
        wf.insert(0, "window", window_label)
        wf.insert(1, "window_start_year", start_year)
        all_results.append(wf)

    return pd.concat(all_results, ignore_index=True)


def summarize_window_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Métricas promedio por (ventana, modelo) ordenadas por log_loss."""
    return (
        results.groupby(["window", "model"])[
            ["log_loss", "brier_score", "roc_auc", "winner_accuracy"]
        ]
        .mean()
        .round(4)
        .sort_values(["window", "log_loss"])
        .reset_index()
    )


def save_window_comparison(results: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "window_comparison_results.csv"
    results.to_csv(out, index=False)
    return out


def best_window(summary: pd.DataFrame) -> str:
    """Retorna la etiqueta de ventana con el mejor log_loss promedio."""
    mean_by_window = (
        summary.groupby("window")["log_loss"].mean().sort_values()
    )
    best = mean_by_window.index[0]
    print(f"\nBest window by mean log_loss across models: {best}")
    print(mean_by_window.to_string())
    return best


if __name__ == "__main__":
    features = pd.read_csv(PROCESSED_DIR / "features_combined.csv")
    results = run_window_comparison(features)
    path = save_window_comparison(results)
    print(f"\nSaved window comparison → {path}")

    summary = summarize_window_comparison(results)
    print("\nSummary (mean metrics per window × model):")
    print(summary.to_string(index=False))
    best = best_window(summary)
    print(f"\nSelected window: {best}")

