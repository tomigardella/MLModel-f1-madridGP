"""Validación temporal walk-forward a lo largo de las temporadas históricas.

Para cada temporada de validación V en [walk_forward_start, walk_forward_end]:
  - Entrenar en todas las temporadas ANTERIORES a V (ventana expansiva)
  - Evaluar en la temporada V

Esto simula cómo se habría desempeñado el modelo de haberse implementado en tiempo real,
sin que la información del futuro se filtre en el entrenamiento.
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
    """Ejecuta la validación walk-forward con ventana expansiva.

    Parámetros
    ----------
    features:
        Conjunto de datos de características combinadas de build_combined_features().
    feature_columns:
        Qué columnas usar como entradas del modelo. Por defecto es MODEL_FEATURE_COLUMNS.

    Retorna
    -------
    pd.DataFrame con columnas: model, validation_season, log_loss, brier_score,
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
    """Agrega los resultados de walk-forward por modelo (promedio a través de las temporadas de validación)."""
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

