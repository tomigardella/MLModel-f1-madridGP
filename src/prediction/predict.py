"""Ejecutor de predicción final para el Gran Premio objetivo.

Este módulo:
  1. Ejecuta la auditoría de fuga de datos (filtro obligatorio)
  2. Entrena el mejor modelo en la ventana histórica óptima
  3. Genera probabilidades de victoria para cada piloto en el GP objetivo
  4. Imprime y guarda la tabla de predicción final
  5. Genera una explicación de las predicciones de los mejores pilotos

Uso
---
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
    """Entrena el modelo seleccionado en la ventana óptima completa (hasta la Ronda 13 de 2026)."""
    from src.modeling.model_registry import get_model_builders

    # Usar todos los datos hasta (pero sin incluir) la carrera objetivo
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

    # Persistir el modelo entrenado
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"final_{best_model_name}_{TARGET['circuit_id']}_{TARGET['year']}.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {model_path}")

    return model


def generate_prediction(model, target: pd.DataFrame) -> pd.DataFrame:
    """Genera el ranking final de probabilidad de victoria para el GP objetivo."""
    predictions = predict_race_probabilities(model, target, MODEL_FEATURE_COLUMNS)
    predictions["win_probability_pct"] = (predictions["win_probability"] * 100).round(1)
    predictions["rank"] = range(1, len(predictions) + 1)
    return predictions


def print_prediction(predictions: pd.DataFrame) -> None:
    """Imprime de forma legible la tabla final de predicciones."""
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
    """Gráfico de barras horizontales de probabilidades de victoria."""
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
    """Guarda la predicción final en un archivo CSV."""
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
    best_window_start: int = 2021,   # sobrescribir a partir de los resultados de window_comparison
    best_model_name: str = "xgboost",  # sobrescribir a partir de los resultados de walk_forward
) -> pd.DataFrame:
    """Ejecuta el pipeline de predicción completo."""
    # Paso 1: Auditoría de fuga (filtro obligatorio)
    print("\nStep 1: Running leakage audit…")
    features = load_features()
    audit = run_leakage_audit(features)
    print_audit(audit)  # lanza excepción si se detecta fuga crítica

    # Paso 2: Entrenar el modelo final
    print("\nStep 2: Training final model…")
    model = train_final_model(features, best_window_start, best_model_name)

    # Paso 3: Generar predicciones
    print("\nStep 3: Generating predictions…")
    target = load_target_features()
    predictions = generate_prediction(model, target)

    # Paso 4: Salida
    print_prediction(predictions)
    plot_prediction(predictions)
    out = save_prediction(predictions)
    print(f"\nPredictions saved → {out}")

    return predictions


if __name__ == "__main__":
    # Estos valores deben actualizarse después de ejecutar window_comparison + walk_forward
    predictions = main(
        best_window_start=2021,   # actualizar después de comparar ventanas
        best_model_name="xgboost",  # actualizar después de la comparación walk-forward
    )

