"""Evaluación de calibración: ¿las probabilidades predichas coinciden con las frecuencias observadas?

En un modelo bien calibrado, cuando el modelo indica "30% de probabilidad de ganar",
el piloto gana aproximadamente el 30% de las veces en escenarios similares.

La calibración en la F1 es intrínsecamente desafiante debido a que:
  - Los ganadores son raros (tasa base de ~5%)
  - Los tamaños de muestra por intervalo de probabilidad son pequeños
  - Los resultados de carrera tienen una alta aleatoriedad inherente (autos de seguridad, fiabilidad, etc.)
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
    """Retorna estadísticas de calibración por intervalo (bin) de probabilidad."""
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
    """Grafica curvas de calibración para múltiples modelos.

    Parámetros
    ----------
    model_probas:
        Diccionario que mapea nombre del modelo → (Series de etiquetas, array de probabilidades)
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

