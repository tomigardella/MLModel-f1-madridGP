"""Métricas de evaluación para modelos de predicción de ganador de carrera.

El enfoque principal de evaluación:
  - Las métricas se calculan a nivel de FILA (una fila = un piloto en una carrera).
  - La precisión del ganador se calcula a nivel de CARRERA: ¿el modelo identificó
    correctamente al piloto con la mayor probabilidad predicha como ganador?
  - Se reportan ambas perspectivas.
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
    """Calcula todas las métricas de evaluación para un modelo entrenado en un dataset.

    Parámetros
    ----------
    model:
        Un Pipeline de sklearn entrenado con un método predict_proba.
    dataset:
        DataFrame con columnas de características y columna objetivo.
    feature_columns:
        Lista de nombres de columnas de características para pasar al modelo.
    target_col:
        Nombre de la columna objetivo binaria (por defecto: "won").

    Retorna
    -------
    dict con claves: log_loss, brier_score, roc_auc, winner_accuracy, n_races
    """
    # Descartar filas donde el target es NaN (fila de la carrera objetivo)
    valid = dataset.dropna(subset=[target_col])
    if len(valid) == 0:
        return {}

    probas = model.predict_proba(valid[feature_columns])[:, 1]
    labels = valid[target_col].astype(int)

    # Métricas a nivel de fila
    ll = log_loss(labels, probas, labels=[0, 1])
    bs = brier_score_loss(labels, probas)

    try:
        auc = roc_auc_score(labels, probas)
    except ValueError:
        auc = float("nan")

    # Precisión del ganador a nivel de carrera
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
    """Genera probabilidades de victoria normalizadas para todos los pilotos en una carrera.

    Las probabilidades brutas de predict_proba se normalizan para
    todos los pilotos de la carrera de modo que sumen 1.0.

    Parámetros
    ----------
    model:
        Un Pipeline de sklearn entrenado.
    race_df:
        DataFrame con una fila por piloto (solo para una carrera individual).
    feature_columns:
        Columnas de características del modelo.

    Retorna
    -------
    pd.DataFrame con columnas agregadas: raw_win_probability, win_probability
    """
    result = race_df.copy()
    raw_probas = model.predict_proba(result[feature_columns])[:, 1]
    result["raw_win_probability"] = raw_probas
    total = raw_probas.sum()
    result["win_probability"] = raw_probas / total if total > 0 else raw_probas
    return result.sort_values("win_probability", ascending=False).reset_index(drop=True)

