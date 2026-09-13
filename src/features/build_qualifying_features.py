"""Construye características derivadas de la clasificación para cada (season, round, driver).

Este módulo es la CONTRIBUCIÓN CENTRAL de este proyecto respecto al
modelo anterior del GP de los Países Bajos. En lugar de usar `grid_position` extraído de
los resultados de la carrera (lo cual puede confundir la posición de clasificación con penalizaciones en la parrilla y
tiene un linaje de datos ambiguo), construimos características directamente a partir del
endpoint de resultados de clasificación.

Contrato anti-fuga (anti-leakage)
---------------------------------
Para cualquier carrera X:
  - Las características de clasificación del EVENTO ACTUAL (qualifying_position, gap_to_pole_sec, etc.)
    utilizan la SESIÓN de clasificación de la propia carrera X — celebrada el sábado antes de la carrera del domingo.
    Estas representan información pre-carrera y su uso es seguro.
  - Las características de clasificación HISTÓRICAS (driver_avg_qualifying_last_N, etc.)
    utilizan únicamente sesiones de clasificación ANTERIORES a la carrera X (mediante shift(1) + rolling).
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Características de clasificación de la carrera actual
# ---------------------------------------------------------------------------

def _add_gap_to_pole(q: pd.DataFrame) -> pd.DataFrame:
    """Agrega la diferencia en segundos con respecto al tiempo de la pole position para cada carrera."""
    pole_times = (
        q[q["qualifying_position"] == 1]
        .groupby(["season", "round"])["best_qualifying_time_sec"]
        .first()
        .rename("pole_time_sec")
        .reset_index()
    )
    q = q.merge(pole_times, on=["season", "round"], how="left")
    q["gap_to_pole_sec"] = q["best_qualifying_time_sec"] - q["pole_time_sec"]
    q.drop(columns=["pole_time_sec"], inplace=True)
    return q


def _add_teammate_gap(q: pd.DataFrame) -> pd.DataFrame:
    """Agrega la diferencia con respecto al compañero de equipo mejor clasificado para cada piloto."""
    team_best = (
        q.groupby(["season", "round", "constructor_id"])["best_qualifying_time_sec"]
        .min()
        .rename("team_best_q_time_sec")
        .reset_index()
    )
    team_best_pos = (
        q.groupby(["season", "round", "constructor_id"])["qualifying_position"]
        .min()
        .rename("team_best_qualifying_pos")
        .reset_index()
    )
    q = q.merge(team_best, on=["season", "round", "constructor_id"], how="left")
    q = q.merge(team_best_pos, on=["season", "round", "constructor_id"], how="left")
    # una diferencia > 0 significa más lento que el compañero; 0 para el compañero más rápido
    q["driver_vs_teammate_q_gap_sec"] = (
        q["best_qualifying_time_sec"] - q["team_best_q_time_sec"]
    )
    q.drop(columns=["team_best_q_time_sec"], inplace=True)
    return q


def _add_qualifying_session_numeric(q: pd.DataFrame) -> pd.DataFrame:
    """Codifica la sesión de clasificación alcanzada como un entero ordinal."""
    session_map = {"Q1": 1, "Q2": 2, "Q3": 3}
    q["qualifying_session_numeric"] = q["qualifying_session_reached"].map(session_map)
    return q


# ---------------------------------------------------------------------------
# Características históricas de clasificación (móviles, desplazadas)
# ---------------------------------------------------------------------------

def _add_historical_qualifying_features(q: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas móviles de posición de clasificación, utilizando estrictamente solo sesiones pasadas."""
    q = q.sort_values(["race_date", "season", "round", "driver_id"]).reset_index(drop=True)
    g = q.groupby("driver_id", sort=False)

    for n in (3, 5):
        q[f"driver_avg_qualifying_last_{n}"] = g["qualifying_position"].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )
        q[f"driver_avg_q_gap_to_pole_last_{n}"] = g["gap_to_pole_sec"].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )

    # Promedio de clasificación en la temporada
    sg = q.groupby(["season", "driver_id"], sort=False)
    q["driver_season_avg_qualifying_before"] = sg["qualifying_position"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    return q


# ---------------------------------------------------------------------------
# Constructor principal
# ---------------------------------------------------------------------------

QUALIFYING_FEATURE_COLUMNS = [
    # Claves
    "season", "round", "driver_id",
    # Clasificación actual (pre-carrera, sábado)
    "qualifying_position",
    "best_qualifying_time_sec",
    "q1_time_sec",
    "q2_time_sec",
    "q3_time_sec",
    "qualifying_session_reached",
    "qualifying_session_numeric",
    "gap_to_pole_sec",
    "driver_vs_teammate_q_gap_sec",
    "team_best_qualifying_pos",
    # Rendimiento histórico en clasificación
    "driver_avg_qualifying_last_3",
    "driver_avg_qualifying_last_5",
    "driver_avg_q_gap_to_pole_last_3",
    "driver_avg_q_gap_to_pole_last_5",
    "driver_season_avg_qualifying_before",
]


def build_qualifying_features(qualifying: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila por (season, round, driver) con características de clasificación.

    Parámetros
    ----------
    qualifying:
        Datos normalizados de clasificación provenientes de load_qualifying().
    """
    q = qualifying.copy()
    q["race_date"] = pd.to_datetime(q["race_date"])
    q = q.sort_values(["race_date", "season", "round", "qualifying_position"]).reset_index(drop=True)

    q = _add_gap_to_pole(q)
    q = _add_teammate_gap(q)
    q = _add_qualifying_session_numeric(q)
    q = _add_historical_qualifying_features(q)

    return q[QUALIFYING_FEATURE_COLUMNS]

