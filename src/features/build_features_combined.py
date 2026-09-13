"""Combina las características de carrera y de clasificación en el dataset principal.

Une las características de carrera (build_race_features) con las de clasificación
(build_qualifying_features) sobre (season, round, driver_id).

Cuando faltan datos de clasificación para una carrera determinada (ej., temporadas más antiguas
o fechas aún no disponibles en la API), las columnas de clasificación contienen NaN. El modelo
maneja estos casos mediante imputación por mediana.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.features.build_race_features import RACE_FEATURE_COLUMNS, build_race_features
from src.features.build_qualifying_features import QUALIFYING_FEATURE_COLUMNS, build_qualifying_features


# Columnas de características finales utilizadas por el modelo
# (excluye metadatos y la variable objetivo — esos se agregan explícitamente)
MODEL_FEATURE_COLUMNS = [
    # === Características del historial de carreras ===
    "driver_career_starts_before",
    "driver_career_wins_before",
    "driver_career_points_before",
    "driver_career_dnf_rate_before",
    "driver_season_points_before",
    "driver_season_wins_before",
    "driver_season_position_before",
    "driver_season_dnf_rate_before",
    "driver_avg_finish_last_3",
    "driver_avg_finish_last_5",
    "driver_avg_finish_last_10",
    "driver_win_rate_last_3",
    "driver_win_rate_last_5",
    "driver_win_rate_last_10",
    "driver_dnf_rate_last_3",
    "driver_dnf_rate_last_5",
    "driver_dnf_rate_last_10",
    "driver_circuit_starts_before",
    "driver_circuit_wins_before",
    "driver_circuit_avg_finish_before",
    "driver_circuit_dnf_rate_before",
    "constructor_career_starts_before",
    "constructor_career_wins_before",
    "constructor_career_points_before",
    "constructor_avg_finish_last_3",
    "constructor_avg_finish_last_5",
    "constructor_win_rate_last_3",
    "constructor_win_rate_last_5",
    "constructor_season_points_before",
    "constructor_season_wins_before",
    "constructor_season_position_before",
    # === Características de clasificación (nuevas en este proyecto) ===
    "qualifying_position",
    "gap_to_pole_sec",
    "driver_vs_teammate_q_gap_sec",
    "team_best_qualifying_pos",
    "qualifying_session_numeric",
    "driver_avg_qualifying_last_3",
    "driver_avg_qualifying_last_5",
    "driver_avg_q_gap_to_pole_last_3",
    "driver_avg_q_gap_to_pole_last_5",
    "driver_season_avg_qualifying_before",
]

METADATA_COLUMNS = [
    "season", "round", "race_date", "race_name", "circuit_id",
    "driver_id", "driver_code", "constructor_id",
]

ALL_COLUMNS = METADATA_COLUMNS + MODEL_FEATURE_COLUMNS + ["won"]


def build_combined_features(
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
) -> pd.DataFrame:
    """Construye y une las características de carrera + clasificación.

    Parámetros
    ----------
    results:
        Provenientes de load_results().
    qualifying:
        Provenientes de load_qualifying().

    Retorna
    -------
    pd.DataFrame
        Una fila por (season, round, driver), que contiene tanto las características de carrera
        como las de clasificación, más la variable objetivo `won`.
    """
    race_feats = build_race_features(results)
    qual_feats = build_qualifying_features(qualifying)

    # Conservar únicamente las columnas de clasificación que no estén ya en las características de carrera
    q_cols = ["season", "round", "driver_id"] + [
        c for c in QUALIFYING_FEATURE_COLUMNS
        if c not in ("season", "round", "driver_id")
    ]

    combined = race_feats.merge(
        qual_feats[q_cols],
        on=["season", "round", "driver_id"],
        how="left",
        validate="one_to_one",
    )

    return combined[ALL_COLUMNS]


def save_combined_features(df: pd.DataFrame) -> Path:
    """Persiste el dataset de características combinadas."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "features_combined.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    from src.config import HISTORY
    from src.data.load_results import load_results
    from src.data.load_qualifying import load_qualifying

    seasons = list(range(HISTORY["start_year"], HISTORY["end_year"] + 1))
    results = load_results(seasons)
    qualifying = load_qualifying(seasons)
    features = build_combined_features(results, qualifying)
    path = save_combined_features(features)
    print(f"Saved {len(features)} combined rows → {path}")
    print(f"Qualifying coverage: {features['qualifying_position'].notna().mean():.1%} of races")
    print(f"Feature columns: {len(MODEL_FEATURE_COLUMNS)}")

