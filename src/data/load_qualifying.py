"""Carga y normaliza los resultados de clasificación de Fórmula 1 a partir de archivos JSON de Jolpica.

Los datos de clasificación son la *nueva fuente principal de características* de este proyecto.
Cada fila representa el mejor tiempo de un piloto en cada sesión de clasificación.

Nota anti-filtración (anti-leakage)
-----------------------------------
Estos datos se recopilan DESPUÉS de la clasificación del sábado pero ANTES de la carrera
del domingo. Son información previa a la carrera y es seguro utilizarlos como características (features).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR

QUALIFYING_COLUMNS = [
    "season",
    "round",
    "race_date",
    "race_name",
    "circuit_id",
    "driver_id",
    "driver_code",
    "constructor_id",
    "qualifying_position",      # posición final de clasificación Q (base 1)
    "q1_time_sec",              # tiempo de vuelta en Q1 en segundos (float) o NaN
    "q2_time_sec",              # tiempo de vuelta en Q2 en segundos (float) o NaN
    "q3_time_sec",              # tiempo de vuelta en Q3 en segundos (float) o NaN
    "best_qualifying_time_sec", # mejor tiempo entre Q1/Q2/Q3 (usualmente Q3 para top 10)
    "qualifying_session_reached",  # "Q1" / "Q2" / "Q3"
]

KEY_COLUMNS = ["season", "round", "driver_id"]

_TIME_RE = re.compile(r"^(\d+):(\d+\.\d+)$")


def _parse_lap_time(raw: str | None) -> float | None:
    """Convierte la cadena 'M:SS.sss' a segundos totales, o devuelve None."""
    if not raw:
        return None
    m = _TIME_RE.match(raw.strip())
    if not m:
        return None
    minutes, seconds = int(m.group(1)), float(m.group(2))
    return minutes * 60.0 + seconds


def _normalize_qualifying(race: dict, result: dict) -> dict:
    """Aplana un resultado de clasificación de Jolpica en un dict plano."""
    q1 = _parse_lap_time(result.get("Q1"))
    q2 = _parse_lap_time(result.get("Q2"))
    q3 = _parse_lap_time(result.get("Q3"))

    times = [t for t in (q1, q2, q3) if t is not None]
    best = min(times) if times else None

    if q3 is not None:
        session_reached = "Q3"
    elif q2 is not None:
        session_reached = "Q2"
    else:
        session_reached = "Q1"

    return {
        "season": int(race["season"]),
        "round": int(race["round"]),
        "race_date": race.get("date"),
        "race_name": race.get("raceName"),
        "circuit_id": race.get("Circuit", {}).get("circuitId"),
        "driver_id": result.get("Driver", {}).get("driverId"),
        "driver_code": result.get("Driver", {}).get("code"),
        "constructor_id": result.get("Constructor", {}).get("constructorId"),
        "qualifying_position": pd.to_numeric(result.get("position"), errors="coerce"),
        "q1_time_sec": q1,
        "q2_time_sec": q2,
        "q3_time_sec": q3,
        "best_qualifying_time_sec": best,
        "qualifying_session_reached": session_reached,
    }


def _qualifying_files(season: int) -> list[Path]:
    """Devuelve las páginas JSON de clasificación para la temporada (*season*).

    Los archivos de rondas individuales (round_XX_qualifying.json) tienen prioridad sobre
    los archivos paginados con offset, ya que las páginas con offset pueden dividir una ronda
    entre páginas, causando datos incompletos para la última ronda de una página.
    """
    q_dir = RAW_DIR / str(season) / "qualifying"
    individual = sorted(q_dir.glob("round_*_qualifying.json"))
    individual_rounds = set()
    for p in individual:
        # Extrae el número de ronda del nombre de archivo: round_14_qualifying.json -> 14
        try:
            individual_rounds.add(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass

    pages = sorted(q_dir.glob("qualifying_offset_*.json"))
    return individual, pages, individual_rounds


def load_qualifying(seasons: list[int]) -> pd.DataFrame:
    """Carga, normaliza y desduplica los resultados de clasificación para las temporadas dadas.

    Los archivos de rondas individuales (round_XX_qualifying.json) tienen prioridad sobre
    los archivos paginados con offset, asegurando datos completos por ronda.

    Retorna
    -------
    pd.DataFrame
        Una fila por (temporada, ronda, piloto), ordenada cronológicamente.
        Las sesiones de clasificación faltantes (algunos años anteriores) son NaN.
    """
    records: list[dict] = []
    seen_rounds: set[tuple] = set()  # (season, round) ya cargadas completamente

    for season in seasons:
        individual_files, offset_files, individual_rounds = _qualifying_files(season)

        # Paso 1: Procesar primero los archivos de rondas individuales (máxima prioridad)
        for path in individual_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for race in payload["MRData"]["RaceTable"].get("Races", []):
                key = (int(race["season"]), int(race["round"]))
                for result in race.get("QualifyingResults", []):
                    records.append(_normalize_qualifying(race, result))
                if race.get("QualifyingResults"):
                    seen_rounds.add(key)

        # Paso 2: Procesar páginas con offset, omitir rondas ya cargadas desde archivos individuales
        for path in offset_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for race in payload["MRData"]["RaceTable"].get("Races", []):
                key = (int(race["season"]), int(race["round"]))
                if key in seen_rounds:
                    continue
                for result in race.get("QualifyingResults", []):
                    records.append(_normalize_qualifying(race, result))
                if race.get("QualifyingResults"):
                    seen_rounds.add(key)

    if not records:
        return pd.DataFrame(columns=QUALIFYING_COLUMNS)

    df = pd.DataFrame(records, columns=QUALIFYING_COLUMNS)
    df = df.drop_duplicates(subset=KEY_COLUMNS, keep="first")
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "season", "round", "qualifying_position"]).reset_index(drop=True)
    return df


def save_processed_qualifying(qualifying: pd.DataFrame) -> Path:
    """Guarda los datos de clasificación normalizados en el directorio de datos procesados."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "qualifying_results.csv"
    qualifying.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    from src.config import HISTORY
    seasons = list(range(HISTORY["start_year"], HISTORY["end_year"] + 1))
    df = load_qualifying(seasons)
    path = save_processed_qualifying(df)
    print(f"Saved {len(df)} qualifying rows -> {path}")
    print(df[["season", "round", "driver_id", "qualifying_position", "best_qualifying_time_sec"]].head(20))

