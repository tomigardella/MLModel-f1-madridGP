"""Load and normalize raw Formula 1 race results from Jolpica JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR

RESULT_COLUMNS = [
    "season",
    "round",
    "race_date",
    "race_name",
    "circuit_id",
    "driver_id",
    "driver_code",
    "constructor_id",
    "grid_position",      # starting grid position (from results endpoint)
    "position_text",      # "1", "2", … or "R", "D", "W", "N", "F", "E"
    "finish_position",    # numeric finish or NaN for DNFs
    "points",
    "laps",
    "status",
    "dnf",                # 1 if did not finish, 0 otherwise
    "won",                # 1 if position_text == "1"
]
KEY_COLUMNS = ["season", "round", "driver_id"]


def _result_files(season: int) -> list[Path]:
    """Return all race result JSON files for *season* (sorted)."""
    season_dir = RAW_DIR / str(season) / "races"
    return sorted(season_dir.glob("results_offset_*.json"))


def _normalize_result(race: dict, result: dict) -> dict:
    """Flatten one nested Jolpica result entry into a plain dict."""
    pos_text = result.get("positionText", "")
    finish_pos = pd.to_numeric(pos_text, errors="coerce")
    # DNF: position text is not a number (R=Retired, D=Disqualified, etc.)
    dnf = 0 if pd.notna(finish_pos) else 1
    return {
        "season": int(race["season"]),
        "round": int(race["round"]),
        "race_date": race.get("date"),
        "race_name": race.get("raceName"),
        "circuit_id": race.get("Circuit", {}).get("circuitId"),
        "driver_id": result.get("Driver", {}).get("driverId"),
        "driver_code": result.get("Driver", {}).get("code"),
        "constructor_id": result.get("Constructor", {}).get("constructorId"),
        "grid_position": pd.to_numeric(result.get("grid"), errors="coerce"),
        "position_text": pos_text,
        "finish_position": finish_pos,
        "points": pd.to_numeric(result.get("points"), errors="coerce"),
        "laps": pd.to_numeric(result.get("laps"), errors="coerce"),
        "status": result.get("status"),
        "dnf": dnf,
        "won": int(pos_text == "1"),
    }


def load_results(seasons: list[int]) -> pd.DataFrame:
    """Load, normalize and deduplicate race results for the requested seasons.

    Parameters
    ----------
    seasons:
        List of season years to load. Each must have raw data downloaded.

    Returns
    -------
    pd.DataFrame
        One row per (season, round, driver) participation, sorted chronologically.
    """
    records: list[dict] = []
    for season in seasons:
        for path in _result_files(season):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for race in payload["MRData"]["RaceTable"].get("Races", []):
                for result in race.get("Results", []):
                    records.append(_normalize_result(race, result))

    df = pd.DataFrame(records, columns=RESULT_COLUMNS)
    df = df.drop_duplicates(subset=KEY_COLUMNS, keep="first")
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "season", "round", "driver_id"]).reset_index(drop=True)
    return df


def save_processed_results(results: pd.DataFrame) -> Path:
    """Save normalized results to the processed data directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "race_results.csv"
    results.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    from src.config import HISTORY
    seasons = list(range(HISTORY["start_year"], HISTORY["end_year"] + 1))
    df = load_results(seasons)
    path = save_processed_results(df)
    print(f"Saved {len(df)} rows → {path}")

