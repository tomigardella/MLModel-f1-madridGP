"""Load and normalize Formula 1 qualifying results from Jolpica JSON files.

Qualifying data is the *new core feature source* of this project.
Each row represents one driver's best time in each qualifying session.

Anti-leakage note
-----------------
These data are collected AFTER qualifying on Saturday but BEFORE the race
on Sunday.  They are pre-race information and are safe to use as features.
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
    "qualifying_position",      # final Q classification position (1-based)
    "q1_time_sec",              # Q1 lap time in seconds (float) or NaN
    "q2_time_sec",              # Q2 lap time in seconds (float) or NaN
    "q3_time_sec",              # Q3 lap time in seconds (float) or NaN
    "best_qualifying_time_sec", # best time across Q1/Q2/Q3 (usually Q3 for top-10)
    "qualifying_session_reached",  # "Q1" / "Q2" / "Q3"
]

KEY_COLUMNS = ["season", "round", "driver_id"]

_TIME_RE = re.compile(r"^(\d+):(\d+\.\d+)$")


def _parse_lap_time(raw: str | None) -> float | None:
    """Convert 'M:SS.sss' string to total seconds, or return None."""
    if not raw:
        return None
    m = _TIME_RE.match(raw.strip())
    if not m:
        return None
    minutes, seconds = int(m.group(1)), float(m.group(2))
    return minutes * 60.0 + seconds


def _normalize_qualifying(race: dict, result: dict) -> dict:
    """Flatten one Jolpica qualifying result into a plain dict."""
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
    """Return all qualifying JSON pages for *season*."""
    q_dir = RAW_DIR / str(season) / "qualifying"
    # Season-wide pages
    pages = sorted(q_dir.glob("qualifying_offset_*.json"))
    # Target-round individual files
    pages += sorted(q_dir.glob("round_*_qualifying.json"))
    return pages


def load_qualifying(seasons: list[int]) -> pd.DataFrame:
    """Load, normalize and deduplicate qualifying results for the given seasons.

    Returns
    -------
    pd.DataFrame
        One row per (season, round, driver), sorted chronologically.
        Missing qualifying sessions (some older years) are NaN.
    """
    records: list[dict] = []
    seen_rounds: set[tuple] = set()  # (season, round) already loaded

    for season in seasons:
        for path in _qualifying_files(season):
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
    """Save normalized qualifying data to the processed directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "qualifying_results.csv"
    qualifying.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    from src.config import HISTORY
    seasons = list(range(HISTORY["start_year"], HISTORY["end_year"] + 1))
    df = load_qualifying(seasons)
    path = save_processed_qualifying(df)
    print(f"Saved {len(df)} qualifying rows → {path}")
    print(df[["season", "round", "driver_id", "qualifying_position", "best_qualifying_time_sec"]].head(20))

