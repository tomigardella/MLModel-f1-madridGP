"""Build the prediction input dataset for the target Grand Prix.

This module appends a synthetic target-race row for each driver
(with no race result yet) to the historical data, then runs the
full feature pipeline to compute pre-race features.

The qualifying features for the target race come directly from the
qualifying results downloaded on Saturday (already available in the API).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import TARGET, RAW_DIR, PROCESSED_DIR
from src.data.load_results import load_results, RESULT_COLUMNS
from src.data.load_qualifying import load_qualifying
from src.features.build_features_combined import build_combined_features, ALL_COLUMNS, MODEL_FEATURE_COLUMNS


def _target_qualifying_data() -> pd.DataFrame:
    """Load the real qualifying results for the target GP from the API cache."""
    year = TARGET["year"]
    round_ = TARGET["round"]
    q_file = RAW_DIR / str(year) / "qualifying" / f"round_{round_:02d}_qualifying.json"

    if not q_file.exists():
        raise FileNotFoundError(
            f"Target qualifying file not found: {q_file}\n"
            f"Run: python -m src.data.download_historical --target-qualifying"
        )

    payload = json.loads(q_file.read_text(encoding="utf-8"))
    races = payload["MRData"]["RaceTable"].get("Races", [])
    if not races or not races[0].get("QualifyingResults"):
        raise RuntimeError("Qualifying results are empty in the cached file.")

    return races[0]


def _candidate_drivers_from_qualifying(race_data: dict) -> pd.DataFrame:
    """Extract the 20-driver grid directly from the target qualifying results."""
    from src.data.load_qualifying import _normalize_qualifying
    records = [
        _normalize_qualifying(race_data, r)
        for r in race_data.get("QualifyingResults", [])
    ]
    df = pd.DataFrame(records)
    return df[["driver_id", "driver_code", "constructor_id"]].drop_duplicates("driver_id")


def build_target_features(results: pd.DataFrame, qualifying: pd.DataFrame) -> pd.DataFrame:
    """Build prediction features for every driver in the target GP lineup.

    Steps
    -----
    1. Load the real qualifying results for the target GP.
    2. Create synthetic zero-result rows for the target race.
    3. Concatenate with historical results and run the full feature pipeline.
    4. Extract only the target-race rows.

    The qualifying features for the target race are already included in
    `qualifying` (loaded from the downloaded qualifying file).

    Returns
    -------
    pd.DataFrame
        One row per driver, with MODEL_FEATURE_COLUMNS populated and
        `won` set to NaN (unknown — we are predicting this).
    """
    race_data = _target_qualifying_data()
    candidates = _candidate_drivers_from_qualifying(race_data)

    target_rows = candidates.assign(
        season=TARGET["year"],
        round=TARGET["round"],
        race_date=TARGET["race_date"],
        race_name=TARGET["gp_name"],
        circuit_id=TARGET["circuit_id"],
        grid_position=pd.NA,
        position_text=pd.NA,
        finish_position=pd.NA,
        points=pd.NA,
        laps=pd.NA,
        status=pd.NA,
        dnf=pd.NA,
        won=pd.NA,
    )

    # Align columns with results schema
    for col in RESULT_COLUMNS:
        if col not in target_rows.columns:
            target_rows[col] = pd.NA
    target_rows = target_rows[RESULT_COLUMNS]

    combined_results = pd.concat([results, target_rows], ignore_index=True)
    features = build_combined_features(combined_results, qualifying)

    target_mask = (
        (features["season"] == TARGET["year"])
        & (features["round"] == TARGET["round"])
    )
    return features[target_mask].reset_index(drop=True)


def save_target_features(df: pd.DataFrame) -> Path:
    """Persist the target prediction input."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gp_slug = TARGET["circuit_id"]
    out = PROCESSED_DIR / f"{gp_slug}_{TARGET['year']}_prediction_input.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    from src.config import HISTORY

    seasons = list(range(HISTORY["start_year"], HISTORY["end_year"] + 1))
    results = load_results(seasons)
    qualifying = load_qualifying(seasons)
    target = build_target_features(results, qualifying)
    path = save_target_features(target)
    print(f"Saved {len(target)} target rows → {path}")
    print(target[["driver_id", "constructor_id", "qualifying_position", "gap_to_pole_sec"]].to_string(index=False))
