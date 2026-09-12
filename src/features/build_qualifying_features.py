"""Build qualifying-derived features for every (season, round, driver).

This module is the CENTRAL CONTRIBUTION of this project relative to the
previous Dutch GP model.  Instead of using `grid_position` extracted from
race results (which can conflate qualifying position with grid penalties and
has an ambiguous data lineage), we build features directly from the
qualifying results endpoint.

Anti-leakage contract
---------------------
For any race X:
  - The TARGET qualifying features (qualifying_position, gap_to_pole_sec, etc.)
    use the qualifying SESSION of race X itself — held on Saturday before Sunday's race.
    These are pre-race information and are safe to use.
  - The HISTORICAL qualifying features (driver_avg_qualifying_last_N, etc.)
    use only qualifying sessions BEFORE race X (via shift(1) + rolling).
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Current-race qualifying features
# ---------------------------------------------------------------------------

def _add_gap_to_pole(q: pd.DataFrame) -> pd.DataFrame:
    """Add gap in seconds to the pole-sitter time for each race."""
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
    """Add gap to best-qualifying teammate for each driver."""
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
    # gap > 0 means slower than teammate; 0 for the fastest teammate
    q["driver_vs_teammate_q_gap_sec"] = (
        q["best_qualifying_time_sec"] - q["team_best_q_time_sec"]
    )
    q.drop(columns=["team_best_q_time_sec"], inplace=True)
    return q


def _add_qualifying_session_numeric(q: pd.DataFrame) -> pd.DataFrame:
    """Encode qualifying session reached as an ordinal integer."""
    session_map = {"Q1": 1, "Q2": 2, "Q3": 3}
    q["qualifying_session_numeric"] = q["qualifying_session_reached"].map(session_map)
    return q


# ---------------------------------------------------------------------------
# Historical qualifying features (rolling, shifted)
# ---------------------------------------------------------------------------

def _add_historical_qualifying_features(q: pd.DataFrame) -> pd.DataFrame:
    """Rolling qualifying position stats, strictly using past sessions only."""
    q = q.sort_values(["race_date", "season", "round", "driver_id"]).reset_index(drop=True)
    g = q.groupby("driver_id", sort=False)

    for n in (3, 5):
        q[f"driver_avg_qualifying_last_{n}"] = g["qualifying_position"].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )
        q[f"driver_avg_q_gap_to_pole_last_{n}"] = g["gap_to_pole_sec"].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )

    # Season qualifying average
    sg = q.groupby(["season", "driver_id"], sort=False)
    q["driver_season_avg_qualifying_before"] = sg["qualifying_position"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    return q


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

QUALIFYING_FEATURE_COLUMNS = [
    # Keys
    "season", "round", "driver_id",
    # Current qualifying (pre-race, Saturday)
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
    # Historical qualifying form
    "driver_avg_qualifying_last_3",
    "driver_avg_qualifying_last_5",
    "driver_avg_q_gap_to_pole_last_3",
    "driver_avg_q_gap_to_pole_last_5",
    "driver_season_avg_qualifying_before",
]


def build_qualifying_features(qualifying: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (season, round, driver) with qualifying features.

    Parameters
    ----------
    qualifying:
        Normalized qualifying data from load_qualifying().
    """
    q = qualifying.copy()
    q["race_date"] = pd.to_datetime(q["race_date"])
    q = q.sort_values(["race_date", "season", "round", "qualifying_position"]).reset_index(drop=True)

    q = _add_gap_to_pole(q)
    q = _add_teammate_gap(q)
    q = _add_qualifying_session_numeric(q)
    q = _add_historical_qualifying_features(q)

    return q[QUALIFYING_FEATURE_COLUMNS]
