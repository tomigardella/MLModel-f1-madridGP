"""Build pre-race features from historical race results.

All features are calculated with a strict temporal guard:
- cumsum().shift(1) ensures only information from BEFORE the current race is used.
- Rolling windows (.rolling(N).mean() on shifted series) respect the same guard.

Anti-leakage contract
---------------------
No feature in this module touches `finish_position`, `points`, `won`, `laps`,
or `status` of the CURRENT race.  Those fields exist in the raw results but
are only used shifted-by-one or aggregated over previous races.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Race-level feature builders
# ---------------------------------------------------------------------------

def _driver_career_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lifetime career stats computed strictly before each race."""
    g = df.groupby("driver_id", sort=False)

    df["driver_career_starts_before"] = g.cumcount()  # 0 for debut race
    df["driver_career_wins_before"] = g["won"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    df["driver_career_points_before"] = g["points"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    df["driver_career_dnf_rate_before"] = g["dnf"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    return df


def _driver_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """Within-season cumulative stats computed before each race."""
    g = df.groupby(["season", "driver_id"], sort=False)

    df["driver_season_points_before"] = g["points"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    df["driver_season_wins_before"] = g["won"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    df["driver_season_dnf_rate_before"] = g["dnf"].transform(
        lambda s: s.expanding().mean().shift(1)
    )

    # Driver standings position (before the current race)
    season_round_counts = (
        df.groupby(["season", "round", "driver_id"], sort=False)
        .first()[["driver_season_points_before", "driver_season_wins_before"]]
        .reset_index()
        .sort_values(
            ["season", "round", "driver_season_points_before", "driver_season_wins_before", "driver_id"],
            ascending=[True, True, False, False, True],
        )
    )
    season_round_counts["driver_season_position_before"] = (
        season_round_counts.groupby(["season", "round"]).cumcount() + 1
    )
    df = df.merge(
        season_round_counts[["season", "round", "driver_id", "driver_season_position_before"]],
        on=["season", "round", "driver_id"],
        how="left",
        validate="many_to_one",
    )
    return df


def _driver_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Short-term form windows (3, 5, 10 races) — shifted to avoid leakage."""
    g = df.groupby("driver_id", sort=False)

    for n in (3, 5, 10):
        df[f"driver_avg_finish_last_{n}"] = g["finish_position"].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )
        df[f"driver_win_rate_last_{n}"] = g["won"].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )
        df[f"driver_dnf_rate_last_{n}"] = g["dnf"].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )

    return df


def _driver_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Historical performance at THIS specific circuit."""
    g = df.groupby(["driver_id", "circuit_id"], sort=False)

    df["driver_circuit_starts_before"] = g.cumcount()
    df["driver_circuit_wins_before"] = g["won"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    df["driver_circuit_avg_finish_before"] = g["finish_position"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["driver_circuit_dnf_rate_before"] = g["dnf"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    return df


def _constructor_race_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-race constructor stats from driver rows."""
    return (
        df.groupby(["season", "round", "constructor_id"], sort=False, as_index=False)
        .agg(
            constructor_points_race=("points", "sum"),
            constructor_wins_race=("won", "sum"),
            constructor_finish_race=("finish_position", "mean"),
            constructor_dnf_race=("dnf", "sum"),
        )
        .sort_values(["season", "round", "constructor_id"])
    )


def _constructor_features(df: pd.DataFrame, race_totals: pd.DataFrame) -> pd.DataFrame:
    """Career and rolling constructor features computed before each race."""
    g = race_totals.groupby("constructor_id", sort=False)

    race_totals["constructor_career_starts_before"] = g.cumcount()
    race_totals["constructor_career_wins_before"] = g["constructor_wins_race"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    race_totals["constructor_career_points_before"] = g["constructor_points_race"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    for n in (3, 5):
        race_totals[f"constructor_avg_finish_last_{n}"] = g["constructor_finish_race"].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )
        race_totals[f"constructor_win_rate_last_{n}"] = g["constructor_wins_race"].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )

    # Season standings
    sg = race_totals.groupby(["season", "constructor_id"], sort=False)
    race_totals["constructor_season_points_before"] = sg["constructor_points_race"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    race_totals["constructor_season_wins_before"] = sg["constructor_wins_race"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    race_totals_sorted = race_totals.sort_values(
        ["season", "round", "constructor_season_points_before", "constructor_season_wins_before", "constructor_id"],
        ascending=[True, True, False, False, True],
    )
    race_totals_sorted["constructor_season_position_before"] = (
        race_totals_sorted.groupby(["season", "round"]).cumcount() + 1
    )

    constructor_cols = [
        "season", "round", "constructor_id",
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
    ]
    return df.merge(
        race_totals_sorted[constructor_cols],
        on=["season", "round", "constructor_id"],
        how="left",
        validate="many_to_one",
    )


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

RACE_FEATURE_COLUMNS = [
    # Keys
    "season", "round", "race_date", "race_name", "circuit_id",
    "driver_id", "driver_code", "constructor_id",
    # Driver career
    "driver_career_starts_before",
    "driver_career_wins_before",
    "driver_career_points_before",
    "driver_career_dnf_rate_before",
    # Driver season
    "driver_season_points_before",
    "driver_season_wins_before",
    "driver_season_position_before",
    "driver_season_dnf_rate_before",
    # Driver rolling form
    "driver_avg_finish_last_3",
    "driver_avg_finish_last_5",
    "driver_avg_finish_last_10",
    "driver_win_rate_last_3",
    "driver_win_rate_last_5",
    "driver_win_rate_last_10",
    "driver_dnf_rate_last_3",
    "driver_dnf_rate_last_5",
    "driver_dnf_rate_last_10",
    # Driver circuit
    "driver_circuit_starts_before",
    "driver_circuit_wins_before",
    "driver_circuit_avg_finish_before",
    "driver_circuit_dnf_rate_before",
    # Constructor
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
    # Target
    "won",
]


def build_race_features(results: pd.DataFrame) -> pd.DataFrame:
    """Return one pre-race feature row per driver per race.

    Parameters
    ----------
    results:
        Normalized race results from load_results().  Must include all
        seasons needed; the caller controls the historical window.
    """
    df = results.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "season", "round", "driver_id"]).reset_index(drop=True)

    df = _driver_career_features(df)
    df = _driver_season_features(df)
    df = _driver_rolling_features(df)
    df = _driver_circuit_features(df)

    race_totals = _constructor_race_totals(df)
    df = _constructor_features(df, race_totals)

    return df[RACE_FEATURE_COLUMNS]

