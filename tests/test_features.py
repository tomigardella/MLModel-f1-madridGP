"""Basic smoke tests for data loading and feature engineering."""

import pandas as pd
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class TestLoadQualifying:
    """Tests for the qualifying data loader."""

    def test_parse_lap_time_valid(self):
        from src.data.load_qualifying import _parse_lap_time
        assert abs(_parse_lap_time("1:31.824") - 91.824) < 0.001
        assert abs(_parse_lap_time("1:32.591") - 92.591) < 0.001

    def test_parse_lap_time_none(self):
        from src.data.load_qualifying import _parse_lap_time
        assert _parse_lap_time(None) is None
        assert _parse_lap_time("") is None

    def test_normalize_qualifying_structure(self):
        from src.data.load_qualifying import _normalize_qualifying
        race = {
            "season": "2026", "round": "14",
            "date": "2026-09-13", "raceName": "Spanish Grand Prix",
            "Circuit": {"circuitId": "madring"}
        }
        result = {
            "position": "1",
            "Driver": {"driverId": "norris", "code": "NOR"},
            "Constructor": {"constructorId": "mclaren"},
            "Q1": "1:33.469", "Q2": "1:32.873", "Q3": "1:31.824"
        }
        row = _normalize_qualifying(race, result)
        assert row["qualifying_position"] == 1
        assert row["qualifying_session_reached"] == "Q3"
        assert abs(row["q3_time_sec"] - 91.824) < 0.001
        assert abs(row["best_qualifying_time_sec"] - 91.824) < 0.001


class TestQualifyingFeatures:
    """Tests for qualifying feature engineering."""

    def _make_qualifying_df(self) -> pd.DataFrame:
        """Minimal multi-race qualifying DataFrame for testing."""
        return pd.DataFrame([
            {"season": 2025, "round": 1, "race_date": "2025-03-16",
             "race_name": "R1", "circuit_id": "c1",
             "driver_id": "drv_a", "driver_code": "A", "constructor_id": "team_x",
             "qualifying_position": 1, "q1_time_sec": 90.0, "q2_time_sec": 89.5, "q3_time_sec": 89.0,
             "best_qualifying_time_sec": 89.0, "qualifying_session_reached": "Q3"},
            {"season": 2025, "round": 1, "race_date": "2025-03-16",
             "race_name": "R1", "circuit_id": "c1",
             "driver_id": "drv_b", "driver_code": "B", "constructor_id": "team_x",
             "qualifying_position": 2, "q1_time_sec": 90.2, "q2_time_sec": 89.8, "q3_time_sec": 89.3,
             "best_qualifying_time_sec": 89.3, "qualifying_session_reached": "Q3"},
            {"season": 2025, "round": 2, "race_date": "2025-03-23",
             "race_name": "R2", "circuit_id": "c2",
             "driver_id": "drv_a", "driver_code": "A", "constructor_id": "team_x",
             "qualifying_position": 2, "q1_time_sec": 92.0, "q2_time_sec": 91.5, "q3_time_sec": 91.2,
             "best_qualifying_time_sec": 91.2, "qualifying_session_reached": "Q3"},
            {"season": 2025, "round": 2, "race_date": "2025-03-23",
             "race_name": "R2", "circuit_id": "c2",
             "driver_id": "drv_b", "driver_code": "B", "constructor_id": "team_y",
             "qualifying_position": 1, "q1_time_sec": 91.8, "q2_time_sec": 91.3, "q3_time_sec": 91.0,
             "best_qualifying_time_sec": 91.0, "qualifying_session_reached": "Q3"},
        ])

    def test_gap_to_pole(self):
        from src.features.build_qualifying_features import build_qualifying_features
        q = self._make_qualifying_df()
        feats = build_qualifying_features(q)
        # Pole-sitter should have gap_to_pole_sec == 0
        pole_rows = feats[feats["qualifying_position"] == 1]
        assert (pole_rows["gap_to_pole_sec"].abs() < 1e-9).all()

    def test_historical_qualifying_feature_no_leakage(self):
        """driver_avg_qualifying_last_3 for round 1 must be NaN (no prior history)."""
        from src.features.build_qualifying_features import build_qualifying_features
        q = self._make_qualifying_df()
        feats = build_qualifying_features(q)
        r1 = feats[feats["round"] == 1]
        assert r1["driver_avg_qualifying_last_3"].isna().all(), \
            "Round 1 must have NaN historical qualifying features (no prior data)"

    def test_qualifying_session_numeric(self):
        from src.features.build_qualifying_features import build_qualifying_features
        q = self._make_qualifying_df()
        feats = build_qualifying_features(q)
        assert (feats["qualifying_session_numeric"] == 3).all()


class TestLeakageAudit:
    """Tests for the leakage audit."""

    def test_forbidden_features_not_in_model_columns(self):
        from src.features.build_features_combined import MODEL_FEATURE_COLUMNS
        forbidden = ["finish_position", "position_text", "points", "laps",
                     "status", "dnf", "won"]
        for col in forbidden:
            assert col not in MODEL_FEATURE_COLUMNS, \
                f"'{col}' must NOT appear in MODEL_FEATURE_COLUMNS!"

    def test_audit_passes_cleanly(self):
        from src.evaluation.leakage_audit import run_leakage_audit, FORBIDDEN_FEATURES
        from src.features.build_features_combined import MODEL_FEATURE_COLUMNS
        # Build a minimal synthetic features DataFrame
        df = pd.DataFrame({col: [0.0] for col in MODEL_FEATURE_COLUMNS})
        audit = run_leakage_audit(df)
        critical = audit[audit["status"].str.startswith("🚨")]
        assert len(critical) == 0, f"Leakage detected: {critical}"

