"""Model registry — defines all candidate models as sklearn-compatible pipelines.

Every model:
  - Uses median imputation for missing values (qualifying coverage < 100%)
  - Uses a MissingIndicator to let the model learn from the pattern of missingness
  - Is reproducible via the global random seed

Adding a new model
------------------
1. Define a `build_<name>()` function returning an sklearn Pipeline.
2. Add it to MODEL_BUILDERS dict at the bottom.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------

def _numeric_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Median imputation + missing indicator + standard scaling."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
    )


def _tree_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Median imputation + missing indicator (no scaling — trees don't need it)."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Individual model builders
# ---------------------------------------------------------------------------

def build_logistic_regression(feature_columns: list[str]) -> Pipeline:
    """Logistic Regression — well calibrated, interpretable baseline."""
    return Pipeline([
        ("preprocessor", _numeric_preprocessor(feature_columns)),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.1,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])


def build_random_forest(feature_columns: list[str], n_estimators: int = 500) -> Pipeline:
    """Random Forest — captures non-linearities, robust to small datasets."""
    return Pipeline([
        ("preprocessor", _tree_preprocessor(feature_columns)),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])


def build_gradient_boosting(feature_columns: list[str]) -> Pipeline:
    """sklearn GradientBoosting — slower but good calibration."""
    return Pipeline([
        ("preprocessor", _tree_preprocessor(feature_columns)),
        ("classifier", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=8,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )),
    ])


def build_xgboost(feature_columns: list[str]) -> Pipeline:
    """XGBoost — strong tabular performance, built-in regularization."""
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=19,  # ~1/win_rate for balanced training
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    return Pipeline([
        ("preprocessor", _tree_preprocessor(feature_columns)),
        ("classifier", clf),
    ])


def build_lightgbm(feature_columns: list[str]) -> Pipeline:
    """LightGBM — fast gradient boosting, good for small-to-medium datasets."""
    if not _HAS_LGB:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
    clf = LGBMClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    return Pipeline([
        ("preprocessor", _tree_preprocessor(feature_columns)),
        ("classifier", clf),
    ])


def build_catboost(feature_columns: list[str]) -> Pipeline:
    """CatBoost — often well-calibrated out of the box."""
    if not _HAS_CAT:
        raise ImportError("catboost is not installed. Run: pip install catboost")
    clf = CatBoostClassifier(
        iterations=300,
        depth=3,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=0,
    )
    return Pipeline([
        ("preprocessor", _tree_preprocessor(feature_columns)),
        ("classifier", clf),
    ])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_model_builders(feature_columns: list[str]) -> dict[str, Pipeline]:
    """Return all available model pipelines keyed by name."""
    builders = {
        "logistic_regression": build_logistic_regression(feature_columns),
        "random_forest": build_random_forest(feature_columns),
        "gradient_boosting": build_gradient_boosting(feature_columns),
    }
    if _HAS_XGB:
        builders["xgboost"] = build_xgboost(feature_columns)
    if _HAS_LGB:
        builders["lightgbm"] = build_lightgbm(feature_columns)
    if _HAS_CAT:
        builders["catboost"] = build_catboost(feature_columns)
    return builders

