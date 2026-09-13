"""Registro de modelos — define todos los modelos candidatos como pipelines compatibles con sklearn.

Cada modelo:
  - Utiliza imputación por mediana para valores faltantes (cobertura de clasificación < 100%)
  - Utiliza un MissingIndicator para permitir que el modelo aprenda del patrón de valores faltantes
  - Es reproducible mediante la semilla aleatoria global

Agregar un nuevo modelo
-----------------------
1. Definir una función `build_<name>()` que retorne un Pipeline de sklearn.
2. Agregarlo al diccionario de constructores en la parte inferior.
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
# Preprocesamiento compartido
# ---------------------------------------------------------------------------

def _numeric_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Imputación por mediana + indicador de faltantes + escalado estándar."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
    )


def _tree_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Imputación por mediana + indicador de faltantes (sin escalado — los árboles no lo necesitan)."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Constructores de modelos individuales
# ---------------------------------------------------------------------------

def build_logistic_regression(feature_columns: list[str]) -> Pipeline:
    """Regresión Logística — línea base bien calibrada e interpretable."""
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
    """Random Forest — captura no linealidades, robusto ante conjuntos de datos pequeños."""
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
    """GradientBoosting de sklearn — más lento pero con buena calibración."""
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
    """XGBoost — sólido rendimiento tabular, regularización incorporada."""
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
        scale_pos_weight=19,  # ~1/win_rate para entrenamiento balanceado
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    return Pipeline([
        ("preprocessor", _tree_preprocessor(feature_columns)),
        ("classifier", clf),
    ])


def build_lightgbm(feature_columns: list[str]) -> Pipeline:
    """LightGBM — gradient boosting rápido, adecuado para conjuntos de datos pequeños a medianos."""
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
    """CatBoost — frecuentemente bien calibrado por defecto."""
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
# Registro
# ---------------------------------------------------------------------------

def get_model_builders(feature_columns: list[str]) -> dict[str, Pipeline]:
    """Retorna todos los pipelines de modelos disponibles indexados por nombre."""
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

