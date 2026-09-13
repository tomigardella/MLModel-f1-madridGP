"""Ajuste de hiperparámetros para el modelo con mejor rendimiento.

Utiliza RandomizedSearchCV para buscar eficientemente en el espacio de hiperparámetros.
Solo se ejecuta en el mejor modelo seleccionado para evitar el sobreajuste del propio
proceso de búsqueda.

Decisión: RandomizedSearchCV en lugar de GridSearch — con conjuntos de datos pequeños de F1
(~3000 filas), una búsqueda en cuadrícula completa (GridSearch) es excesiva y corre el riesgo
de filtración del conjunto de búsqueda. La cantidad de iteraciones se mantiene conservadora (n_iter=20)
por la misma razón.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline

from src.config import PROCESSED_DIR, RANDOM_SEED, VALIDATION
from src.features.build_features_combined import MODEL_FEATURE_COLUMNS

PARAM_GRIDS = {
    "xgboost": {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.15],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__reg_alpha": [0, 0.1, 0.5, 1.0],
        "classifier__reg_lambda": [0.5, 1.0, 2.0],
    },
    "lightgbm": {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__min_child_samples": [5, 10, 20],
        "classifier__num_leaves": [15, 31, 63],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 500, 1000],
        "classifier__max_depth": [None, 5, 10, 15],
        "classifier__min_samples_leaf": [3, 5, 10],
        "classifier__max_features": ["sqrt", "log2", 0.5],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 4],
        "classifier__min_samples_leaf": [5, 8, 15],
        "classifier__subsample": [0.7, 0.8, 0.9],
    },
}


def tune_model(
    model: Pipeline,
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_iter: int = 20,
) -> tuple[Pipeline, dict]:
    """Ejecuta RandomizedSearchCV usando una división predefinida de entrenamiento/validación.

    Parámetros
    ----------
    model:
        Pipeline de sklearn sin ajustar.
    model_name:
        Clave en PARAM_GRIDS.
    train, val:
        Conjuntos temporales de entrenamiento y validación.
    feature_columns:
        Columnas de características del modelo.
    n_iter:
        Número de configuraciones aleatorias de parámetros a probar.

    Retorna
    -------
    (best_pipeline, best_params)
    """
    if feature_columns is None:
        feature_columns = MODEL_FEATURE_COLUMNS

    if model_name not in PARAM_GRIDS:
        print(f"  [warn] No param grid for {model_name}, returning default model")
        return model, {}

    param_grid = PARAM_GRIDS[model_name]

    # Combinar train+val para poder usar PredefinedSplit
    combined = pd.concat([train, val], ignore_index=True)
    test_fold = [-1] * len(train) + [0] * len(val)
    ps = PredefinedSplit(test_fold)

    X = combined[feature_columns]
    y = combined["won"].astype(int)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=ps,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1,
        refit=True,
    )
    search.fit(X, y)

    print(f"  Best log_loss: {-search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


if __name__ == "__main__":
    from src.config import VALIDATION
    from src.modeling.model_registry import get_model_builders

    features = pd.read_csv(PROCESSED_DIR / "features_combined.csv")
    features = features.dropna(subset=["won"])

    # Usar las últimas 2 temporadas como validación, el resto como entrenamiento
    val_season = VALIDATION["test_season"]
    train = features[features["season"] < val_season - 1]
    val = features[features["season"].isin([val_season - 1, val_season])]

    models = get_model_builders(MODEL_FEATURE_COLUMNS)
    for name in ["xgboost", "lightgbm"]:
        if name in models:
            print(f"\nTuning {name}...")
            best_model, best_params = tune_model(models[name], name, train, val)
            out = PROCESSED_DIR / f"best_params_{name}.json"
            import json
            with open(out, "w") as f:
                json.dump({k: str(v) for k, v in best_params.items()}, f, indent=2)
            print(f"  Saved best params -> {out}")

