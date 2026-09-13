"""Ejecuta el pipeline completo de evaluación: auditoría de fuga + calibración + SHAP.

Uso: python -m src.evaluation.run_evaluation --model gradient_boosting --window 2018
"""

from __future__ import annotations

import argparse
import pickle

import matplotlib
matplotlib.use('Agg')  # backend no interactivo para ejecución headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FIGURES_DIR, MODEL_DIR, VALIDATION
from src.features.build_features_combined import MODEL_FEATURE_COLUMNS
from src.evaluation.leakage_audit import run_leakage_audit, print_audit
from src.evaluation.calibration import calibration_table, plot_calibration_curves
from src.evaluation.metrics import evaluate, predict_race_probabilities


def run_evaluation(model_name: str = "gradient_boosting", window_start: int = 2018) -> None:
    """Evaluación completa: auditoría de fuga + calibración + SHAP + tabla de métricas."""
    from src.modeling.model_registry import get_model_builders

    features = pd.read_csv(PROCESSED_DIR / "features_combined.csv")
    TARGET_YEAR = 2026
    TARGET_ROUND = 14

    # --- Paso 1: Auditoría de fuga ---
    print("\n=== LEAKAGE AUDIT ===")
    audit = run_leakage_audit(features)
    print_audit(audit)
    audit_path = PROCESSED_DIR.parent / "reports" / "leakage_audit.csv"
    audit_path.parent.mkdir(exist_ok=True)
    audit.to_csv(audit_path, index=False)
    print(f"Saved -> {audit_path}")

    # --- Paso 2: Entrenar en la ventana completa, evaluar en la temporada de prueba ---
    print(f"\n=== MODEL EVALUATION (train from {window_start}, test = {VALIDATION['test_season']}) ===")
    train = features[
        (features["season"] >= window_start) &
        (features["season"] < VALIDATION["test_season"])
    ].dropna(subset=["won"])

    test = features[
        features["season"] == VALIDATION["test_season"]
    ].dropna(subset=["won"])

    print(f"Train: {len(train)} rows, {train.groupby(['season','round']).ngroups} races")
    print(f"Test:  {len(test)} rows,  {test.groupby(['season','round']).ngroups} races")

    models = get_model_builders(MODEL_FEATURE_COLUMNS)
    results = []
    model_probas_for_cal = {}

    for name, model in models.items():
        model.fit(train[MODEL_FEATURE_COLUMNS], train["won"].astype(int))
        m = evaluate(model, test, MODEL_FEATURE_COLUMNS)
        m["model"] = name
        results.append(m)
        probas = model.predict_proba(test[MODEL_FEATURE_COLUMNS])[:, 1]
        model_probas_for_cal[name] = (test["won"].astype(int), probas)
        print(f"  {name:<25} LL={m['log_loss']:.4f}  BS={m['brier_score']:.4f}  "
              f"AUC={m['roc_auc']:.4f}  WA={m['winner_accuracy']:.3f}")

    metrics_df = pd.DataFrame(results).set_index("model").sort_values("log_loss")
    metrics_path = PROCESSED_DIR / f"evaluation_metrics_{window_start}.csv"
    metrics_df.to_csv(metrics_path)
    print(f"\nSaved metrics -> {metrics_path}")

    # --- Paso 3: Curvas de calibración ---
    print("\n=== CALIBRATION CURVES ===")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_calibration_curves(
        model_probas_for_cal,
        title=f"Calibration curves (test = {VALIDATION['test_season']})",
        save_path=FIGURES_DIR / "calibration_curves.png",
    )

    # --- Paso 4: Análisis SHAP para el mejor modelo ---
    print(f"\n=== SHAP ANALYSIS ({model_name}) ===")
    best_model = models[model_name]
    best_model.fit(train[MODEL_FEATURE_COLUMNS], train["won"].astype(int))
    try:
        from src.evaluation.shap_analysis import run_shap_analysis
        shap_df = run_shap_analysis(best_model, test, MODEL_FEATURE_COLUMNS, model_name)
        print("\nTop 15 features by SHAP importance:")
        print(shap_df.head(15)[["rank", "feature", "mean_abs_shap"]].to_string(index=False))
    except Exception as e:
        print(f"  [warn] SHAP skipped: {e}")

    # --- Paso 5: Resumen de walk-forward por temporada ---
    print("\n=== PER-SEASON WALK-FORWARD SUMMARY ===")
    try:
        wf = pd.read_csv(PROCESSED_DIR / "walk_forward_results.csv")
        summary = wf.groupby("model")[["log_loss", "brier_score", "winner_accuracy"]].mean().round(4)
        print(summary.sort_values("log_loss").to_string())
    except FileNotFoundError:
        print("  walk_forward_results.csv not found — run walk_forward.py first")

    print("\n=== EVALUATION COMPLETE ===")
    print(f"Best model by LL on test set: {metrics_df.index[0]}")
    print(f"  Log Loss:        {metrics_df.iloc[0]['log_loss']:.4f}")
    print(f"  Winner Accuracy: {metrics_df.iloc[0]['winner_accuracy']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gradient_boosting")
    p.add_argument("--window", type=int, default=2018)
    args = p.parse_args()
    run_evaluation(args.model, args.window)

