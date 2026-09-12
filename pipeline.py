"""End-to-end orchestration pipeline for the F1 GP winner prediction project.

This script runs the full pipeline from data download to final prediction.
All parameters are read from config/race_config.yaml.

Steps
-----
1.  Download historical race results (Jolpica API)
2.  Download historical qualifying results (Jolpica API)
3.  Download target-race qualifying (already available for Madrid 2026)
4.  Load and normalize race results
5.  Load and normalize qualifying results
6.  Build combined features (race + qualifying)
7.  Build target prediction input
8.  Run naïve baselines
9.  Run walk-forward validation
10. Run historical window comparison
11. Run leakage audit
12. Train final model and generate prediction
13. Run SHAP analysis

Usage
-----
    # Full pipeline
    python pipeline.py

    # Skip download (if data already cached)
    python pipeline.py --skip-download

    # Quick run (only prediction, using cached features)
    python pipeline.py --predict-only --model xgboost --window 2021
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HISTORY, TARGET, PROCESSED_DIR, FIGURES_DIR, MODEL_DIR


def step(n: int, label: str) -> None:
    print(f"\n{'='*70}")
    print(f"  Step {n}: {label}")
    print(f"{'='*70}")


def run_pipeline(
    skip_download: bool = False,
    predict_only: bool = False,
    best_model: str | None = None,
    best_window: int | None = None,
) -> dict:
    """Execute the full ML pipeline.

    Parameters
    ----------
    skip_download:  If True, skip API downloads (use cached raw data).
    predict_only:   If True, skip validation and go straight to prediction.
    best_model:     Model name override (default: determined by walk-forward).
    best_window:    Window start year override (default: determined by comparison).
    """
    t0 = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict = {}

    # -----------------------------------------------------------------------
    # PHASE 1: DATA DOWNLOAD
    # -----------------------------------------------------------------------
    if not skip_download:
        step(1, "Downloading race results (Jolpica API)")
        from src.data.download_historical import (
            download_race_results, download_schedule, download_target_qualifying
        )
        for year in range(HISTORY["start_year"], HISTORY["end_year"] + 1):
            download_schedule(year)
            pages = download_race_results(year)
            print(f"  {year}: {len(pages)} result page(s)")

        step(2, "Downloading qualifying results (Jolpica API)")
        from src.data.download_historical import download_qualifying_season
        for year in range(HISTORY["start_year"], HISTORY["end_year"] + 1):
            pages = download_qualifying_season(year)
            print(f"  {year}: {len(pages)} qualifying page(s)")

        step(3, f"Downloading target qualifying ({TARGET['gp_name']} {TARGET['year']})")
        path = download_target_qualifying(TARGET["year"], TARGET["round"])
        print(f"  Saved → {path}")
    else:
        print("\n[skip-download] Using cached raw data.")

    # -----------------------------------------------------------------------
    # PHASE 2: LOAD & FEATURE ENGINEERING
    # -----------------------------------------------------------------------
    step(4, "Loading and normalizing race results")
    from src.data.load_results import load_results
    seasons = list(range(HISTORY["start_year"], HISTORY["end_year"] + 1))
    results = load_results(seasons)
    print(f"  Loaded {len(results)} race result rows")

    step(5, "Loading and normalizing qualifying results")
    from src.data.load_qualifying import load_qualifying
    qualifying = load_qualifying(seasons)
    print(f"  Loaded {len(qualifying)} qualifying rows")
    print(f"  Coverage: {qualifying['qualifying_position'].notna().mean():.1%}")

    step(6, "Building combined feature dataset")
    from src.features.build_features_combined import build_combined_features, save_combined_features
    features = build_combined_features(results, qualifying)
    feat_path = save_combined_features(features)
    print(f"  {len(features)} rows × {features.shape[1]} columns → {feat_path}")
    outputs["features"] = str(feat_path)

    step(7, "Building target prediction input")
    from src.features.build_prediction import build_target_features, save_target_features
    target_feats = build_target_features(results, qualifying)
    target_path = save_target_features(target_feats)
    print(f"  {len(target_feats)} drivers → {target_path}")
    outputs["target_features"] = str(target_path)

    if predict_only:
        print("\n[predict-only] Skipping validation steps.")
        final_model = best_model or "xgboost"
        final_window = best_window or 2021
    else:
        # -------------------------------------------------------------------
        # PHASE 3: BASELINES
        # -------------------------------------------------------------------
        step(8, "Naïve baselines")
        from src.modeling.baseline import evaluate_qualifying_baseline, evaluate_win_rate_baseline
        q_base = evaluate_qualifying_baseline(features)
        wr_base = evaluate_win_rate_baseline(features)
        print(f"  Qualifying position baseline: {q_base}")
        print(f"  Career win rate baseline:     {wr_base}")
        outputs["baselines"] = {"qualifying": q_base, "win_rate": wr_base}

        # -------------------------------------------------------------------
        # PHASE 4: WALK-FORWARD VALIDATION
        # -------------------------------------------------------------------
        step(9, "Walk-forward temporal validation (all models)")
        from src.modeling.walk_forward import run_walk_forward, summarize_walk_forward, save_walk_forward
        wf = run_walk_forward(features)
        wf_path = save_walk_forward(wf)
        print(f"\n  Results saved → {wf_path}")
        wf_summary = summarize_walk_forward(wf)
        print(wf_summary.to_string())
        outputs["walk_forward"] = str(wf_path)

        # -------------------------------------------------------------------
        # PHASE 5: WINDOW COMPARISON
        # -------------------------------------------------------------------
        step(10, "Historical window comparison (3yr / 5yr / 7yr / all)")
        from src.modeling.window_comparison import (
            run_window_comparison, summarize_window_comparison,
            save_window_comparison, best_window as find_best_window
        )
        wc = run_window_comparison(features)
        wc_path = save_window_comparison(wc)
        wc_summary = summarize_window_comparison(wc)
        print(f"\n  Results saved → {wc_path}")
        print(wc_summary.to_string(index=False))
        outputs["window_comparison"] = str(wc_path)

        # Select best combination
        best_combo = (
            wc_summary.sort_values("log_loss").iloc[0]
        )
        final_model = best_model or best_combo["model"]
        final_window = best_window or int(wc_summary.groupby("window")["log_loss"].mean()
                                         .sort_values().index[0].replace("yr", "") if "yr" in
                                         wc_summary["window"].iloc[0] else best_combo["window_start_year"])

        print(f"\n  → Selected model: {final_model}")
        print(f"  → Selected window: starts {final_window}")

    # -----------------------------------------------------------------------
    # PHASE 6: LEAKAGE AUDIT
    # -----------------------------------------------------------------------
    step(11, "Data leakage audit (mandatory gate)")
    from src.evaluation.leakage_audit import run_leakage_audit, print_audit
    audit = run_leakage_audit(features)
    print_audit(audit)  # raises RuntimeError if critical leakage found
    audit_path = PROCESSED_DIR.parent / "reports" / "leakage_audit.csv"
    audit_path.parent.mkdir(exist_ok=True)
    audit.to_csv(audit_path, index=False)
    outputs["leakage_audit"] = str(audit_path)

    # -----------------------------------------------------------------------
    # PHASE 7: FINAL PREDICTION
    # -----------------------------------------------------------------------
    step(12, f"Final prediction — {TARGET['gp_name']} {TARGET['year']}")
    from src.prediction.predict import (
        train_final_model, generate_prediction, print_prediction,
        plot_prediction, save_prediction
    )
    model = train_final_model(features, final_window, final_model)
    target_df = pd.read_csv(target_path)
    predictions = generate_prediction(model, target_df)
    print_prediction(predictions)
    plot_prediction(predictions)
    pred_path = save_prediction(predictions)
    outputs["predictions"] = str(pred_path)

    # -----------------------------------------------------------------------
    # PHASE 8: SHAP ANALYSIS
    # -----------------------------------------------------------------------
    step(13, "SHAP interpretability analysis")
    try:
        from src.evaluation.shap_analysis import run_shap_analysis
        from src.config import VALIDATION
        test_data = features[features["season"] == VALIDATION["test_season"]].dropna(subset=["won"])
        shap_df = run_shap_analysis(model, test_data, from_features_module(), final_model)
        print(f"\n  Top 10 features by SHAP importance:")
        print(shap_df.head(10)[["rank", "feature", "mean_abs_shap"]].to_string(index=False))
        outputs["shap"] = str(PROCESSED_DIR / f"shap_importance_{final_model}.csv")
    except Exception as e:
        print(f"  [warn] SHAP analysis skipped: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Pipeline completed in {elapsed:.1f}s")
    print(f"{'='*70}")
    return outputs


def from_features_module():
    from src.features.build_features_combined import MODEL_FEATURE_COLUMNS
    return MODEL_FEATURE_COLUMNS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F1 GP winner prediction pipeline")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--model", type=str, default=None, help="Model name override")
    p.add_argument("--window", type=int, default=None, help="Window start year override")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = run_pipeline(
        skip_download=args.skip_download,
        predict_only=args.predict_only,
        best_model=args.model,
        best_window=args.window,
    )
    print("\nOutputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")

