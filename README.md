# 🏎️ F1 Grand Prix Winner Prediction — Madrid 2026

> **Pre-race machine learning prediction of the 2026 Spanish Grand Prix winner**  
> A complete end-to-end ML pipeline built for portfolio demonstration.

---

## Project Overview

This project builds a full machine learning pipeline to predict the **probability of each driver winning** the 2026 Spanish Grand Prix at the Madring circuit in Madrid.

The key constraint: **only information available before the race on Sunday is used**.  
Qualifying results from Saturday are explicitly included — the central improvement over the previous Dutch GP model.

**Business Question:**  
*"We are standing in the Madring paddock on Saturday night, after qualifying. Who has the highest probability of winning tomorrow's race?"*

---

## Key Features of This Project

| Feature | Description |
|---|---|
| ✅ **No data leakage** | Strict temporal guard: shift(1) + cumsum pattern on all rolling features |
| ✅ **Qualifying features** | gap_to_pole_sec, teammate gap, Q session reached — not available in the previous model |
| ✅ **Walk-forward validation** | Expanding window across 5 validation seasons (2021–2025) |
| ✅ **Window comparison** | 3yr / 5yr / 7yr / all — best window selected empirically |
| ✅ **6 models compared** | LR, RF, GBT, XGBoost, LightGBM, CatBoost |
| ✅ **Leakage audit** | Formal audit table — pipeline will not run if leakage is detected |
| ✅ **SHAP interpretability** | Explains why the model favors specific drivers |
| ✅ **Reproducible** | Seeds fixed, `race_config.yaml` parametrizes the whole pipeline |
| ✅ **Reusable** | Change `circuit_id`, `round`, `year` to predict any GP |

---

## Project Structure

```
f1_madrid2026/
├── config/
│   └── race_config.yaml          # ← Single source of truth for the target GP
├── data/
│   ├── raw/                       # API JSON responses (gitignored, reproducible)
│   ├── processed/                 # Cleaned CSVs and features
│   └── external/                  # Circuit metadata
├── src/
│   ├── config.py                  # Central config loader
│   ├── data/
│   │   ├── download_historical.py # Jolpica API downloader
│   │   ├── load_results.py        # Race results normalizer
│   │   └── load_qualifying.py     # Qualifying results normalizer (NEW)
│   ├── features/
│   │   ├── build_race_features.py       # Historical race features
│   │   ├── build_qualifying_features.py # Qualifying features (NEW)
│   │   ├── build_features_combined.py   # Joined feature dataset
│   │   └── build_prediction.py         # Target GP prediction input
│   ├── modeling/
│   │   ├── baseline.py            # Naïve baselines for benchmarking
│   │   ├── model_registry.py      # All 6 model definitions
│   │   ├── walk_forward.py        # Temporal validation
│   │   └── window_comparison.py   # Historical window analysis
│   ├── evaluation/
│   │   ├── metrics.py             # Log loss, Brier, AUC, winner accuracy
│   │   ├── calibration.py         # Calibration curves
│   │   ├── leakage_audit.py       # Formal leakage audit (mandatory gate)
│   │   └── shap_analysis.py       # SHAP feature importance
│   └── prediction/
│       └── predict.py             # Final prediction runner
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_window_comparison.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_final_prediction.ipynb
├── models/                        # Serialized fitted models
├── reports/
│   ├── figures/                   # Charts and plots
│   └── leakage_audit.csv
├── tests/
├── pipeline.py                    # End-to-end orchestrator
└── requirements.txt
```

---

## Quick Start

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd ModeloML_F1GP_Madrid

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python pipeline.py
```

This will:
1. Download historical data (2018–2026) from the Jolpica API
2. Download qualifying results for the Madrid GP (already available)
3. Build all features
4. Run baselines, walk-forward validation, and window comparison
5. Run the leakage audit
6. Train the best model and generate the final prediction

### 3. Quick prediction (using cached data)

```bash
# If you already ran the pipeline once:
python pipeline.py --skip-download

# Override model and window directly:
python pipeline.py --skip-download --model xgboost --window 2021
```

### 4. Notebooks

```bash
jupyter notebook notebooks/
```

Run notebooks in order (01 → 06) for the full analytical narrative.

---

## Data Sources

### Primary: Jolpica API (Ergast-compatible)

- **URL:** `https://api.jolpi.ca/ergast/f1`
- **Coverage:** 1950–present (race results), 2003–present (qualifying)
- **Used for:** Race results, qualifying results, schedules
- **Advantages:** Free, no authentication, REST JSON, stable

### Optional: FastF1

- **Package:** `fastf1`
- **Used for:** Detailed session timing if more granular qualifying data is needed
- **Not required** for the core pipeline

---

## Feature Engineering

### Race History Features (from Jolpica race results)

| Feature | Description |
|---|---|
| `driver_career_wins_before` | Career wins before this race |
| `driver_season_points_before` | Championship points in current season |
| `driver_season_position_before` | Championship standing before this race |
| `driver_avg_finish_last_3/5/10` | Rolling average finish position |
| `driver_win_rate_last_3/5/10` | Rolling win rate |
| `driver_circuit_avg_finish_before` | Average finish at this specific circuit |
| `constructor_season_position_before` | Constructor championship standing |
| `constructor_avg_finish_last_3/5` | Team's recent form |

### Qualifying Features (NEW — from Jolpica qualifying results)

| Feature | Description |
|---|---|
| `qualifying_position` | Final qualifying classification (1 = pole) |
| `gap_to_pole_sec` | Time gap to pole-sitter in seconds |
| `driver_vs_teammate_q_gap_sec` | Gap to best-qualifying teammate |
| `team_best_qualifying_pos` | Best qualifying position for the team |
| `qualifying_session_numeric` | Q1=1, Q2=2, Q3=3 reached |
| `driver_avg_qualifying_last_3/5` | Historical qualifying form |

---

## Modeling Approach

### Problem Formulation

Binary classification per driver per race: `winner = 0/1`  
Win probabilities are softmax-normalized across all drivers per race so they sum to ~100%.

### Models Evaluated

| Model | Rationale |
|---|---|
| Logistic Regression | Excellent calibration, interpretable, baseline |
| Random Forest | Non-linearities, robust to small datasets |
| Gradient Boosting | Balance of complexity and performance |
| **XGBoost** | State-of-the-art tabular, built-in regularization |
| LightGBM | Fast, good with moderate-size data |
| CatBoost | Often well-calibrated, handles imbalance |

Final model selected based on **walk-forward Log Loss** — not by assumption.

### Temporal Validation

```
Train → 2018-2020   Validate → 2021
Train → 2018-2021   Validate → 2022
Train → 2018-2022   Validate → 2023
Train → 2018-2023   Validate → 2024
Train → 2018-2024   Validate → 2025
```

No random train/test split. No data from the future is used to train.

---

## Data Leakage Prevention

> **This is the most critical methodological aspect of the project.**

The following fields are **never** used as model features:
- `finish_position` (race result)
- `points` from the current race
- `laps`, `status` from the current race
- `won` (the target itself)

All historical features use `shift(1)` to ensure only past races are used.  
Qualifying features for the target race use the Saturday qualifying session — pre-race information, safe to include.

A formal leakage audit is run automatically before any prediction is generated.

---

## Results

> See `reports/` and `data/processed/` after running the pipeline.

### Final Prediction — Spanish Grand Prix 2026

| # | Driver | Team | Win % |
|---|---|---|---|
| 1 | Norris | McLaren | ~% |
| 2 | Antonelli | Mercedes | ~% |
| 3 | Verstappen | Red Bull | ~% |
| ... | ... | ... | ... |

*Results populated after running `python pipeline.py`*

---

## Evaluation Metrics

| Metric | Why it matters |
|---|---|
| **Log Loss** | Primary — penalizes overconfident wrong predictions |
| **Brier Score** | Calibration — squared probability error |
| **ROC-AUC** | Discrimination ability |
| **Winner Accuracy** | Did we predict the correct winner per race? |
| **Calibration Curve** | Does P=0.3 really mean 30% observed win rate? |

---

## Limitations

- **Madring is a new circuit** — no historical circuit-specific data available for Madrid
- **Small dataset** — ~200 races × ~20 drivers = ~4,000 rows; model complexity is constrained accordingly
- **F1 has high inherent randomness** — safety cars, mechanical failures, weather, and strategy decisions are not predictable from pre-race features alone
- **Grid penalties not modeled** — qualifying position may differ from starting grid if penalties are applied
- **Driver market changes** — 2026 is a new regulation era; some driver-team combinations have no history

---

## Future Improvements

- Incorporate sprint race results as additional signal
- Add weather forecast data (if demonstrated to improve validation metrics)
- Model grid penalties explicitly (qualify vs. actual starting position)
- Add pit stop strategy data (FastF1)
- Use Platt scaling or isotonic regression for better probability calibration
- Extend to predict podium probability (top 3) in addition to winner

---

## Stack

Python 3.11+ · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · SHAP · matplotlib · seaborn · FastF1 · PyYAML · Jupyter

---

## Reproducibility

All random seeds are set via `random_seed: 42` in `config/race_config.yaml`.  
Raw data is fully reproducible by re-running the download step.  
No absolute paths — all paths are relative to the project root.

To adapt this pipeline for a different Grand Prix, edit `config/race_config.yaml`:

```yaml
target:
  gp_name: "Singapore Grand Prix"
  year: 2026
  round: 17
  circuit_id: "marina_bay"
  race_date: "2026-10-11"
  qualifying_date: "2026-10-10"
```

---

## Author

Developed as part of a personal Data Engineering / Machine Learning portfolio.  
Demonstrates: ETL, Feature Engineering, Temporal Validation, Model Evaluation, Interpretability, and Reproducibility.

