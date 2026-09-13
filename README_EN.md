# 🏎️ F1 Grand Prix Winner Prediction — Spanish GP 2026 (Madrid)

> **Machine Learning pipeline to predict the 2026 Spanish Grand Prix winner**  
> End-to-end Data Science / ML portfolio project.

---

## Project Overview

This project builds a complete Machine Learning pipeline to predict the **win probability for each driver** at the 2026 Spanish Grand Prix at the brand-new Madring circuit in Madrid.

The fundamental constraint: **only information available before Sunday's race is used**. Saturday qualifying results are explicitly included as features — the central improvement over the previous Dutch GP model.

**Business Question:**  
*"We are standing in the Madring paddock on Saturday night, after qualifying. Who has the highest probability of winning tomorrow's race?"*

---

## Key Features

| Feature | Description |
|---|---|
| ✅ **No data leakage** | Strict temporal guard: `shift(1)` + `cumsum` pattern on all historical features |
| ✅ **Qualifying features** | `gap_to_pole_sec`, teammate gap, session reached — absent in the previous model |
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
ModeloML_F1GP_Madrid/
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
│   │   ├── walk_forward.py        # Temporal walk-forward validation
│   │   ├── window_comparison.py   # Historical window analysis
│   │   └── hyperparameter_tuning.py # RandomizedSearchCV tuning
│   ├── evaluation/
│   │   ├── metrics.py             # Log loss, Brier, AUC, winner accuracy
│   │   ├── calibration.py         # Calibration curves
│   │   ├── leakage_audit.py       # Formal leakage audit (mandatory gate)
│   │   ├── shap_analysis.py       # SHAP feature importance
│   │   └── run_evaluation.py      # Complete evaluation runner
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
│   └── test_features.py           # Unit tests (8 tests, all passing)
├── pipeline.py                    # End-to-end orchestrator
├── generate_final_prediction.py   # Standalone prediction script
├── requirements.txt
├── README_EN.md                   # This file (English)
└── README.md                      # Spanish version
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
3. Build all features (41 features, 93.2% qualifying coverage)
4. Run baselines, walk-forward validation, and window comparison
5. Run the leakage audit
6. Train the best model and generate the final prediction

### 3. Quick runs (with cached data)

```bash
# If you already ran the pipeline once:
python pipeline.py --skip-download

# Override model and window:
python pipeline.py --skip-download --model gradient_boosting --window 2021

# Standalone final prediction:
python generate_final_prediction.py
```

### 4. Full evaluation

```bash
python -m src.evaluation.run_evaluation --model gradient_boosting --window 2018
```

### 5. Unit tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Data Sources

### Primary: Jolpica API (Ergast-compatible)

- **URL:** `https://api.jolpi.ca/ergast/f1`
- **Coverage:** 1950–present (race results), 2003–present (qualifying)
- **Used for:** Race results, qualifying times, schedules
- **Advantages:** Free, no authentication, REST JSON, stable

### Optional: FastF1

- **Package:** `fastf1`
- **Used for:** Detailed timing data if more granular qualifying features are needed
- **Not required** for the core pipeline

---

## Feature Engineering

### Race History Features (from Jolpica race results endpoint)

| Feature | Description |
|---|---|
| `driver_career_wins_before` | Career wins before this race |
| `driver_season_points_before` | Championship points in current season |
| `driver_season_position_before` | Championship standing before this race |
| `driver_avg_finish_last_3/5/10` | Rolling average finish position |
| `driver_win_rate_last_3/5/10` | Rolling win rate |
| `driver_circuit_avg_finish_before` | Historical average finish at this specific circuit |
| `constructor_season_position_before` | Constructor championship standing |
| `constructor_avg_finish_last_3/5` | Team's recent form |

### Qualifying Features (NEW — from Jolpica qualifying endpoint)

| Feature | Description |
|---|---|
| `qualifying_position` | Final qualifying classification (1 = pole) |
| `gap_to_pole_sec` | Time gap to pole-sitter in seconds |
| `driver_vs_teammate_q_gap_sec` | Gap to best-qualifying teammate |
| `team_best_qualifying_pos` | Best qualifying position for the team |
| `qualifying_session_numeric` | Session reached: Q1=1, Q2=2, Q3=3 |
| `driver_avg_qualifying_last_3/5` | Historical qualifying form (rolling) |

---

## Modeling Approach

### Problem Formulation

Binary classification per driver per race: `won = 0/1`  
Win probabilities are softmax-normalized across all drivers per race so they sum to ~100%.

### Models Evaluated

| Model | Rationale |
|---|---|
| Logistic Regression | Well calibrated, interpretable, baseline |
| Random Forest | Captures non-linearities, robust to small datasets |
| **Gradient Boosting** ⭐ | Best balance of complexity and performance |
| XGBoost | State-of-the-art tabular, built-in regularization |
| LightGBM | Fast, good with moderate-size data |
| CatBoost | Often well-calibrated out of the box, handles imbalance |

The final model is selected based on **walk-forward Log Loss** — not by assumption.

### Temporal Walk-Forward Validation

```
Train: 2018-2020   Validate: 2021
Train: 2018-2021   Validate: 2022
Train: 2018-2022   Validate: 2023
Train: 2018-2023   Validate: 2024
Train: 2018-2024   Validate: 2025
```

No random train/test split. No data from the future leaks into training.

---

## Data Leakage Prevention

> **This is the most critical methodological aspect of the project.**

The following fields are **never** used as model features:
- `finish_position` (race result)
- `points` from the current race
- `laps`, `status` from the current race
- `won` (the target variable itself)

All historical features use `shift(1)` to ensure only past races are used.  
Qualifying features for the target race use Saturday's qualifying session — pre-race information, safe to include.

**A formal leakage audit runs automatically** before any prediction is generated.

---

## Results

### Model Comparison — Walk-Forward (2021–2025, full history)

| Model | Log Loss | Brier Score | ROC-AUC | Winner Accuracy |
|---|---|---|---|---|
| **Gradient Boosting** | **0.1072** | **0.0309** | **0.950** | **60.8%** |
| XGBoost | 0.1276 | 0.0388 | 0.941 | 53.6% |
| Random Forest | 0.1342 | 0.0363 | 0.951 | 52.1% |
| LightGBM | 0.1412 | 0.0413 | 0.930 | 50.2% |
| CatBoost | 0.1473 | 0.0446 | 0.940 | 53.4% |
| Logistic Regression | 0.2608 | 0.0797 | 0.942 | 54.8% |

> **Naïve baselines:** Qualifying P1 always wins = 55.4% · Career win rate = 34.4%  
> Gradient Boosting (60.8%) **beats the qualifying baseline** — the model adds real value.

### Historical Window Comparison (Gradient Boosting)

| Window | Log Loss | Winner Accuracy |
|---|---|---|
| **5yr (2021–)** | **0.1129** | **63.5%** |
| 7yr (2019–) | 0.1067 | 56.5% |
| all (2018–) | 0.1072 | 60.8% |
| 3yr (2023–) | 0.1474 | 54.2% |

> **Best window: 5yr** — captures the current competitive era without outdated dynamics from 2018–2020.

### SHAP Feature Importance (Top 10)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `qualifying_position` | 0.307 |
| 2 | `constructor_avg_finish_last_3` | 0.201 |
| 3 | `driver_win_rate_last_10` | 0.162 |
| 4 | `gap_to_pole_sec` | 0.150 |
| 5 | `driver_avg_finish_last_3` | 0.145 |
| 6 | `constructor_career_points_before` | 0.099 |
| 7 | `driver_avg_finish_last_10` | 0.089 |
| 8 | `driver_career_points_before` | 0.088 |
| 9 | `driver_season_points_before` | 0.066 |
| 10 | `driver_season_avg_qualifying_before` | 0.053 |

> Qualifying position and gap to pole are the **top 2 features by SHAP** — validating the central methodological contribution of this project.

### Final Prediction — Spanish Grand Prix 2026

> **Model:** Gradient Boosting · **Window:** 5yr (2021–2026 R13) · **Qualifying:** Saturday 2026-09-12

| # | Driver | Team | Qualifying | Win % |
|---|---|---|---|---|
| 1 | **Norris** | McLaren | P1 | **45.8%** |
| 2 | **Antonelli** | Mercedes | P2 | **41.9%** |
| 3 | Verstappen | Red Bull | P3 | 4.6% |
| 4 | Russell | Mercedes | P6 | 1.3% |
| 5 | Hamilton | Ferrari | P4 | 1.1% |
| 6 | Leclerc | Ferrari | P5 | 0.8% |
| 7 | Piastri | McLaren | P7 | 0.4% |
| ... | ... | ... | ... | ... |

> **Predicted winner: Lando Norris (McLaren)** — Pole position, strongest recent form.  
> McLaren + Mercedes combined: **>87%** win probability. Madring is a new circuit — all drivers start with zero circuit-specific history.

---

## Evaluation Metrics

| Metric | Why it matters |
|---|---|
| **Log Loss** | Primary — penalizes overconfident wrong predictions |
| **Brier Score** | Calibration — squared probability error |
| **ROC-AUC** | Discrimination ability |
| **Winner Accuracy** | Did we correctly identify the race winner? |
| **Calibration Curve** | Does P=0.3 really mean 30% observed win rate? |

---

## Limitations

- **Madring is a new circuit** — no historical circuit-specific data available; all drivers start with zero circuit features
- **Small dataset** — ~200 races × ~20 drivers = ~4,000 rows; model complexity constrained accordingly
- **High inherent randomness in F1** — safety cars, reliability, weather, and strategy are not predictable from pre-race features alone
- **Grid penalties not modeled** — qualifying position may differ from actual starting grid if penalties are applied
- **2026 regulation change** — team/driver dynamics in this era are different; patterns may transfer imperfectly

---

## Future Improvements

- Incorporate sprint race results as additional signal
- Add weather forecast data (if demonstrated to improve validation metrics)
- Model grid penalties explicitly
- Add pit stop strategy data (FastF1)
- Platt scaling or isotonic regression for better probability calibration
- Extend to predict podium probability (top 3)

---

## Tech Stack

Python 3.12 · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · SHAP · matplotlib · FastF1 · PyYAML · Jupyter

---

## Reproducibility

All random seeds are set via `random_seed: 42` in `config/race_config.yaml`.  
Raw data is fully reproducible by re-running the download step.  
No absolute paths — all paths are relative to the project root.

### Adapt for a different Grand Prix

Edit only `config/race_config.yaml`:

```yaml
target:
  gp_name: "Singapore Grand Prix"
  year: 2026
  round: 17
  circuit_id: "marina_bay"
  race_date: "2026-10-11"
  qualifying_date: "2026-10-10"
```

Then run:
```bash
python pipeline.py --skip-download  # if data is already downloaded
```

---

## Author

Developed as a personal Data Engineering / Machine Learning portfolio project.  
Demonstrates: ETL, Feature Engineering, Temporal Validation, Model Evaluation, Interpretability, and Reproducibility.

---

## License

Personal / portfolio use. Data sourced from the public Jolpica API (Ergast-compatible).

---

> 📄 **Versión en español:** [README.md](README.md)

