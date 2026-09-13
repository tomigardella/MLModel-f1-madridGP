"""Genera la predicción final para el GP de España 2026 (Madrid).

Modelo: Gradient Boosting
Ventana: 5 años (2021-2026 Ronda 13)

Uso: python generate_final_prediction.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROCESSED_DIR, FIGURES_DIR, TARGET
from src.features.build_features_combined import MODEL_FEATURE_COLUMNS
from src.modeling.model_registry import get_model_builders
from src.evaluation.metrics import predict_race_probabilities

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(PROCESSED_DIR / 'features_combined.csv')
target = pd.read_csv(PROCESSED_DIR / 'madring_2026_prediction_input.csv')

# Configuración óptima determinada empíricamente
BEST_MODEL = 'gradient_boosting'
BEST_WINDOW_START = 2021   # ventana de 5 años

train = features[
    (features['season'] >= BEST_WINDOW_START) &
    ~((features['season'] == TARGET['year']) & (features['round'] >= TARGET['round']))
].dropna(subset=['won'])

print(f"Training: {len(train)} rows, "
      f"{train.groupby(['season','round']).ngroups} races, "
      f"{train['season'].min()}-{train['season'].max()}")

models = get_model_builders(MODEL_FEATURE_COLUMNS)
model = models[BEST_MODEL]
model.fit(train[MODEL_FEATURE_COLUMNS], train['won'].astype(int))
print("Model trained successfully")

predictions = predict_race_probabilities(model, target, MODEL_FEATURE_COLUMNS)
predictions['win_probability_pct'] = (predictions['win_probability'] * 100).round(1)
predictions['rank'] = range(1, len(predictions) + 1)

# --- Imprimir tabla ---
print()
print('=' * 65)
print('  GRAN PREMIO DE ESPANA 2026 (MADRID)')
print('  Model: Gradient Boosting | Window: 5yr (2021-2026 Rnd 13)')
print('=' * 65)
print(f"  {'#':<4} {'Driver':<20} {'Team':<20} {'Qual':>4}  {'Win%':>6}")
print('-' * 65)
for _, row in predictions.iterrows():
    qpos = int(row['qualifying_position']) if pd.notna(row['qualifying_position']) else 0
    drv = str(row['driver_id'])[:18]
    team = str(row['constructor_id'])[:18]
    print(f"  {int(row['rank']):<4} {drv:<20} {team:<20} P{qpos:<3}  {row['win_probability_pct']:>5.1f}%")
total = predictions['win_probability_pct'].sum()
print(f"\n  Total: {total:.1f}%")
print('=' * 65)

# --- Guardar CSV ---
save_cols = ['rank', 'driver_id', 'driver_code', 'constructor_id',
             'qualifying_position', 'gap_to_pole_sec',
             'raw_win_probability', 'win_probability', 'win_probability_pct']
avail = [c for c in save_cols if c in predictions.columns]
out = PROCESSED_DIR / 'final_prediction_madring_2026.csv'
predictions[avail].to_csv(out, index=False)
print(f"\nCSV saved -> {out}")

# --- Gráfico de barras ---
colors_map = {
    'mclaren': '#FF8000',
    'mercedes': '#00D2BE',
    'red_bull': '#1E41FF',
    'ferrari': '#DC0000',
    'alpine': '#0090FF',
    'aston_martin': '#006F62',
    'rb': '#6692FF',
    'haas': '#B6BABD',
    'williams': '#005AFF',
    'audi': '#E10600',
    'cadillac': '#AAAAAA',
}

top = predictions.head(12).copy()
bar_colors = [colors_map.get(c, '#999999') for c in top['constructor_id']]

short_names = {
    'max_verstappen': 'Verstappen',
    'arvid_lindblad': 'Lindblad',
}

def fmt_label(row):
    name = short_names.get(row['driver_id'], row['driver_id'].capitalize())
    qp = int(row['qualifying_position']) if pd.notna(row['qualifying_position']) else 0
    return f"{name} (P{qp})"

labels = [fmt_label(row) for _, row in top.iterrows()]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(
    range(len(top)),
    top['win_probability_pct'].values,
    color=bar_colors,
    edgecolor='black',
    linewidth=0.5,
)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(labels, fontsize=11)
ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10)
ax.set_xlabel('Win probability (%)', fontsize=12)
ax.set_title(
    'Pre-race win probability\nSpanish Grand Prix 2026 - Madring',
    fontsize=13, fontweight='bold',
)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

chart_path = FIGURES_DIR / 'final_prediction_madrid_2026.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart saved -> {chart_path}")
print("\nDone.")

