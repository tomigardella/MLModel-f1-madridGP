# 🏎️ Predicción del Ganador — Gran Premio de España 2026 (Madrid)

> **Pipeline de Machine Learning para predecir el ganador del Gran Premio de España 2026 (Madrid)**  
> Proyecto de portfolio de Data Science / ML — ejecutable de extremo a extremo.

---

## Descripción general

Este proyecto construye un pipeline completo de Machine Learning para predecir la **probabilidad de victoria de cada piloto** en el Gran Premio de España 2026, celebrado en el nuevo circuito Madring de Madrid.

La restricción fundamental: **solo se utiliza información disponible antes del inicio de la carrera del domingo**. Los datos de clasificación del sábado se incorporan explícitamente como features — la mejora central respecto al modelo anterior del GP de Países Bajos.

**Pregunta de negocio:**  
*"Estamos en el paddock del Madring el sábado por la noche, después de la clasificación. ¿Quién tiene mayor probabilidad de ganar la carrera de mañana?"*

---

## Características principales del proyecto

| Característica | Descripción |
|---|---|
| ✅ **Sin fuga de datos** | Guardia temporal estricta: patrón `shift(1)` + `cumsum` en todas las features históricas |
| ✅ **Features de clasificación** | `gap_to_pole_sec`, diferencia con compañero de equipo, sesión alcanzada — ausentes en el modelo anterior |
| ✅ **Validación temporal walk-forward** | Ventana expandida sobre 5 temporadas de validación (2021–2025) |
| ✅ **Comparación de ventanas históricas** | 3yr / 5yr / 7yr / todo — la mejor ventana se selecciona empíricamente |
| ✅ **6 modelos comparados** | LR, RF, GBT, XGBoost, LightGBM, CatBoost |
| ✅ **Auditoría de leakage** | Tabla formal — el pipeline no corre si detecta fuga de datos |
| ✅ **Interpretabilidad SHAP** | Explica por qué el modelo favorece a determinados pilotos |
| ✅ **Reproducible** | Semillas fijas, `race_config.yaml` parametriza todo el pipeline |
| ✅ **Reutilizable** | Cambiar `circuit_id`, `round` y `year` permite predecir cualquier GP |

---

## Estructura del proyecto

```
ModeloML_F1GP_Madrid/
├── config/
│   └── race_config.yaml          # ← Única fuente de verdad para el GP objetivo
├── data/
│   ├── raw/                       # Respuestas JSON de la API (en .gitignore, reproducibles)
│   ├── processed/                 # CSVs limpios y features
│   └── external/                  # Metadatos del circuito
├── src/
│   ├── config.py                  # Cargador central de configuración
│   ├── data/
│   │   ├── download_historical.py # Descargador de la API Jolpica
│   │   ├── load_results.py        # Normalizador de resultados de carrera
│   │   └── load_qualifying.py     # Normalizador de clasificación (NUEVO)
│   ├── features/
│   │   ├── build_race_features.py       # Features históricas de carrera
│   │   ├── build_qualifying_features.py # Features de clasificación (NUEVO)
│   │   ├── build_features_combined.py   # Dataset combinado de features
│   │   └── build_prediction.py         # Input de predicción para el GP objetivo
│   ├── modeling/
│   │   ├── baseline.py            # Baselines naïve para comparación
│   │   ├── model_registry.py      # Definiciones de los 6 modelos
│   │   ├── walk_forward.py        # Validación temporal expanding window
│   │   ├── window_comparison.py   # Análisis de ventanas históricas
│   │   └── hyperparameter_tuning.py # Tuning con RandomizedSearchCV
│   ├── evaluation/
│   │   ├── metrics.py             # Log loss, Brier, AUC, winner accuracy
│   │   ├── calibration.py         # Curvas de calibración
│   │   ├── leakage_audit.py       # Auditoría formal de fuga de datos (puerta obligatoria)
│   │   ├── shap_analysis.py       # Importancia de features por SHAP
│   │   └── run_evaluation.py      # Script de evaluación completa
│   └── prediction/
│       └── predict.py             # Generador de predicción final
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_window_comparison.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_final_prediction.ipynb
├── models/                        # Modelos serializados (.pkl)
├── reports/
│   ├── figures/                   # Gráficos y visualizaciones
│   └── leakage_audit.csv          # Auditoría de fuga de datos
├── tests/
│   └── test_features.py           # Tests unitarios (8 tests)
├── pipeline.py                    # Orquestador end-to-end
├── generate_final_prediction.py   # Script de predicción final
├── requirements.txt
├── README.md                      # Este archivo (español)
└── README_EN.md                   # English version
```

---

## Inicio rápido

### 1. Clonar y configurar el entorno

```bash
git clone <url-del-repositorio>
cd ModeloML_F1GP_Madrid

python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Ejecutar el pipeline completo

```bash
python pipeline.py
```

Este comando:
1. Descarga datos históricos 2018–2026 desde la API Jolpica
2. Descarga los datos de clasificación del GP de Madrid (ya disponibles)
3. Construye todas las features (41 features, cobertura de qualifying 93.2%)
4. Ejecuta baselines, validación walk-forward y comparación de ventanas
5. Ejecuta la auditoría de leakage
6. Entrena el mejor modelo y genera la predicción final

### 3. Predicción rápida (con datos ya descargados)

```bash
# Si ya corriste el pipeline una vez:
python pipeline.py --skip-download

# Sobrescribir modelo y ventana directamente:
python pipeline.py --skip-download --model gradient_boosting --window 2021

# Solo la predicción final:
python generate_final_prediction.py
```

### 4. Evaluación completa

```bash
python -m src.evaluation.run_evaluation --model gradient_boosting --window 2018
```

### 5. Tests unitarios

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Fuentes de datos

### Principal: API Jolpica (compatible con Ergast)

- **URL:** `https://api.jolpi.ca/ergast/f1`
- **Cobertura:** 1950–presente (resultados de carrera), 2003–presente (clasificación)
- **Uso:** Resultados de carrera, tiempos de clasificación, calendarios
- **Ventajas:** Gratuita, sin autenticación, REST JSON, estable

### Opcional: FastF1

- **Paquete:** `fastf1`
- **Uso:** Datos de timing más granulares si se desea ampliar las features
- **No requerido** para el pipeline principal

---

## Ingeniería de Features

### Features de historial de carrera (fuente: resultados Jolpica)

| Feature | Descripción |
|---|---|
| `driver_career_wins_before` | Victorias de carrera acumuladas antes de esta fecha |
| `driver_season_points_before` | Puntos del campeonato en la temporada actual |
| `driver_season_position_before` | Posición en el campeonato antes de esta carrera |
| `driver_avg_finish_last_3/5/10` | Promedio de posición de llegada en las últimas 3/5/10 carreras |
| `driver_win_rate_last_3/5/10` | Tasa de victorias en las últimas 3/5/10 carreras |
| `driver_circuit_avg_finish_before` | Promedio histórico en este circuito específico |
| `constructor_season_position_before` | Posición del constructor en el campeonato |
| `constructor_avg_finish_last_3/5` | Forma reciente del equipo |

### Features de clasificación (NUEVO — fuente: endpoint de qualifying Jolpica)

| Feature | Descripción |
|---|---|
| `qualifying_position` | Posición final en clasificación (1 = pole) |
| `gap_to_pole_sec` | Diferencia con el tiempo de pole en segundos |
| `driver_vs_teammate_q_gap_sec` | Diferencia con el compañero de equipo en clasificación |
| `team_best_qualifying_pos` | Mejor posición de clasificación del equipo |
| `qualifying_session_numeric` | Sesión alcanzada: Q1=1, Q2=2, Q3=3 |
| `driver_avg_qualifying_last_3/5` | Forma histórica en clasificación |

---

## Enfoque de modelado

### Formulación del problema

Clasificación binaria por piloto por carrera: `ganó = 0/1`  
Las probabilidades se normalizan con softmax por carrera para que sumen ~100%.

### Modelos evaluados

| Modelo | Justificación |
|---|---|
| Regresión Logística | Excelente calibración, interpretable, baseline |
| Random Forest | Captura no linealidades, robusto con datasets pequeños |
| **Gradient Boosting** ⭐ | Mejor balance complejidad-rendimiento |
| XGBoost | State-of-the-art tabular, regularización integrada |
| LightGBM | Rápido, buen desempeño con datos de tamaño moderado |
| CatBoost | Bien calibrado out-of-the-box, maneja desbalance |

El modelo final se selecciona por **Log Loss en validación walk-forward**, no por suposición.

### Validación temporal walk-forward

```
Entrenamiento: 2018-2020   Validación: 2021
Entrenamiento: 2018-2021   Validación: 2022
Entrenamiento: 2018-2022   Validación: 2023
Entrenamiento: 2018-2023   Validación: 2024
Entrenamiento: 2018-2024   Validación: 2025
```

Sin división aleatoria train/test. Sin datos del futuro en el entrenamiento.

---

## Prevención de fuga de datos

> **Este es el aspecto metodológico más crítico del proyecto.**

Los siguientes campos **nunca** se usan como features del modelo:
- `finish_position` (resultado de la carrera)
- `points` de la carrera actual
- `laps`, `status` de la carrera actual
- `won` (la variable objetivo)

Todas las features históricas usan `shift(1)` para garantizar que solo se use información de carreras pasadas.  
Las features de clasificación del GP objetivo provienen de la sesión del sábado — información pre-carrera, válida.

**Una auditoría formal de leakage corre automáticamente** antes de generar cualquier predicción.

---

## Resultados

### Comparación de modelos — Walk-Forward (2021–2025, historia completa)

| Modelo | Log Loss | Brier Score | ROC-AUC | Winner Accuracy |
|---|---|---|---|---|
| **Gradient Boosting** | **0.1072** | **0.0309** | **0.950** | **60.8%** |
| XGBoost | 0.1276 | 0.0388 | 0.941 | 53.6% |
| Random Forest | 0.1342 | 0.0363 | 0.951 | 52.1% |
| LightGBM | 0.1412 | 0.0413 | 0.930 | 50.2% |
| CatBoost | 0.1473 | 0.0446 | 0.940 | 53.4% |
| Regresión Logística | 0.2608 | 0.0797 | 0.942 | 54.8% |

> **Baselines naïve:** El piloto de pole siempre gana = 55.4% · Tasa de victorias histórica = 34.4%  
> Gradient Boosting (60.8%) **supera el baseline de pole** — el modelo agrega valor real.

### Comparación de ventanas históricas (Gradient Boosting)

| Ventana | Log Loss | Winner Accuracy |
|---|---|---|
| **5yr (2021–)** | **0.1129** | **63.5%** |
| 7yr (2019–) | 0.1067 | 56.5% |
| todo (2018–) | 0.1072 | 60.8% |
| 3yr (2023–) | 0.1474 | 54.2% |

> **Mejor ventana: 5yr** — captura la era competitiva actual sin ruido de temporadas antiguas.

### Importancia de features SHAP (Top 10)

| Ranking | Feature | Importancia |
|---|---|---|
| 1 | `qualifying_position` | 0.307 |
| 2 | `constructor_avg_finish_last_3` | 0.201 |
| 3 | `driver_win_rate_last_10` | 0.162 |
| 4 | `gap_to_pole_sec` | 0.150 |
| 5 | `driver_avg_finish_last_3` | 0.145 |

> La posición de clasificación y la diferencia con la pole son las **2 features más importantes por SHAP** — validando la contribución metodológica central del proyecto.

### Predicción final — Gran Premio de España 2026

> **Modelo:** Gradient Boosting · **Ventana:** 5yr (2021–2026 Rnd 13) · **Clasificación:** sábado 12/09/2026

| # | Piloto | Equipo | Clasificación | Prob. victoria |
|---|---|---|---|---|
| 1 | **Norris** | McLaren | P1 | **45.8%** |
| 2 | **Antonelli** | Mercedes | P2 | **41.9%** |
| 3 | Verstappen | Red Bull | P3 | 4.6% |
| 4 | Russell | Mercedes | P6 | 1.3% |
| 5 | Hamilton | Ferrari | P4 | 1.1% |
| 6 | Leclerc | Ferrari | P5 | 0.8% |
| 7 | Piastri | McLaren | P7 | 0.4% |
| ... | ... | ... | ... | ... |

> **Ganador predicho: Lando Norris (McLaren)** — pole position, mejor forma reciente del piloto y del equipo.  
> McLaren + Mercedes combinados: **>87%** de probabilidad. El Madring es un circuito nuevo sin historia — todos los pilotos tienen features de circuito en cero.

---

## Métricas de evaluación

| Métrica | Por qué importa |
|---|---|
| **Log Loss** | Principal — penaliza predicciones con alta confianza que resultan incorrectas |
| **Brier Score** | Calibración — error cuadrático en las probabilidades |
| **ROC-AUC** | Capacidad discriminativa del modelo |
| **Winner Accuracy** | ¿Predijimos correctamente al ganador en cada carrera? |
| **Curva de calibración** | ¿Cuando el modelo dice 30%, ocurre en el 30% de los casos? |

---

## Limitaciones

- **Madring es un circuito nuevo** — sin datos históricos en F1 para ningún piloto; todos arrancan desde cero en circuit-features
- **Dataset pequeño** — ~200 carreras × ~20 pilotos = ~4.000 filas; la complejidad del modelo está limitada en consecuencia
- **Alta aleatoriedad intrínseca en F1** — autos de seguridad, fallas mecánicas, clima y estrategias de pit no son predecibles desde features pre-carrera
- **Penalizaciones de grilla no modeladas** — la posición de clasificación puede diferir de la posición real de salida si hay penalizaciones
- **Cambio de reglamento 2026** — las dinámicas de equipo y piloto en esta era son distintas; el modelo aprende patrones que pueden transferirse de forma imperfecta

---

## Mejoras futuras

- Incorporar resultados de sprint races como señal adicional
- Agregar datos de pronóstico del tiempo (si demuestran mejorar las métricas)
- Modelar penalizaciones de grilla explícitamente
- Incorporar datos de pit stops (FastF1)
- Calibración isotónica o de Platt para mejorar la calibración de probabilidades
- Extender para predecir probabilidad de podio (top 3)

---

## Stack tecnológico

Python 3.12 · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · SHAP · matplotlib · FastF1 · PyYAML · Jupyter

---

## Reproducibilidad

Todas las semillas aleatorias se fijan con `random_seed: 42` en `config/race_config.yaml`.  
Los datos raw son completamente reproducibles re-ejecutando el paso de descarga.  
No hay rutas absolutas — todas las rutas son relativas a la raíz del proyecto.

### Adaptar para otro Gran Premio

Editar solo `config/race_config.yaml`:

```yaml
target:
  gp_name: "Gran Premio de Singapur"
  year: 2026
  round: 17
  circuit_id: "marina_bay"
  race_date: "2026-10-11"
  qualifying_date: "2026-10-10"
```

Luego ejecutar:
```bash
python pipeline.py --skip-download  # si los datos ya están descargados
```

---

## Autor

Desarrollado como parte de un portfolio personal de Data Engineering / Machine Learning.  
Demuestra: ETL, Ingeniería de Features, Validación Temporal, Evaluación de Modelos, Interpretabilidad y Reproducibilidad.

---

## Licencia

Uso personal / portfolio. Los datos provienen de la API pública Jolpica (compatible con Ergast).

---

> 📄 **English version:** [README_EN.md](README_EN.md)
