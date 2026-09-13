"""Cargador central de configuración para el pipeline de predicción del GP de F1."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[1]
_CONFIG_PATH = PROJECT_ROOT / "config" / "race_config.yaml"


def load_config() -> dict:
    """Carga y devuelve la configuración de la carrera desde el archivo YAML."""
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Singleton cargado una sola vez al importar
CONFIG = load_config()

# Accesores de conveniencia
TARGET = CONFIG["target"]
HISTORY = CONFIG["history"]
VALIDATION = CONFIG["validation"]
WINDOWS = CONFIG["windows"]
RANDOM_SEED = CONFIG["random_seed"]

# Accesos directos a directorios
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

