"""Central configuration loader for the F1 GP prediction pipeline."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[1]
_CONFIG_PATH = PROJECT_ROOT / "config" / "race_config.yaml"


def load_config() -> dict:
    """Load and return the race configuration from YAML."""
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Singleton loaded once at import time
CONFIG = load_config()

# Convenience accessors
TARGET = CONFIG["target"]
HISTORY = CONFIG["history"]
VALIDATION = CONFIG["validation"]
WINDOWS = CONFIG["windows"]
RANDOM_SEED = CONFIG["random_seed"]

# Directory shortcuts
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"
