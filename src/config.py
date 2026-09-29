"""Central configuration: file locations, model settings and shared constants."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = ROOT / "models"

FEATURES_CSV = DATA_DIR / "eth_food_prices_features_cereals.csv"
NATIONAL_MAIZE_CSV = DATA_DIR / "maize_white_retail_national_monthly.csv"
FORWARD_FORECAST_CSV = DATA_DIR / "13_forward_forecast.csv"
SUMMARY_JSON = RESULTS_DIR / "model_summary.json"

FINAL_CLASSIFIER_PATH = MODELS_DIR / "13_final_spike_classifier.joblib"
FEATURE_CONFIG_PATH = MODELS_DIR / "10_feature_config_cereals.json"
MODEL_CARD_PATH = MODELS_DIR / "13_model_card.json"

with open(FEATURE_CONFIG_PATH, encoding="utf-8") as _f:
    _cfg = json.load(_f)

NUMERIC_FEATURES: list[str] = _cfg["numeric_features"]
CATEGORICAL_FEATURES: list[str] = _cfg["categorical_features"]
CLASSIFIER_TARGET: str = _cfg["target"]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Decision threshold chosen in notebook 10. Deliberately below 0.5: a missed spike is costlier than a
# false alarm in an early-warning setting.
DECISION_THRESHOLD: float = float(_cfg.get("decision_threshold", 0.3))

# Bands used only to make risk scores easier to read. Scores are model outputs used for ranking,
# not calibrated probabilities (see docs/model_card.md).
RISK_BANDS = [(0.5, "High"), (DECISION_THRESHOLD, "Elevated"), (0.0, "Baseline")]

APP_TITLE = "Ethiopia Food Prices - Cereals & Tubers"
