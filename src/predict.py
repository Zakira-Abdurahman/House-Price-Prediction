"""Final-model logic (notebook 13): persistence + Drift forecasts and the spike-risk classifier."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_FEATURES,
    CLASSIFIER_TARGET,
    DECISION_THRESHOLD,
    FEATURE_COLUMNS,
    FINAL_CLASSIFIER_PATH,
    NUMERIC_FEATURES,
    RISK_BANDS,
)


# ----------------------------------------------------------------------------------------------
# Price forecasts
# ----------------------------------------------------------------------------------------------
def persistence_forecast(current_price):
    """Final panel-level price method (notebook 11/13): next month's price = this month's price."""
    return current_price


def drift_forecast(series: pd.Series, horizon: int = 1) -> np.ndarray:
    """Drift method (notebook 06): last value plus the average historical change per step."""
    values = np.asarray(series, dtype=float)
    if len(values) < 2:
        raise ValueError("Drift needs at least two observations")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    slope = (values[-1] - values[0]) / (len(values) - 1)
    return values[-1] + slope * np.arange(1, horizon + 1)


# ----------------------------------------------------------------------------------------------
# Spike-risk classifier
# ----------------------------------------------------------------------------------------------
def load_classifier(path=FINAL_CLASSIFIER_PATH):
    """Load the final Hist Gradient Boosting spike classifier (trusted, project-owned artifact)."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Re-run notebook 13 to regenerate it.")
    return joblib.load(path)


def training_categories(df: pd.DataFrame) -> dict[str, pd.Index]:
    """Category levels the classifier was trained with (sorted unique values over its training rows)."""
    train_rows = df.dropna(subset=NUMERIC_FEATURES + [CLASSIFIER_TARGET])
    return {c: pd.Categorical(train_rows[c]).categories for c in CATEGORICAL_FEATURES}


def latest_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Series reported in the most recent month that have every numeric feature available."""
    last = df["date"].max()
    return df[df["date"] == last].dropna(subset=NUMERIC_FEATURES).copy()


def risk_band(score: float) -> str:
    for cutoff, label in RISK_BANDS:
        if score >= cutoff:
            return label
    return RISK_BANDS[-1][1]


def score_rows(rows: pd.DataFrame, categories: dict[str, pd.Index], model=None) -> pd.DataFrame:
    """Score feature rows with the classifier and attach the persistence price forecast."""
    model = model or load_classifier()
    X = rows[FEATURE_COLUMNS].copy()
    for c in CATEGORICAL_FEATURES:
        # Values unseen in training become missing, which the model handles natively. Values are
        # masked to NaN *before* constructing the Categorical, since assigning a dtype whose
        # categories don't cover every value is deprecated as of pandas 3.
        known = X[c].where(X[c].isin(categories[c]))
        X[c] = pd.Categorical(known, categories=categories[c])
    out = rows[["date", "market", "admin1", "commodity", "commodity_unit", "unit", "usdprice"]].copy()
    out["forecast_next_price"] = persistence_forecast(out["usdprice"])
    out["spike_risk_next_month"] = model.predict_proba(X)[:, 1]
    out["risk_band"] = out["spike_risk_next_month"].map(risk_band)
    out["flagged"] = out["spike_risk_next_month"] >= DECISION_THRESHOLD
    return out.sort_values("spike_risk_next_month", ascending=False).reset_index(drop=True)


def score_latest(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Forecast the month after the last month in ``df`` for every series with a complete feature history."""
    return score_rows(latest_feature_rows(df), training_categories(df), model=model)
