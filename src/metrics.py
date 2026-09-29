"""Forecast error metrics (same definitions used across the notebooks)."""
from __future__ import annotations

import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def mae(y, yhat) -> float:
    """Mean absolute error, in the units of the target (USD)."""
    y, yhat = _arr(y), _arr(yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    """Root mean squared error; punishes large misses more than MAE."""
    y, yhat = _arr(y), _arr(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y, yhat) -> float:
    """Mean absolute percentage error (%). Undefined when an actual value is zero."""
    y, yhat = _arr(y), _arr(yhat)
    if np.any(y == 0):
        raise ValueError("MAPE is undefined when any actual value is 0")
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def precision_at_k_pct(y_true, scores, k_pct: float) -> float:
    """Share of true positives among the top ``k_pct`` percent highest scores."""
    y_true, scores = _arr(y_true), _arr(scores)
    k = max(1, int(len(y_true) * k_pct / 100))
    top = np.argsort(-scores)[:k]
    return float(y_true[top].mean())
