"""Unit tests for src/metrics.py - pure functions, no data files needed."""
import numpy as np
import pytest

from src.metrics import mae, mape, precision_at_k_pct, rmse


def test_mae_zero_for_perfect_forecast():
    assert mae([1, 2, 3], [1, 2, 3]) == 0


def test_mae_known_value():
    # |1-2| + |2-2| + |3-2| = 2, mean = 2/3
    assert mae([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)


def test_rmse_at_least_mae():
    # RMSE >= MAE always holds (Jensen's inequality on squared vs. absolute error).
    y, yhat = [10, 20, 30, 100], [12, 18, 33, 40]
    assert rmse(y, yhat) >= mae(y, yhat)


def test_rmse_punishes_one_big_miss_more_than_mae():
    y = [10, 10, 10, 10]
    small_misses = [11, 11, 11, 11]
    one_big_miss = [10, 10, 10, 14]  # same total absolute error (4) as small_misses
    assert mae(y, small_misses) == pytest.approx(mae(y, one_big_miss))
    assert rmse(y, one_big_miss) > rmse(y, small_misses)


def test_mape_is_percent_of_actual():
    assert mape([100], [90]) == pytest.approx(10.0)


def test_mape_raises_on_zero_actual():
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])


def test_precision_at_k_pct_perfect_ranking():
    y = [0] * 90 + [1] * 10
    scores = list(range(100))  # positives (index 90-99) score highest
    assert precision_at_k_pct(y, scores, 10) == pytest.approx(1.0)


def test_precision_at_k_pct_worst_ranking():
    y = [0] * 90 + [1] * 10
    scores = list(range(99, -1, -1))  # positives score lowest
    assert precision_at_k_pct(y, scores, 10) == pytest.approx(0.0)


def test_precision_at_k_pct_matches_base_rate_for_random_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    scores = rng.random(5000)  # unrelated to y
    # With enough rows a random ranking's top-k precision should land close to the base rate.
    assert precision_at_k_pct(y, scores, 20) == pytest.approx(y.mean(), abs=0.03)


def test_precision_at_k_pct_rounds_up_to_at_least_one_row():
    y = [0, 0, 0, 1]
    assert precision_at_k_pct(y, [1, 2, 3, 4], 1) == pytest.approx(1.0)  # 1% of 4 rows -> still checks top 1
