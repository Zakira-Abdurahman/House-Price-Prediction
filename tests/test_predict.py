"""Tests for src/predict.py - the final production logic (notebook 13)."""
import numpy as np
import pandas as pd
import pytest

from src import predict as P
from src.config import CATEGORICAL_FEATURES, DECISION_THRESHOLD, NUMERIC_FEATURES


# ---------------------------------------------------------------------------------------------
# Pure functions: no model or data file needed
# ---------------------------------------------------------------------------------------------
def test_persistence_forecast_is_identity():
    prices = pd.Series([10.0, 20.0, 30.0])
    pd.testing.assert_series_equal(P.persistence_forecast(prices), prices)


def test_drift_forecast_flat_series_is_flat():
    series = pd.Series([50.0] * 12)
    fc = P.drift_forecast(series, horizon=3)
    np.testing.assert_allclose(fc, [50.0, 50.0, 50.0])


def test_drift_forecast_matches_hand_calculation():
    # slope = (last - first) / (n-1) = (30-10)/4 = 5
    series = pd.Series([10.0, 15.0, 20.0, 25.0, 30.0])
    fc = P.drift_forecast(series, horizon=2)
    np.testing.assert_allclose(fc, [35.0, 40.0])


def test_drift_forecast_requires_two_points():
    with pytest.raises(ValueError):
        P.drift_forecast(pd.Series([1.0]))


def test_drift_forecast_rejects_non_positive_horizon():
    with pytest.raises(ValueError):
        P.drift_forecast(pd.Series([1.0, 2.0]), horizon=0)


def test_risk_band_boundaries():
    assert P.risk_band(0.9) == "High"
    assert P.risk_band(0.5) == "High"  # boundary is inclusive
    assert P.risk_band(0.49) == "Elevated"
    assert P.risk_band(DECISION_THRESHOLD) == "Elevated"
    assert P.risk_band(DECISION_THRESHOLD - 0.001) == "Baseline"
    assert P.risk_band(0.0) == "Baseline"


# ---------------------------------------------------------------------------------------------
# Functions that load the real feature table and/or the trained model
# ---------------------------------------------------------------------------------------------
def test_load_classifier_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        P.load_classifier(tmp_path / "does_not_exist.joblib")


def test_classifier_loads_and_predicts_proba(classifier):
    assert hasattr(classifier, "predict_proba")


def test_latest_feature_rows_are_from_the_max_date_and_complete(features_df):
    rows = P.latest_feature_rows(features_df)
    assert (rows["date"] == features_df["date"].max()).all()
    assert rows[NUMERIC_FEATURES].isna().sum().sum() == 0


def test_training_categories_cover_the_data(features_df):
    cats = P.training_categories(features_df)
    assert set(cats.keys()) == set(CATEGORICAL_FEATURES)
    for c in CATEGORICAL_FEATURES:
        assert len(cats[c]) > 0


def test_score_rows_output_shape_and_types(features_df, classifier):
    rows = P.latest_feature_rows(features_df).head(20)
    cats = P.training_categories(features_df)
    out = P.score_rows(rows, cats, model=classifier)
    assert len(out) == len(rows)
    assert out["spike_risk_next_month"].between(0, 1).all()
    assert set(out["risk_band"].unique()).issubset({"High", "Elevated", "Baseline"})
    assert (out["flagged"] == (out["spike_risk_next_month"] >= DECISION_THRESHOLD)).all()


def test_score_rows_sorted_descending_by_risk(features_df, classifier):
    rows = P.latest_feature_rows(features_df)
    cats = P.training_categories(features_df)
    out = P.score_rows(rows, cats, model=classifier)
    risk = out["spike_risk_next_month"].to_numpy()
    assert np.all(risk[:-1] >= risk[1:])


def test_score_rows_forecast_equals_persistence(features_df, classifier):
    rows = P.latest_feature_rows(features_df).head(20)
    cats = P.training_categories(features_df)
    out = P.score_rows(rows, cats, model=classifier)
    pd.testing.assert_series_equal(out["forecast_next_price"], out["usdprice"], check_names=False)


def test_score_latest_matches_notebook_13_forward_forecast(features_df, classifier):
    """Regression test: the dashboard's live scoring must reproduce notebook 13's saved output exactly."""
    scored = P.score_latest(features_df, model=classifier)
    reference = pd.read_csv("data/13_forward_forecast.csv")
    merged = scored.merge(reference, on=["market", "admin1", "commodity"], suffixes=("", "_ref"))
    assert len(merged) == len(reference) == len(scored)
    np.testing.assert_allclose(merged["spike_risk_next_month"], merged["spike_risk_next_month_ref"], atol=1e-8)


def test_score_rows_handles_unseen_category_without_crashing(features_df, classifier):
    rows = P.latest_feature_rows(features_df).head(3).copy()
    cats = P.training_categories(features_df)
    rows["admin1"] = "Nonexistent Region"
    out = P.score_rows(rows, cats, model=classifier)  # must not raise
    assert out["spike_risk_next_month"].between(0, 1).all()
