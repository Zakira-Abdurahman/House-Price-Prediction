"""Tests for src/config.py - catches a stale/malformed feature-config JSON before it reaches the app."""
from src import config as C


def test_paths_exist_on_disk():
    for p in (C.FEATURES_CSV, C.NATIONAL_MAIZE_CSV, C.FORWARD_FORECAST_CSV,
              C.FINAL_CLASSIFIER_PATH, C.FEATURE_CONFIG_PATH, C.MODEL_CARD_PATH):
        assert p.exists(), f"missing required file: {p}"


def test_feature_columns_is_numeric_then_categorical_concatenated():
    assert C.FEATURE_COLUMNS == C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES


def test_feature_lists_non_empty_and_disjoint():
    assert len(C.NUMERIC_FEATURES) > 0
    assert len(C.CATEGORICAL_FEATURES) > 0
    assert set(C.NUMERIC_FEATURES).isdisjoint(C.CATEGORICAL_FEATURES)


def test_category_column_not_in_features():
    # 'category' is constant ("cereals and tubers") in this scope and was deliberately dropped
    # as a model feature in notebooks 09/10 - a regression here would silently feed a useless column.
    assert "category" not in C.FEATURE_COLUMNS


def test_decision_threshold_in_unit_interval():
    assert 0.0 < C.DECISION_THRESHOLD < 1.0


def test_risk_bands_sorted_descending_and_cover_zero():
    cutoffs = [c for c, _ in C.RISK_BANDS]
    assert cutoffs == sorted(cutoffs, reverse=True)
    assert cutoffs[-1] == 0.0
