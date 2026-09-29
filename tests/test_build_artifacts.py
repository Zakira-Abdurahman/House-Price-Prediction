"""Tests for the model_summary.json produced by scripts/build_artifacts.py.

These do not re-run the (slow) script; they check the committed output is well-formed and
internally consistent, which is what the dashboard actually depends on.
"""
import json

from src.config import RESULTS_DIR, SUMMARY_JSON


def _load():
    assert SUMMARY_JSON.exists(), "run `python scripts/build_artifacts.py` before testing"
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


def test_summary_has_expected_top_level_keys():
    s = _load()
    for key in ("regression_panel", "classification", "topk_precision", "bootstrap_mae_gap",
                "single_series", "log_target_experiment", "importance", "threshold_metrics",
                "alignment_check"):
        assert key in s, f"missing key: {key}"


def test_regression_panel_persistence_beats_ml_models():
    s = _load()
    scores = {r["Model"]: r["RMSE"] for r in s["regression_panel"]}
    persistence = scores["Persistence (naive)"]
    ml_models = {k: v for k, v in scores.items() if k != "Persistence (naive)"}
    assert all(persistence < rmse for rmse in ml_models.values()), (
        "This project's headline finding is that persistence beats every ML regressor; "
        "if this now fails, the dashboard's text claims need updating too.")


def test_bootstrap_confirms_persistence_gap_is_positive():
    s = _load()
    b = s["bootstrap_mae_gap"]
    assert b["ci95_low"] > 0, "bootstrap CI should be entirely above zero: persistence's edge should be real"
    assert b["ci95_low"] < b["ci95_high"]


def test_classification_hist_gb_has_best_pr_auc():
    s = _load()
    scores = {r["Model"]: r["PR-AUC (avg precision)"] for r in s["classification"]}
    assert scores["Hist Gradient Boosting"] == max(scores.values())


def test_topk_precision_beats_base_rate_for_hist_gb():
    s = _load()
    for row in s["topk_precision"]:
        assert row["Hist Gradient Boosting"] >= row["base_rate"], (
            f"top-{row['top_pct']}% precision should be at or above the base rate")


def test_topk_precision_decreasing_as_k_grows():
    s = _load()
    values = [row["Hist Gradient Boosting"] for row in s["topk_precision"]]
    assert values == sorted(values, reverse=True)


def test_alignment_check_exact_share_is_majority():
    s = _load()
    a = s["alignment_check"]
    assert 0.5 < a["share_next_target_exact_all_rows"] <= 1.0
    assert a["test_rows_exact"] <= a["test_rows"]


def test_importance_tables_sorted_descending():
    s = _load()
    for key in ("regressor", "classifier"):
        values = [row["importance"] for row in s["importance"][key]]
        assert values == sorted(values, reverse=True)


def test_threshold_metrics_consistent_with_confusion_counts():
    s = _load()
    t = s["threshold_metrics"]
    assert t["caught"] <= t["true_spikes"]
    assert t["caught"] <= t["flagged"]
    assert t["precision"] == round(t["caught"] / t["flagged"], 4)
    assert t["recall"] == round(t["caught"] / t["true_spikes"], 4)


def test_results_csvs_referenced_by_the_script_exist():
    for name in ("09_regression_model_scores_cereals.csv", "09_regression_test_predictions_cereals.csv",
                "10_classification_model_scores_cereals.csv", "10_classification_test_predictions_cereals.csv"):
        assert (RESULTS_DIR / name).exists()
