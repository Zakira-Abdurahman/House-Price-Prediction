"""Build data/results/model_summary.json, the file the dashboard's "Models & Method" page reads.

Inputs (produced by notebooks 09 and 10 and kept in the repo):
  data/results/09_regression_model_scores_cereals.csv, 09_regression_test_predictions_cereals.csv
  data/results/10_classification_model_scores_cereals.csv, 10_classification_test_predictions_cereals.csv
  models/experiments/*.joblib   (hold-out models, trained only on data before the 2025-07 cutoff)

Run from the repository root:   python scripts/build_artifacts.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RESULTS_DIR, SUMMARY_JSON  # noqa: E402
from src.metrics import mae, mape, precision_at_k_pct, rmse  # noqa: E402

CUTOFF = pd.Timestamp("2025-07-01")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXP = ROOT / "models" / "experiments"


def holdout_split(df: pd.DataFrame, target: str):
    m = df.dropna(subset=NUMERIC_FEATURES + [target]).copy()
    train, test = m[m["date"] < CUTOFF].copy(), m[m["date"] >= CUTOFF].copy()
    for c in CATEGORICAL_FEATURES:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    return train, test


def importance_table(model, X, y, scoring) -> list[dict]:
    r = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring=scoring, n_jobs=-1)
    t = pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean, "std": r.importances_std})
    return t.sort_values("importance", ascending=False).round(5).to_dict("records")


def main() -> None:
    df = D.load_features()

    reg_scores = pd.read_csv(RESULTS_DIR / "09_regression_model_scores_cereals.csv")
    clf_scores = pd.read_csv(RESULTS_DIR / "10_classification_model_scores_cereals.csv")
    reg_pred = pd.read_csv(RESULTS_DIR / "09_regression_test_predictions_cereals.csv")
    clf_pred = pd.read_csv(RESULTS_DIR / "10_classification_test_predictions_cereals.csv")

    # --- bootstrap: is persistence's advantage over the best ML regressor real? -------------------
    target = reg_pred["target_next_usdprice"].to_numpy()
    err_ml = np.abs(target - reg_pred["pred_hist_gb"].to_numpy())
    err_pe = np.abs(target - reg_pred["pred_persistence"].to_numpy())
    rng = np.random.default_rng(42)
    diffs = np.empty(2000)
    for i in range(len(diffs)):
        idx = rng.integers(0, len(target), len(target))
        diffs[i] = err_ml[idx].mean() - err_pe[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    # --- top-k precision for the spike classifiers ------------------------------------------------
    y = clf_pred["target_next_is_spike"].astype(int).to_numpy()
    cols = {"Logistic Regression": "proba_logistic", "Random Forest": "proba_random_forest",
            "Hist Gradient Boosting": "proba_hist_gb"}
    topk = [{"top_pct": k, **{name: precision_at_k_pct(y, clf_pred[c], k) for name, c in cols.items()},
             "base_rate": float(y.mean())} for k in (1, 2, 5, 10, 20)]

    # --- warning quality at the operating threshold (final model family: Hist Gradient Boosting) -------
    from src.config import DECISION_THRESHOLD
    flag = clf_pred["proba_hist_gb"].to_numpy() >= DECISION_THRESHOLD
    tp = int((flag & (y == 1)).sum())
    threshold_metrics = {"threshold": DECISION_THRESHOLD, "flagged": int(flag.sum()), "true_spikes": int(y.sum()),
                         "caught": tp, "precision": round(tp / max(1, int(flag.sum())), 4),
                         "recall": round(tp / max(1, int(y.sum())), 4)}

    # --- calendar-alignment robustness check ---------------------------------------------------------
    # Notebook 08 built lags/targets by row position within each series, so after a reporting gap "next
    # month" can be 2+ months away. Re-score the saved test predictions on rows whose target is exactly
    # the next calendar month to confirm the conclusions do not depend on the misaligned rows.
    grp = df.groupby(["market", "commodity_unit"])
    nxt = grp["date"].shift(-1)
    df["gap_next"] = (nxt.dt.year - df["date"].dt.year) * 12 + (nxt.dt.month - df["date"].dt.month)
    lag12 = grp["date"].shift(12)
    gap12 = (df["date"].dt.year - lag12.dt.year) * 12 + (df["date"].dt.month - lag12.dt.month)
    keys = ["market", "commodity_unit", "date"]
    reg_x = pd.read_csv(RESULTS_DIR / "09_regression_test_predictions_cereals.csv", parse_dates=["date"]).merge(
        df[keys + ["gap_next"]], on=keys, how="left")
    clf_x = pd.read_csv(RESULTS_DIR / "10_classification_test_predictions_cereals.csv", parse_dates=["date"]).merge(
        df[keys + ["gap_next"]], on=keys, how="left")
    reg_x, clf_x = reg_x[reg_x["gap_next"] == 1], clf_x[clf_x["gap_next"] == 1]
    yx = clf_x["target_next_is_spike"].astype(int)
    reg_cols = {"Persistence (naive)": "pred_persistence", "Hist Gradient Boosting": "pred_hist_gb",
                "Ridge (linear)": "pred_ridge", "Random Forest": "pred_random_forest"}
    alignment = {
        "share_next_target_exact_all_rows": round(float((df["gap_next"].dropna() == 1).mean()), 4),
        "share_lag12_exact_all_rows": round(float((gap12.dropna() == 12).mean()), 4),
        "test_rows": int(len(reg_pred)), "test_rows_exact": int(len(reg_x)),
        "regression_exact_only": [{"model": n, "mae": round(mae(reg_x["target_next_usdprice"], reg_x[c]), 3),
                                   "rmse": round(rmse(reg_x["target_next_usdprice"], reg_x[c]), 3),
                                   "mape": round(mape(reg_x["target_next_usdprice"], reg_x[c]), 2)}
                                  for n, c in reg_cols.items()],
        "classification_exact_only": {
            "positives": int(yx.sum()), "base_rate": round(float(yx.mean()), 4),
            "pr_auc": {n: round(float(average_precision_score(yx, clf_x[c])), 4) for n, c in cols.items()},
            "top1pct_hit_rate_hgb": round(precision_at_k_pct(yx, clf_x["proba_hist_gb"], 1), 4)},
    }

    # --- permutation importance on the hold-out test set -------------------------------------------
    _, reg_test = holdout_split(df, "target_next_usdprice")
    _, clf_test = holdout_split(df, "target_next_is_spike")
    reg_imp = importance_table(joblib.load(EXP / "09_hist_gb_regressor_cereals.joblib"), reg_test[FEATURES],
                               reg_test["target_next_usdprice"], "neg_mean_absolute_error")
    clf_imp = importance_table(joblib.load(EXP / "10_hist_gb_classifier_cereals.joblib"), clf_test[FEATURES],
                               clf_test["target_next_is_spike"].astype(int), "average_precision")

    summary = {
        "generated": date.today().isoformat(),
        "test_window": {"start": "2025-07", "end": "2026-06", "regression_rows": int(len(reg_pred)),
                        "classification_rows": int(len(clf_pred)), "positives": int(y.sum())},
        "regression_panel": reg_scores.round(4).to_dict("records"),
        "classification": clf_scores.round(4).to_dict("records"),
        "topk_precision": topk,
        "threshold_metrics": threshold_metrics,
        "alignment_check": alignment,
        "bootstrap_mae_gap": {"description": "Best ML MAE minus persistence MAE, 2000 bootstrap resamples",
                              "ci95_low": round(float(lo), 3), "ci95_high": round(float(hi), 3)},
        # Copied from executed notebook outputs (notebook 07 and notebook 13) - single national series.
        "single_series": [
            {"model": "Drift (baseline)", "rmse": 5.06, "source": "notebook 06"},
            {"model": "ARIMA(3,1,0)", "rmse": 5.14, "source": "notebook 07"},
            {"model": "Moving avg k=12 (baseline)", "rmse": 5.28, "source": "notebook 06"},
            {"model": "Naive (baseline)", "rmse": 5.32, "source": "notebook 06"},
            {"model": "Moving avg k=3 (baseline)", "rmse": 5.60, "source": "notebook 06"},
            {"model": "SARIMA(1,1,1)x(0,1,1,12)", "rmse": 6.48, "source": "notebook 07"},
            {"model": "Seasonal naive (baseline)", "rmse": 7.18, "source": "notebook 06"},
        ],
        "log_target_experiment": [
            {"model": "Persistence (naive)", "mae": 3.05, "rmse": 6.42, "mape": 7.7},
            {"model": "Hist GB - raw USD target", "mae": 5.37, "rmse": 10.99, "mape": 54.6},
            {"model": "Hist GB - log(1+price) target", "mae": 6.12, "rmse": 12.56, "mape": 18.1},
        ],
        "importance": {"regressor": reg_imp, "classifier": clf_imp},
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY_JSON.relative_to(ROOT)}")
    print("bootstrap CI:", round(float(lo), 2), round(float(hi), 2))
    print("top regressor features:", [r["feature"] for r in reg_imp[:3]])
    print("top classifier features:", [r["feature"] for r in clf_imp[:3]])


if __name__ == "__main__":
    main()
