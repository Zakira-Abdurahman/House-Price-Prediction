# Model Card

Human-readable version of `models/13_model_card.json` (generated 2026-09-27, data through 2026-07-15).
Regenerate the JSON by re-running notebook `13`; this file should be updated to match by hand whenever
that happens (see `docs/deployment.md`, "Updating the model or data").

## Scope

**Cereals and tubers only**: Maize, Teff, Wheat, Sorghum, Rice, Barley, Potatoes, and related items.
Retail prices only (no wholesale). Ethiopia, WFP market monitoring data.

## Models in production

| Task | Method | Test metric | Notebook(s) |
|---|---|---|---|
| National white-maize price, 1 month ahead | Drift (naive + trend extrapolation) | RMSE 5.06 USD | `06` |
| Next-month price, every market x commodity series | **Persistence** (next month = this month) | RMSE 6.42 USD | `09`, `13` |
| Next-month spike risk (classification) | Hist Gradient Boosting, class-weighted, retrained on all data | PR-AUC 0.120 (base rate 0.032) | `10`, `13` |

### Why persistence, not machine learning, for price forecasting

Ridge regression, Random Forest, and Hist Gradient Boosting were all tested against a wide feature set
(lags, rolling statistics, spike history, calendar effects, regional price premium - see `08`). None beat
the simple persistence rule; a 2,000-resample bootstrap confirmed the gap is real, not sampling noise
(95% CI for the ML disadvantage: 1.96 to 2.71 USD in mean absolute error). A log-price retrain (`13.2`)
narrowed the percentage-error gap but not the dollar-error gap. **Known limitation**: cereals are reported
in two units (`KG` and `100 KG`, roughly a 50x price-level difference), which partly explains why a single
pooled regression struggles; see notebook `09` section 9.8 for the full diagnosis, and the caveat in the
dashboard about a robustness check that found this explains only part of the gap.

### Why Hist Gradient Boosting for spike classification

It had the best PR-AUC and the best precision in the highest-risk slice of predictions (top 1%: roughly
32% precision vs. a 3.2% base rate - about a 10x lift) among Logistic Regression, Random Forest, and
Hist Gradient Boosting. **Known limitation**: only about 3% of rows are spikes, so precision even at the
tuned decision threshold (0.3) is modest (around 20% in the held-out test window) - this is a
prioritisation tool, not a reliable alarm.

## Intended use

Early-warning shortlist and price-trend reference for food-security monitoring in Ethiopia's cereal and
tuber markets. Meant to help an analyst decide **where to look first**, not to replace on-the-ground
market verification.

## Not intended for

- Automated trading or procurement decisions.
- Commodity categories other than cereals and tubers (the models were never trained on, e.g., livestock
  or vegetables - see the project's earlier full-panel run in `notebooks/archive_full_panel/` for why
  that broader scope was abandoned: mixing livestock "Head" prices with crop prices made the regression
  problem substantially worse).
- Markets or commodity series with under 12 months of reporting history (excluded from training and from
  the live scoring in `src/predict.py::latest_feature_rows`).

## Data and retraining

- Training data: `data/eth_food_prices_features_cereals.csv`, built by notebook `08` from the cleaned
  WFP price dataset (see the project's earlier `01_02`-`04` notebooks).
- The spike classifier was retrained on **all** available labelled data (no held-out test set) before
  being shipped as the production artifact (`models/13_final_spike_classifier.joblib`), which is
  standard practice once model selection is finished - see `docs/deployment.md` for how to refresh it.
- Test metrics quoted above come from the **held-out** version of each model (trained on data before
  2025-07, tested on 2025-07 to 2026-06), not the final retrained-on-everything production model, since
  a model can't be honestly graded on data it was trained on.

## Risk score interpretation

`spike_risk_next_month` in the dashboard is a **model output used for ranking**, not a calibrated
probability - class weighting during training inflates the raw scores. Use the relative ordering and the
risk bands (High / Elevated / Baseline) rather than treating a score as "this will happen X% of the
time."
