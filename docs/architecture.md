# Architecture

## System overview

```
notebooks/ (offline, one-time / on-demand)
   01-02  clean            data/eth_food_prices_features_cereals.csv
   03     explore                 |
   04     spike-flag              |
   05-07  time series/baselines   |  <- committed to the repo, the app reads these directly
   08     feature engineering ----+
   09-10  train models     data/results/*_cereals.csv, models/experiments/*.joblib
   11-12  compare/explain          |
   13     finalize          models/13_final_spike_classifier.joblib, models/13_model_card.json
                                   |
scripts/build_artifacts.py --------+---> data/results/model_summary.json
                                   |
src/  (shared logic, no Streamlit imports - independently testable)
   config.py    file paths, feature lists, thresholds
   data.py      load/filter/aggregate the feature table
   predict.py   persistence + Drift forecasts, spike-risk scoring
   metrics.py   MAE / RMSE / MAPE / precision@k
   viz.py       Plotly figure builders
   dashboard.py Streamlit-specific caching and page setup
                                   |
app/  (Streamlit UI, thin - delegates all logic to src/)
   Home.py                     overview + headline risk
   pages/1_Price_Explorer.py   filter/compare prices by market, region, commodity
   pages/2_Spike_Analysis.py   when/where/what spikes
   pages/3_Forecast_and_Risk.py  ranking, per-series lookup, national maize forecast
   pages/4_Models_and_Method.py  scores, bootstrap check, explainability, limitations
```

## Design decisions

**`src/` has no Streamlit imports except `dashboard.py`.** Every other module (`data`, `predict`,
`metrics`, `viz`) is plain pandas/numpy/sklearn/plotly, callable and testable without a running
Streamlit process. `dashboard.py` is the only place `@st.cache_data`/`@st.cache_resource` appear; it
wraps the pure functions in `data.py`/`predict.py` rather than duplicating their logic.

**The app never trains anything.** `models/13_final_spike_classifier.joblib` is a committed artifact
produced by notebook `13`. `src/predict.py` loads it and computes forecasts/scores at request time -
this keeps the app fast to start (no training on boot) and means the exact same predictions can be
reproduced from the notebook.

**`scripts/build_artifacts.py` is the bridge between notebook outputs and the dashboard.** Rather than
have the "Models & Method" page recompute a bootstrap significance test or permutation importances on
every page load (slow, and it would need the held-out training split reconstructed at runtime), this
script runs those once and writes `data/results/model_summary.json`. This is also why
`models/experiments/*.joblib` (the *held-out* versions of the regressor/classifier, trained only on data
before the test cutoff) are kept separately from `models/13_final_spike_classifier.joblib` (the
production model, retrained on everything) - the summary's metrics need the held-out models to be honest,
while the live dashboard needs the production model.

**One commodity+unit at a time, always.** Cereals are reported in two units (`KG` and `100 KG`) with a
roughly 50x difference in price level (this is the central finding in notebook `09`/`11` about why
pooled ML regression underperforms). `src/data.filter_prices` and the Price Explorer page enforce
filtering to one `commodity_unit` before plotting, so the UI can never silently average across units.

**Persistence, not a trained model, is the shipped price forecast.** `src/predict.persistence_forecast`
is the identity function; there is no `models/*price*.joblib` in production. This was a deliberate
result of the modelling work (`09`, `11`, `13`), not an oversight - see `docs/model_card.md`.

## Data flow at request time

1. `src/dashboard.get_features()` loads and caches `data/eth_food_prices_features_cereals.csv` (cached
   per Streamlit session via `st.cache_data`).
2. `src/dashboard.get_classifier()` loads and caches the joblib model (`st.cache_resource`, since it's
   not a DataFrame).
3. `src/dashboard.get_scored()` calls `src/predict.score_latest`, which finds every series that reported
   in the most recent month with a complete feature history, encodes categoricals to match the model's
   training categories, and scores them - this is the exact logic notebook `13` used to produce
   `data/13_forward_forecast.csv`, which `tests/test_predict.py` checks bit-for-bit.
4. Pages read from these cached functions and `data/results/model_summary.json`; no page recomputes a
   model score or a significance test from scratch.

## Adding a new page

1. Create `app/pages/N_Name.py` (the leading number controls sidebar order).
2. Start with `sys.path.insert(...)` + `from src.dashboard import page_setup; page_setup(...)`,
   matching the existing pages.
3. Put any new data logic in `src/data.py` (or a new `src/` module) rather than inline in the page, and
   add tests for it in `tests/` - the existing pages are intentionally thin.
