# Notebooks

The full analysis and modelling pipeline, in order. Numbers match the stage names used throughout the
project's reports. These are development artifacts: the running dashboard (`app/`) does **not** execute
any notebook - it reads the CSV/JSON/joblib files these notebooks produced, committed under `data/` and
`models/`.

| # | Notebook | Produces |
|---|---|---|
| 01-02 | `01_02_data_cleaning_and_preprocessing.ipynb` | Cleaned WFP price dataset |
| 03 | `03_exploratory_data_analysis.ipynb` | Trend/regional/seasonal analysis |
| 04 | `04_price_spike_analysis_cereals.ipynb` | Spike-flagged dataset (rolling z-score) |
| 05 | `05_time_series_analysis.ipynb` | National maize series: decomposition, stationarity |
| 06 | `06_baseline_forecasting.ipynb` | Naive/drift/moving-average baselines |
| 07 | `07_arima_sarima_forecasting.ipynb` | ARIMA/SARIMA on the national series |
| 08 | `08_feature_engineering_cereals.ipynb` | `data/eth_food_prices_features_cereals.csv` |
| 09 | `09_machine_learning_forecasting_cereals.ipynb` | `data/results/09_*_cereals.csv` |
| 10 | `10_spike_prediction_cereals.ipynb` | `data/results/10_*_cereals.csv` |
| 11 | `11_model_evaluation_and_selection_cereals.ipynb` | Cross-model comparison, bootstrap significance |
| 12 | `12_model_explainability_cereals.ipynb` | Permutation importance, SHAP |
| 13 | `13_final_model_and_forecasting.ipynb` | `models/13_final_spike_classifier.joblib`, `models/13_model_card.json` |

`archive_full_panel/` holds the original run of notebooks 04/08/09/10/11 across **all** WFP food
categories (not just cereals & tubers) - superseded because mixing crop prices with livestock ("Head"
unit) prices made the price-regression problem substantially worse. Kept for reference; see
`docs/model_card.md`.

To reproduce `data/results/model_summary.json` (what the dashboard's *Models & Method* page reads) after
re-running these notebooks, run `python scripts/build_artifacts.py` from the repository root.
