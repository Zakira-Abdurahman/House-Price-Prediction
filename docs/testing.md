# Testing

## Running the suite

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                                   # 73 tests, a few seconds
pytest -q --cov=src --cov-report=term-missing   # with coverage (99% of src/)
pytest -q -m "not slow"                     # skip the Streamlit page integration tests
```

`docker compose run --rm test` runs the same suite inside the exact image that gets deployed.

## What's covered, and why

| File | Tests | What it protects against |
|---|---|---|
| `test_metrics.py` | 10 | MAE/RMSE/MAPE/precision@k formulas silently changing meaning |
| `test_data.py` | 19 | Loading, filtering, and aggregation logic disagreeing with a manual groupby |
| `test_predict.py` | 15 | Forecast/scoring logic drifting from notebook 13's saved output |
| `test_config.py` | 6 | A stale or malformed `models/10_feature_config_cereals.json` |
| `test_build_artifacts.py` | 10 | The project's headline findings quietly flipping (see below) |
| `test_app_pages.py` | 13 | A page crashing on load or on a specific user interaction |

Total: **73 tests**, ~99% line coverage of `src/` (the two uncovered lines are defensive branches not
reachable with the current committed data - see the coverage report for exact lines).

## The most important tests in this suite

`tests/test_predict.py::test_score_latest_matches_notebook_13_forward_forecast` re-runs the dashboard's
live scoring function and asserts it reproduces `data/13_forward_forecast.csv` (notebook 13's saved
output) to 1e-8. If anyone edits `src/predict.py` in a way that changes what the app predicts, this is
the test that catches it - a bug here would mean the dashboard quietly shows different numbers than the
model was actually validated on.

`tests/test_build_artifacts.py` encodes the project's actual conclusions as assertions - for example,
`test_regression_panel_persistence_beats_ml_models` fails if any ML regressor's RMSE ever drops below
persistence's. This is deliberate: these tests are not "does the code run" checks, they are "does the
story the dashboard tells still match the data" checks. If a future model update makes one of these
fail, that is real news (a genuine improvement, or a regression) and the dashboard's text needs to
change alongside the model, not just the model file.

## Streamlit page tests (`test_app_pages.py`)

Use Streamlit's official `streamlit.testing.v1.AppTest`, which runs a page script in a simulated session
and lets a test interact with widgets (select a different commodity, toggle currency, move a slider) the
same way a user would, then asserts no exception was raised. These tests load the real feature table and
the real model (no mocking) - they are the closest thing to "does the deployed app actually work" that
runs without a browser. Marked `slow` (a few seconds each, since they load ~24k rows and a model) so they
can be skipped during fast local iteration with `-m "not slow"`.

## What is *not* tested here

- **Visual appearance** (layout, colors, chart rendering) - there is no browser-based screenshot testing.
  `docs/architecture.md` and manual review are the only checks on this.
- **The notebooks themselves** - they are treated as the source of the committed artifacts, not as code
  under test. Re-running a notebook and confirming its output still matches what's committed
  (`data/eth_food_prices_features_cereals.csv`, etc.) is a manual step, documented in
  `docs/deployment.md`.
- **Load/performance testing** - not needed at the current data size (~24k rows, <10 MB), but would be a
  reasonable addition before a much larger scope or a much larger audience.

## CI

`.github/workflows/ci.yml` runs the full suite (with coverage) on every push and pull request, then - in
a separate job that only runs if tests pass - builds the Docker image and smoke-tests the running
container (waits for its health check, then requests the home page). See `docs/deployment.md`.
