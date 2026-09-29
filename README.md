# 🌾 Ethiopia Food Prices — Cereals & Tubers Dashboard

A Streamlit dashboard for exploring WFP retail food price data for Ethiopia and getting a next-month
spike-risk early warning, built on top of a 13-stage notebook pipeline (data cleaning through model
explainability and finalization).

**Headline finding**: for next-month price forecasting, a simple *persistence* rule ("next month = this
month") beat every machine-learning model tested, and a bootstrap test confirmed that isn't noise. For
spike-risk classification, a Hist Gradient Boosting model **does** meaningfully beat chance (about a 10x
precision lift in its highest-risk 1% of predictions). Both results, including the negative one, are
reported honestly on the dashboard's *Models & Method* page. See `docs/model_card.md` for the full story.

## Quick start

```bash
# Option 1: Docker (recommended - no local Python setup needed)
docker compose up --build
# open http://localhost:8501

# Option 2: local Python
pip install -r requirements.txt
streamlit run app/Home.py
```

## Repository layout

```
app/                  Streamlit pages (thin - delegates logic to src/)
src/                  Shared, independently-tested logic (data, predict, metrics, viz)
scripts/              build_artifacts.py: turns notebook outputs into data/results/model_summary.json
tests/                73 tests, ~99% coverage of src/ (pytest)
notebooks/            The 13-stage analysis/modelling pipeline that produced data/ and models/
data/, models/        Committed artifacts the running app reads (never retrains at runtime)
docs/                 Architecture, user guide, model card, testing, deployment
Dockerfile, docker-compose.yml, .github/workflows/ci.yml   Containerization and CI/CD
```

## Documentation

| Doc | For |
|---|---|
| [`docs/user_guide.md`](docs/user_guide.md) | Using the dashboard, page by page |
| [`docs/model_card.md`](docs/model_card.md) | What each model does, why it was chosen, its limitations |
| [`docs/architecture.md`](docs/architecture.md) | How the code is organized and why |
| [`docs/testing.md`](docs/testing.md) | Running and understanding the test suite |
| [`docs/deployment.md`](docs/deployment.md) | Deploying via Docker, Streamlit Cloud, or a cloud platform |
| [`notebooks/README.md`](notebooks/README.md) | The 13-stage analysis pipeline, one row per stage |

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q --cov=src --cov-report=term-missing   # 73 tests, ~99% coverage
python -m pyflakes src app scripts tests        # lint
python scripts/build_artifacts.py               # regenerate data/results/model_summary.json
```

## Scope and limitations (short version — see `docs/model_card.md` for the full version)

- Cereals and tubers only (Maize, Teff, Wheat, Sorghum, Rice, Barley, Potatoes, and related items),
  retail prices only.
- Spike-risk scores are for **ranking**, not calibrated probabilities.
- A decision-support tool for food-security monitoring, not a substitute for on-the-ground verification,
  and not intended for automated trading or procurement decisions.

## License and data source

Price data: [WFP Food Prices](https://data.humdata.org/dataset/wfp-food-prices-for-ethiopia) via the
Humanitarian Data Exchange. This repository's code has no license file yet — add one appropriate to your
intended use before distributing.
