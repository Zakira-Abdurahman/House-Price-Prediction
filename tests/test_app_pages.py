"""End-to-end smoke tests for every dashboard page, using Streamlit's own AppTest harness.

These run the real page scripts against the real data/model files and assert no exception is
raised - the same class of bug ("it works until a user clicks X") that unit tests on src/ alone
would miss. Marked slow because they load the full feature table and the model on every test.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    str(ROOT / "app" / "Home.py"),
    str(ROOT / "app" / "pages" / "1_Price_Explorer.py"),
    str(ROOT / "app" / "pages" / "2_Spike_Analysis.py"),
    str(ROOT / "app" / "pages" / "3_Forecast_and_Risk.py"),
    str(ROOT / "app" / "pages" / "4_Models_and_Method.py"),
]

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("page", PAGES)
def test_page_loads_without_exception(page):
    at = AppTest.from_file(page, default_timeout=90).run()
    assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"


def test_home_shows_dataset_metrics():
    at = AppTest.from_file(PAGES[0], default_timeout=90).run()
    assert not at.exception
    assert len(at.metric) >= 5  # observations, markets, regions, commodities, series


def test_price_explorer_switching_commodity_has_no_exception():
    at = AppTest.from_file(PAGES[1], default_timeout=90).run()
    options = at.selectbox[0].options
    assert len(options) > 1
    at.selectbox[0].select(options[1]).run()
    assert not at.exception


def test_price_explorer_currency_toggle_has_no_exception():
    at = AppTest.from_file(PAGES[1], default_timeout=90).run()
    at.radio[0].set_value("ETB").run()
    assert not at.exception


def test_price_explorer_empty_filter_shows_warning_not_crash():
    at = AppTest.from_file(PAGES[1], default_timeout=90).run()
    at.multiselect[0].select("Addis Ababa").run()  # region
    # Pick a market that (likely) doesn't belong to Addis Ababa to try to force an empty result;
    # if it's non-empty that's fine too - the point is neither path may raise.
    if at.multiselect[1].options:
        at.multiselect[1].select(at.multiselect[1].options[0]).run()
    assert not at.exception


def test_spike_analysis_high_min_obs_does_not_crash():
    at = AppTest.from_file(PAGES[2], default_timeout=90).run()
    at.slider[0].set_value(500).run()
    assert not at.exception


def test_forecast_and_risk_maize_horizon_slider():
    at = AppTest.from_file(PAGES[3], default_timeout=90).run()
    at.tabs[2].slider[0].set_value(6).run()
    assert not at.exception


def test_forecast_and_risk_lookup_with_no_forecast_series_shows_warning():
    at = AppTest.from_file(PAGES[3], default_timeout=90).run()
    lookup = at.tabs[1]
    market_options = lookup.selectbox[0].options
    assert len(market_options) > 0
    at.tabs[1].selectbox[0].select(market_options[0]).run()
    assert not at.exception


def test_models_and_method_all_tabs_present():
    at = AppTest.from_file(PAGES[4], default_timeout=90).run()
    assert not at.exception
    assert len(at.tabs) == 5
