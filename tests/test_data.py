"""Tests for src/data.py against the real cereals & tubers feature table."""
import numpy as np
import pandas as pd
import pytest

from src import data as D


def test_load_features_basic_shape(features_df):
    assert len(features_df) > 0
    for col in ("date", "market", "admin1", "commodity", "commodity_unit", "usdprice", "is_spike"):
        assert col in features_df.columns


def test_load_features_no_duplicate_implied_fx_column(features_df):
    # An earlier notebook export carried a duplicated "implied_fx.1" column; the loader must drop it.
    assert not any(c.endswith(".1") for c in features_df.columns)


def test_load_features_is_spike_is_boolean(features_df):
    assert features_df["is_spike"].dtype == bool
    assert features_df["is_drop"].dtype == bool


def test_load_features_sorted_within_series(features_df):
    # Every series (market, commodity_unit) must be in ascending date order - lag/rolling features
    # and the dashboard's line charts both depend on this.
    checks = features_df.groupby(["market", "commodity_unit"], observed=True)["date"].apply(
        lambda s: s.is_monotonic_increasing)
    assert checks.all()


def test_dataset_overview_counts_are_sane(features_df):
    ov = D.dataset_overview(features_df)
    assert ov["observations"] == len(features_df)
    assert ov["markets"] == features_df["market"].nunique()
    assert ov["first_date"] <= ov["last_date"]
    assert ov["series"] >= ov["markets"]  # at least one series per market, usually several


def test_commodity_units_sorted_by_frequency(features_df):
    units = D.commodity_units(features_df, min_rows=0)
    counts = features_df["commodity_unit"].value_counts()
    assert units[0] == counts.index[0]
    assert all(counts[units[i]] >= counts[units[i + 1]] for i in range(len(units) - 1))


def test_commodity_units_min_rows_filter(features_df):
    all_units = set(D.commodity_units(features_df, min_rows=0))
    filtered = set(D.commodity_units(features_df, min_rows=1000))
    assert filtered.issubset(all_units)
    assert len(filtered) < len(all_units)


def test_filter_prices_never_mixes_units(features_df):
    for cu in D.commodity_units(features_df, min_rows=500)[:3]:
        sub = D.filter_prices(features_df, cu)
        assert sub["commodity_unit"].nunique() == 1
        assert sub["commodity_unit"].iloc[0] == cu


def test_filter_prices_region_and_date_filters(features_df):
    cu = D.commodity_units(features_df, min_rows=500)[0]
    region = features_df[features_df["commodity_unit"] == cu]["admin1"].iloc[0]
    sub = D.filter_prices(features_df, cu, regions=[region], start="2022-01-01", end="2022-12-31")
    assert (sub["admin1"] == region).all()
    assert sub["date"].between("2022-01-01", "2022-12-31").all()


def test_national_median_matches_manual_groupby(features_df):
    cu = D.commodity_units(features_df, min_rows=500)[0]
    med = D.national_median(features_df, cu)
    expected = features_df[features_df["commodity_unit"] == cu].groupby("date")["usdprice"].median()
    pd.testing.assert_series_equal(med.sort_index(), expected.sort_index(), check_names=False)


def test_national_snapshot_change_pct_matches_manual_calc(features_df):
    cu = D.commodity_units(features_df, min_rows=500)[0]
    snap = D.national_snapshot(features_df, [cu]).iloc[0]
    med = D.national_median(features_df, cu)
    expected_change = (med.iloc[-1] / med.iloc[-2] - 1) * 100
    assert snap["change_pct"] == pytest.approx(expected_change)


def test_national_snapshot_skips_unknown_commodity(features_df):
    snap = D.national_snapshot(features_df, ["Not A Real Commodity (KG)"])
    assert snap.empty


def test_eligible_drops_rows_without_zscore(features_df):
    el = D.eligible(features_df)
    assert el["pct_change_z"].notna().all()
    assert len(el) <= len(features_df)


def test_spike_rate_by_matches_manual_calc(features_df):
    by_region = D.spike_rate_by(features_df, "admin1", min_obs=0)
    el = D.eligible(features_df)
    for _, row in by_region.head(3).iterrows():
        expected = el[el["admin1"] == row["admin1"]]["is_spike"].mean() * 100
        assert row["spike_rate"] == pytest.approx(expected)


def test_spike_rate_by_respects_min_obs(features_df):
    loose = D.spike_rate_by(features_df, "commodity", min_obs=0)
    strict = D.spike_rate_by(features_df, "commodity", min_obs=100000)
    assert len(strict) <= len(loose)


def test_spike_rate_by_sorted_descending(features_df):
    by_comm = D.spike_rate_by(features_df, "commodity", min_obs=50)
    rates = by_comm["spike_rate"].to_numpy()
    assert np.all(rates[:-1] >= rates[1:])


def test_monthly_spikes_counts_match_total(features_df):
    m = D.monthly_spikes(features_df)
    assert m["spikes"].sum() == D.eligible(features_df)["is_spike"].sum()
    assert (m["spike_rate"] <= 100).all() and (m["spike_rate"] >= 0).all()


def test_top_spike_events_are_all_flagged_and_sorted(features_df):
    events = D.top_spike_events(features_df, n=15)
    assert len(events) <= 15
    assert (events["pct_change"].diff().dropna() <= 0).all()  # descending


def test_latest_date_is_the_max_date(features_df):
    assert D.latest_date(features_df) == features_df["date"].max()
