"""Data access and aggregation helpers. Pure pandas: no Streamlit imports, so everything is unit-testable."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES_CSV, NATIONAL_MAIZE_CSV


def load_features(path: Path | str = FEATURES_CSV) -> pd.DataFrame:
    """Load the cereals & tubers feature table (retail prices with spike flags and model features)."""
    df = pd.read_csv(path, parse_dates=["date"])
    # Older exports of notebook 08 carried a duplicated implied_fx column; ignore it if present.
    df = df.drop(columns=[c for c in df.columns if c.endswith(".1")], errors="ignore")
    for col in ("is_spike", "is_drop"):
        df[col] = df[col].astype(bool)
    return df.sort_values(["market", "commodity_unit", "date"]).reset_index(drop=True)


def load_national_maize(path: Path | str = NATIONAL_MAIZE_CSV) -> pd.Series:
    """National median retail price (USD / 100 KG) of white maize, one value per month."""
    s = pd.read_csv(path, index_col="date", parse_dates=True)["usdprice"]
    return s.sort_index()


def latest_date(df: pd.DataFrame) -> pd.Timestamp:
    return df["date"].max()


def dataset_overview(df: pd.DataFrame) -> dict:
    """Headline numbers for the home page."""
    return {
        "observations": int(len(df)),
        "markets": int(df["market"].nunique()),
        "regions": int(df["admin1"].nunique()),
        "commodities": int(df["commodity"].nunique()),
        "series": int(df[["market", "commodity_unit"]].drop_duplicates().shape[0]),
        "first_date": df["date"].min(),
        "last_date": df["date"].max(),
    }


def commodity_units(df: pd.DataFrame, min_rows: int = 0) -> list[str]:
    """Commodity + unit labels (e.g. 'Maize (white) (100 KG)'), most reported first."""
    counts = df["commodity_unit"].value_counts()
    return [c for c, n in counts.items() if n >= min_rows]


def filter_prices(
    df: pd.DataFrame,
    commodity_unit: str,
    regions: list[str] | None = None,
    markets: list[str] | None = None,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Rows for one commodity+unit (never mix units), optionally limited by region, market and dates."""
    out = df[df["commodity_unit"] == commodity_unit]
    if regions:
        out = out[out["admin1"].isin(regions)]
    if markets:
        out = out[out["market"].isin(markets)]
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out


def national_median(df: pd.DataFrame, commodity_unit: str, value_col: str = "usdprice") -> pd.Series:
    """Median across markets, per month, for one commodity+unit."""
    sub = df[df["commodity_unit"] == commodity_unit]
    return sub.groupby("date")[value_col].median().sort_index()


def national_snapshot(df: pd.DataFrame, commodity_units_: list[str]) -> pd.DataFrame:
    """Latest national median per commodity, its change vs. the previous reported month, and market count."""
    rows = []
    for cu in commodity_units_:
        sub = df[df["commodity_unit"] == cu]
        if sub.empty:
            continue
        med = sub.groupby("date")["usdprice"].median().sort_index()
        n_markets = sub[sub["date"] == med.index[-1]]["market"].nunique()
        prev = med.iloc[-2] if len(med) > 1 else np.nan
        rows.append(
            {
                "commodity_unit": cu,
                "latest_month": med.index[-1],
                "median_usd": float(med.iloc[-1]),
                "change_pct": float((med.iloc[-1] / prev - 1) * 100) if prev and not np.isnan(prev) else np.nan,
                "markets_reporting": int(n_markets),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Spike helpers
# ----------------------------------------------------------------------------------------------
def eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where a spike z-score could be computed (a series needs about 6 months of history first)."""
    return df[df["pct_change_z"].notna()]


def spike_rate_by(df: pd.DataFrame, by: str, min_obs: int = 200) -> pd.DataFrame:
    """Spike rate (%) per group, counted only over rows that have a z-score."""
    el = eligible(df)
    g = el.groupby(by, observed=True).agg(spike_rate=("is_spike", "mean"), drop_rate=("is_drop", "mean"),
                                          observations=("is_spike", "size"))
    g = g[g["observations"] >= min_obs].copy()
    g["spike_rate"] *= 100
    g["drop_rate"] *= 100
    return g.sort_values("spike_rate", ascending=False).reset_index()


def monthly_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """Spike counts and rate (%) per calendar month."""
    el = eligible(df)
    g = el.groupby(el["date"].dt.to_period("M")).agg(spikes=("is_spike", "sum"), observations=("is_spike", "size"))
    g["spike_rate"] = g["spikes"] / g["observations"] * 100
    g.index = g.index.to_timestamp()
    return g.reset_index().rename(columns={"date": "month"})


def top_spike_events(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    cols = ["date", "admin1", "market", "commodity", "unit", "usdprice", "pct_change", "pct_change_z"]
    return df[df["is_spike"]].sort_values("pct_change", ascending=False)[cols].head(n).reset_index(drop=True)
