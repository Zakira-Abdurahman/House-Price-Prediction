"""Home page: dataset overview, latest national snapshot and next-month risk headline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import data as D  # noqa: E402
from src import viz  # noqa: E402
from src.config import APP_TITLE, DECISION_THRESHOLD  # noqa: E402
from src.dashboard import get_features, get_scored, money, page_setup  # noqa: E402

page_setup("Overview", "🌾")
df = get_features()
ov = D.dataset_overview(df)

st.title(f"🌾 {APP_TITLE}")
st.markdown(
    "Monthly retail prices of cereals and tubers from **WFP market monitoring in Ethiopia**, with spike "
    "detection and a next-month **spike early-warning model**. Prices are in **US dollars** per reporting "
    "unit (per KG or per 100 KG, always shown next to the commodity name)."
)

c = st.columns(5)
c[0].metric("Price observations", f"{ov['observations']:,}")
c[1].metric("Markets", ov["markets"])
c[2].metric("Regions", ov["regions"])
c[3].metric("Commodities", ov["commodities"])
c[4].metric("Market x commodity series", f"{ov['series']:,}")
st.caption(f"Coverage: {ov['first_date']:%b %Y} to {ov['last_date']:%b %Y}. Reporting is sparse before 2020, "
           "so older trends rest on few markets.")

st.subheader("Latest national snapshot")
staples = ["Maize (white) (100 KG)", "Teff (mixed) (100 KG)", "Wheat (100 KG)", "Sorghum (white) (100 KG)"]
snap = D.national_snapshot(df, staples)
cols = st.columns(len(snap))
for col, (_, r) in zip(cols, snap.iterrows()):
    delta = None if r["change_pct"] != r["change_pct"] else f"{r['change_pct']:+.1f}% vs previous month"
    col.metric(r["commodity_unit"], money(r["median_usd"]), delta)
st.caption("National median across the markets that reported in the latest month "
           f"({snap['markets_reporting'].min()} to {snap['markets_reporting'].max()} markets depending on the commodity).")

series = {cu: D.national_median(df, cu) for cu in staples}
series = {k: v[v.index >= "2019-12-01"] for k, v in series.items()}
st.plotly_chart(viz.trend_lines(series, "National median retail price since 2020 (USD per 100 KG)", "USD per 100 KG"),
                width="stretch")

st.subheader("Next-month spike risk: headline")
scored = get_scored()
flagged = int(scored["flagged"].sum())
c1, c2, c3 = st.columns(3)
c1.metric("Series scored", len(scored))
c2.metric(f"Flagged (score >= {DECISION_THRESHOLD})", flagged)
c3.metric("Forecast month", f"{(D.latest_date(df) + pd.DateOffset(months=1)):%B %Y}")
top = scored.head(5)[["market", "admin1", "commodity", "spike_risk_next_month", "risk_band"]].rename(
    columns={"market": "Market", "admin1": "Region", "commodity": "Commodity",
             "spike_risk_next_month": "Risk score", "risk_band": "Band"})
st.dataframe(top, hide_index=True, width="stretch",
             column_config={"Risk score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
st.info("Open **Forecast & Risk** in the sidebar for the full ranking, per-series lookup and the national maize forecast.")

st.subheader("Where to go next")
st.markdown(
    "- **Price Explorer**: compare markets and regions for any commodity, with flagged spikes.\n"
    "- **Spike Analysis**: when, where and in which commodities prices jump.\n"
    "- **Forecast & Risk**: next-month price forecast and spike-risk ranking.\n"
    "- **Models & Method**: how the models were built, compared and explained, including what did *not* work."
)
