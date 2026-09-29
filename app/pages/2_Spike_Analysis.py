"""Spike Analysis: when, where and in which commodities prices jump."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src import data as D  # noqa: E402
from src import viz  # noqa: E402
from src.dashboard import get_features, page_setup  # noqa: E402

page_setup("Spike Analysis", "⚡")
df = get_features()

st.title("⚡ Spike Analysis")
with st.expander("How is a spike defined?", expanded=False):
    st.markdown(
        "For every market and commodity series we take the **month-over-month % change in USD price**, "
        "compare it with that series' own last 12 months of changes, and compute a **z-score**. "
        "A month is a **spike** when its z-score is above 2 (an unusually large rise for that series) and a "
        "**drop** when it is below -2. Using each series' own history means a volatile crop is not judged by "
        "the same fixed threshold as a stable one.\n\n"
        "Rates below are the share of months that spiked **among months where a z-score could be computed** "
        "(a series needs about 6 months of history first)."
    )

with st.sidebar:
    st.header("Filters")
    years = sorted(df["date"].dt.year.unique())
    y0, y1 = st.select_slider("Years", options=years, value=(years[0], years[-1]))
    regions = st.multiselect("Regions", sorted(df["admin1"].unique()))
    commodities = st.multiselect("Commodities", sorted(df["commodity"].unique()))
    min_obs = st.slider("Minimum observations per group", 50, 500, 200, 50,
                        help="Hides groups with too few observations for a stable rate.")

view = df[(df["date"].dt.year >= y0) & (df["date"].dt.year <= y1)]
if regions:
    view = view[view["admin1"].isin(regions)]
if commodities:
    view = view[view["commodity"].isin(commodities)]

el = D.eligible(view)
if el.empty:
    st.warning("No observations with a spike score match these filters.")
    st.stop()

k = st.columns(4)
k[0].metric("Months scored", f"{len(el):,}")
k[1].metric("Spikes", f"{int(el['is_spike'].sum()):,}")
k[2].metric("Spike rate", f"{el['is_spike'].mean() * 100:.2f}%")
k[3].metric("Drop rate", f"{el['is_drop'].mean() * 100:.2f}%")

st.plotly_chart(viz.monthly_bars(D.monthly_spikes(view)), width="stretch")
st.caption("Bursts of spikes in the same months across many series point to shared shocks (currency, fuel, "
           "supply disruption) rather than single-crop noise.")

left, right = st.columns(2)
by_region = D.spike_rate_by(view, "admin1", min_obs)
by_comm = D.spike_rate_by(view, "commodity", min_obs).head(15)
with left:
    if by_region.empty:
        st.info("No region has enough observations. Lower the minimum.")
    else:
        st.plotly_chart(viz.hbar(by_region, "spike_rate", "admin1", "Spike rate by region", "Spike rate (%)",
                                 color=viz.GREEN, fmt=".1f"), width="stretch")
with right:
    if by_comm.empty:
        st.info("No commodity has enough observations. Lower the minimum.")
    else:
        st.plotly_chart(viz.hbar(by_comm, "spike_rate", "commodity", "Spike rate by commodity (top 15)",
                                 "Spike rate (%)", color=viz.ORANGE, fmt=".1f"), width="stretch")

st.subheader("Largest flagged spikes")
events = D.top_spike_events(view, 20)
st.dataframe(events, hide_index=True, width="stretch",
             column_config={"date": st.column_config.DateColumn("Month", format="MMM YYYY"),
                            "pct_change": st.column_config.NumberColumn("Change (%)", format="%.0f"),
                            "pct_change_z": st.column_config.NumberColumn("z-score", format="%.1f")})
st.warning("The very largest jumps (thousands of %) usually come from thinly reported series or markets "
           "reopening after disruption (for example Tigray in 2022). Treat a spike flag on a thin series as "
           "weaker evidence than one on a well-covered series.")
