"""Price Explorer: filter and compare retail prices for one commodity (one unit at a time)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src import data as D  # noqa: E402
from src import viz  # noqa: E402
from src.dashboard import get_features, page_setup  # noqa: E402

page_setup("Price Explorer", "📈")
df = get_features()

st.title("📈 Price Explorer")
st.caption("Compare markets for one commodity at a time. Units differ across commodities (KG vs 100 KG), "
           "so prices are never mixed between commodities.")

options = D.commodity_units(df, min_rows=50)
default = options.index("Maize (white) (100 KG)") if "Maize (white) (100 KG)" in options else 0

with st.sidebar:
    st.header("Filters")
    cu = st.selectbox("Commodity (unit)", options, index=default)
    currency = st.radio("Currency", ["USD", "ETB"], horizontal=True,
                        help="ETB is the local-currency price. It rises faster than USD because the birr depreciated.")
    sub_all = D.filter_prices(df, cu)
    regions = st.multiselect("Regions", sorted(sub_all["admin1"].unique()))
    market_pool = sub_all if not regions else sub_all[sub_all["admin1"].isin(regions)]
    markets = st.multiselect("Markets", sorted(market_pool["market"].unique()))
    lo, hi = sub_all["date"].min().date(), sub_all["date"].max().date()
    start, end = st.slider("Date range", min_value=lo, max_value=hi, value=(lo, hi), format="MMM YYYY")
    show_spikes = st.checkbox("Mark flagged spikes", value=True)
    show_national = st.checkbox("Show national median", value=True)

value_col = "usdprice" if currency == "USD" else "price"
unit = sub_all["unit"].iloc[0]
sub = D.filter_prices(df, cu, regions, markets, start, end)

if sub.empty:
    st.warning("No observations match these filters. Try widening the date range or clearing region/market filters.")
    st.stop()

national = D.national_median(D.filter_prices(df, cu, start=start, end=end), cu, value_col) if show_national else None
st.plotly_chart(
    viz.price_lines(sub, value_col, f"{currency} per {unit}", f"{cu}: retail price by market ({currency})",
                    national=national, show_spikes=show_spikes),
    width="stretch")
if sub["market"].nunique() > 12:
    st.caption(f"Showing the 12 best-reported of {sub['market'].nunique()} markets. Narrow the filters to see others.")

k = st.columns(4)
k[0].metric("Observations", f"{len(sub):,}")
k[1].metric("Markets", sub["market"].nunique())
k[2].metric("Latest median", f"{sub[sub['date'] == sub['date'].max()][value_col].median():,.2f} {currency}")
k[3].metric("Flagged spikes", int(sub["is_spike"].sum()))

tab1, tab2 = st.tabs(["Summary by market", "Data"])
with tab1:
    summary = (sub.groupby(["admin1", "market"], observed=True)
               .agg(observations=(value_col, "size"), first_month=("date", "min"), last_month=("date", "max"),
                    median_price=(value_col, "median"), spikes=("is_spike", "sum"))
               .reset_index().rename(columns={"admin1": "Region", "market": "Market"})
               .sort_values("observations", ascending=False))
    st.dataframe(summary, hide_index=True, width="stretch")
with tab2:
    table = sub[["date", "admin1", "market", "commodity", "unit", "price", "usdprice", "is_spike", "pct_change"]]
    st.dataframe(table.sort_values("date", ascending=False), hide_index=True, width="stretch")
    st.download_button("Download filtered data (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_prices.csv", mime="text/csv")
