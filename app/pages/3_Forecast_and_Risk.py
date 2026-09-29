"""Forecast & Risk: next-month spike-risk ranking, per-series lookup and the national maize (Drift) forecast."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import data as D  # noqa: E402
from src import predict as P  # noqa: E402
from src import viz  # noqa: E402
from src.config import DECISION_THRESHOLD  # noqa: E402
from src.dashboard import get_features, get_national_maize, get_scored, get_summary, money, page_setup  # noqa: E402

page_setup("Forecast & Risk", "🔮")
df = get_features()
scored = get_scored()
last = D.latest_date(df)
next_month = last + pd.DateOffset(months=1)

st.title("🔮 Forecast & Risk")
st.markdown(f"Forecasts for **{next_month:%B %Y}**, made from data through **{last:%B %Y}**.")

with st.expander("How to read this page", expanded=False):
    st.markdown(
        "- **Price forecast** uses *persistence*: next month's price equals this month's. In testing this simple rule "
        "beat every machine-learning model we tried (see *Models & Method*).\n"
        "- **Spike risk** comes from a Hist Gradient Boosting classifier. It is better than chance but modest: "
        "treat it as a **ranking of where to look first**, not a certain alarm. Scores are **not calibrated probabilities**.\n"
        f"- Bands: **High** >= 0.50, **Elevated** >= {DECISION_THRESHOLD}, otherwise **Baseline**.\n"
        "- Only series that reported in the latest month and have at least a year of history are scored."
    )

tab_rank, tab_lookup, tab_maize = st.tabs(["Risk ranking", "Series lookup", "National maize forecast"])

# ----------------------------------------------------------------------------------------------
with tab_rank:
    f1, f2, f3, f4 = st.columns(4)
    regions = f1.multiselect("Region", sorted(scored["admin1"].unique()), key="rk_region")
    comms = f2.multiselect("Commodity", sorted(scored["commodity"].unique()), key="rk_comm")
    min_score = f3.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05, key="rk_min")
    top_n = f4.slider("Bars to show", 5, 30, 12, key="rk_n")

    view = scored
    if regions:
        view = view[view["admin1"].isin(regions)]
    if comms:
        view = view[view["commodity"].isin(comms)]
    view = view[view["spike_risk_next_month"] >= min_score]

    m = st.columns(4)
    m[0].metric("Series shown", len(view))
    m[1].metric("High", int((view["risk_band"] == "High").sum()))
    m[2].metric("Elevated", int((view["risk_band"] == "Elevated").sum()))
    m[3].metric("Baseline", int((view["risk_band"] == "Baseline").sum()))

    if view.empty:
        st.info("No series match these filters.")
    else:
        top = view.head(top_n).copy()
        top["label"] = top["market"].astype(str) + " - " + top["commodity"].astype(str)
        st.plotly_chart(viz.hbar(top, "spike_risk_next_month", "label", f"Highest spike-risk series for {next_month:%B %Y}",
                                 "Risk score (ranking, not a calibrated probability)", color=viz.RED), width="stretch")
        table = view.rename(columns={"market": "Market", "admin1": "Region", "commodity": "Commodity", "unit": "Unit",
                                     "usdprice": "Latest price (USD)", "forecast_next_price": "Forecast price (USD)",
                                     "spike_risk_next_month": "Risk score", "risk_band": "Band"})
        table = table[["Market", "Region", "Commodity", "Unit", "Latest price (USD)", "Forecast price (USD)", "Risk score", "Band"]]
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"Risk score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        st.download_button("Download ranking (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name=f"spike_risk_{next_month:%Y_%m}.csv", mime="text/csv")

# ----------------------------------------------------------------------------------------------
with tab_lookup:
    st.markdown("Pick a market and commodity to see its price history, next-month forecast and risk score.")
    c1, c2 = st.columns(2)
    market = c1.selectbox("Market", sorted(df["market"].unique()), index=None, placeholder="Choose a market")
    if market:
        options = sorted(df[df["market"] == market]["commodity_unit"].unique())
        cu = c2.selectbox("Commodity (unit)", options)
        hist = df[(df["market"] == market) & (df["commodity_unit"] == cu)].sort_values("date")
        row = scored[(scored["market"] == market) & (scored["commodity_unit"] == cu)]
        st.caption(f"{len(hist)} monthly observations, {hist['date'].min():%b %Y} to {hist['date'].max():%b %Y}. "
                   f"Region: {hist['admin1'].iloc[0]}.")
        if row.empty:
            st.warning("This series has no forecast: it did not report in "
                       f"{last:%B %Y} or has less than a year of price history.")
            fig = viz.history_with_forecast(hist["date"], hist["usdprice"], hist["date"].iloc[-1], hist["usdprice"].iloc[-1],
                                            f"{cu} - {market}", "USD", None, None)
            fig.data = fig.data[:-1]
        else:
            r = row.iloc[0]
            k = st.columns(4)
            k[0].metric(f"Latest price ({r['date']:%b %Y})", money(r["usdprice"]))
            k[1].metric(f"Forecast ({next_month:%b %Y})", money(r["forecast_next_price"]), "persistence rule", delta_color="off")
            k[2].metric("Spike risk score", f"{r['spike_risk_next_month']:.2f}")
            k[3].metric("Band", r["risk_band"])
            sp = hist[hist["is_spike"]]
            fig = viz.history_with_forecast(hist["date"], hist["usdprice"], next_month, float(r["forecast_next_price"]),
                                            f"{cu} - {market}", "USD per reporting unit", sp["date"], sp["usdprice"])
        st.plotly_chart(fig, width="stretch")
        recent = hist.tail(6)[["date", "usdprice", "pct_change", "pct_change_z", "is_spike", "price_premium_pct"]]
        st.markdown("**Recent months**")
        st.dataframe(recent.sort_values("date", ascending=False), hide_index=True, width="stretch",
                     column_config={"date": st.column_config.DateColumn("Month", format="MMM YYYY"),
                                    "pct_change": st.column_config.NumberColumn("Change (%)", format="%.1f"),
                                    "pct_change_z": st.column_config.NumberColumn("z-score", format="%.2f"),
                                    "price_premium_pct": st.column_config.NumberColumn("vs national median (%)", format="%.1f")})

# ----------------------------------------------------------------------------------------------
with tab_maize:
    maize = get_national_maize()
    horizon = st.slider("Months ahead", 1, 6, 1, key="mz_h")
    fc = P.drift_forecast(maize, horizon)
    dates = [maize.index[-1] + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    slope = (maize.iloc[-1] - maize.iloc[0]) / (len(maize) - 1)

    k = st.columns(3)
    k[0].metric(f"Latest ({maize.index[-1]:%b %Y})", money(maize.iloc[-1]))
    k[1].metric(f"Drift forecast ({dates[-1]:%b %Y})", money(fc[-1]))
    k[2].metric("Average drift per month", f"{slope:+.3f} USD")

    recent = maize[maize.index >= maize.index[-1] - pd.DateOffset(years=5)]
    import plotly.graph_objects as go  # noqa: E402
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines", name="National median",
                             line=dict(color=viz.ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=[recent.index[-1]] + dates, y=[recent.iloc[-1]] + list(fc), mode="lines+markers",
                             name="Drift forecast", line=dict(color=viz.ORANGE, dash="dash")))
    fig.update_layout(title="White maize, national median retail price (USD per 100 KG)", yaxis_title="USD per 100 KG", **viz.LAYOUT)
    st.plotly_chart(fig, width="stretch")

    rmse = next(r["rmse"] for r in get_summary()["single_series"] if r["model"].startswith("Drift"))
    st.info(f"Drift extends the average change seen since {maize.index[0]:%Y}. On the held-out last 12 months it had an "
            f"RMSE of about **{rmse:.2f} USD** on a price near 40 USD, the best of seven methods tested. "
            "Because it is a straight-line extrapolation, use longer horizons with caution.")
