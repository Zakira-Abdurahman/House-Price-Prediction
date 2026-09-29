"""Plotly figure builders used by the dashboard pages (kept separate so they can be tested)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

NAVY, ACCENT, RED, GREEN, ORANGE, GRAY = "#1F3864", "#2E5A88", "#C44E52", "#55A868", "#DD8452", "#8A8F98"
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))


def price_lines(df: pd.DataFrame, value_col: str, ylabel: str, title: str, national: pd.Series | None = None,
                show_spikes: bool = True, max_markets: int = 12) -> go.Figure:
    """One line per market (limited to the best-reported ones), optional national median and spike markers."""
    fig = go.Figure()
    markets = df["market"].value_counts().index[:max_markets]
    for m in markets:
        sub = df[df["market"] == m].sort_values("date")
        fig.add_trace(go.Scatter(x=sub["date"], y=sub[value_col], mode="lines", name=str(m), line=dict(width=1.4),
                                 opacity=0.75, hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>" + str(m) + "</extra>"))
    if national is not None and len(national):
        fig.add_trace(go.Scatter(x=national.index, y=national.values, mode="lines", name="National median",
                                 line=dict(color="black", width=3), hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>National median</extra>"))
    if show_spikes:
        sp = df[df["market"].isin(markets) & df["is_spike"]]
        if len(sp):
            fig.add_trace(go.Scatter(x=sp["date"], y=sp[value_col], mode="markers", name="Flagged spike",
                                     marker=dict(color=RED, size=9, symbol="triangle-up", line=dict(color="white", width=1)),
                                     hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>Spike</extra>"))
    fig.update_layout(title=title, yaxis_title=ylabel, hovermode="closest", **LAYOUT)
    return fig


def trend_lines(series_by_name: dict[str, pd.Series], title: str, ylabel: str) -> go.Figure:
    fig = go.Figure()
    for name, s in series_by_name.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name, line=dict(width=2),
                                 hovertemplate="%{x|%b %Y}: %{y:,.2f}<extra>" + name + "</extra>"))
    fig.update_layout(title=title, yaxis_title=ylabel, **LAYOUT)
    return fig


def hbar(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, color: str = ACCENT, fmt: str = ".2f") -> go.Figure:
    d = df.iloc[::-1]  # so the largest value is drawn at the top
    fig = go.Figure(go.Bar(x=d[x], y=d[y], orientation="h", marker_color=color,
                           text=d[x].map(lambda v: format(v, fmt)), textposition="outside", cliponaxis=False))
    fig.update_layout(title=title, xaxis_title=xlabel, height=max(320, 26 * len(d) + 120), **{**LAYOUT, "legend": None})
    return fig


def monthly_bars(m: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=m["month"], y=m["spikes"], marker_color=RED,
                           hovertemplate="%{x|%b %Y}: %{y} spikes<extra></extra>"))
    fig.update_layout(title="Flagged spikes per month", yaxis_title="Spike count", **{**LAYOUT, "legend": None})
    return fig


def history_with_forecast(dates: pd.Series, values: pd.Series, next_date: pd.Timestamp, forecast: float,
                          title: str, ylabel: str, spike_dates: pd.Series | None = None,
                          spike_values: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, mode="lines+markers", name="Reported price",
                             line=dict(color=ACCENT, width=2), marker=dict(size=4)))
    if spike_dates is not None and len(spike_dates):
        fig.add_trace(go.Scatter(x=spike_dates, y=spike_values, mode="markers", name="Flagged spike",
                                 marker=dict(color=RED, size=10, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=[dates.iloc[-1], next_date], y=[values.iloc[-1], forecast], mode="lines+markers",
                             name="Forecast", line=dict(color=ORANGE, width=2, dash="dash"),
                             marker=dict(size=[0, 11], symbol="diamond")))
    fig.update_layout(title=title, yaxis_title=ylabel, **LAYOUT)
    return fig


def model_bars(df: pd.DataFrame, label: str, value: str, title: str, xlabel: str, highlight: str | None = None,
               lower_is_better: bool = True) -> go.Figure:
    d = df.sort_values(value, ascending=lower_is_better).reset_index(drop=True)
    colors = [GREEN if (highlight and m == highlight) else ACCENT for m in d[label]]
    fig = go.Figure(go.Bar(x=d[value], y=d[label], orientation="h", marker_color=colors,
                           text=d[value].map(lambda v: f"{v:.3g}"), textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title, xaxis_title=xlabel, height=max(260, 55 * len(d) + 110), **{**LAYOUT, "legend": None})
    return fig
