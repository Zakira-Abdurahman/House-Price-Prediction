# User Guide

A tour of the five pages, for someone opening the dashboard for the first time.

## Overview (Home)

The landing page. Top row: how much data is behind the dashboard (observations, markets, regions,
commodities, series) and the date range covered. **Latest national snapshot** shows the four best-covered
staples' current median price and change from the previous reported month. Below that, a trend chart
since 2020, then a **headline** for next-month spike risk: how many series were scored, how many were
flagged, and a preview of the top 5 highest-risk series. Use the sidebar to jump to any other page.

## Price Explorer

Compare retail prices across markets for **one commodity at a time** (prices are never mixed across
units - a KG price and a 100 KG price for different commodities would not be comparable on the same
axis). Sidebar filters:

- **Commodity (unit)**: pick from the dropdown; the unit (KG or 100 KG) is shown in parentheses.
- **Currency**: USD (recommended for comparing across years - it removes the effect of the Ethiopian
  birr's depreciation) or ETB (the local-currency price actually paid).
- **Regions / Markets**: narrow to specific areas; leave blank to see the best-reported markets nationally.
- **Date range**, **mark flagged spikes**, **show national median** toggle extra detail on the chart.

Below the chart: quick stats, a per-market summary table, and a raw data table you can download as CSV.

## Spike Analysis

Explains and explores the project's spike definition (expand "How is a spike defined?" at the top). A
spike is a month where a series' price jump is unusually large **compared to that series' own recent
history** (not a fixed percentage for every commodity). Filter by year range, region, commodity, and a
minimum-observations threshold (hides regions/commodities too thin to give a stable rate). The page shows
spike counts over time (bursts often mean a shared shock - currency, fuel, supply disruption - rather than
one crop's problem), spike rate by region and by commodity, and a table of the single largest flagged
jumps (read the warning below that table - the very largest jumps are often thin/reopening markets, weaker
evidence than a spike on a well-covered series).

## Forecast & Risk

The most actionable page, in three tabs:

- **Risk ranking**: every scored series (reported in the latest month, at least a year of history) ranked
  by next-month spike-risk score. Filter by region/commodity/minimum score, download the ranking as CSV.
  Remember: scores are for **ranking**, not calibrated probabilities - use the risk bands (High /
  Elevated / Baseline) rather than reading a score as a percent chance.
- **Series lookup**: pick a specific market and commodity to see its price history, flagged spikes, and
  (if it qualifies) its next-month forecast and risk score. If a series didn't report in the latest month
  or has under a year of history, the page says so rather than guessing.
- **National maize forecast**: the single-series Drift forecast for white maize nationally, with an
  adjustable horizon (1-6 months). Longer horizons are straight-line extrapolations - treat them with
  more caution than the 1-month forecast.

## Models & Method

For anyone who wants to know *why* the dashboard makes the forecasts it does, in five tabs: **Overview**
(what's in production and why, plus the full notebook pipeline), **Price forecasting** (why persistence
beat every machine-learning model, with the bootstrap significance check and the log-price experiment),
**Spike classification** (ranking-quality metrics, hit rate in the highest-risk slice, and what the
operating threshold catches), **Explainability** (which inputs the models actually rely on), and
**Limitations** (an honest list of what this dashboard should not be used for). Start here if a number
elsewhere in the dashboard seems surprising.

## Tips

- Every chart is interactive: hover for exact values, drag to zoom, double-click to reset.
- Tables with a download button export exactly what's currently filtered, not the whole dataset.
- The sidebar always shows the data's last reporting month - forecasts are always "the month after that."
