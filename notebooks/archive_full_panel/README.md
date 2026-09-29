# Archived: full-panel (all food categories) run

These five notebooks are the **original** versions of 04/08/09/10/11, run across every WFP food category
(cereals, pulses, oil, dairy, meat, livestock, vegetables) rather than cereals & tubers only.

They are kept for reference, not reproduced by any script here, and **not** what the dashboard is built
on - the current pipeline (`../04_price_spike_analysis_cereals.ipynb` onward) restricts to cereals &
tubers, which resolved a real problem found in this full-panel run: pooling livestock prices (quoted per
"Head", often hundreds of USD) with crop prices (quoted per KG or 100 KG, usually under 100 USD) made the
next-month price regression noticeably worse than even the cereals-only version, which itself already
struggled to beat a simple persistence forecast (see `docs/model_card.md`).
