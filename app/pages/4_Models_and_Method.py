"""Models & Method: how the models were built, compared and explained, including what did not work."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import viz  # noqa: E402
from src.dashboard import get_model_card, get_summary, page_setup  # noqa: E402

page_setup("Models & Method", "🧪")
S = get_summary()
card = get_model_card()

st.title("🧪 Models & Method")
st.markdown("A summary of the modelling work behind the dashboard. Every number here comes from the project's "
            "held-out test window (**July 2025 to June 2026**), never from data the models trained on.")

tab_over, tab_price, tab_spike, tab_explain, tab_limits = st.tabs(
    ["Overview", "Price forecasting", "Spike classification", "Explainability", "Limitations"])

with tab_over:
    st.subheader("What is used in production")
    final = pd.DataFrame([
        {"Task": "National white-maize price", "Method": "Drift (trend extrapolation)", "Why": "Lowest error of 7 methods tested"},
        {"Task": "Price for each market and crop", "Method": "Persistence (next month = this month)",
         "Why": "Beat every ML model, with a statistically clear margin"},
        {"Task": "Next-month spike risk", "Method": "Hist Gradient Boosting classifier",
         "Why": "Best ranking quality (PR-AUC) and best hit rate at the top of the list"},
    ])
    st.dataframe(final, hide_index=True, width="stretch")
    st.subheader("The pipeline")
    steps = pd.DataFrame([
        ["01-02", "Data, Clean", "Profile and clean 62,880 WFP rows"],
        ["03", "EDA", "Trends, regions, seasonality, retail vs wholesale"],
        ["04", "Spike analysis", "Rolling z-score spike definition"],
        ["05-07", "Time series, baselines, ARIMA/SARIMA", "Study one national series; simple methods win"],
        ["08", "Feature engineering", "Leakage-safe lags, rolling stats, spike history, calendar"],
        ["09-10", "ML forecasting, spike classification", "Regression loses to persistence; classifier beats chance"],
        ["11-12", "Model comparison, explainability", "Bootstrap check; permutation importance and SHAP"],
        ["13", "Final models", "Log-price experiment, retrain, forward forecast, model card"],
    ], columns=["Stage", "Name", "What happened"])
    st.dataframe(steps, hide_index=True, width="stretch")
    st.caption(f"Model card generated {card['generated']} with data through {card['data_through']}.")

with tab_price:
    reg = pd.DataFrame(S["regression_panel"])
    st.subheader("All markets and crops: next-month price")
    st.plotly_chart(viz.model_bars(reg, "Model", "RMSE", "Test RMSE (USD, lower is better)", "RMSE (USD)",
                                   highlight="Persistence (naive)"), width="stretch")
    st.dataframe(reg.rename(columns={"MAPE (%)": "MAPE (%)"}), hide_index=True, width="stretch")
    b = S["bootstrap_mae_gap"]
    st.success(f"**Persistence wins, and it is not luck.** Resampling the test rows 2,000 times, the best ML model's "
               f"average error was worse than persistence by **{b['ci95_low']} to {b['ci95_high']} USD** (95% range, "
               "entirely above zero).")
    st.subheader("Why did machine learning not help?")
    st.markdown("Price history already carries most of the information, and the models mostly re-learn it (see the "
                "*Explainability* tab). Cereal prices are also quoted per KG or per 100 KG, a roughly 50x difference in "
                "level, which makes one pooled model harder to fit. A log-price target was tested to address this:")
    logx = pd.DataFrame(S["log_target_experiment"]).rename(columns={"model": "Model", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE (%)"})
    st.dataframe(logx, hide_index=True, width="stretch")
    st.caption("The log target cut percentage error (MAPE) a lot but made dollar error worse, and still did not beat persistence.")
    st.subheader("Single national series (white maize)")
    single = pd.DataFrame(S["single_series"]).rename(columns={"model": "Model", "rmse": "RMSE (USD)", "source": "Source"})
    st.dataframe(single, hide_index=True, width="stretch")
    st.caption("Different problem from the table above (one smooth national series), so the two RMSE scales are not comparable.")

with tab_spike:
    clf = pd.DataFrame(S["classification"])
    tw = S["test_window"]
    st.subheader("Ranking quality")
    st.plotly_chart(viz.model_bars(clf, "Model", "PR-AUC (avg precision)", "PR-AUC (higher is better)", "PR-AUC",
                                   highlight="Hist Gradient Boosting", lower_is_better=False), width="stretch")
    st.dataframe(clf, hide_index=True, width="stretch")
    base = S["topk_precision"][0]["base_rate"]
    st.caption(f"Only {base * 100:.1f}% of test rows are spikes ({tw['positives']} real spikes in {tw['classification_rows']:,} rows), "
               "so PR-AUC is compared with that base rate, not with 0.5.")
    st.subheader("Hit rate in the highest-risk slice")
    tk = pd.DataFrame(S["topk_precision"])
    tk["top_pct"] = tk["top_pct"].map(lambda v: f"Top {v}%")
    tk = tk.rename(columns={"top_pct": "Reviewed", "base_rate": "Base rate"})
    pct = tk.set_index("Reviewed")[["Logistic Regression", "Random Forest", "Hist Gradient Boosting", "Base rate"]] * 100
    st.bar_chart(pct[["Hist Gradient Boosting", "Base rate"]], color=[viz.ACCENT, viz.GRAY])
    st.dataframe(pct.round(1).astype(str) + "%", width="stretch")
    t = S["threshold_metrics"]
    st.info(f"At the operating threshold of {t['threshold']}, the model flagged {t['flagged']} series and caught "
            f"{t['caught']} of {t['true_spikes']} real spikes (precision {t['precision'] * 100:.0f}%, recall "
            f"{t['recall'] * 100:.0f}%). It is a prioritisation aid, not an alarm.")

with tab_explain:
    st.markdown("**Permutation importance**: how much the score gets worse when one input is shuffled. Bigger means the "
                "model leans on it more. Computed on the held-out test window.")
    c1, c2 = st.columns(2)
    reg_imp = pd.DataFrame(S["importance"]["regressor"]).head(8)
    clf_imp = pd.DataFrame(S["importance"]["classifier"]).head(8)
    with c1:
        st.plotly_chart(viz.hbar(reg_imp, "importance", "feature", "Price model", "Increase in MAE (USD) when shuffled",
                                 color=viz.ACCENT, fmt=".2f"), width="stretch")
    with c2:
        st.plotly_chart(viz.hbar(clf_imp, "importance", "feature", "Spike classifier", "Drop in PR-AUC when shuffled",
                                 color=viz.RED, fmt=".3f"), width="stretch")
    top_reg = ", ".join(reg_imp["feature"].head(3))
    top_clf = ", ".join(clf_imp["feature"].head(2))
    st.markdown(f"- The **price model** relies mainly on `{top_reg}`: it largely re-derives the persistence rule, which "
                "explains why it cannot beat it.\n"
                f"- The **spike classifier** relies most on `{top_clf}` (how expensive a market is versus the national "
                "median, and how long since its last spike). Its other inputs matter far less.")
    st.caption("SHAP breakdowns of individual predictions are in notebook 12.")

with tab_limits:
    st.markdown(
        "- **Prices**: the production price forecast is the simple persistence rule. The dashboard does not claim machine "
        "learning improves price forecasts, because testing showed it does not.\n"
        "- **Spike risk is modest**: better than chance, far from reliable. Small test window with only "
        f"{S['test_window']['positives']} real spikes, so the numbers are noisy.\n"
        "- **Scores are not probabilities**: class weighting inflates them. Use the ranking and bands.\n"
        "- **The deployed classifier was retrained on all data**, including the test window, so there is no fresh "
        "hold-out estimate for that exact model. Reported metrics come from the hold-out version.\n"
        "- **Coverage**: cereals and tubers, retail prices only, and only series with at least a year of history "
        "that reported in the latest month.\n"
        f"- **Reporting gaps**: {(1 - S['alignment_check']['share_next_target_exact_all_rows']) * 100:.0f}% of rows "
        "have a reporting gap before their 'next month' label, so that label is really 'next reported month', "
        "sometimes 2 or more months later. Restricting to only the "
        f"{S['alignment_check']['test_rows_exact']:,} calendar-exact test rows changes the numbers slightly "
        f"(persistence RMSE {S['alignment_check']['regression_exact_only'][0]['rmse']:.2f} vs "
        f"{next(r['RMSE'] for r in S['regression_panel'] if 'Persistence' in r['Model'])}) but not the "
        "conclusion: persistence still wins, and Hist Gradient Boosting is still the best spike ranker.\n"
        "- **Data quality**: reporting is sparse before 2020, and a 10-month gap (Sep 2018 to Jun 2019) in the "
        "national maize series was filled by interpolation.\n"
        "- **Not for**: automated trading or procurement decisions, or other commodity categories."
    )
