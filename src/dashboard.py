"""Shared Streamlit plumbing: cached loaders, page setup and small UI helpers."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from . import data as D
from . import predict as P
from .config import APP_TITLE, MODEL_CARD_PATH, SUMMARY_JSON

DISCLAIMER = ("Decision-support tool for food-security monitoring. Spike-risk scores rank series for review; "
              "they are not calibrated probabilities. Confirm alerts with local market information.")


@st.cache_data(show_spinner="Loading price data...")
def get_features() -> pd.DataFrame:
    return D.load_features()


@st.cache_resource(show_spinner="Loading model...")
def get_classifier():
    return P.load_classifier()


@st.cache_data(show_spinner="Scoring series...")
def get_scored() -> pd.DataFrame:
    return P.score_latest(get_features(), model=get_classifier())


@st.cache_data
def get_national_maize() -> pd.Series:
    return D.load_national_maize()


@st.cache_data
def get_summary() -> dict:
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


@st.cache_data
def get_model_card() -> dict:
    return json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))


def page_setup(title: str, icon: str) -> None:
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"{title} | {APP_TITLE}", page_icon=icon, layout="wide")
    st.markdown(
        "<style>[data-testid='stMetricValue']{font-size:1.6rem}"
        ".block-container{padding-top:2rem}</style>", unsafe_allow_html=True)
    df = get_features()
    with st.sidebar:
        st.caption(f"Data through **{D.latest_date(df):%B %Y}**")
        st.caption(DISCLAIMER)


def money(v: float) -> str:
    return f"${v:,.2f}"
