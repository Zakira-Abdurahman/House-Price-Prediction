"""Shared fixtures. Everything loads the real project artifacts (small enough to load fast, and the
whole point of these tests is to catch a data/model file going stale or out of sync with the code)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src import predict as P  # noqa: E402


@pytest.fixture(scope="session")
def features_df() -> pd.DataFrame:
    return D.load_features()


@pytest.fixture(scope="session")
def classifier():
    return P.load_classifier()


@pytest.fixture(scope="session")
def scored(features_df, classifier) -> pd.DataFrame:
    return P.score_latest(features_df, model=classifier)
