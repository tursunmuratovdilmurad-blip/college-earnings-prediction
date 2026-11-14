"""Shared helpers for the Streamlit app: file paths, cached loaders and small formatters."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "MD_EARN_WNE_P10"
TARGET_LABEL = "Median earnings 10 years after entry (USD)"


@st.cache_data
def load_metadata() -> dict:
    """Settings, feature lists and scores saved by the notebook."""
    return json.loads((MODELS_DIR / "metadata.json").read_text())


@st.cache_data
def load_raw_subset() -> pd.DataFrame:
    """The columns of the raw College Scorecard file used in the project, before cleaning."""
    return pd.read_csv(DATA_DIR / "raw_subset.csv", low_memory=False)


@st.cache_data
def load_model_table() -> pd.DataFrame:
    """The cleaned table: one row per institution, with engineered features and the split label."""
    return pd.read_csv(DATA_DIR / "model_table.csv")


@st.cache_data
def load_report_table(name: str, **kwargs) -> pd.DataFrame:
    """A result table written by the notebook, for example 'metrics_test.csv'."""
    return pd.read_csv(REPORTS_DIR / name, **kwargs)


@st.cache_resource
def load_final_model():
    """The final model (preprocessing + SVR) used for live predictions."""
    return joblib.load(MODELS_DIR / load_metadata()["best_model"]["file"])


def usd(value: float) -> str:
    """Format a number as whole dollars, e.g. 45574.2 -> '$45,574'."""
    return f"${value:,.0f}"


def usd_md(value: float) -> str:
    """Same as usd(), with the dollar sign escaped so Markdown does not read it as a formula."""
    return usd(value).replace("$", "\\$")
