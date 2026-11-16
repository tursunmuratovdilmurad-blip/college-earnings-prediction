"""Shared helpers for the Streamlit app: file paths, cached loaders and small formatters."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "MD_EARN_WNE_P10"
TARGET_LABEL = "Median earnings 10 years after entry (USD)"

# Codes used in the raw College Scorecard file
CONTROL_MAP = {1: "Public", 2: "Private nonprofit", 3: "Private for-profit"}
PREDDEG_MAP = {0: "Not classified", 1: "Certificate", 2: "Associate", 3: "Bachelor", 4: "Graduate"}
REGION_MAP = {0: "U.S. Service Schools", 1: "New England", 2: "Mid East", 3: "Great Lakes", 4: "Plains",
              5: "Southeast", 6: "Southwest", 7: "Rocky Mountains", 8: "Far West", 9: "Outlying Areas"}

# One colour per type of ownership, the same in every chart
OWNERSHIP_COLORS = {"Public": "#2a78d6", "Private nonprofit": "#1baf7a", "Private for-profit": "#eb7a2d"}

# Plain-language names for raw columns that are not model features
RAW_LABELS = {
    "UNITID": "College ID (one per campus)", "OPEID6": "Institution ID used for federal aid",
    "INSTNM": "College name", "CITY": "City", "STABBR": "State", "LATITUDE": "Latitude", "LONGITUDE": "Longitude",
    "MAIN": "Main campus (1 = yes)", "CURROPER": "Currently operating (1 = yes)", "LOCALE": "Locale code (12 levels)",
    "MENONLY": "Men-only college", "WOMENONLY": "Women-only college", "OPENADMP": "Open admission (1 = yes, 2 = no)",
    "ADM_RATE": "Admission rate", "SAT_AVG": "Average SAT score",
    "UGDS_AIAN": "Share American Indian / Alaska Native", "UGDS_NHPI": "Share Native Hawaiian / Pacific Islander",
    "UGDS_2MOR": "Share two or more races", "UGDS_UNKN": "Share race unknown", "UGDS_MEN": "Share of men",
    "TUITIONFEE_IN": "In-state tuition and fees (USD)", "TUITIONFEE_OUT": "Out-of-state tuition and fees (USD)",
    "TUITIONFEE_PROG": "Tuition and fees, programme-year colleges (USD)",
    "COSTT4_A": "Cost of attendance, academic-year colleges (USD)",
    "COSTT4_P": "Cost of attendance, programme-year colleges (USD)",
    "NPT4_PUB": "Net price, public colleges (USD)", "NPT4_PRIV": "Net price, private colleges (USD)",
    "C150_4": "Completion rate, 4-year colleges", "C150_L4": "Completion rate, shorter colleges",
    "RET_FT4": "Retention rate, 4-year colleges", "RET_FTL4": "Retention rate, shorter colleges",
    "DEPENDENT": "Share of dependent students",
    "MD_EARN_WNE_P10": "Median earnings 10 years after entry (USD): the target",
    "MD_EARN_WNE_P6": "Median earnings 6 years after entry (USD)",
    "COUNT_WNE_P10": "Number of former students behind the earnings figure",
    "SPLIT": "Train, validation or test set",
}


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


@st.cache_data
def load_fit_data():
    """The colleges the final model was trained on (training + validation sets).

    Returns two tables with the same rows: the features before preprocessing, and the same
    features after the saved pipeline has imputed, capped, scaled and encoded them.
    """
    table = load_model_table()
    rows = table[table["SPLIT"].isin(["train", "validation"])]
    before = rows[load_metadata()["features"]["all"]]
    prep = load_final_model().regressor_.named_steps["prep"]
    after = pd.DataFrame(prep.transform(before), columns=prep.get_feature_names_out(), index=before.index)
    return before, after


@st.cache_data
def numeric_preprocessing_details() -> pd.DataFrame:
    """What the saved pipeline does to each numeric feature, in the feature's original units."""
    prep = load_final_model().regressor_.named_steps["prep"]
    before, _ = load_fit_data()
    rows = []
    for name, pipeline, columns in prep.transformers_:
        if name not in ("log_num", "num"):
            continue
        winsor = pipeline.named_steps["winsor"]
        uses_log = name == "log_num"
        to_original = np.expm1 if uses_log else (lambda v: v)
        for i, column in enumerate(columns):
            values = before[column]
            low, high = to_original(winsor.lower_bounds_[i]), to_original(winsor.upper_bounds_[i])
            rows.append({
                "feature": column,
                "log transform": uses_log,
                "missing": int(values.isna().sum()),
                "missing %": values.isna().mean() * 100,
                "filled with (median)": values.median(),
                "lower cap": low,
                "upper cap": high,
                "values capped": int(((values < low - 1e-9) | (values > high + 1e-9)).sum()),
            })
    return pd.DataFrame(rows).set_index("feature")


def column_label(column: str) -> str:
    """Plain-language name of a column, or an empty string if there is none."""
    labels = load_metadata()["features"]["labels"]
    if column in labels:
        return labels[column]
    if column in RAW_LABELS:
        return RAW_LABELS[column]
    if column.startswith("PCIP"):
        return f"Share of awards in subject group {column[4:]}"
    return ""


def show_chart(fig) -> None:
    """Draw a Plotly chart across the full width of its container."""
    st.plotly_chart(fig, use_container_width=True)


def usd(value: float) -> str:
    """Format a number as whole dollars, e.g. 45574.2 -> '$45,574'."""
    return f"${value:,.0f}"


def usd_md(value: float) -> str:
    """Same as usd(), with the dollar sign escaped so Markdown does not read it as a formula."""
    return usd(value).replace("$", "\\$")
