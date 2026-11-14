"""Data overview page: browse the raw and cleaned tables, their columns and their summary statistics."""
import pandas as pd
import streamlit as st

from src.app_utils import (CONTROL_MAP, PREDDEG_MAP, TARGET, column_label, load_model_table,
                           load_raw_subset, usd)

ID_COLUMNS = ["INSTNM", "CITY", "STABBR", "CONTROL", "PREDDEG"]

st.title("Data overview")
st.write(
    "Look at the data before and after cleaning. Use the filters on the left to narrow the table down, "
    "then check the columns, their types and their summary statistics in the tabs below."
)

view = st.radio("Which table do you want to see?",
                ["Raw data (before cleaning)", "Cleaned data (used for the models)"], horizontal=True)
is_raw = view.startswith("Raw")

df = (load_raw_subset() if is_raw else load_model_table()).copy()
file_types = df.dtypes                                       # how each column is stored in the file
if is_raw:
    # The raw file stores these as number codes. Names are easier to read and to filter.
    df["CONTROL"] = df["CONTROL"].map(CONTROL_MAP)
    df["PREDDEG"] = df["PREDDEG"].map(PREDDEG_MAP)
earnings = pd.to_numeric(df[TARGET], errors="coerce")       # in the raw file some values are the text "PS"

# ── Filters ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    ownership = st.multiselect("Ownership", sorted(df["CONTROL"].dropna().unique()))
    degree = st.multiselect("Main degree awarded", sorted(df["PREDDEG"].dropna().unique()))
    states = st.multiselect("State", sorted(df["STABBR"].dropna().unique()))
    name = st.text_input("College name contains")
    low, high = int(earnings.min()), int(earnings.max())
    earn_range = st.slider("Earnings after 10 years (USD)", low, high, (low, high), step=1000)
    keep_missing = True
    if is_raw:
        keep_missing = st.checkbox("Include colleges with no earnings figure", value=True)

mask = pd.Series(True, index=df.index)
if ownership:
    mask &= df["CONTROL"].isin(ownership)
if degree:
    mask &= df["PREDDEG"].isin(degree)
if states:
    mask &= df["STABBR"].isin(states)
if name:
    mask &= df["INSTNM"].str.contains(name, case=False, na=False, regex=False)
in_range = earnings.between(*earn_range)
mask &= (in_range | earnings.isna()) if keep_missing else in_range

shown = df[mask]
shown_earnings = earnings[mask]

# ── Headline numbers ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Colleges shown", f"{len(shown):,}", help=f"Out of {len(df):,} in this table.")
col2.metric("Columns", df.shape[1])
col3.metric("Missing cells", f"{shown.isna().mean().mean():.1%}" if len(shown) else "n/a")
col4.metric("Median earnings", usd(shown_earnings.median()) if shown_earnings.notna().any() else "n/a")

if shown.empty:
    st.warning("No colleges match these filters. Try removing one.")
    st.stop()

tab_table, tab_columns, tab_stats = st.tabs(["Table", "Columns and types", "Summary statistics"])

# ── Tab 1: the table itself ──────────────────────────────────────────────────
with tab_table:
    default_cols = ID_COLUMNS + ["UGDS", TARGET] + ([] if is_raw else ["SPLIT"])
    columns = st.multiselect("Columns to show", list(df.columns), default=default_cols)
    if columns:
        st.dataframe(shown[columns], hide_index=True, width="stretch")
        st.download_button("Download these rows as CSV", shown[columns].to_csv(index=False),
                           file_name="colleges.csv", mime="text/csv")
    else:
        st.info("Pick at least one column.")

# ── Tab 2: columns and data types ────────────────────────────────────────────
with tab_columns:
    info = pd.DataFrame({
        "Column": df.columns,
        "Meaning": [column_label(c) for c in df.columns],
        "Type": ["text" if file_types[c] == object else "whole number" if "int" in str(file_types[c]) else "number"
                 for c in df.columns],
        "Filled": shown.notna().sum().values,
        "Missing %": (shown.isna().mean() * 100).round(1).values,
        "Different values": shown.nunique().values,
    })
    if is_raw:
        info["Hidden for privacy (PS)"] = [(shown[c] == "PS").sum() if shown[c].dtype == object else 0
                                           for c in df.columns]
    type_counts = info["Type"].value_counts()
    st.write(f"This table has {len(df.columns)} columns: "
             + ", ".join(f"{n} {t}" for t, n in type_counts.items()) + ".")
    if is_raw:
        st.write(
            "In the raw file some number columns are stored as text. The reason is the code PS, which the "
            "government writes when it hides a value to protect the privacy of small groups of students. "
            "During cleaning these cells are turned into proper missing values."
        )
    st.dataframe(info, hide_index=True, width="stretch")

# ── Tab 3: summary statistics ────────────────────────────────────────────────
with tab_stats:
    numeric = shown.drop(columns=[c for c in ID_COLUMNS + ["UNITID", "OPEID6", "SPLIT"] if c in shown.columns])
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    default_stats = [c for c in [TARGET, "UGDS", "INEXPFTE", "MD_FAMINC", "PCTPELL", "AVGFACSAL"] if c in numeric]
    picked = st.multiselect("Columns to summarise", list(numeric.columns), default=default_stats,
                            format_func=lambda c: f"{c}: {column_label(c)}" if column_label(c) else c)
    if picked:
        part = numeric[picked]
        stats = pd.DataFrame({
            "Count": part.count(),
            "Mean": part.mean(),
            "Median": part.median(),
            "Std. deviation": part.std(),
            "Variance": part.var(),
            "Min": part.min(),
            "Max": part.max(),
            "Skewness": part.skew(),
        })
        st.dataframe(stats.style.format("{:,.2f}").format("{:,.0f}", subset=["Count"]), width="stretch")
        st.caption(
            "The mean and the median both show the middle of a column. When the mean is far above the median, "
            "a few very large values are pulling it up, and the skewness is well above zero."
        )
    else:
        st.info("Pick at least one column.")
