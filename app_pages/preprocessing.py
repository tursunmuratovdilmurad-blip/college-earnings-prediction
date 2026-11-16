"""Preprocessing page: what cleaning, imputation, outlier capping and scaling did to the data."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import (column_label, load_fit_data, load_metadata, load_report_table,
                           numeric_preprocessing_details, show_chart)

meta = load_metadata()
before, after = load_fit_data()
details = numeric_preprocessing_details()
numeric_features = list(details.index)


def name_of(column: str) -> str:
    label = column_label(column)
    return f"{column}: {label}" if label else column


st.title("Preprocessing")
st.write(
    "Raw data cannot go straight into a model. This page shows each step that prepares it, with the data "
    "before and after. Everything that is learned from the data, such as medians and limits, comes from the "
    f"{len(before):,} colleges used to train the final model, never from the test set."
)

tab_clean, tab_missing, tab_outliers, tab_scaling = st.tabs(
    ["Cleaning steps", "Missing values", "Outliers", "Scaling"])

# ── Tab 1: cleaning ──────────────────────────────────────────────────────────
with tab_clean:
    audit = load_report_table("cleaning_audit.csv")
    removed = audit[audit["unit"] == "rows removed"]
    repaired = audit[audit["unit"] == "rows repaired"]
    start, end = int(audit["rows_remaining"].iloc[0]), int(removed["rows_remaining"].iloc[-1])

    col1, col2, col3 = st.columns(3)
    col1.metric("Colleges in the raw file", f"{start:,}")
    col2.metric("Colleges after cleaning", f"{end:,}")
    col3.metric("Removed", f"{start - end:,}", help="Rows that could not be used for training.")

    st.subheader("Which colleges were removed")
    steps = pd.DataFrame({
        "Step": ["Raw file", "No earnings figure", "College has closed", "Graduate-only or unclassified",
                 "Earnings based on fewer than 30 people", "One row per institution"],
        "Colleges left": [start] + removed["rows_remaining"].tolist()[1:],
    })
    fig = px.funnel(steps, x="Colleges left", y="Step", color_discrete_sequence=["#2a78d6"])
    fig.update_layout(margin=dict(t=20), yaxis_title="")
    show_chart(fig)
    st.write(
        "The last step matters most. The government works out earnings once per institution, but every branch "
        "campus is its own row with the same earnings. Keeping one row per institution stops copies of the same "
        "answer from landing in both the training and the test data."
    )

    st.subheader("Which values were repaired")
    st.write("When one value in a row was impossible, only that value was set to missing and the row was kept.")
    st.dataframe(repaired[["rule", "affected"]].rename(columns={"rule": "Rule", "affected": "Rows affected"}),
                 hide_index=True, width="stretch")

# ── Tab 2: missing values ────────────────────────────────────────────────────
with tab_missing:
    missing_before = int(before.isna().sum().sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Missing cells before", f"{missing_before:,}")
    col2.metric("Missing cells after", f"{int(after.isna().sum().sum()):,}")
    col3.metric("Rows complete before", f"{before.notna().all(axis=1).mean():.0%}",
                help="Share of colleges with no gap in any feature. Deleting the rest would lose about half the data.")

    st.write(
        "Gaps in number columns are filled with the median of the training data, because the median is not "
        "pulled by a few giant values the way the mean is. A yes/no column is added for each filled feature, "
        "so the model still knows the value was missing. Text columns get their own label, 'Missing'."
    )
    dropped = meta["features"]["dropped_for_missingness"]
    if dropped:
        st.write(f"One rule came first: a feature with more than half of its values missing is dropped. "
                 f"That removed {', '.join(dropped)}.")

    share = (before.isna().mean() * 100).sort_values(ascending=False)
    share = share[share > 0].head(15).iloc[::-1]
    fig = px.bar(x=share.values, y=[column_label(c) or c for c in share.index], orientation="h",
                 labels={"x": "Missing before filling (%)", "y": ""}, color_discrete_sequence=["#eb7a2d"])
    fig.update_layout(margin=dict(t=20), height=480)
    st.subheader("Features with the most gaps")
    show_chart(fig)

    st.subheader("What each number feature was filled with")
    table = details[["missing", "missing %", "filled with (median)"]].copy()
    table.insert(0, "meaning", [column_label(c) for c in table.index])
    table = table[table["missing"] > 0].sort_values("missing", ascending=False)
    st.dataframe(table.style.format({"missing": "{:,.0f}", "missing %": "{:.1f}", "filled with (median)": "{:,.2f}"}),
                 width="stretch")

# ── Tab 3: outliers ──────────────────────────────────────────────────────────
with tab_outliers:
    st.write(
        "Extreme values can pull a model around. They are not deleted here. Any value below the 1st percentile "
        "or above the 99th percentile of the training data is pulled back to that limit, so the college stays "
        "in the data and only the extreme number is softened."
    )
    feature = st.selectbox("Feature", numeric_features, index=numeric_features.index("UGDS"),
                           format_func=name_of, key="outlier_feature")
    row = details.loc[feature]
    values = before[feature].dropna().astype(float)
    capped = values.clip(float(row["lower cap"]), float(row["upper cap"]))
    q1, q3 = values.quantile([0.25, 0.75])
    iqr_share = ((values < q1 - 1.5 * (q3 - q1)) | (values > q3 + 1.5 * (q3 - q1))).mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lower limit", f"{row['lower cap']:,.2f}")
    col2.metric("Upper limit", f"{row['upper cap']:,.2f}")
    col3.metric("Values capped", f"{int(row['values capped']):,}", help=f"Out of {len(values):,} colleges.")
    col4.metric("Flagged by the 1.5 x IQR rule", f"{iqr_share:.1%}",
                help="The usual textbook rule. On lopsided data it marks far too many real colleges as outliers, "
                     "which is why it is not used here.")

    fig = go.Figure()
    fig.add_trace(go.Box(x=values, name="Before", marker_color="#eb7a2d", boxpoints="outliers"))
    fig.add_trace(go.Box(x=capped, name="After capping", marker_color="#1baf7a", boxpoints="outliers"))
    fig.update_layout(margin=dict(t=20), height=320, showlegend=False,
                      xaxis_title=column_label(feature) or feature)
    fig.update_yaxes(autorange="reversed")                   # "Before" on top, "After capping" below it
    show_chart(fig)
    st.caption(f"Largest value before: {values.max():,.2f}. Largest value after capping: {capped.max():,.2f}.")

# ── Tab 4: scaling ───────────────────────────────────────────────────────────
with tab_scaling:
    st.write(
        "Features come in very different sizes: enrolment is in thousands and a share is between 0 and 1. "
        "Two of the models are sensitive to this, so every number feature is rescaled to a mean of 0 and a "
        "standard deviation of 1. Features with a long tail to the right also get a log transform first."
    )
    feature = st.selectbox("Feature", numeric_features, index=numeric_features.index("INEXPFTE"),
                           format_func=name_of, key="scaling_feature")
    uses_log = bool(details.loc[feature, "log transform"])
    raw_values, new_values = before[feature].dropna(), after[feature]

    left, right = st.columns(2)
    with left:
        st.subheader("Before")
        fig = px.histogram(raw_values, nbins=50, color_discrete_sequence=["#eb7a2d"])
        fig.update_layout(showlegend=False, margin=dict(t=20), height=340, yaxis_title="Number of colleges",
                          xaxis_title=column_label(feature) or feature)
        show_chart(fig)
    with right:
        st.subheader("After")
        fig = px.histogram(new_values, nbins=50, color_discrete_sequence=["#1baf7a"])
        fig.update_layout(showlegend=False, margin=dict(t=20), height=340, yaxis_title="Number of colleges",
                          xaxis_title="Standardised value")
        show_chart(fig)

    steps = (["log transform"] if uses_log else []) + ["capping at the 1st and 99th percentile",
                                                       "scaling to mean 0 and standard deviation 1",
                                                       "filling gaps with the median"]
    st.write("Steps applied to this feature: " + ", then ".join(steps) + ".")
    compare = pd.DataFrame({"Before": [raw_values.mean(), raw_values.std(), raw_values.min(), raw_values.max(),
                                       raw_values.skew()],
                            "After": [new_values.mean(), new_values.std(), new_values.min(), new_values.max(),
                                      new_values.skew()]},
                           index=["Mean", "Std. deviation", "Min", "Max", "Skewness"])
    st.dataframe(compare.style.format("{:,.2f}"), width="stretch")
