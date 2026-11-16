"""Data overview page: browse the raw and cleaned tables, their columns, statistics and charts."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_utils import (CONTROL_MAP, OWNERSHIP_COLORS, PREDDEG_MAP, REGION_MAP, TARGET, column_label,
                           load_model_table, load_raw_subset, show_chart, usd)

ID_COLUMNS = ["INSTNM", "CITY", "STABBR", "CONTROL", "PREDDEG"]
NOT_NUMERIC = ID_COLUMNS + ["UNITID", "OPEID6", "SPLIT", "REGION", "LOCALE_TYPE"]
GROUPS = {"Ownership": "CONTROL", "Main degree awarded": "PREDDEG", "Region": "REGION", "Locale": "LOCALE_TYPE"}
KEY_COLUMNS = [TARGET, "MD_FAMINC", "AVGFACSAL", "INEXPFTE", "TUITFTE", "UGDS", "PCTPELL", "PCTFLOAN",
               "FIRST_GEN", "AGE_ENTRY", "UG25ABV", "STUFACR", "MEDIAN_HH_INC", "POVERTY_RATE"]


def name_of(column: str) -> str:
    """Column code with its plain-language name, for menus and axis titles."""
    label = column_label(column)
    return f"{column}: {label}" if label else column


def axis_name(column: str) -> str:
    """Short axis title: the plain-language name if there is one, otherwise the column code."""
    return column_label(column).replace(": the target", "") or column

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
    df["REGION"] = df["REGION"].map(REGION_MAP)
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

# Number columns only, with the text code "PS" turned into a missing value so they can be summarised and plotted
numeric = shown.drop(columns=[c for c in NOT_NUMERIC if c in shown.columns]).apply(pd.to_numeric, errors="coerce")
number_columns = list(numeric.columns)
key_columns = [c for c in KEY_COLUMNS if c in numeric]

tab_table, tab_columns, tab_stats, tab_dist, tab_groups, tab_scatter, tab_corr = st.tabs(
    ["Table", "Columns and types", "Summary statistics", "Distributions", "Group comparison",
     "Relationships", "Correlations"])

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
    default_stats = [c for c in [TARGET, "UGDS", "INEXPFTE", "MD_FAMINC", "PCTPELL", "AVGFACSAL"] if c in numeric]
    picked = st.multiselect("Columns to summarise", number_columns, default=default_stats, format_func=name_of)
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

# ── Tab 4: distribution of one column ────────────────────────────────────────
with tab_dist:
    st.write("A histogram shows how the values of one column are spread out. The box above it marks the "
             "middle half of the colleges, and the dots are unusually high or low values.")
    left, right = st.columns([3, 1])
    column = left.selectbox("Column", number_columns, index=number_columns.index(TARGET), format_func=name_of,
                            key="dist_column")
    use_log = right.checkbox("Log scale", key="dist_log",
                             help="Squeezes a long tail of very large values so the shape is easier to see.")
    values = numeric[column].dropna()
    if use_log:
        values = np.log10(values[values > 0])
    if values.empty:
        st.info("This column has no values for the colleges shown.")
    else:
        title = axis_name(column) + (" (log10)" if use_log else "")
        fig = px.histogram(values.to_frame(name=title), x=title, nbins=50, marginal="box",
                           color_discrete_sequence=["#2a78d6"])
        fig.update_layout(yaxis_title="Number of colleges", bargap=0.03, margin=dict(t=30))
        show_chart(fig)
        st.caption(f"Mean {values.mean():,.2f}, median {values.median():,.2f}, skewness {values.skew():.2f}. "
                   "A skewness near 0 means the two sides are balanced. Above 1 means a long tail to the right.")

# ── Tab 5: one column compared across groups ─────────────────────────────────
with tab_groups:
    st.write("A box plot compares groups side by side. The line inside each box is the median, the box holds "
             "the middle half of the colleges, and the dots are outliers.")
    groups = {label: col for label, col in GROUPS.items() if col in shown.columns}
    left, right = st.columns(2)
    column = left.selectbox("Column", number_columns, index=number_columns.index(TARGET), format_func=name_of,
                            key="group_column")
    group_label = right.selectbox("Compare by", list(groups), key="group_by")
    group_col = groups[group_label]
    value_name = axis_name(column)
    data = pd.DataFrame({group_label: shown[group_col], value_name: numeric[column],
                         "College": shown["INSTNM"]}).dropna(subset=[group_label, value_name])
    if data.empty:
        st.info("There is nothing to compare for the colleges shown.")
    else:
        order = data.groupby(group_label)[value_name].median().sort_values(ascending=False).index.tolist()
        colors = OWNERSHIP_COLORS if group_col == "CONTROL" else None
        fig = px.box(data, x=group_label, y=value_name, color=group_label, hover_name="College",
                     category_orders={group_label: order}, color_discrete_map=colors)
        fig.update_layout(showlegend=False, margin=dict(t=30))
        show_chart(fig)
        summary = data.groupby(group_label)[value_name].agg(["count", "median", "mean", "std"]).loc[order]
        summary.columns = ["Colleges", "Median", "Mean", "Std. deviation"]
        st.dataframe(summary.style.format("{:,.2f}").format("{:,.0f}", subset=["Colleges"]), width="stretch")

# ── Tab 6: two columns against each other ────────────────────────────────────
with tab_scatter:
    st.write("A scatter plot puts one dot per college and shows whether two columns move together, and "
             "whether the link is straight, curved or not there at all.")
    col_x, col_y, col_opt = st.columns([2, 2, 1])
    default_x = "MD_FAMINC" if "MD_FAMINC" in numeric else number_columns[0]
    x = col_x.selectbox("Horizontal axis", number_columns, index=number_columns.index(default_x),
                        format_func=name_of, key="scatter_x")
    y = col_y.selectbox("Vertical axis", number_columns, index=number_columns.index(TARGET),
                        format_func=name_of, key="scatter_y")
    log_x = col_opt.checkbox("Log scale (horizontal)", value=True, key="scatter_log")
    data = pd.DataFrame({"x": numeric[x], "y": numeric[y], "Ownership": shown["CONTROL"],
                         "College": shown["INSTNM"]}).dropna()
    if log_x:
        data = data[data["x"] > 0]
    if len(data) < 3:
        st.info("There are not enough colleges with both values to draw this chart.")
    else:
        fig = px.scatter(data, x="x", y="y", color="Ownership", hover_name="College", log_x=log_x, opacity=0.5,
                         color_discrete_map=OWNERSHIP_COLORS, category_orders={"Ownership": list(OWNERSHIP_COLORS)},
                         labels={"x": axis_name(x), "y": axis_name(y)})
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(margin=dict(t=30), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        show_chart(fig)
        if data["x"].nunique() < 2 or data["y"].nunique() < 2:
            st.caption("The correlation cannot be worked out here, because one of the two columns has the same "
                       "value for every college shown.")
        else:
            rho = data["x"].corr(data["y"], method="spearman")
            st.caption(f"Spearman correlation: {rho:+.2f}, from {len(data):,} colleges. Values near +1 or -1 mean "
                       "a strong link, and values near 0 mean no clear link.")

# ── Tab 7: correlation matrix ────────────────────────────────────────────────
with tab_corr:
    st.write("The correlation matrix shows every pair of columns at once. Red means two columns rise together, "
             "blue means one rises as the other falls.")
    left, right = st.columns([4, 1])
    picked = left.multiselect("Columns", number_columns, default=key_columns, format_func=name_of, key="corr_columns")
    method = right.radio("Method", ["Spearman", "Pearson"], key="corr_method",
                         help="Spearman works on ranks, so extreme values and curved links do not distort it.")
    # a column with one single value has no correlation with anything, so it is left out
    flat = [c for c in picked if numeric[c].nunique() < 2]
    picked = [c for c in picked if c not in flat]
    if flat:
        st.caption("Left out because every college shown has the same value (or no value): " + ", ".join(flat) + ".")
    if len(picked) < 2:
        st.info("Pick at least two columns that have more than one value.")
    else:
        corr = numeric[picked].corr(method=method.lower())
        # full names down the side, short column codes along the bottom so the labels stay readable
        fig = px.imshow(corr.values, x=picked, y=[axis_name(c) for c in picked], text_auto=".2f",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=max(420, 34 * len(picked) + 120), margin=dict(t=30),
                          coloraxis_colorbar=dict(title="Correlation"))
        fig.update_xaxes(tickangle=-45)
        show_chart(fig)
        if TARGET in picked and len(picked) > 2:
            with_target = corr[TARGET].drop(TARGET).sort_values()
            bars = px.bar(x=with_target.values, y=[axis_name(c) for c in with_target.index], orientation="h",
                          color=with_target.values, color_continuous_scale="RdBu_r", range_color=(-1, 1),
                          labels={"x": f"{method} correlation with earnings", "y": ""})
            bars.update_layout(coloraxis_showscale=False, margin=dict(t=30),
                               height=max(300, 28 * len(with_target) + 100))
            st.subheader("Correlation with earnings")
            show_chart(bars)
