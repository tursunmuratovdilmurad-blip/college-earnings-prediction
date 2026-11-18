"""Preprocessing page: cleaning, new features, imputation, outlier capping, scaling and encoding."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import (TARGET, column_label, load_fit_data, load_metadata, load_model_table,
                           load_preprocessor, load_raw_subset, load_report_table, load_state_encoding,
                           numeric_preprocessing_details, section_picker, show_chart)

SUBJECT_GROUPS = {
    "STEM_SHARE": ["PCIP11", "PCIP14", "PCIP15", "PCIP26", "PCIP27", "PCIP40", "PCIP41"],
    "HEALTH_SHARE": ["PCIP51"],
    "BUSINESS_SHARE": ["PCIP52"],
    "TRADES_SHARE": ["PCIP12", "PCIP46", "PCIP47", "PCIP48", "PCIP49"],
    "ARTS_HUM_SHARE": ["PCIP16", "PCIP23", "PCIP24", "PCIP38", "PCIP39", "PCIP50", "PCIP54"],
    "EDU_SOCIAL_SHARE": ["PCIP13", "PCIP19", "PCIP42", "PCIP44", "PCIP45"],
}
# new feature -> (raw columns it is built from, how, why)
MERGED = {
    "NET_PRICE": (["NPT4_PUB", "NPT4_PRIV"], "Public colleges fill one column and private colleges the other"),
    "COMPLETION_RATE": (["C150_4", "C150_L4"], "4-year colleges fill one column and shorter colleges the other"),
    "RETENTION_RATE": (["RET_FT4", "RET_FTL4"], "4-year colleges fill one column and shorter colleges the other"),
    "COST_ATTEND": (["COSTT4_A", "COSTT4_P"], "Academic-year colleges fill one, programme-year colleges the other"),
    "TUITION": (["TUITIONFEE_IN", "TUITIONFEE_PROG"], "Academic-year colleges fill one, programme-year colleges the other"),
}
HIGHDEG_NAMES = {1: "Certificate", 2: "Associate degree", 3: "Bachelor's degree", 4: "Graduate degree"}

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

section = section_picker(['Cleaning steps', 'New features', 'Missing values', 'Outliers', 'Scaling', 'Encoding', 'One college'], key="prep_section")

# ── Tab 1: cleaning ──────────────────────────────────────────────────────────
if section == 'Cleaning steps':
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
if section == 'Missing values':
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
if section == 'Outliers':
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
    limit = lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"   # no decimals on large numbers, so it fits
    col1.metric("Lower limit", limit(row["lower cap"]), help=f"Exact value: {row['lower cap']:,.4f}")
    col2.metric("Upper limit", limit(row["upper cap"]), help=f"Exact value: {row['upper cap']:,.4f}")
    col3.metric("Values capped", f"{int(row['values capped']):,}", help=f"Out of {len(values):,} colleges.")
    col4.metric("IQR rule flags", f"{iqr_share:.1%}",
                help="Share of colleges the usual 1.5 x IQR rule would call outliers. On lopsided data it marks "
                     "far too many real colleges as outliers, which is why it is not used here.")

    fig = go.Figure()
    fig.add_trace(go.Box(x=values, name="Before", marker_color="#eb7a2d", boxpoints="outliers"))
    fig.add_trace(go.Box(x=capped, name="After capping", marker_color="#1baf7a", boxpoints="outliers"))
    fig.update_layout(margin=dict(t=20), height=320, showlegend=False,
                      xaxis_title=column_label(feature) or feature)
    fig.update_yaxes(autorange="reversed")                   # "Before" on top, "After capping" below it
    show_chart(fig)
    st.caption(f"Largest value before: {values.max():,.2f}. Largest value after capping: {capped.max():,.2f}.")

# ── Tab 4: scaling ───────────────────────────────────────────────────────────
if section == 'Scaling':
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

# ── Tab: new features ────────────────────────────────────────────────────────
if section == 'New features':
    table = load_model_table()
    raw = load_raw_subset()
    raw = raw[raw["UNITID"].isin(table["UNITID"])].apply(pd.to_numeric, errors="coerce")

    st.write(
        "Some useful information is hidden in the raw columns and has to be built. Each new feature uses only "
        "values from the college's own row, so building them cannot leak anything from the test set."
    )

    st.subheader("One piece of information split into two columns")
    st.write("Many raw columns come in pairs where each college fills in only one. On their own they look "
             "half empty. Joined into one column, almost nothing is missing.")
    rows = []
    for feature, (sources, _) in MERGED.items():
        for source in sources:
            rows.append({"Feature": column_label(feature), "Column": source, "Missing (%)": raw[source].isna().mean() * 100,
                         "Stage": "Raw columns"})
        rows.append({"Feature": column_label(feature), "Column": feature, "Missing (%)": table[feature].isna().mean() * 100,
                     "Stage": "Joined feature"})
    pairs = pd.DataFrame(rows)
    fig = px.bar(pairs, x="Missing (%)", y="Feature", color="Stage", barmode="group", orientation="h",
                 hover_name="Column", color_discrete_map={"Raw columns": "#eb7a2d", "Joined feature": "#1baf7a"})
    fig.update_layout(margin=dict(t=20), height=420, yaxis_title="", legend_title="")
    show_chart(fig)

    st.subheader("All new features")
    recipes = [{"Feature": f, "Meaning": column_label(f), "Built from": " or ".join(src), "Why": why}
               for f, (src, why) in MERGED.items()]
    recipes.append({"Feature": "ADM_RATE_FILLED", "Meaning": column_label("ADM_RATE_FILLED"),
                    "Built from": "ADM_RATE, OPENADMP",
                    "Why": "Colleges that accept everyone do not report a rate. For them it is known to be 1"})
    for feature, sources in SUBJECT_GROUPS.items():
        recipes.append({"Feature": feature, "Meaning": column_label(feature), "Built from": " + ".join(sources),
                        "Why": "38 thin subject columns are added up into 6 groups that are easier to learn from"})
    recipes.append({"Feature": "PROGRAM_HHI", "Meaning": column_label("PROGRAM_HHI"),
                    "Built from": "all 38 subject shares, squared and added up",
                    "Why": "Tells a one-subject school (near 1) from a broad university (near 0)"})
    recipes.append({"Feature": "INSTR_PER_TUITION", "Meaning": column_label("INSTR_PER_TUITION"),
                    "Built from": "INEXPFTE / TUITFTE",
                    "Why": "How much of what students pay goes into teaching them"})
    recipes.append({"Feature": "LOCALE_TYPE", "Meaning": column_label("LOCALE_TYPE"),
                    "Built from": "first digit of LOCALE",
                    "Why": "Keeps city / suburb / town / rural in 4 groups, not 12"})
    recipes = pd.DataFrame(recipes)
    st.dataframe(recipes, hide_index=True, width="stretch")

    st.subheader("Do the new features help?")
    numeric_new = [f for f in recipes["Feature"] if f != "LOCALE_TYPE"]
    link = table[numeric_new].corrwith(table[TARGET], method="spearman").sort_values()
    fig = px.bar(x=link.values, y=[column_label(c) for c in link.index], orientation="h", color=link.values,
                 color_continuous_scale="RdBu_r", range_color=(-1, 1),
                 labels={"x": "Spearman correlation with earnings", "y": ""})
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=20), height=460)
    show_chart(fig)
    st.caption(
        "The share of trades programmes and the share of STEM programmes are among the strongest signals in the "
        "whole dataset. The completion rate looks weak on its own, because short certificate courses are easy "
        "to finish but lead to low pay. It becomes useful together with the level of degree."
    )

# ── Tab: encoding ────────────────────────────────────────────────────────────
if section == 'Encoding':
    prep = load_preprocessor()
    st.write("Models work with numbers, so text columns have to be turned into numbers. Three methods are used, "
             "each where it fits.")

    st.subheader("One-hot encoding: one yes/no column per option")
    st.write("Used when a column has few options and no natural order. Numbering the options 1, 2, 3 would "
             "invent an order that does not exist.")
    encoder = prep.named_transformers_["ohe"].named_steps["encode"]
    onehot_columns = meta["features"]["onehot"]
    feature = st.selectbox("Column", onehot_columns, format_func=name_of, key="onehot_feature")
    options = list(encoder.categories_[onehot_columns.index(feature)])
    grid = pd.DataFrame(np.eye(len(options), dtype=int), index=options, columns=[f"{feature}_{o}" for o in options])
    grid.index.name = "Original value"
    st.dataframe(grid, width="stretch")
    st.caption(f"{column_label(feature)} has {len(options)} options, so it becomes {len(options)} columns of 0 and 1.")

    st.subheader("Ordered numbers: when the options have a natural order")
    st.write("The highest degree a college offers goes from certificate up to graduate degree, so it is stored "
             "as a single number that keeps that order.")
    ordered = pd.DataFrame({"Highest degree offered": list(HIGHDEG_NAMES.values()),
                            "Value in the data": list(HIGHDEG_NAMES),
                            "Value given to the model": range(len(HIGHDEG_NAMES))})
    st.dataframe(ordered, hide_index=True, width="stretch")

    st.subheader("Target encoding: for a column with many options")
    states, overall = load_state_encoding()
    st.write(
        f"State has {len(states)} options. One-hot encoding would add {len(states)} thin columns. Target encoding "
        "replaces each state with one number: the typical earnings of colleges in that state. During training, "
        "each college's value is worked out from other colleges only, so its own answer cannot leak into its own "
        "feature. The chart shows the values used when predicting for a new college."
    )
    states = states.sort_values("Value used (USD)")
    fig = px.bar(states, x="State", y="Value used (USD)", hover_data=["Colleges", "Median earnings (USD)"],
                 color_discrete_sequence=["#2a78d6"])
    fig.add_hline(y=overall, line_dash="dash", line_color="grey", annotation_text="overall")
    fig.update_layout(margin=dict(t=20), height=380)
    show_chart(fig)
    st.dataframe(states.sort_values("Colleges").style.format(
        {"Value used (USD)": "{:,.0f}", "Median earnings (USD)": "{:,.0f}"}), hide_index=True, width="stretch")
    st.caption("Sorted by number of colleges. The value used is a blend of the state's own average and the overall "
               "average. It stays closest to the state's own figure when the state has many colleges or when its "
               "colleges earn similar amounts.")

# ── Tab: one college through every step ──────────────────────────────────────
if section == 'One college':
    table = load_model_table()
    prep = load_preprocessor()
    st.write("Pick a college to see its own values before and after preprocessing.")
    names = (table["INSTNM"] + " (" + table["CITY"] + ", " + table["STABBR"] + ")").tolist()
    choices = {}                                                # text shown in the menu -> position in the table
    for i, name in enumerate(names):
        choices[name if name not in choices else f"{name} [{i}]"] = i   # keeps every entry unique
    default = next((i for i, n in enumerate(choices) if n.startswith("Massachusetts Institute of Technology")), 0)
    position = choices[st.selectbox("College", list(choices), index=default, key="college_pick")]
    college = table.iloc[[position]]
    features = meta["features"]["all"]
    result = pd.Series(prep.transform(college[features])[0], index=prep.get_feature_names_out())

    col1, col2, col3 = st.columns(3)
    col1.metric("Earnings after 10 years", f"${college[TARGET].iloc[0]:,.0f}")
    col2.metric("Data set", college["SPLIT"].iloc[0].capitalize())
    col3.metric("Model inputs", len(result),
                help=f"{len(features)} features go in. One-hot columns and 'was missing' flags bring the total up.")

    st.subheader("Number features")
    rows = []
    for feature in numeric_features:
        value, info = college[feature].iloc[0], details.loc[feature]
        notes = []
        if pd.isna(value):
            notes.append("missing, filled with the median")
        elif value < info["lower cap"] - 1e-9 or value > info["upper cap"] + 1e-9:
            notes.append("capped")
        if info["log transform"]:
            notes.append("log transform")
        rows.append({"Feature": feature, "Meaning": column_label(feature), "Original value": value,
                     "After preprocessing": result[feature], "What happened": ", ".join(notes)})
    st.dataframe(pd.DataFrame(rows).style.format({"Original value": "{:,.3f}", "After preprocessing": "{:+.2f}"},
                                                  na_rep="missing"), hide_index=True, width="stretch")
    st.caption("After preprocessing, 0 means the same as an average college, +1 means one standard deviation "
               "above it and -1 one below.")

    st.subheader("Text and yes/no features")
    rows = []
    for feature in meta["features"]["onehot"]:
        value = college[feature].iloc[0]
        label = "Missing" if pd.isna(value) else value
        rows.append({"Feature": feature, "Meaning": column_label(feature), "Original value": str(label),
                     "After preprocessing": f"{feature}_{label} = 1, the other {feature} columns = 0"})
    state = college["STABBR"].iloc[0]
    rows.append({"Feature": "STABBR", "Meaning": column_label("STABBR"), "Original value": state,
                 "After preprocessing": f"{result['STABBR']:+.2f} (target encoded, then scaled)"})
    degree = college["HIGHDEG"].iloc[0]
    rows.append({"Feature": "HIGHDEG", "Meaning": "Highest degree offered",
                 "Original value": f"{int(degree)} ({HIGHDEG_NAMES.get(int(degree), '')})",
                 "After preprocessing": f"{result['HIGHDEG']:.0f}"})
    for feature in meta["features"]["binary"]:
        value = college[feature].iloc[0]
        rows.append({"Feature": feature, "Meaning": column_label(feature),
                     "Original value": "missing" if pd.isna(value) else ("yes" if value == 1 else "no"),
                     "After preprocessing": f"{result[feature]:.0f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

