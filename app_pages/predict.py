"""Prediction page: describe a college and get its predicted graduate earnings from the final model."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_utils import (TARGET, column_label, load_final_model, load_metadata, load_model_table,
                           load_report_table, most_typical_college, show_chart, usd, usd_md)

meta = load_metadata()
table = load_model_table()
model = load_final_model()
FEATURES = meta["features"]["all"]
STATS = meta["numeric_stats"]
BEST = meta["best_model"]["name"]

# How each number feature is entered
SHARES = ["PPTUG_EF", "UG25ABV", "PCTPELL", "PCTFLOAN", "UGDS_WOMEN", "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP",
          "UGDS_ASIAN", "UGDS_NRA", "PFTFAC", "ADM_RATE_FILLED", "COMPLETION_RATE", "RETENTION_RATE", "FIRST_GEN",
          "MARRIED", "STEM_SHARE", "HEALTH_SHARE", "BUSINESS_SHARE", "TRADES_SHARE", "ARTS_HUM_SHARE",
          "EDU_SOCIAL_SHARE"]                                   # stored as 0 to 1, shown as 0 to 100%
PERCENTS = ["POVERTY_RATE", "UNEMP_RATE", "PCT_BA"]              # already stored as 0 to 100
WHOLE = ["UGDS", "NUMBRANCH"]
SUBJECTS = ["STEM_SHARE", "HEALTH_SHARE", "BUSINESS_SHARE", "TRADES_SHARE", "ARTS_HUM_SHARE", "EDU_SOCIAL_SHARE"]
HIGHDEG_NAMES = {1: "Certificate", 2: "Associate degree", 3: "Bachelor's degree", 4: "Graduate degree"}
YES_NO = {"HBCU": "Historically Black college", "DISTANCEONLY": "Teaches online only",
          "OPEN_ADMISSION": "Accepts every applicant (open admission)", "SINGLE_SEX": "Single-sex college"}
GROUPS = {
    "Size and students": ["UGDS", "PPTUG_EF", "UG25ABV", "UGDS_WOMEN", "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP",
                          "UGDS_ASIAN", "UGDS_NRA", "PCTPELL", "PCTFLOAN"],
    "Prices, spending and staff": ["TUITION", "COST_ATTEND", "NET_PRICE", "TUITFTE", "INEXPFTE", "AVGFACSAL",
                                   "PFTFAC", "STUFACR"],
    "Admission and progress": ["ADM_RATE_FILLED", "COMPLETION_RATE", "RETENTION_RATE"],
    "Student background": ["MD_FAMINC", "FIRST_GEN", "AGE_ENTRY", "MARRIED"],
    "Economy of students' home areas": ["MEDIAN_HH_INC", "POVERTY_RATE", "UNEMP_RATE", "PCT_BA"],
    "What the college teaches": SUBJECTS + ["PROGRAM_HHI"],
}


def typical_values() -> dict:
    """The middle value of every feature, used where a college has no published value."""
    values = {f: STATS[f]["median"] for f in STATS}
    values.update(meta["category_modes"])
    return values


edited = set()                                                  # features the user has changed on this run


def number_field(feature: str, start: float, key: str) -> float:
    """One input for a number feature, in the units people expect. Returns the value in the model's units.

    The box shows a rounded number. While it is left untouched the exact starting value is used, so an
    unchanged real college gets exactly the model's own prediction.
    """
    label, high = column_label(feature), STATS[feature]["max"]
    if feature in SHARES:
        default = float(round(min(max(start, 0.0), 1.0) * 100, 1))
        shown = st.slider(label, 0.0, 100.0, default, step=0.1, format="%.1f%%", key=key)
        value = shown / 100
    elif feature in PERCENTS:
        top = float(np.ceil(high / 10) * 10)
        default = float(round(min(start, top), 1))
        shown = value = st.slider(label, 0.0, top, default, step=0.1, format="%.1f%%", key=key)
    elif feature == "PROGRAM_HHI":
        default = float(round(start, 2))
        shown = value = st.slider(label, 0.0, 1.0, default, step=0.01, key=key,
                                  help="Near 1: the college teaches one subject. Near 0: many subjects.")
    elif feature in WHOLE:
        default = int(round(start))
        shown = st.number_input(label, min_value=1, max_value=int(high * 2), value=default, step=1, key=key)
        value = float(shown)
    else:
        default = float(round(start, 1))
        shown = value = st.number_input(label, min_value=0.0, max_value=float(high * 2), value=default,
                                        step=100.0 if high > 2000 else 0.5,
                                        format="%.0f" if high > 2000 else "%.1f", key=key)
    if shown != default:
        edited.add(feature)
        return float(value)
    return float(start)


st.title("Predict earnings")
st.write(
    "Describe a college and the final model estimates the median earnings of its former students ten years "
    "after they started. Pick a real college to start from, then change any value below. "
    "The prediction updates straight away."
)

# ── Starting point ───────────────────────────────────────────────────────────
names = (table["INSTNM"] + " (" + table["CITY"] + ", " + table["STABBR"] + ")")
typical = most_typical_college()
choices = [typical] + [i for i in names.sort_values().index if i != typical]
start = st.selectbox(
    "Start from", choices, key="start_from",
    format_func=lambda i: f"{names[i]}, the most typical college in the data" if i == typical else names[i])
base = table.loc[start, FEATURES].to_dict()
missing_in_base = [f for f in FEATURES if pd.isna(base[f])]
for feature in missing_in_base:                                 # a gap in the real data starts from the typical value
    base[feature] = typical_values()[feature]
tag = str(start)                                                # new widgets, and so new defaults, per starting point

result_box = st.container()

# ── Inputs ───────────────────────────────────────────────────────────────────
st.subheader("Describe the college")
values = {}
with st.expander("Type of college", expanded=True):
    left, right = st.columns(2)
    for i, feature in enumerate(["CONTROL", "PREDDEG", "REGION", "LOCALE_TYPE", "STABBR"]):
        options = meta["categories"][feature]
        values[feature] = (left if i % 2 == 0 else right).selectbox(
            column_label(feature), options, index=options.index(base[feature]), key=f"{feature}_{tag}")
        if values[feature] != base[feature]:
            edited.add(feature)
    degree_names = list(HIGHDEG_NAMES.values())
    degree = right.selectbox("Highest degree offered", degree_names, index=int(base["HIGHDEG"]) - 1,
                             key=f"HIGHDEG_{tag}")
    values["HIGHDEG"] = degree_names.index(degree) + 1
    if values["HIGHDEG"] != int(base["HIGHDEG"]):
        edited.add("HIGHDEG")
    values["NUMBRANCH"] = int(number_field("NUMBRANCH", base["NUMBRANCH"], f"NUMBRANCH_{tag}"))
    for feature, label in YES_NO.items():
        values[feature] = float(st.checkbox(label, value=bool(base[feature]), key=f"{feature}_{tag}"))
        if values[feature] != float(base[feature]):
            edited.add(feature)
    values["MULTI_CAMPUS"] = float(values["NUMBRANCH"] > 1)     # follows from the number of campuses

for group, features in GROUPS.items():
    with st.expander(group):
        left, right = st.columns(2)
        for i, feature in enumerate(features):
            with (left if i % 2 == 0 else right):
                values[feature] = number_field(feature, base[feature], f"{feature}_{tag}")

# Values that follow from other answers
if values["OPEN_ADMISSION"] == 1:
    values["ADM_RATE_FILLED"] = 1.0                             # an open-admission college admits everyone
values["INSTR_PER_TUITION"] = values["INEXPFTE"] / values["TUITFTE"] if values["TUITFTE"] > 0 else base["INSTR_PER_TUITION"]

# ── Prediction ───────────────────────────────────────────────────────────────
row = pd.DataFrame([values])[FEATURES]
row[["HIGHDEG", "NUMBRANCH"]] = row[["HIGHDEG", "NUMBRANCH"]].astype(int)
prediction = float(model.predict(row)[0])

tests = load_report_table("test_predictions.csv")
ratio = (tests["actual"] / tests["pred_best"])[tests["CONTROL"] == values["CONTROL"]]
low, high = prediction * ratio.quantile(0.10), prediction * ratio.quantile(0.90)
overall = float(table[TARGET].median())
changed = sorted(edited)
real = float(table.loc[start, TARGET])
kind = values["CONTROL"].lower()

with result_box:
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted earnings after 10 years", usd(prediction))
    col2.metric("Real figure for the starting college", usd(real),
                help="The published earnings of the college you started from.")
    col3.metric("Median of all colleges", usd(overall))
    st.write(
        f"Likely range: **{usd_md(low)} to {usd_md(high)}**. In testing, 8 out of 10 {kind} colleges had true "
        "earnings inside a range of this width around the prediction."
    )
    if changed:
        st.caption(f"You have changed {len(changed)} value(s), so the prediction is no longer for the exact "
                   "college you started from.")

    spread = px.histogram(table, x=TARGET, nbins=70, color_discrete_sequence=["#9ec5f4"],
                          labels={TARGET: "Earnings after 10 years (USD)"})
    spread.add_vrect(x0=low, x1=high, fillcolor="#2a78d6", opacity=0.18, line_width=0)
    spread.add_vline(x=prediction, line_color="#2a78d6", line_width=3, annotation_text="prediction")
    if not changed:
        spread.add_vline(x=real, line_color="#eb7a2d", line_dash="dash", annotation_text="real figure",
                         annotation_position="bottom right")
    spread.update_layout(margin=dict(t=30), height=260, yaxis_title="Number of colleges")
    show_chart(spread)
    st.caption(f"The bars show the real earnings of all {len(table):,} colleges. The line is this prediction and the "
               "shaded band is its likely range.")

    # things the user should know about this prediction
    if sum(values[s] for s in SUBJECTS) > 1.001:
        st.warning("The subject shares add up to more than 100%. Lower some of them for a sensible prediction.")
    if values["HIGHDEG"] < {"Certificate": 1, "Associate": 2, "Bachelor": 3}[values["PREDDEG"]]:
        st.warning("The highest degree offered is lower than the main degree awarded. That combination cannot exist.")
    number_inputs = [f for group in GROUPS.values() for f in group]
    unusual = [column_label(f) for f in number_inputs if not STATS[f]["p01"] <= values[f] <= STATS[f]["p99"]]
    if unusual:
        st.info("These values are outside the range of almost all colleges the model learned from, so it treats "
                "them as the nearest usual value: " + ", ".join(unusual) + ".")
    if missing_in_base:
        st.caption("This college has no published value for: " + ", ".join(column_label(f) for f in missing_in_base)
                   + ". The typical value is shown instead.")

st.caption(
    f"Model: {BEST}. The prediction is for a whole college, not for one student, and it shows what colleges with "
    "these features tend to lead to. It does not prove that changing a value would change earnings."
)
