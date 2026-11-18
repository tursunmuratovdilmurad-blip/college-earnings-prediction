"""Prediction page: describe a college and get its predicted graduate earnings from the final model."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import (TARGET, column_label, load_all_preprocessed, load_final_model, load_metadata,
                           load_model_table, load_preprocessor, load_report_table, most_typical_college,
                           section_picker, show_chart, usd, usd_md)

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


MONEY = ["TUITION", "COST_ATTEND", "NET_PRICE", "TUITFTE", "INEXPFTE", "AVGFACSAL", "MD_FAMINC", "MEDIAN_HH_INC"]


def show_value(feature: str, value) -> str:
    """A feature value written the way a person would read it."""
    if isinstance(value, str):
        return value
    if feature in SHARES:
        return f"{value * 100:.1f}%"
    if feature in PERCENTS:
        return f"{value:.1f}%"
    if feature in MONEY:
        return f"${value:,.0f}"
    if feature in YES_NO or feature == "MULTI_CAMPUS":
        return "yes" if value == 1 else "no"
    if feature == "HIGHDEG":
        return HIGHDEG_NAMES[int(value)]
    return f"{value:,.2f}" if abs(value) < 100 else f"{value:,.0f}"


def as_rows(records: list) -> pd.DataFrame:
    """Turn a list of feature dictionaries into the table the model expects."""
    frame = pd.DataFrame(records)[FEATURES]
    frame[["HIGHDEG", "NUMBRANCH"]] = frame[["HIGHDEG", "NUMBRANCH"]].astype(int)
    return frame


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
order = [typical] + [i for i in names.sort_values().index if i != typical]
choices = {}                                                    # text shown in the menu -> row of the college
for i in order:
    text = f"{names[i]}, the most typical college in the data" if i == typical else names[i]
    choices[text if text not in choices else f"{text} [{i}]"] = i   # keeps every entry unique
start = choices[st.selectbox("Start from", list(choices), key="start_from")]
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
row = as_rows([values])
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
    col1.metric("Prediction", usd(prediction),
                help="Predicted median earnings of former students 10 years after they started.")
    col2.metric("Real figure", usd(real), help="The published earnings of the college you started from.")
    col3.metric("All colleges", usd(overall), help="Median earnings of all colleges in the data.")
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

# ── Understand the prediction ────────────────────────────────────────────────
st.subheader("Understand the prediction")
section = section_picker(['What pushes it up or down', 'Similar colleges', 'What if'], key="predict_section")

if section == 'What pushes it up or down':
    st.write(
        "For each feature, the model is asked what it would predict for the same college with the typical "
        "value of that one feature. The difference shows how much this college's own value pushes the "
        "prediction up or down."
    )
    typical = typical_values()
    variants = as_rows([dict(values, **{feature: typical[feature]}) for feature in FEATURES])
    effect = prediction - model.predict(variants)
    pushes = pd.DataFrame({"Feature": [column_label(f) for f in FEATURES],
                           "This college": [show_value(f, values[f]) for f in FEATURES],
                           "Typical college": [show_value(f, typical[f]) for f in FEATURES],
                           "Effect on the prediction (USD)": effect})
    pushes = pushes.reindex(pushes["Effect on the prediction (USD)"].abs().sort_values(ascending=False).index)
    top = pushes.head(12).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["Effect on the prediction (USD)"], y=top["Feature"] + " = " + top["This college"], orientation="h",
        marker_color=np.where(top["Effect on the prediction (USD)"] >= 0, "#2a78d6", "#e34a4a")))
    fig.update_layout(margin=dict(t=20), height=440, xaxis_title="Effect on the predicted earnings (USD)")
    show_chart(fig)
    st.caption("Blue bars raise the prediction and red bars lower it, compared with the typical value. The effects "
               "do not add up exactly to the prediction, because features also work together.")
    with st.expander("All features"):
        st.dataframe(pushes.style.format({"Effect on the prediction (USD)": "{:+,.0f}"}), hide_index=True,
                     width="stretch")

if section == 'Similar colleges':
    st.write("These are the real colleges most like the one described above, after all features are put on the "
             "same scale. If the prediction is sensible, their real earnings should be close to it.")
    count = st.slider("Number of similar colleges", 5, 25, 10, key="similar_count")
    everyone = load_all_preprocessed()
    described = load_preprocessor().transform(row)[0]
    distance = np.sqrt(((everyone - described) ** 2).sum(axis=1))
    if not changed:
        distance = distance.drop(start)                         # do not compare the starting college with itself
    nearest = distance.nsmallest(count)
    similar = table.loc[nearest.index, ["INSTNM", "STABBR", "CONTROL", "PREDDEG", TARGET]].copy()
    similar["Difference"] = nearest.values
    similar_median = float(similar[TARGET].median())

    col1, col2, col3 = st.columns(3)
    col1.metric("This prediction", usd(prediction))
    col2.metric("Similar colleges", usd(similar_median), help="Median real earnings of the colleges listed below.")
    col3.metric("Gap", usd(abs(prediction - similar_median)))

    chart = similar.sort_values(TARGET)
    fig = px.bar(chart, x=TARGET, y="INSTNM", orientation="h", color_discrete_sequence=["#9ec5f4"],
                 labels={TARGET: "Real earnings after 10 years (USD)", "INSTNM": ""})
    fig.add_vline(x=prediction, line_color="#2a78d6", line_width=3, annotation_text="prediction")
    fig.update_layout(margin=dict(t=30), height=max(320, 28 * count + 100))
    show_chart(fig)
    similar = similar.rename(columns={"INSTNM": "College", "STABBR": "State", "CONTROL": "Ownership",
                                      "PREDDEG": "Main degree", TARGET: "Real earnings (USD)"})
    st.dataframe(similar.style.format({"Real earnings (USD)": "{:,.0f}", "Difference": "{:.2f}"}),
                 hide_index=True, width="stretch")
    st.caption("Difference is the distance between two colleges across all features. Smaller means more alike.")

if section == 'What if':
    st.write("Change one feature across its usual range while everything else stays as described above, and "
             "see how the prediction moves.")
    number_inputs = [f for group in GROUPS.values() for f in group]
    if values["OPEN_ADMISSION"] == 1:
        # an open-admission college admits everyone, so its admission rate cannot be varied on its own
        number_inputs.remove("ADM_RATE_FILLED")
    feature = st.selectbox("Feature to change", number_inputs, index=number_inputs.index("MD_FAMINC"),
                           format_func=lambda f: f"{f}: {column_label(f)}", key="whatif_feature")
    grid = np.linspace(STATS[feature]["p01"], STATS[feature]["p99"], 40)
    records = []
    for point in grid:
        record = dict(values, **{feature: float(point)})
        if feature in ("INEXPFTE", "TUITFTE") and record["TUITFTE"] > 0:
            record["INSTR_PER_TUITION"] = record["INEXPFTE"] / record["TUITFTE"]
        records.append(record)
    curve = model.predict(as_rows(records))
    scale = 100 if feature in SHARES else 1
    unit = " (%)" if feature in SHARES else ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid * scale, y=curve, mode="lines", line=dict(color="#2a78d6", width=3),
                             name="Prediction"))
    fig.add_trace(go.Scatter(x=[values[feature] * scale], y=[prediction], mode="markers", name="This college",
                             marker=dict(color="#eb7a2d", size=13)))
    fig.update_layout(margin=dict(t=20), height=400, xaxis_title=column_label(feature) + unit,
                      yaxis_title="Predicted earnings (USD)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    show_chart(fig)
    st.caption(
        f"Across the usual range of this feature the prediction goes from {usd(curve.min())} to {usd(curve.max())}. "
        "This shows what the model expects for colleges with different values. It is not proof that changing "
        "the value at one college would change what its students earn."
    )

st.caption(
    f"Model: {BEST}. The prediction is for a whole college, not for one student, and it shows what colleges with "
    "these features tend to lead to. It does not prove that changing a value would change earnings."
)
