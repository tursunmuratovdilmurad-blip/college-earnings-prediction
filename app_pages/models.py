"""Model comparison page: the models, their scores, how they were tuned and where they go wrong."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import OWNERSHIP_COLORS, column_label, load_metadata, load_report_table, show_chart, usd

meta = load_metadata()
BEST = meta["best_model"]["name"]
BASELINE = "Baseline (median)"
MODELS = [m for m in meta["models"] if m != BASELINE]
COLORS = {"ElasticNet": "#8d8d8d", "Random Forest": "#1baf7a", "Hist Gradient Boosting": "#eb7a2d",
          "SVR (RBF)": "#2a78d6", BASELINE: "#c9c9c9"}
SLUG = {"ElasticNet": "elasticnet", "Random Forest": "random_forest",
        "Hist Gradient Boosting": "hist_gradient_boosting", "SVR (RBF)": "svr_rbf"}

ABOUT = {
    BASELINE: ("Predicts the same middle value for every college", "Nothing", "Not a real model. It is the score to beat"),
    "ElasticNet": ("Gives each feature a weight and adds them up. A penalty keeps the weights small",
                   "Each feature pushes earnings up or down by a fixed amount", "Cannot follow curves"),
    "Random Forest": ("Builds 500 decision trees on random samples of the data and averages them",
                      "Colleges can be grouped by yes/no questions", "Predicts in steps and can memorise the training data"),
    "Hist Gradient Boosting": ("Builds small trees one after another. Each tree fixes the mistakes of the ones before",
                               "Many small corrections reach the truth", "Fits the training data very closely"),
    "SVR (RBF)": ("Fits a smooth curve through the data and ignores small errors",
                  "Colleges that look alike earn alike", "Needs scaled data and is harder to explain"),
}
METRICS = {
    "RMSE": "Root mean squared error in dollars. Large misses count extra. Lower is better.",
    "MAE": "Mean absolute error in dollars: how far off the model is on average. Lower is better.",
    "MedAE": "Median absolute error in dollars: how far off the model is for a typical college. Lower is better.",
    "MAPE_%": "Mean absolute percentage error: how far off the model is in percent. Lower is better.",
    "R2": "Share of the differences between colleges that the model explains. Higher is better, 1 is perfect.",
}
# hyperparameter -> (plain meaning, range that was searched, use a log axis)
PARAMS = {
    "ElasticNet": {"alpha": ("How strongly large weights are punished", "0.00001 to 1", True),
                   "l1_ratio": ("Mix of the two penalties. 1 is lasso only, which can switch features off",
                                "0.05 to 1", False)},
    "Random Forest": {"n_estimators": ("Number of trees", "300 or 500", False),
                      "max_depth": ("How deep each tree may grow", "8, 12, 16, 24 or no limit", False),
                      "min_samples_leaf": ("Smallest group of colleges a tree may end on", "1, 2, 4 or 8", False),
                      "max_features": ("Share of the features tried at each split", "0.2 to 1", False)},
    "Hist Gradient Boosting": {"learning_rate": ("How big each correction step is", "0.01 to 0.3", True),
                               "max_iter": ("Number of trees", "150 to 800", False),
                               "max_leaf_nodes": ("Leaves per tree", "8, 15, 31 or 63", False),
                               "min_samples_leaf": ("Smallest group of colleges a tree may end on", "5, 10, 20 or 40", False),
                               "l2_regularization": ("Penalty on extreme predictions", "0.001 to 10", True),
                               "max_features": ("Share of the features tried at each split", "0.5, 0.8 or 1", False)},
    "SVR (RBF)": {"C": ("How hard the model chases every single point", "0.1 to 100", True),
                  "gamma": ("How local the curve is. Small values give a smoother curve", "0.0001 to 0.1", True),
                  "epsilon": ("Size of error that is ignored, on the log earnings scale", "0.01 to 0.2", False)},
}

scores = load_report_table("metrics_all_splits.csv", header=[0, 1], index_col=0)
cv_summary = load_report_table("cv_summary.csv", index_col=0)

st.title("Model comparison")
st.write(
    "Four models that work in different ways were trained on the same data and compared with a simple baseline. "
    f"The model chosen in the end is **{BEST}**."
)

tab_models, tab_scores, tab_tuning, tab_pred, tab_groups, tab_drivers = st.tabs(
    ["The models", "Scores", "Tuning", "Predictions", "Errors by group", "What drives the predictions"])

# ── Tab 1: what the models are ───────────────────────────────────────────────
with tab_models:
    st.write("The models were picked because they make different guesses about what the true pattern looks like. "
             "If they had all been similar, comparing them would say little about the data.")
    about = pd.DataFrame([{"Model": name, "How it works": how, "What it assumes": assumes, "Weak spot": weak}
                          for name, (how, assumes, weak) in ABOUT.items()])
    st.dataframe(about, hide_index=True, width="stretch")
    st.write(
        "Every model predicts the logarithm of earnings, and its answer is turned back into dollars before it is "
        "scored. On raw dollars a model would spend most of its effort on a few very high earners."
    )

# ── Tab 2: scores ────────────────────────────────────────────────────────────
with tab_scores:
    test = scores["test"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best RMSE on the test set", usd(test.loc[BEST, "RMSE"]))
    col2.metric("Typical error (MAE)", usd(test.loc[BEST, "MAE"]))
    col3.metric("Error in percent (MAPE)", f"{test.loc[BEST, 'MAPE_%']:.1f}%")
    col4.metric("Differences explained (R²)", f"{test.loc[BEST, 'R2']:.0%}")

    left, right = st.columns([1, 1])
    split_names = {"Test": "test", "Validation": "validation", "Training": "train"}
    split = split_names[left.radio("Data split", list(split_names), horizontal=True, key="score_split")]
    metric = right.selectbox("Score for the chart", list(METRICS), key="score_metric")
    st.caption(METRICS[metric])

    table = scores[split].sort_values("RMSE")
    money = {"RMSE": "${:,.0f}", "MAE": "${:,.0f}", "MedAE": "${:,.0f}", "MAPE_%": "{:.1f}%", "R2": "{:.3f}"}
    best_mark = "font-weight: bold; color: #1baf7a"           # readable in both the light and the dark theme
    styled = (table.style.format(money)
              .highlight_min(subset=["RMSE", "MAE", "MedAE", "MAPE_%"], props=best_mark)
              .highlight_max(subset=["R2"], props=best_mark))
    st.dataframe(styled, width="stretch")
    st.caption("The best value in each column is shown in green.")

    ordered = table[metric].sort_values(ascending=(metric != "R2"))
    fig = px.bar(x=ordered.index, y=ordered.values, color=ordered.index, color_discrete_map=COLORS,
                 labels={"x": "", "y": metric})
    fig.update_layout(showlegend=False, margin=dict(t=20), height=360)
    show_chart(fig)

    st.subheader("Did the models learn the pattern or memorise the examples?")
    st.write("A model that memorises has a very low error on the data it trained on and a much higher error on "
             "new data. The closer the bars of one model are to each other, the more its score can be trusted.")
    by_split = pd.DataFrame({"Training": scores[("train", "RMSE")], "Cross-validation": cv_summary["cv_RMSE"],
                             "Validation": scores[("validation", "RMSE")], "Test": scores[("test", "RMSE")]}).loc[MODELS]
    long = by_split.reset_index(names="Model").melt(id_vars="Model", var_name="Data", value_name="RMSE (USD)")
    fig = px.bar(long, x="Model", y="RMSE (USD)", color="Data", barmode="group",
                 color_discrete_sequence=["#eb7a2d", "#9ec5f4", "#2a78d6", "#104281"])
    fig.update_layout(margin=dict(t=20), height=380, xaxis_title="", legend_title="")
    show_chart(fig)

    st.subheader("How sure are these scores?")
    st.write("The test set has only 596 colleges, so each score would shift a little with a different sample. "
             "The lines show the range each model's RMSE stayed in when the test set was resampled 2,000 times.")
    ci = load_report_table("metrics_test_bootstrap_ci.csv", index_col=0).loc[lambda d: d.index != BASELINE]
    fig = go.Figure(go.Scatter(
        x=ci["RMSE"], y=ci.index, mode="markers", marker=dict(size=11, color=[COLORS[m] for m in ci.index]),
        error_x=dict(type="data", symmetric=False, array=ci["RMSE_hi"] - ci["RMSE"],
                     arrayminus=ci["RMSE"] - ci["RMSE_lo"], thickness=3, color="#8d8d8d")))
    fig.update_layout(margin=dict(t=20), height=300, xaxis_title="RMSE on the test set (USD), with 95% range")
    fig.update_yaxes(autorange="reversed")
    show_chart(fig)
    ranked = test.loc[MODELS, "RMSE"].sort_values()
    first, second = ranked.index[0], ranked.index[1]
    st.info(
        f"The ranges of {first} and {second} overlap almost fully, so the two cannot be told apart on this test "
        f"set. {BEST} was chosen because it had the lowest error on the validation set, which was the rule set "
        "before the test set was opened, and because its training and test errors are closest together."
    )

# ── Tab 3: tuning ────────────────────────────────────────────────────────────
with tab_tuning:
    st.write(
        "Each model has settings that are not learned from the data and must be chosen, called hyperparameters. "
        "They were tested with 5-fold cross-validation on the training set: the data is cut into five parts, the "
        "model trains on four and is checked on the fifth, and every part gets a turn as the check."
    )
    tuned = st.selectbox("Model", MODELS, index=MODELS.index(BEST), key="tuning_model")
    summary = cv_summary.loc[tuned]
    best_params = meta["models"][tuned]["best_params"]
    if tuned == "ElasticNet":
        st.write("Search method: **grid search**. This model has only two settings, so every combination was tried.")
    else:
        st.write("Search method: **random search**. This model has several settings, and a full grid would need "
                 "thousands of runs. Random combinations find good values with far fewer tries.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Settings tried", int(summary["candidates"]))
    col2.metric("Models trained", int(summary["fits"]), help="Settings tried times five folds.")
    col3.metric("Best cross-validation RMSE", usd(summary["cv_RMSE"]),
                help=f"Give or take {usd(summary['cv_RMSE_std'])} between the five folds.")

    st.subheader("Settings and the best values found")
    rows = []
    for name, (meaning, searched, _) in PARAMS[tuned].items():
        value = best_params.get(name)
        rows.append({"Setting": name, "What it controls": meaning, "Range searched": searched,
                     "Best value": "no limit" if value is None else str(value)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("How much does each setting matter?")
    st.write("Each dot is one combination of settings that was tried. A flat cloud means the setting hardly "
             "matters. A clear slope shows where the good values are.")
    results = load_report_table(f"cv_results_{SLUG[tuned]}.csv")
    setting = st.selectbox("Setting", list(PARAMS[tuned]), key="tuning_setting",
                           format_func=lambda p, names=PARAMS[tuned]: f"{p}: {names[p][0]}")
    column = f"param_regressor__model__{setting}"
    use_log = PARAMS[tuned][setting][2]
    values = results[column] if use_log else results[column].fillna("no limit").astype(str)
    points = pd.concat([
        pd.DataFrame({setting: values, "RMSE (USD)": -results["mean_test_rmse"], "Checked on": "Held-out folds"}),
        pd.DataFrame({setting: values, "RMSE (USD)": -results["mean_train_rmse"], "Checked on": "Training folds"}),
    ])
    if not use_log:
        order = sorted(points[setting].unique(), key=lambda v: (v == "no limit", float(v) if v.replace(".", "", 1).isdigit() else 1e9, v))
    fig = px.scatter(points, x=setting, y="RMSE (USD)", color="Checked on", log_x=use_log,
                     color_discrete_map={"Held-out folds": "#2a78d6", "Training folds": "#eb7a2d"},
                     category_orders=None if use_log else {setting: order})
    fig.update_traces(marker=dict(size=9, opacity=0.75))
    fig.update_layout(margin=dict(t=20), height=400, legend_title="")
    if not use_log:
        fig.update_xaxes(type="category")
    show_chart(fig)
    st.caption("Blue dots are the errors on data the model did not train on. Orange dots are the errors on its own "
               "training data. A large gap between them means the model is memorising.")

# ── Shared data for the evaluation tabs ──────────────────────────────────────
predictions = load_report_table("test_predictions.csv")


def results_for(model: str) -> pd.DataFrame:
    """Actual and predicted earnings of the test colleges for one model, with the error of each college."""
    out = predictions[["INSTNM", "STABBR", "CONTROL", "PREDDEG", "actual"]].copy()
    out["predicted"] = predictions[f"pred_{model}"]
    out["error"] = out["actual"] - out["predicted"]
    out["error %"] = out["error"].abs() / out["actual"] * 100
    return out


# ── Tab 4: predicted against actual ──────────────────────────────────────────
with tab_pred:
    st.write(f"These charts use the {len(predictions):,} test colleges, which no model saw during training.")
    shown = st.selectbox("Model", MODELS, index=MODELS.index(BEST), key="pred_model")
    res = results_for(shown)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Within 10% of the true figure", f"{(res['error %'] <= 10).mean():.0%}")
    col2.metric("Within 20%", f"{(res['error %'] <= 20).mean():.0%}")
    col3.metric("Average error", usd(res["error"].abs().mean()))
    col4.metric("Largest miss", usd(res["error"].abs().max()))

    st.subheader("Predicted against actual earnings")
    st.write("Each dot is one college. A perfect model would put every dot on the dashed line. Dots below the line "
             "are colleges the model predicted too low.")
    low, high = res["actual"].min() * 0.9, res["actual"].max() * 1.03
    fig = px.scatter(res, x="actual", y="predicted", color="CONTROL", hover_name="INSTNM", opacity=0.6,
                     color_discrete_map=OWNERSHIP_COLORS, category_orders={"CONTROL": list(OWNERSHIP_COLORS)},
                     labels={"actual": "Actual earnings (USD)", "predicted": "Predicted earnings (USD)",
                             "CONTROL": "Ownership"})
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect prediction",
                             line=dict(color="grey", dash="dash")))
    fig.update_layout(margin=dict(t=20), height=480, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    show_chart(fig)

    st.subheader("Errors")
    left, right = st.columns(2)
    with left:
        st.write("Error of each college against its prediction. A good model shows a flat cloud around zero.")
        fig = px.scatter(res, x="predicted", y="error", hover_name="INSTNM", opacity=0.55,
                         color_discrete_sequence=[COLORS[shown]],
                         labels={"predicted": "Predicted earnings (USD)", "error": "Actual minus predicted (USD)"})
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(margin=dict(t=20), height=380)
        show_chart(fig)
    with right:
        st.write("How the errors are spread. Most should be small, and the two sides should be about equal.")
        fig = px.histogram(res, x="error", nbins=60, color_discrete_sequence=[COLORS[shown]],
                           labels={"error": "Actual minus predicted (USD)"})
        fig.add_vline(x=0, line_dash="dash", line_color="grey")
        fig.update_layout(margin=dict(t=20), height=380, yaxis_title="Number of colleges")
        show_chart(fig)

# ── Tab 5: errors by group ───────────────────────────────────────────────────
with tab_groups:
    st.write("One overall score can hide groups where a model does badly. This tab splits the test errors.")
    left, right = st.columns(2)
    picked = left.selectbox("Model", MODELS, index=MODELS.index(BEST), key="group_model")
    group_names = {"Ownership": "CONTROL", "Main degree awarded": "PREDDEG"}
    group_label = right.selectbox("Split by", list(group_names), key="group_split")
    res = results_for(picked)
    group_col = group_names[group_label]

    grouped = res.groupby(group_col).agg(Colleges=("actual", "size"), MAE=("error", lambda e: e.abs().mean()),
                                         MAPE=("error %", "mean"), Bias=("error", "mean")).sort_values("MAPE")
    fig = px.bar(grouped.reset_index(), x=group_col, y="MAPE", color=group_col,
                 color_discrete_map=OWNERSHIP_COLORS if group_col == "CONTROL" else None,
                 labels={group_col: "", "MAPE": "Average error (%)"})
    fig.update_layout(showlegend=False, margin=dict(t=20), height=340)
    show_chart(fig)
    st.dataframe(grouped.rename(columns={"MAE": "Average error (USD)", "MAPE": "Average error (%)",
                                         "Bias": "Average of actual minus predicted (USD)"})
                 .style.format({"Average error (USD)": "{:,.0f}", "Average error (%)": "{:.1f}",
                                "Average of actual minus predicted (USD)": "{:+,.0f}"}), width="stretch")
    st.caption("A positive value in the last column means the model predicts too low for that group on average.")

    st.subheader("Low earners and high earners")
    st.write("The test colleges are sorted by their true earnings and cut into ten equal groups. The bars show "
             "whether the model predicts too low or too high in each group.")
    decile = pd.qcut(res["actual"], 10, labels=[f"{i}" for i in range(1, 11)])
    bias = res.groupby(decile, observed=True)["error"].mean()
    fig = px.bar(x=bias.index.astype(str), y=bias.values, color=np.where(bias.values >= 0, "Predicted too low",
                                                                         "Predicted too high"),
                 color_discrete_map={"Predicted too low": "#2a78d6", "Predicted too high": "#e34a4a"},
                 labels={"x": "Group (1 = lowest earnings, 10 = highest)", "y": "Average of actual minus predicted (USD)",
                         "color": ""})
    fig.update_layout(margin=dict(t=20), height=360)
    show_chart(fig)
    st.caption("Every model predicts too low for the highest earners. There are few such colleges to learn from, "
               "and what makes them special is not in the data.")

    st.subheader("Largest errors")
    worst = res.reindex(res["error"].abs().sort_values(ascending=False).index).head(10)
    worst = worst.rename(columns={"INSTNM": "College", "STABBR": "State", "CONTROL": "Ownership",
                                  "PREDDEG": "Main degree", "actual": "Actual (USD)", "predicted": "Predicted (USD)",
                                  "error": "Error (USD)"}).drop(columns="error %")
    st.dataframe(worst.style.format({"Actual (USD)": "{:,.0f}", "Predicted (USD)": "{:,.0f}",
                                     "Error (USD)": "{:+,.0f}"}), hide_index=True, width="stretch")

# ── Tab 6: what drives the predictions ───────────────────────────────────────
with tab_drivers:
    st.write(
        f"To see what {BEST} relies on, one feature at a time was shuffled and the model was scored again. "
        "If the error goes up a lot, the model depends on that feature. This was measured on the validation set, "
        "not on the training data."
    )
    importance = load_report_table("permutation_importance.csv")
    count = st.slider("Number of features to show", 5, len(importance), 15, key="driver_count")
    top = importance.head(count).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["rmse_increase"], y=[column_label(f) or f for f in top["feature"]], orientation="h",
                           marker_color="#2a78d6", error_x=dict(type="data", array=top["std"], color="#8d8d8d")))
    fig.update_layout(margin=dict(t=20), height=max(320, 26 * count + 100),
                      xaxis_title="Increase in RMSE when the feature is shuffled (USD)")
    show_chart(fig)
    first, second = importance.iloc[0], importance.iloc[1]
    st.info(
        f"{column_label(first['feature'])} is far ahead of everything else: shuffling it adds "
        f"{first['rmse_increase'] / second['rmse_increase']:.1f} times as much error as the next feature. The model "
        "predicts earnings mostly from who enrols at a college, and only partly from what the college does. "
        "A high prediction is not proof of good teaching."
    )
    st.caption("When two features carry similar information they share the credit, so a low bar does not prove a "
               "feature is useless.")

