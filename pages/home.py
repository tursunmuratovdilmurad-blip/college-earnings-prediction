"""Home page: what the project is about and the headline numbers."""
import streamlit as st

from src.app_utils import load_metadata, usd, usd_md

meta = load_metadata()
best_name = meta["best_model"]["name"]
best_test = meta["models"][best_name]["test"]
baseline_test = meta["models"]["Baseline (median)"]["test"]

st.title("College Earnings Prediction")
st.write(
    "How much do the former students of a U.S. college earn ten years after they started? "
    "The government publishes this number, but it arrives ten years late and is missing for many colleges. "
    "This project predicts it from facts that are known much earlier, such as the type of college, "
    "its prices, its students and what it teaches."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Colleges in the model", f"{meta['dataset']['rows_modelled']:,}")
col2.metric("Features used", len(meta["features"]["all"]))
col3.metric("Final model", best_name)
col4.metric("Typical error", usd(best_test["MAE"]),
            help="Mean absolute error on the held-out test set: colleges the model never saw in training.")

st.subheader("In short")
st.write(
    f"The final model explains {best_test['R2']:.0%} of the differences in earnings between colleges it has "
    f"never seen. Its error is about one third of a simple baseline that predicts the same value for every "
    f"college ({usd_md(best_test['RMSE'])} against {usd_md(baseline_test['RMSE'])} RMSE)."
)

st.subheader("The data")
st.write(
    f"The data is the U.S. Department of Education College Scorecard "
    f"([download page]({meta['dataset']['url']})). The raw file has {meta['dataset']['rows_raw']:,} colleges and "
    f"{meta['dataset']['columns_raw']:,} columns. After cleaning, {meta['dataset']['rows_modelled']:,} colleges remain, "
    f"split into {meta['split']['train']:,} for training, {meta['split']['validation']:,} for validation and "
    f"{meta['split']['test']:,} for the final test."
)

st.info(
    "Each row is a whole college, not a person. The model shows links between types of colleges and earnings. "
    "It does not prove that a college causes those earnings.",
)
