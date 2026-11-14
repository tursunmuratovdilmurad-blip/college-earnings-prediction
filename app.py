"""College Earnings Prediction - Streamlit app entry point.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="College Earnings Prediction", layout="wide")

pages = [
    st.Page("pages/home.py", title="Home", default=True),
]

st.navigation(pages).run()
