"""College Earnings Prediction - Streamlit app entry point.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="College Earnings Prediction", layout="wide")

pages = [
    st.Page("app_pages/home.py", title="Home", default=True),
    st.Page("app_pages/data_overview.py", title="Data overview"),
    st.Page("app_pages/preprocessing.py", title="Preprocessing"),
]

st.navigation(pages).run()
