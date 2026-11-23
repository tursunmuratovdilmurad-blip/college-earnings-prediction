"""Every page of the Streamlit app opens without an error, and the app predicts what the model predicts."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.app_utils import load_final_model, load_metadata, load_model_table, most_typical_college, usd

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "app_pages").glob("*.py"))


def open_page(page):
    return AppTest.from_file(str(ROOT / page), default_timeout=300).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_opens_without_an_error(page):
    at = open_page(page)
    assert not at.exception, [e.value for e in at.exception]


def test_prediction_page_shows_what_the_model_predicts():
    """The page starts from the most typical college and must show exactly the saved model's prediction."""
    at = open_page("app_pages/predict.py")
    shown = next(m.value for m in at.metric if m.label == "Prediction")

    college = load_model_table().loc[[most_typical_college()], load_metadata()["features"]["all"]]
    assert shown == usd(load_final_model().predict(college)[0])
