"""Tests for the saved models: they load, and they give the same results as the notebook."""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
META = json.loads((ROOT / "models" / "metadata.json").read_text())
FEATURES = META["features"]["all"]
TARGET = META["target"]["name"]


def load(file):
    path = ROOT / "models" / file
    if not path.exists():
        pytest.skip(f"{file} is not in the repository (it is too large); the notebook creates it")
    return joblib.load(path)


def predict(model, X):
    with warnings.catch_warnings():         # ElasticNet can trigger harmless matmul warnings on some Macs
        warnings.simplefilter("ignore", RuntimeWarning)
        return model.predict(X)


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(ROOT / "data" / "processed" / "model_table.csv")


@pytest.fixture(scope="module")
def test_set(table):
    rows = table[table["SPLIT"] == "test"]
    return rows[FEATURES], rows[TARGET]


@pytest.fixture(scope="module")
def final_model():
    return load(META["best_model"]["file"])


def test_split_sizes_match_the_metadata(table):
    assert table["SPLIT"].value_counts().to_dict() == META["split"]


def test_final_model_expects_the_features_in_the_metadata(final_model):
    assert list(final_model.feature_names_in_) == FEATURES


@pytest.mark.parametrize("name", list(META["models"]))
def test_each_model_reproduces_its_saved_test_scores(name, test_set):
    X, y = test_set
    saved = META["models"][name]
    pred = predict(load(saved["file"]), X)
    assert np.sqrt(mean_squared_error(y, pred)) == pytest.approx(saved["test"]["RMSE"], rel=1e-9)
    assert mean_absolute_error(y, pred) == pytest.approx(saved["test"]["MAE"], rel=1e-9)
    assert r2_score(y, pred) == pytest.approx(saved["test"]["R2"], rel=1e-9)


def test_final_model_reproduces_its_saved_test_scores(final_model, test_set):
    X, y = test_set
    pred = final_model.predict(X)
    saved = META["best_model"]["test"]
    assert np.sqrt(mean_squared_error(y, pred)) == pytest.approx(saved["RMSE"], rel=1e-9)
    assert r2_score(y, pred) == pytest.approx(saved["R2"], rel=1e-9)


def test_predictions_match_the_saved_prediction_table(table):
    saved = pd.read_csv(ROOT / "reports" / "test_predictions.csv", index_col=0)
    X = table.loc[saved.index, FEATURES]
    assert (table.loc[saved.index, "INSTNM"] == saved["INSTNM"]).all()
    pred = predict(load(META["models"]["SVR (RBF)"]["file"]), X)
    np.testing.assert_allclose(pred, saved["pred_SVR (RBF)"], rtol=0, atol=0.01)


def test_predictions_are_sensible(final_model, test_set):
    X, y = test_set
    pred = final_model.predict(X)
    assert np.isfinite(pred).all()
    assert pred.min() > 0.5 * y.min() and pred.max() < 2 * y.max()


def test_a_college_with_missing_numbers_still_gets_a_prediction(final_model, test_set):
    """The pipeline fills in missing values itself, so a college with gaps can still be scored."""
    X, _ = test_set
    row = X.iloc[[0]].copy()
    row[META["features"]["numeric"]] = np.nan
    pred = final_model.predict(row)
    assert np.isfinite(pred).all() and pred[0] > 0
