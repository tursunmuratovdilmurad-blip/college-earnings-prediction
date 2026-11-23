"""Tests for the Winsorizer, the custom transformer that caps outliers."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.transformers import Winsorizer


@pytest.fixture
def train():
    """Two columns with the values 0 to 100, so the 10% and 90% quantiles are exactly 10 and 90."""
    values = np.arange(101, dtype=float)
    return pd.DataFrame({"a": values, "b": values * 2})


def test_learns_the_quantiles_of_each_column(train):
    w = Winsorizer(lower=0.1, upper=0.9).fit(train)
    np.testing.assert_allclose(w.lower_bounds_, [10, 20])
    np.testing.assert_allclose(w.upper_bounds_, [90, 180])


def test_caps_values_outside_the_bounds_and_keeps_the_rest(train):
    w = Winsorizer(lower=0.1, upper=0.9).fit(train)
    new = pd.DataFrame({"a": [-500.0, 50.0, 500.0], "b": [0.0, 100.0, 1000.0]})
    np.testing.assert_allclose(w.transform(new), [[10, 20], [50, 100], [90, 180]])


def test_uses_the_training_bounds_on_new_data(train):
    """The bounds come from fit only, so extreme new data cannot move them (no leakage)."""
    w = Winsorizer(lower=0.1, upper=0.9).fit(train)
    w.transform(pd.DataFrame({"a": [1e9, -1e9], "b": [1e9, -1e9]}))
    np.testing.assert_allclose(w.lower_bounds_, [10, 20])
    np.testing.assert_allclose(w.upper_bounds_, [90, 180])


def test_keeps_the_number_of_rows():
    """Outliers are capped, never deleted."""
    data = np.random.default_rng(0).standard_cauchy((200, 3))
    w = Winsorizer().fit(data)
    out = w.transform(data)
    assert out.shape == data.shape
    assert (out >= w.lower_bounds_).all() and (out <= w.upper_bounds_).all()


def test_ignores_missing_values_when_fitting_and_keeps_them():
    data = pd.DataFrame({"a": [np.nan, 0.0, 1.0, 2.0, 3.0, 4.0, np.nan]})
    w = Winsorizer(lower=0.25, upper=0.75).fit(data)
    np.testing.assert_allclose(w.lower_bounds_, [1])
    np.testing.assert_allclose(w.upper_bounds_, [3])
    out = w.transform(data)
    assert np.isnan(out[0, 0]) and np.isnan(out[-1, 0])
    np.testing.assert_allclose(out[1:-1, 0], [1, 1, 2, 3, 3])


def test_lower_0_and_upper_1_change_nothing(train):
    w = Winsorizer(lower=0.0, upper=1.0).fit(train)
    np.testing.assert_array_equal(w.transform(train), train.to_numpy())


@pytest.mark.parametrize("lower, upper", [(0.9, 0.1), (0.5, 0.5), (-0.1, 0.9), (0.1, 1.5)])
def test_rejects_wrong_quantiles(train, lower, upper):
    with pytest.raises(ValueError):
        Winsorizer(lower=lower, upper=upper).fit(train)


def test_must_be_fitted_before_transform(train):
    with pytest.raises(NotFittedError):
        Winsorizer().transform(train)


def test_keeps_the_column_names(train):
    assert list(Winsorizer().fit(train).get_feature_names_out()) == ["a", "b"]
    assert list(Winsorizer().fit(train.to_numpy()).get_feature_names_out()) == ["x0", "x1"]


def test_works_inside_a_scikit_learn_pipeline(train):
    pipeline = make_pipeline(Winsorizer(lower=0.1, upper=0.9), StandardScaler())
    copy = clone(pipeline)
    assert copy.get_params()["winsorizer__upper"] == 0.9
    out = copy.fit_transform(train)
    assert out.shape == train.shape
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-12)
