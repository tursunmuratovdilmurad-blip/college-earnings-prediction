"""Custom scikit-learn transformers shared by the notebook and the Streamlit app.

They live in an importable module (rather than in a notebook cell) so that the
pipelines serialised with ``joblib`` can be loaded again by any other process.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip every column to quantile bounds learned from the training data.

    Outliers are capped instead of deleted, so no rows are lost and the bounds
    are estimated on the training fold only (no leakage). Missing values are
    ignored when the bounds are learned and are passed through untouched.

    Parameters
    ----------
    lower, upper : float
        Lower and upper quantiles (between 0 and 1) used as clipping bounds.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
        self.feature_names_in_ = getattr(X, "columns", None)
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self.feature_names_in_ is not None:
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
