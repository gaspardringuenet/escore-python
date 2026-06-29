import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore


class SvDifferenceExtractor(TransformerMixin, BaseEstimator):
    """
    Converts acoustic data by subtracting a reference channel to the others.
    X shape: (n_samples, n_channels) -> (n_samples, n_channels-1)

    Parameters
    ----------
    ref_channel_idx : int, optional
        The index of the reference column to subtract to the others, by default None.

    """

    def __init__(self, ref_channel_idx: int | None = None):
        self.ref_channel_idx = ref_channel_idx

    def fit(self, X, y=None):

        X = validate_data(self, X, y)

        # Basic: use 0 as default
        if self.ref_channel_idx is None:
            self.ref_channel_idx_ = 0
        else:
            self.ref_channel_idx_ = self.ref_channel_idx
        return self

    def transform(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False).copy()

        # Compute MVBS differences as features
        selector = [i for i in range(X.shape[1]) if i != self.ref_channel_idx_]
        X_ref = X[:, self.ref_channel_idx_]
        return X[:, selector] - X_ref[:, np.newaxis]

    def get_params(self, deep=True):
        return {"ref_channel_idx": self.ref_channel_idx}

    def set_params(self, **params):
        if "ref_channel_idx" in params:
            self.ref_channel_idx = params["ref_channel_idx"]
        return self


class PlusSvExtractor(TransformerMixin, BaseEstimator):
    """Sum channels to compute ΣSv (or ΣMVBS)."""

    def fit(self, X, y=None):
        X = validate_data(self, X, y)
        return self

    def transform(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False).copy()

        # Compute ΣMVBS
        return np.sum(X, axis=-1, keepdims=True)  # shape (n_samples, 1)


class MeanSvExtractor(TransformerMixin, BaseEstimator):
    """Mean of Sv accros channels."""

    def fit(self, X, y=None):
        X = validate_data(self, X, y)
        return self

    def transform(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False).copy()

        # Compute ΣMVBS
        return np.mean(X, axis=-1, keepdims=True)  # shape (n_samples, 1)
