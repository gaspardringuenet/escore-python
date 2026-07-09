import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore


class DummyCluster(ClusterMixin, BaseEstimator):
    def __init__(self, predict_value=0):
        self.predict_value = predict_value

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

    def fit(self, X, y=None):

        # Input validation
        X = validate_data(self, X)

        self.predict_value_ = self.predict_value
        return self

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False)

        return np.ones(X.shape[0])
