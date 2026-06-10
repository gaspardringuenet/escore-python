import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data


class EscoreClassifier(ClassifierMixin, BaseEstimator):
    """Implement"""

    def __init__(self, demo_param="demo"):
        self.demo_param = demo_param

    def fit(self, X, y):

        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # Return the classifier
        return self

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False)
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(EscoreClassifier())
