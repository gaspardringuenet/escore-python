from operator import ge, gt, le, lt
from typing import Callable, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore

ComparisonOperator = Callable[[float, float], bool]


class FixedSvThresholds(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        column_idx: List[int] = [0],
        thresholds: List[Tuple[ComparisonOperator, float]] = [(ge, -40)],
    ):
        """Classify data based on fixed threshold values.

        Parameters
        ----------
        column_idx : List[int], optional
            Indices of the column to apply thresholds to.
        thresholds : List[Tuple[ComparisonOperator, float]], optional
            List of (comparison_operator, value) tuples.
            Use operator.ge (>=), operator.le (<=), operator.gt (>), operator.lt (<)
            from the built-in operator module.
            Example [(ge, -40), (le, 0)]
        """

        if not len(column_idx) == len(thresholds):
            raise ValueError("column_idx and thresholds arguments must have same length.")

        # validate operators
        valid_operators = {ge, le, gt, lt}
        for op, _ in thresholds:
            if op not in valid_operators:
                raise ValueError("Operators must be from the operator module: ge, le, gt, lt")

        self.column_idx = column_idx
        self.thresholds = thresholds

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

    def fit(self, X, y=None):

        # Input validation
        X = validate_data(self, X)

        if not set(self.column_idx) <= set(range(X.shape[1])):
            raise ValueError(f"Parameter column_idx is out of bound from input X with {X.shape[1]} features.")

        self.column_idx_ = np.array(self.column_idx)
        self.thresholds_ = np.array(self.thresholds)

        return self

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False)

        preds = np.ones(X.shape[0], dtype=bool)  # shape (n_samples,)

        # Apply thresholds
        for col_idx, (op, thresh_val) in zip(self.column_idx_, self.thresholds_):
            preds &= op(X[:, col_idx], thresh_val)

        return preds.astype("uint8")  # return as integer
