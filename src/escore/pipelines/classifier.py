from typing import List, Literal

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from sklearn.utils.multiclass import unique_labels
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


class FixedSvThresholds(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        column_idx: List[int] = [0],
        thresh_vals: List[float] = [-40],
        thresh_types: List[Literal["above", "below"]] = ["above"],
    ):
        """Classify data based on fixed threshold values."""

        if not len(column_idx) == len(thresh_vals) == len(thresh_types):
            raise ValueError("columns and threshold arguments must have same length.")
        if not set(thresh_types) <= {"above", "below"}:
            raise ValueError("threshold types must be either 'above' or 'below'.")

        # Find duplicated: they must be max 2 and have different types
        counter = {x: {"n": 0, "types": set()} for x in set(column_idx)}
        for i, x in enumerate(column_idx):
            counter[x]["n"] += 1
            counter[x]["types"].add(thresh_types[i])
        for x, count_dict in counter.items():
            if count_dict["n"] > 2:
                raise ValueError(
                    "Cannot specify more than 2 thresholds for one given column id."
                )
            if len(count_dict["types"]) > 2:
                raise ValueError(
                    "Cannot specify more than 2 thresholds types for one column id."
                )

        self.column_idx = column_idx
        self.thresh_vals = thresh_vals
        self.thresh_types = thresh_types

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

    def fit(self, X, y=None):

        # Input validation
        X = validate_data(self, X)

        if not set(self.column_idx) <= set(range(X.shape[1])):
            raise ValueError(
                f"Parameter column_idx is out of bound from input X with {X.shape[1]} features."
            )

        self.column_idx_ = np.array(self.column_idx).squeeze()
        self.thresh_vals_ = np.array(self.thresh_vals).squeeze()
        self.thresh_types_ = np.array(self.thresh_types).squeeze()

        return self

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False)

        preds = np.ones(X.shape[0], dtype=bool)  # shape (n_samples,)

        # Apply "above" thresholds
        id_mask = self.thresh_types_ == "above"
        if id_mask.sum() > 0:
            idx = self.column_idx_[id_mask]
            threshs = self.thresh_vals_[id_mask]
            above_preds = X[:, idx] >= threshs  # same number of cols -> cast on rows
            preds &= above_preds.squeeze()

        # Apply "below" thresholds
        id_mask = ~id_mask
        if id_mask.sum() > 0:
            idx = self.column_idx_[id_mask]
            threshs = self.thresh_vals_[id_mask]
            below_preds = X[:, idx] <= threshs
            preds &= (
                below_preds.squeeze()
            )  # for each col we can specify above and below

        return preds.astype("uint8")  # return as integer


class EscoreClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, e_thresh=1e-2, q_thresh=0.5):
        """
        Scikit-learn estimator implementing the Escore model.

        Parameters
        ----------
        e_thresh : _type_, optional
            Minimal likelihood for best class, by default 1e-2
        q_thresh : float, optional
            Maximal likelihood ratio between best and second best class, by default 0.5
        """
        self.e_thresh = e_thresh
        self.q_thresh = q_thresh

    def fit(self, X, y):
        """Fit the model: learn each class' gaussian distribution mean & cov matrix"""

        # Input validation
        X, y = validate_data(self, X, y)

        # Store classes
        self.classes_ = unique_labels(y)

        # Check that number of classes is sufficient
        if self.classes_.size < 2:
            raise ValueError(
                f"EscoreClassifier requires >= 2 classes. Currently has {self.classes_.size}."
            )

        # Compute gaussian distributions means & cov matrices
        mean_list, cov_list = [], []
        for cls in self.classes_:
            X_cls = X[y == cls]
            mean_list.append(X_cls.mean(axis=0))
            cov = np.cov(X_cls.T)
            cov_list.append(cov)

        # Store fitted parameters
        self.means_ = np.stack(mean_list, axis=0)
        self.covs_ = np.stack(cov_list, axis=0)

        return self

    def predict(self, X):
        """Predict class with highest likelihood. Only predicts where thresholds are respected."""

        # Fetch best & 2nd best class
        likelihoods = self.predict_proba(X)
        sorted = np.sort(likelihoods, axis=1)
        best, second = sorted[:, -1], sorted[:, -2]

        # Init prediction array
        preds = np.empty(X.shape[0])

        # Fill & apply thresholds
        # absolute threshold
        e_valid = best >= self.e_thresh
        # relative threshold
        # if best < e_thresh, use e_thresh
        # -> avoids RunTimeWarning & has no consequence (e_tresh values overwrite q_tresh values)
        q_valid = second / np.maximum(best, self.e_thresh) <= self.q_thresh
        valid = e_valid & q_valid
        preds[valid] = np.argmax(likelihoods, axis=1)[valid]
        preds[~q_valid] = -1
        preds[~e_valid] = -2

        return preds

    def predict_proba(self, X):
        """
        Predict likelihood P(X|y) = PDF_y(X)
        where PDF is a Gaussian PDF.
        """

        # Compute log likelihoods and return exponent
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X):
        """
        Predict log likelihood log P(X|y) = log PDF_y(X)
        where PDF is a Gaussian PDF.
        """

        n_samples, n_features = X.shape

        # Compute mahalanobis distance to each class
        mahalanobis_dists = self.predict_distances(X)

        # Init log probas array
        log_probas = np.zeros((n_samples, self.classes_.size))

        # Loop through classes, compute and fill
        for i, _ in enumerate(self.classes_):
            det_cov = np.linalg.det(self.covs_[i])
            normalization = 1.0 / np.sqrt((2 * np.pi) ** n_features * det_cov)
            log_probas[:, i] = np.log(normalization) - 0.5 * mahalanobis_dists[:, i]

        return log_probas

    def predict_distances(self, X):
        """
        Predict mahalanobis distance to each classe.
        """

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False)

        # Compute distances
        distances = np.zeros((X.shape[0], self.classes_.size))
        for i, _ in enumerate(self.classes_):
            mu, cov_inv = self.means_[i], np.linalg.inv(self.covs_[i])
            centered = X - mu
            distances[:, i] = np.sum((centered @ cov_inv) * centered, axis=1)

        return distances
