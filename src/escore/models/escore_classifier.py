import numpy as np
from scipy.stats import anderson, kstest, normaltest, shapiro
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore

from escore.models.transformers import SvDifferenceExtractor


def make_escore() -> Pipeline:
    """
    Create a ready-made Escore pipeline.

    Returns
    -------
    Pipeline
        - 'preprocessor' (`SvDifferenceExtractor`) computing ΔMVBS with respect to the first channel,
        - 'classifier' (`EscoreClassifier`) with default thresholds
    """

    model = Pipeline(
        [
            ("preprocessor", SvDifferenceExtractor(ref_channel_idx=0)),
            ("classifier", EscoreClassifier()),
        ]
    )

    return model


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

        self.diagnostics = EscoreDiagnostics(self)

    def fit(self, X, y):
        """Fit the model: learn each class' gaussian distribution mean & cov matrix"""

        # Input validation
        X, y = validate_data(self, X, y)

        # Store classes
        self.classes_ = unique_labels(y)

        # Check that number of classes is sufficient
        if self.classes_.size < 2:
            raise ValueError(f"EscoreClassifier requires >= 2 classes. Currently has {self.classes_.size}.")

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
        likelihoods = np.exp(self.decision_function(X))
        sorted_liks = np.sort(likelihoods, axis=1)
        best, second = sorted_liks[:, -1], sorted_liks[:, -2]

        # Get argmax predictions
        preds = np.argmax(likelihoods, axis=1)

        # Apply thresholds
        e_valid = best >= self.e_thresh
        q_valid = (
            second / np.maximum(best, self.e_thresh)  # use e_thresh if best < e_thresh to avoid warning
            <= self.q_thresh
        )
        valid = e_valid & q_valid

        # Convert to class label
        preds = self.classes_[preds]
        preds[~valid] = -1  # sklearn compatible rejection code

        return preds

    def predict_proba(self, X):
        """
        Predict normalized likelihoods P(X|y) = PDF_y(X) where PDF is Gaussian

        Note: Returns likelihoods, not posterior probabilities P(y|X),
        since class priors are unknown for the target population.
        """

        # Compute exponents of log likelihoods and return exponent
        likelihoods = np.exp(self.decision_function(X))

        # Normalize to sum to 1
        return likelihoods / likelihoods.sum(axis=1, keepdims=True)

    def decision_function(self, X):
        """
        Predict log likelihoods log P(X|y) = log PDF_y(X) where PDF is Gaussian
        """

        n_samples, n_features = X.shape

        # Compute mahalanobis distance to each class
        mahalanobis_dists = self._predict_distance(X)

        # Init log probas array
        log_probas = np.zeros((n_samples, self.classes_.size))

        # Loop through classes, compute and fill
        for i, _ in enumerate(self.classes_):
            det_cov = np.linalg.det(self.covs_[i])
            normalization = 1.0 / np.sqrt((2 * np.pi) ** n_features * det_cov)
            log_probas[:, i] = np.log(normalization) - 0.5 * mahalanobis_dists[:, i]

        return log_probas

    def get_rejection_reasons(self, X):
        """
        Diagnostic method: return which samples failed which threshold.

        Returns
        -------
        dict with:
            - 'e_failures': bool array, samples failing e_tresh
            - 'q_failures': bool array, samples failing q_thresh
            - 'best_likelihoods': best likelihood per sample
            - 'likelihood_ratios': second/best ratio per sample
        """

        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        likelihoods = np.exp(self.decision_function(X))
        sorted_liks = np.sort(likelihoods, axis=1)
        best, second = sorted_liks[:, -1], sorted_liks[:, -2]

        return {
            "e_failures": best < self.e_thresh,
            "q_failures": (second / np.maximum(best, self.e_thresh)) > self.q_thresh,
            "best_likelihoods": best,
            "likelihood_ratios": second / np.maximum(best, self.e_thresh),
        }

    def _predict_distance(self, X):
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


class EscoreDiagnostics:
    def __init__(self, parent: EscoreClassifier):
        self.parent = parent

    def normality(self, X, y):
        """Run normality tests per class and per feature. Return p-values."""
        results = {}
        for cls in self.parent.classes_:
            X_cls = X[y == cls]
            X_cls = np.asarray(X_cls)

            # Test each feature independently
            feature_results = {}
            for feat_idx in range(X_cls.shape[1]):
                feature_data = X_cls[:, feat_idx]
                feature_results[f"feature_{feat_idx}"] = {
                    "shapiro_p": shapiro(feature_data)[1],
                    "anderson_stat": anderson(feature_data).statistic,  # type: ignore
                    "kstest_p": kstest(feature_data, "norm")[1],
                }
            results[cls] = feature_results
        return results

    def qq_plots(self, X, y):
        """Generate Q-Q plot data for visual inspection."""
        from scipy import stats

        results = {}
        for cls in self.parent.classes_:
            X_cls = np.asarray(X[y == cls])
            feature_results = {}
            for feat_idx in range(X_cls.shape[1]):
                feature_data = X_cls[:, feat_idx]
                n = len(feature_data)

                # Compute theoretical quantiles at empirical CDF positions
                quantile_positions = np.arange(1, n + 1) / (n + 1)
                theoretical_quantiles = stats.norm.ppf(quantile_positions)

                # Sample quantiles: sorted data (order statistics)
                sample_quantiles = np.sort(feature_data)

                feature_results[f"feature_{feat_idx}"] = {
                    "sample_quantiles": sample_quantiles,
                    "theoretical_quantiles": theoretical_quantiles,
                }
            results[cls] = feature_results
        return results
