import os
import tempfile
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering


class EscoreClassifier(BaseEstimator):
    """Echoclass classifier based on squared Mahalanobis distance.

    Echotype mean ΔMVBS vectors are clustered hierarchically (Ward linkage) into
    ``n_classes`` echoclasses. Each class is described by the mean and covariance
    of its echotype means, not of pooled pixels. Pixels are assigned to the class
    with the smallest squared Mahalanobis distance.

    Parameters
    ----------
    n_classes : int
        Number of echoclasses.
    max_squared_distance : float, tuple of float, or None, default None
        Pixels whose smallest squared distance is greater than or equal to this
        value are left unclassified (label -1). A tuple supplies one threshold
        per class; only eligible classes compete. None disables rejection.

    Attributes
    ----------
    feature_names_in_ : tuple of str
        Feature names and order seen by ``fit``.
    labels_ : ndarray of shape (n_echotypes,)
        Echoclass of each training echotype.
    means_ : ndarray of shape (n_classes, n_features)
        Mean of each class.
    covariances_ : ndarray of shape (n_classes, n_features, n_features)
        Covariance of each class, estimated from its echotype means.
    """

    def __init__(
        self, *, n_classes: int, max_squared_distance: float | tuple[float, ...] | None = None
    ) -> None:
        self.n_classes = n_classes
        self.max_squared_distance = max_squared_distance

    def fit(
        self,
        X: pd.DataFrame,
        y: None = None,
        *,
        channels: tuple[str, ...] | tuple[float, ...] | None = None,
    ) -> Self:
        """Cluster echotype means and estimate one Gaussian per echoclass.

        Parameters
        ----------
        X : DataFrame
            One row per accepted echotype and one named column per ΔMVBS feature.
        y : None
            Unused.
        channels : tuple, optional
            Channel labels in reference-first order, corresponding to X's features.

        Raises
        ------
        ValueError
            If ``n_classes`` or ``max_squared_distance`` is invalid, there are fewer
            echotypes than classes, a class has too few echotypes to estimate a
            full-rank covariance, or ``X`` contains non-finite values.
        """

        # Input validation
        if isinstance(self.n_classes, bool) or not isinstance(self.n_classes, int):
            raise ValueError("n_classes must be an integer.")  # noqa
        if self.n_classes < 1:
            raise ValueError("n_classes must be at least 1.")
        self._thresholds()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        names = tuple(X.columns)
        values = _check_values(X, names)
        n_samples, n_features = values.shape
        if not n_features or not X.columns.is_unique or not all(isinstance(n, str) for n in names):
            raise ValueError("X must have unique string feature names and at least one feature.")
        if channels is not None:
            if len(channels) != n_features + 1 or len(set(channels)) != len(channels):
                raise ValueError("channels must contain distinct reference and feature channels.")
            expected = tuple(f"{channel} - {channels[0]}" for channel in channels[1:])
            if names != expected:
                raise ValueError(f"X features do not match channels: expected {expected}.")
        if n_samples < self.n_classes:
            raise ValueError(f"Cannot fit {self.n_classes} classes from {n_samples} echotype(s).")

        # Step 1 - Cluster echotypes into echoclasses
        clusterer: AgglomerativeClustering = AgglomerativeClustering(
            n_clusters=self.n_classes,
            linkage="ward",
        )
        labels = clusterer.fit_predict(values)

        # Step 2 - Fit the ellipses
        means, covariances = [], []
        for class_id in range(self.n_classes):
            members = values[labels == class_id]
            if len(members) < n_features + 1:
                raise ValueError(
                    f"Class {class_id} has {len(members)} echotype(s); at least "
                    f"{n_features + 1} are needed to estimate its covariance."
                )
            covariance = np.atleast_2d(np.cov(members, rowvar=False))
            if np.linalg.matrix_rank(covariance) < n_features or np.linalg.cond(covariance) > 1e12:
                raise ValueError(f"Class {class_id} has a singular or ill-conditioned covariance.")
            means.append(members.mean(axis=0))
            covariances.append(covariance)

        # Update parameters
        self.feature_names_in_ = names
        self.labels_ = labels
        self.means_ = np.array(means)
        self.covariances_ = np.array(covariances)
        self._precisions = np.linalg.inv(self.covariances_)
        self.channels_ = tuple(channels) if channels is not None else None
        self.training_index_ = X.index.copy()
        return self

    def transform(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Return squared Mahalanobis distances with shape ``(n_samples, n_classes)``."""
        self._check_fitted()
        diff = _check_values(X, self.feature_names_in_)[:, None, :] - self.means_[None, :, :]
        return np.einsum("nkd,kde,nke->nk", diff, self._precisions, diff)

    def predict(self, X: pd.DataFrame) -> NDArray[np.intp]:
        """Return the nearest echoclass per row, or -1 if it is beyond the threshold."""
        distances = self.transform(X)
        return self._classify_distances(distances)

    def _thresholds(self) -> np.ndarray:
        threshold = self.max_squared_distance
        if threshold is None:
            return np.full(self.n_classes, np.inf)
        try:
            values = np.asarray(threshold, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Thresholds must be positive finite numbers.") from exc
        if values.ndim == 0:
            values = np.full(self.n_classes, values)
        if values.shape != (self.n_classes,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Provide a positive finite threshold for each class, or None.")
        return values

    def _classify_distances(self, distances: np.ndarray) -> np.ndarray:
        return _labels_for_distances(distances, self._thresholds())

    def save(self, path: str | Path) -> None:
        """Save the fitted model; load joblib files only from trusted sources."""
        self._check_fitted()
        path = Path(path)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            temp_path = Path(file.name)
        try:
            joblib.dump(self, temp_path)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a trusted model file; joblib loading can execute arbitrary code."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("File does not contain an EscoreClassifier.")
        model._check_fitted()
        return model

    def _check_fitted(self) -> None:
        if not hasattr(self, "feature_names_in_"):
            raise ValueError("This EscoreClassifier is not fitted yet; call fit first.")


def _check_values(X: pd.DataFrame, names: tuple[str, ...]) -> NDArray[np.float64]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if tuple(X.columns) != names:
        raise ValueError(f"X columns {tuple(X.columns)} do not match features {names}.")
    values = X.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("X contains non-finite values.")
    return values


def _labels_for_distances(distances: np.ndarray, thresholds: np.ndarray | float) -> np.ndarray:
    eligible = np.where(distances < thresholds, distances, np.inf)
    labels = np.argmin(eligible, axis=-1).astype(np.intp)
    return np.where(np.isfinite(eligible).any(axis=-1), labels, -1)
