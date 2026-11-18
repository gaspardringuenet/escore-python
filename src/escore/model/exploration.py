"""Explore Ward clusterings independently of class covariance fitting."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .classifier import _check_values


class WardExploration:
    """A Ward hierarchy of echotype means, reusable across candidate cuts."""

    def __init__(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not X.columns.is_unique or not all(isinstance(name, str) for name in X.columns):
            raise ValueError("X must have unique string feature names.")
        values = _check_values(X, tuple(X.columns))
        if len(values) < 2 or values.shape[1] == 0:
            raise ValueError("Ward clustering requires at least two rows and one feature.")
        self.data = X.copy()
        self.linkage = linkage(values, method="ward", metric="euclidean")

    def labels(self, n_classes: int) -> np.ndarray:
        """Return a zero-based class label for each training row."""
        if isinstance(n_classes, bool) or not isinstance(n_classes, int):
            raise ValueError("n_classes must be an integer.")
        if not 1 <= n_classes <= len(self.data):
            raise ValueError("n_classes must be between 1 and the number of rows.")
        return cut_tree(self.linkage, n_clusters=n_classes).ravel()

    def scores(self, counts: list[int] | tuple[int, ...]) -> pd.DataFrame:
        """Evaluate valid cuts; undefined metrics are reported as NaN with a reason."""
        rows = []
        values = self.data.to_numpy(dtype=float)
        for count in counts:
            labels = self.labels(count)
            n_labels = len(np.unique(labels))
            row = {"n_classes": count, "silhouette": np.nan, "calinski_harabasz": np.nan,
                   "davies_bouldin": np.nan, "error": ""}
            if not 2 <= n_labels < len(values):
                row["error"] = "Metrics require between 2 and n_samples - 1 distinct clusters."
            else:
                for name, metric in (
                    ("silhouette", silhouette_score),
                    ("calinski_harabasz", calinski_harabasz_score),
                    ("davies_bouldin", davies_bouldin_score),
                ):
                    try:
                        row[name] = metric(values, labels)
                    except ValueError as exc:
                        row["error"] += f"{name}: {exc}; "
            rows.append(row)
        return pd.DataFrame(rows, columns=[
            "n_classes", "silhouette", "calinski_harabasz", "davies_bouldin", "error"
        ]).set_index("n_classes")

    def plot(self, *, ax=None):
        """Draw the dendrogram and return its axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        dendrogram(self.linkage, ax=ax)
        ax.set_ylabel("Ward linkage distance")
        return ax
