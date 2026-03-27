from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, set_link_color_palette
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class EchoclassClassifier:
    """Hierarchical clustering of echotypes into echo-classes"""

    def __init__(self, method: str = 'ward', scale: bool = True):
        """
        Args:
            method (str, optional): Linkage method. Defaults to 'ward'.
            scale (bool, optional): Whether to scale data. Defaults to True.
        """
        self.method: str = method
        self.scaler = StandardScaler() if scale else None
        self.linkage_matrix: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def fit(self, X: np.ndarray | pd.DataFrame):
        """Compute hierarchical clustering"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.fit_transform(X) if self.scaler else X
        self.linkage_matrix = linkage(X_scaled, method=self.method)
        return self
    
    def predict(self, n_clusters: int) -> np.ndarray:
        """Cut dendrogram into n clusters."""
        if self.linkage_matrix is None:
            raise ValueError("Classifier not fitted.")
        self.labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        return self.labels
    
    def fit_predict(self, X: np.ndarray | pd.DataFrame, n_clusters: int) -> np.ndarray:
        """Compute hierarchical clustering and predict with n clusters."""
        self.fit(X)
        return self.predict(n_clusters)
    
    def get_dendrogram(self, n_clusters: int = None, no_plot: bool = False):
        """Return data for plotting dendrogram"""
        if self.linkage_matrix is None:
            raise ValueError("Classifier not fitted.")
        set_link_color_palette([f"C{i}" for i in range(10)])
        if n_clusters is None:
            return dendrogram(self.linkage_matrix, color_threshold=0, above_threshold_color="black", no_plot=no_plot)
        else:
            n_samples = self.linkage_matrix.shape[0] + 1
            if n_clusters >= n_samples:
                raise ValueError(f"n_clusters must be less than {n_samples}")
            threshold_idx = n_samples - n_clusters + 1
            color_threshold = self.linkage_matrix[threshold_idx - 1, 2]
            return dendrogram(self.linkage_matrix, color_threshold=color_threshold, above_threshold_color="black", no_plot=no_plot)