from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def evaluate_n_clusters(
    X: np.ndarray | pd.DataFrame,
    clusterer: Any = None,
    get_labels_fn: Callable[[Any, np.ndarray, int], np.ndarray] | None = None,
    n_clusters_range: range = range(2, 11),
    scale: bool = True,
    **clusterer_kwargs,
) -> pd.DataFrame:
    """Evalutate clustering quality for different number of clusters using multiple metrics.

    Args:
        X (np.ndarray | pd.DataFrame): Feature matrix for clustering.
        clusterer (Any, optional): Clustering model with fit/predict interface
            (e.g. EchoclassClassifier, KMeans). Defaults to None.
        get_labels_fn (Callable[[Any, np.ndarray, int], np.ndarray] | None, optional):
            Function to extract labels from clusterer.
            Signature: get_labels_fn(clusterer, X, n_clusters) -> np.ndarray.
            If None, assumes clusterer has fit_predict(X, n_clusters) method.
            Defaults to None.
        n_clusters_range (range, optional): Range of clusters numbers to evaluate.
            Defaults to range(2, 11).
        scale (bool, optional): Whether to scale features before clustering. Defaults to True.
        **clusterer_kwargs: Additional arguments passed to clusterer initialization.

    Raises:
        ValueError: When clusterer does not have a fit_predict method and no get_labels_fn is provided.

    Returns:
        pd.DataFrame: Metrics for each number of clusters.
    """

    # Data preprocessing
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = StandardScaler().fit_transform(X) if scale else X

    # Default to hierarchical clustering if no clusterer provided
    if clusterer is None:
        method = clusterer_kwargs.pop("method", "ward")
        Z = linkage(X, method=method)

        def get_labels_fn(_, __, n):
            return fcluster(Z, n, criterion="maxclust")
    elif get_labels_fn is None:
        # Assume clusterer as fit_predict(features_df, n_clusters) method
        if not hasattr(clusterer, "fit_predict"):
            raise ValueError(
                "Clusterer must have fit_predict(X, n_clusters) method "
                "or user must provide custom get_labels_fn(clusterer, X, n_clusters)"
            )

        def get_labels_fn(clr, X, n):
            return clr.fit_predict(X, n)

    results = []
    for n_clusters in n_clusters_range:
        labels = get_labels_fn(clusterer, X, n_clusters)

        # Compute metrics if we have more than one cluster
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
        else:
            silhouette = np.nan
            davies_bouldin = np.nan
            calinski_harabasz = np.nan

        results.append(
            {
                "n_clusters": n_clusters,
                "silhouette_score": silhouette,
                "davis_bouldin_index": davies_bouldin,
                "calinski_harabasz_index": calinski_harabasz,
            }
        )

    return pd.DataFrame(results)


def plot_cluster_evaluation(
    evaluation_df: pd.DataFrame, figsize: tuple = (15, 4), layout: str = "constrained"
) -> Figure:
    """Plot cluster evaluation metrics.

    Args:
        evaluation_df (pd.DataFrame): Output from evaluate_n_clusters().
        figsize (tuple, optional): Figure size_. Defaults to (15, 4).
        layout (str, optional): Matplotlib layout. Defaults to "constrained".

    Returns:
        Figure: Plot.
    """

    fig, axes = plt.subplots(1, 3, figsize=figsize, layout=layout)

    # Silhouette score
    axes[0].plot(evaluation_df["n_clusters"], evaluation_df["silhouette_score"], "o-", linewidth=2)
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].grid(True, alpha=0.3)

    # Davis-Bouldin Index
    axes[1].plot(
        evaluation_df["n_clusters"], evaluation_df["davis_bouldin_index"], "o-", linewidth=2
    )
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Davis-Bouldin Index")
    axes[1].grid(True, alpha=0.3)

    # Calinski-Harabasz Index
    axes[2].plot(
        evaluation_df["n_clusters"], evaluation_df["calinski_harabasz_index"], "o-", linewidth=2
    )
    axes[2].set_xlabel("Number of Clusters")
    axes[2].set_ylabel("Calinski-Harabasz Index")
    axes[2].grid(True, alpha=0.3)

    return fig


def validate_classification(
    features_df: pd.DataFrame,
    test_size: float = 0.3,
    n_estimators: int = 500,
    random_state: int = 42,
) -> dict:
    """Validate echo-class classification using Random Forest.

    Args:
        features_df (pd.DataFrame): Feature matrix with 'echo_class' column.
        test_size (float, optional): Proportion of data for testing. Defaults to 0.3.
        n_estimators (int, optional): Number of trees in Random Forest. Defaults to 500.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        dict: Results including model, predictions, confusion matrix, importance.
    """

    feature_cols = [c for c in features_df.columns if c != "label"]
    X = features_df[feature_cols].values
    y = features_df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model: RandomForestClassifier = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=np.nan),
        "feature_importance": pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False),
        "train_score": model.score(X_train, y_train),
        "test_score": model.score(X_test, y_test),
    }
