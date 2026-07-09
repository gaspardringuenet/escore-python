import warnings
from typing import Callable, List, Literal

import numpy as np
import pandas as pd
from sklearn import metrics

warnings.filterwarnings("ignore", category=RuntimeWarning)


def n_clusters_metrics(
    X: np.ndarray | pd.DataFrame,
    preprocessor,
    classifier,
    n_clusters_range: range = range(2, 10),
    metrics_list: List[Callable] | Literal["default"] = "default",
):
    # Preprocess once
    X_preproc = preprocessor.fit_transform(X)

    # Override default metrics
    if metrics_list == "default":
        m_list: List[Callable] = [
            metrics.silhouette_score,
            metrics.calinski_harabasz_score,
            metrics.davies_bouldin_score,
        ]
    else:
        m_list = metrics_list

    # Init scores dicts
    scores = {"n_clusters": [], **{m.__name__: [] for m in m_list}}

    # Collect metrics for each n
    for n in n_clusters_range:
        classifier.set_params(n_clusters=n)

        # Predict
        labels = classifier.fit_predict(X_preproc)

        # Append scores lists
        scores["n_clusters"].append(n)
        for m in m_list:
            scores[m.__name__].append(m(X_preproc, labels))

    # Format scores dataframe
    scores = pd.DataFrame(scores)
    scores = scores.melt(
        id_vars=["n_clusters"],
        value_vars=[m.__name__ for m in m_list],
        var_name="metric",
    )

    return scores


def n_clusters_metrics_raw(
    data_raw: pd.DataFrame,
    preprocessor,
    classifier,
    group_var: str = "echotype_id",
    n_clusters_range: range = range(2, 10),
    metrics_list: List[Callable] | Literal["default"] = "default",
):

    # Aggregate data
    X_raw = data_raw.drop(group_var, axis=1)
    X_agg = data_raw.groupby(group_var).mean()
    X_agg_preproc = preprocessor.fit_transform(X_agg)
    X_raw_preproc = preprocessor.fit_transform(X_raw)

    # Override default metrics
    if metrics_list == "default":
        m_list: List[Callable] = [
            metrics.silhouette_score,
            metrics.calinski_harabasz_score,
            metrics.davies_bouldin_score,
        ]
    else:
        m_list = metrics_list

    # Init scores dicts
    scores = {"n_clusters": [], **{m.__name__: [] for m in m_list}}

    # Collect metrics for each n
    for n in n_clusters_range:
        classifier.set_params(n_clusters=n)

        # Predict on aggregated data
        labels_agg = classifier.fit_predict(X_agg_preproc)

        # Left join to raw
        X_agg["labels"] = labels_agg
        data_raw = data_raw.join(X_agg[["labels"]], on=group_var, how="left")

        # Extract labels
        labels_raw = data_raw["labels"]
        data_raw = data_raw.drop("labels", axis=1)

        # Append scores lists
        scores["n_clusters"].append(n)
        for m in m_list:
            scores[m.__name__].append(m(X_raw_preproc, labels_raw))

    # Format scores dataframe
    scores = pd.DataFrame(scores)
    scores = scores.melt(
        id_vars=["n_clusters"],
        value_vars=[m.__name__ for m in m_list],
        var_name="metric",
    )

    return scores
