"""Explore k-means cluster counts for a single ROI."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

from .config import ExtractionConfig
from .data import EchotypeCandidate


def explore_kmeans(
    candidate: EchotypeCandidate, config: ExtractionConfig, counts: list[int] | tuple[int, ...]
) -> pd.DataFrame:
    """Evaluate Calinski–Harabasz scores on the candidate's smoothed valid pixels."""
    sv = candidate.smoothed_sv
    if config.clustering_features == "adjacent":
        delta = sv.sel(channel=list(config.clustering_channels or config.channels)).diff("channel")
    else:
        delta = sv.isel(channel=slice(1, None)) - sv.isel(channel=0, drop=True)
    values = np.asarray(delta.stack(pixel=("ping_time", "depth")).transpose("pixel", "channel"))
    values = values[np.isfinite(values).all(axis=1)]
    rows = []
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count < len(values):
            raise ValueError("Each count must be between 2 and the number of valid pixels minus one.")
        labels = KMeans(n_clusters=count, n_init=config.kmeans_n_init,
                        random_state=config.random_state).fit_predict(values)
        if not 2 <= len(np.unique(labels)) < len(values):
            rows.append({"n_clusters": count, "calinski_harabasz": np.nan,
                         "error": "Fewer than two distinct clusters."})
        else:
            rows.append({"n_clusters": count, "calinski_harabasz": calinski_harabasz_score(values, labels),
                         "error": ""})
    return pd.DataFrame(rows, columns=["n_clusters", "calinski_harabasz", "error"]).set_index("n_clusters")
