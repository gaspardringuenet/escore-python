from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ExtractionConfig:
    """Extraction settings; channel labels are ordered with the reference first.

    Attributes
    ----------
    channels : tuple of str or tuple of float
        Channel labels, ordered with the reference first. At least two, unique.
        All labels must be of the same type.
    kernel : ndarray of float, optional
        Non-negative 2D kernel. Must contain a positive value.
    clusterer : {"gmm", "kmeans"}, default "gmm"
        Clustering algorithm.
    gmm_components : tuple of int, default (1, 2, 3)
        Component counts to try for GMM. Positive integers.
    kmeans_clusters : int, default 2
        Number of clusters for k-means.
    kmeans_n_init : int, default 10
        Number of k-means initializations.
    random_state : int, default 0
        Random seed in ``[0, 2**32 - 1]``. Used during clusterer fitting.
    confidence_threshold : float, optional
        Value in ``[0, 1]``. GMM only. Samples with prediction probabilities below this threshold are rejected.
    clustering_features : {"reference", "adjacent"}
        Feature geometry used for clustering. Final echotype means always use
        reference-relative differences.
    clustering_channels : tuple, optional
        Channel order for adjacent differences; defaults to ``channels``.
    min_sv, max_sv : float, optional
        Require all raw channels at a pixel to be within these inclusive bounds.
    """

    channels: tuple[str, ...] | tuple[float, ...]
    kernel: npt.NDArray[np.float64] | None = None
    clusterer: Literal["gmm", "kmeans"] = "gmm"
    gmm_components: tuple[int, ...] = (1, 2, 3)
    kmeans_clusters: int = 2
    kmeans_n_init: int = 10
    random_state: int = 0
    confidence_threshold: float | None = None
    clustering_features: Literal["reference", "adjacent"] = "reference"
    clustering_channels: tuple[str, ...] | tuple[float, ...] | None = None
    min_sv: float | None = None
    max_sv: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.channels, str):
            raise TypeError("channels must be a sequence of channel labels.")
        channels = tuple(self.channels)
        all_str = all(isinstance(c, str) and c for c in channels)
        all_float = all(isinstance(c, float) for c in channels)
        if len(channels) < 2 or not (all_str or all_float):
            raise ValueError(
                "channels must contain at least two labels, all non-empty strings or all floats."
            )
        if len(set(channels)) != len(channels):
            raise ValueError("channels must be unique.")
        object.__setattr__(self, "channels", channels)
        if self.clustering_features not in ("reference", "adjacent"):
            raise ValueError("clustering_features must be 'reference' or 'adjacent'.")
        if self.clustering_channels is not None:
            order = tuple(self.clustering_channels)
            if self.clustering_features != "adjacent" or len(order) != len(channels) or set(order) != set(channels):
                raise ValueError("clustering_channels must be a permutation of channels for adjacent features.")
            object.__setattr__(self, "clustering_channels", order)
        for bound in (self.min_sv, self.max_sv):
            if bound is not None and (isinstance(bound, bool) or not np.isfinite(bound)):
                raise ValueError("Sv bounds must be finite or None.")
        if self.min_sv is not None and self.max_sv is not None and self.min_sv > self.max_sv:
            raise ValueError("min_sv must not exceed max_sv.")
        if self.kernel is not None:
            kernel = np.array(self.kernel, dtype=float, copy=True)
            if (
                kernel.ndim != 2
                or not kernel.size
                or not np.isfinite(kernel).all()
                or np.any(kernel < 0)
                or not np.any(kernel > 0)
            ):
                raise ValueError("kernel must be a non-empty, finite, non-negative 2D array.")
            kernel.setflags(write=False)
            object.__setattr__(self, "kernel", kernel)
        if self.clusterer not in ("gmm", "kmeans"):
            raise ValueError("clusterer must be 'gmm' or 'kmeans'.")
        components = tuple(self.gmm_components)
        if not components or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in components
        ):
            raise ValueError("gmm_components must contain positive integers.")
        object.__setattr__(self, "gmm_components", components)
        for name in ("kmeans_clusters", "kmeans_n_init"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer.")
        if not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be between 0 and 2**32 - 1.")
        if self.confidence_threshold is not None:
            threshold = self.confidence_threshold
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                raise ValueError("confidence_threshold must be between 0 and 1.")
            if self.clusterer != "gmm":
                raise ValueError("confidence_threshold is only supported for GMM.")
