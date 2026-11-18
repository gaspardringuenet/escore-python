import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Self

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .config import ExtractionConfig


@dataclass
class EchotypeCandidate:
    """ROI data and clustering results before any review decision.

    Contains ROI shape data as well as the successive outputs from extraction
    processing.

    Attributes
    ----------
    roi_id : int
        The region_id of the source ROI.
    metadata: Series
        The Regions2D.data row describing the ROI. Contains metadata (e.g. region_name)
        as well as the actual (time, depth) shape of the ROI polygon.
    raw_Sv: DataArray
        Sv data contained in the ROI.
    smoothed_Sv: DataArray
        Sv data smoothed using the configuration's kernel. ΔSv for clustering is computed
        from this array.
    labels: DataArray
        Clustering labels for each echogram pixel.
    filtered_labels: DataArray
        labels with an optional GMM confidence threshold applied. If clusterer is not "gmm"
        or if no confidence threshold is specified, this equals labels.
    model: GaussianMixture | KMeans
        Fitted clusterer.
    proposed_cluster: int
        Id of the cluster most likely containing the echotype.
        Current rule: The largest cluster is selected, unless the second largest is within
        30% of its pixel count; then the one whose mean ΔSv lies furthest from zero is xselected.
    """

    roi_id: int
    metadata: pd.Series
    raw_sv: xr.DataArray
    smoothed_sv: xr.DataArray
    labels: xr.DataArray
    filtered_labels: xr.DataArray
    model: GaussianMixture | KMeans
    proposed_cluster: int

    @property
    def cluster_ids(self) -> tuple[int, ...]:
        labels = np.asarray(self.labels.values)
        return tuple(int(label) for label in np.unique(labels[labels >= 0]))

    def mask(self, cluster_id: int) -> xr.DataArray:
        """Return the confidence-filtered mask for a fitted cluster."""
        if cluster_id not in self.cluster_ids:
            raise ValueError(f"Cluster {cluster_id} was not fitted for ROI {self.roi_id}.")
        return self.filtered_labels == cluster_id

    def selected_sv(self, cluster_id: int) -> xr.DataArray:
        """Apply a cluster mask to the raw, not smoothed, Sv."""
        return self.raw_sv.where(self.mask(cluster_id))

    def retained_share(self, cluster_id: int) -> float:
        """Fraction of initially labelled pixels retained after confidence filtering."""
        return float(self.mask(cluster_id).sum().item() / (self.labels >= 0).sum().item())


@dataclass(frozen=True)
class ExtractionFailure:
    roi_id: int
    stage: Literal["mask", "cluster"]
    message: str


@dataclass
class ExtractionResults:
    """Candidates and per-ROI failures associated with one extraction run.

    Attributes
    ----------
    extraction_id : str
        Identifier of the extraction run.
    config : ExtractionConfig
        Extraction settings used to produce results.
    candidates : dict[int, EchotypeCandidate]
        Echotype candidates, i.e. segments produced by clustering ROI data. Keys
        are region ids.
    failures : dict[int, ExtractionFailure]
        Failures with stage and explanation. Keys are region ids.
    """

    extraction_id: str
    config: ExtractionConfig
    candidates: dict[int, EchotypeCandidate]
    failures: dict[int, ExtractionFailure]

    def save(self, path: str | Path) -> None:
        """Save a self-contained snapshot; files must only be loaded from trusted sources."""
        path = Path(path)
        candidates = {
            roi_id: replace(
                candidate,
                raw_sv=candidate.raw_sv.compute(),
                smoothed_sv=candidate.smoothed_sv.compute(),
                labels=candidate.labels.compute(),
                filtered_labels=candidate.filtered_labels.compute(),
            )
            for roi_id, candidate in self.candidates.items()
        }
        snapshot = replace(self, candidates=candidates)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
        try:
            joblib.dump(snapshot, temp_path, compress=3)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a trusted snapshot. Joblib files can execute arbitrary code when loaded."""
        results = joblib.load(path)
        if not isinstance(results, cls):
            raise TypeError("File does not contain Escore extraction results.")
        return results

    def get_candidates(self, region_id: int) -> tuple[EchotypeCandidate, ...]:
        """Get echotype candidate(s) for a given region id."""
        match: list = []
        for c in self.candidates.values():
            if c.roi_id == region_id:
                match.append(c)
        return tuple(match)
