from pathlib import Path
from uuid import uuid4

import echoregions as er
import numpy as np
import pandas as pd
import xarray as xr
from echoregions.regions2d import Regions2D
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from ._masking import MaskingError, select_in_region
from ._processing import (
    convolve_sv,
    filter_labels_by_confidence,
    find_best_cluster,
    fit_best_gmm,
    get_labels,
    labels_to_grid,
)
from .config import ExtractionConfig
from .data import EchotypeCandidate, ExtractionFailure, ExtractionResults


def _validate_sv(sv: xr.DataArray, config: ExtractionConfig) -> xr.DataArray:
    if not isinstance(sv, xr.DataArray):
        raise TypeError("sv must be an xarray.DataArray.")
    if set(sv.dims) != {"channel", "ping_time", "depth"}:
        raise ValueError("sv must have exactly the channel, ping_time and depth dimensions.")
    if not np.issubdtype(sv.dtype, np.number) or np.issubdtype(sv.dtype, np.complexfloating):
        raise ValueError("sv must contain real numeric values.")
    for dim in ("channel", "ping_time", "depth"):
        if dim not in sv.indexes or not sv.indexes[dim].is_unique or sv.sizes[dim] == 0:
            raise ValueError(f"sv requires a non-empty, unique {dim} coordinate.")
    for dim in ("ping_time", "depth"):
        if not sv.indexes[dim].is_monotonic_increasing:
            raise ValueError(f"sv {dim} must be sorted in increasing order.")
    missing = set(config.channels) - set(sv.indexes["channel"])
    if missing:
        raise ValueError(f"Channels not found in sv: {sorted(missing)}")
    return sv.sel(channel=list(config.channels)).transpose("channel", "ping_time", "depth")


def _cluster_candidate(
    roi_id: int,
    metadata: pd.Series,
    raw: xr.DataArray,
    config: ExtractionConfig,
) -> EchotypeCandidate:
    if config.min_sv is not None:
        raw = raw.where((raw >= config.min_sv).all("channel"))
    if config.max_sv is not None:
        raw = raw.where((raw <= config.max_sv).all("channel"))
    smoothed = raw if config.kernel is None else convolve_sv(raw, config.kernel)
    if config.clustering_features == "adjacent":
        order = config.clustering_channels or config.channels
        delta = smoothed.sel(channel=list(order)).diff("channel")
    else:
        delta = smoothed.isel(channel=slice(1, None)) - smoothed.isel(channel=0, drop=True)
    stacked = delta.stack(pixel=("ping_time", "depth")).transpose("pixel", "channel")
    data = np.asarray(stacked.values)

    # check validity along the feature axis (1D case: data is (n_samples,))
    valid = np.isfinite(data).all(axis=1) if data.ndim == 2 else np.isfinite(data)

    if not np.any(valid):
        raise ValueError("ROI has no pixels with finite values in every selected channel.")

    if config.clusterer == "gmm":
        model = fit_best_gmm(
            data,
            "2D",
            n_components_range=config.gmm_components,
            random_state=config.random_state,
        )
    else:
        model = KMeans(
            n_clusters=config.kmeans_clusters,
            n_init=config.kmeans_n_init,  # type: ignore
            random_state=config.random_state,
        ).fit(data[valid])
    labels = get_labels(data, model)
    proposed_cluster, _ = find_best_cluster(labels, data)
    filtered = labels
    if isinstance(model, GaussianMixture) and config.confidence_threshold is not None:
        filtered = filter_labels_by_confidence(data, labels, model, config.confidence_threshold)
    return EchotypeCandidate(
        roi_id=roi_id,
        metadata=metadata.copy(),
        raw_sv=raw,
        smoothed_sv=smoothed,
        labels=labels_to_grid(stacked, labels),
        filtered_labels=labels_to_grid(stacked, filtered),
        model=model,
        proposed_cluster=proposed_cluster,
    )


def extract_echotypes(
    sv: xr.DataArray,
    regions: Regions2D | str | Path,
    *,
    config: ExtractionConfig,
) -> ExtractionResults:
    """Extract candidates; report expected ROI masking and clustering failures separately.

    Parameters
    ----------
    sv : DataArray
        Volume backscattering data.
    regions : Regions2D or str or Path
        Selected regions of interest. Either directly a Regions2D object, or the path
        (as str or Path) to a Regions2D-compatible CSV file.
    config : ExtractionConfig
        Extraction settings to use.

    Returns
    -------
    ExtractionResults
        Object containing both successes (EchotypeCandidate's) and failures (ExtractionFailure's)
        for this extraction run.
    """
    sv = _validate_sv(sv, config)
    if isinstance(regions, (str, Path)):
        regions = er.read_regions_csv(regions)
    if not isinstance(regions, Regions2D):
        raise TypeError("regions must be a Regions2D or a CSV path.")
    if "region_id" not in regions.data:
        raise ValueError("regions must contain region_id.")
    ids = regions.data.loc[:, "region_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("region_id values must be non-null and unique.")
    if any(
        isinstance(i, (bool, np.bool_))
        or not isinstance(i, (int, float, np.integer, np.floating))
        or not np.isfinite(i)
        or int(i) != i
        for i in ids
    ):
        raise ValueError("region_id values must be integers.")

    candidates: dict[int, EchotypeCandidate] = {}
    failures: dict[int, ExtractionFailure] = {}
    for _, metadata in tqdm(
        regions.data.iterrows(), desc="Processing ROIs", total=len(regions.data)
    ):
        roi_id = int(metadata.at["region_id"])
        try:
            raw = select_in_region(sv, regions, metadata, roi_id)
            raw.load()
        except MaskingError as exc:
            failures[roi_id] = ExtractionFailure(roi_id, "mask", str(exc))
            continue
        try:
            candidates[roi_id] = _cluster_candidate(roi_id, metadata, raw, config)
        except ValueError as exc:
            failures[roi_id] = ExtractionFailure(roi_id, "cluster", str(exc))
    return ExtractionResults(str(uuid4()), config, candidates, failures)
