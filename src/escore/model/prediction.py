"""Coordinate-preserving classification of multifrequency Sv."""

import numpy as np
import xarray as xr

from .classifier import EscoreClassifier, _labels_for_distances


def classify_sv(
    model: EscoreClassifier,
    sv: xr.DataArray,
    *,
    channels: tuple[str, ...] | tuple[float, ...] | None = None,
    min_sv: float | None = None,
) -> xr.Dataset:
    """Classify Sv cells, using the reference channel followed by feature channels.

    Labels are zero-based; -1 is unclassified, -2 is missing, and -3 is below
    ``min_sv`` at every channel. Scores are squared Mahalanobis distances.
    ``best_squared_distance`` is the nearest distance before thresholding;
    ``assigned_squared_distance`` is the winning eligible distance or NaN.
    Dask-backed inputs remain lazy. Apply any desired smoothing before calling.
    """
    model._check_fitted()
    if not isinstance(sv, xr.DataArray) or "channel" not in sv.dims:
        raise ValueError("sv must be a DataArray with a channel dimension.")
    if "channel" not in sv.indexes or not sv.indexes["channel"].is_unique:
        raise ValueError("sv requires unique channel coordinates.")
    selected = channels if channels is not None else model.channels_
    if selected is None:
        raise ValueError("Provide channels in reference-first order, or fit with channels.")
    if len(selected) != len(model.feature_names_in_) + 1 or len(set(selected)) != len(selected):
        raise ValueError("channels must contain distinct reference and feature channels.")
    if model.channels_ is not None and tuple(selected) != model.channels_:
        raise ValueError("channels do not match those recorded by the model.")
    names = tuple(f"{channel} - {selected[0]}" for channel in selected[1:])
    if names != model.feature_names_in_:
        raise ValueError("channels do not match the model feature names and order.")
    if min_sv is not None and (isinstance(min_sv, bool) or not np.isfinite(min_sv)):
        raise ValueError("min_sv must be finite or None.")
    values = sv.sel(channel=list(selected))
    if "class" in values.dims:
        raise ValueError("sv cannot contain a dimension named 'class'.")
    thresholds = model._thresholds()
    means = model.means_
    precisions = model._precisions

    def classify(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        finite = np.isfinite(block).all(axis=-1)
        low = finite & (np.all(block < min_sv, axis=-1) if min_sv is not None else False)
        valid = finite & ~low
        delta = block[..., 1:] - block[..., :1]
        diff = delta[..., None, :] - means
        distances = np.einsum("...kd,kde,...ke->...k", diff, precisions, diff)
        distances = np.where(valid[..., None], distances, np.nan)
        eligible = np.where(distances < thresholds, distances, np.inf)
        labels = _labels_for_distances(distances, thresholds)
        labels = np.where(~finite, -2, np.where(low, -3, labels)).astype(np.int64)
        best = np.where(valid, np.min(distances, axis=-1), np.nan)
        assigned = np.where(labels >= 0, np.min(eligible, axis=-1), np.nan)
        return labels, best, assigned, distances

    labels, best, assigned, distances = xr.apply_ufunc(
        classify,
        values,
        input_core_dims=[["channel"]],
        output_core_dims=[[], [], [], ["class"]],
        dask="parallelized",
        output_dtypes=[np.int64, np.float64, np.float64, np.float64],
        dask_gufunc_kwargs={"output_sizes": {"class": model.n_classes}, "allow_rechunk": True},
    )
    distances = distances.assign_coords({"class": np.arange(model.n_classes)})
    result = xr.Dataset(
        {
            "label": labels,
            "best_squared_distance": best,
            "assigned_squared_distance": assigned,
            "squared_distance": distances,
        }
    )
    result.attrs.update(
        {
            "reference_channel": str(selected[0]),
            "channels": [str(c) for c in selected],
            "min_sv": "none" if min_sv is None else float(min_sv),
            "squared_distance_thresholds": [float(x) for x in thresholds],
            "threshold_boundary": "distance < threshold",
        }
    )
    return result
