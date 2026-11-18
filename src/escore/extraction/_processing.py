"""Numerical operations shared by extraction and the existing pipeline."""

from typing import Literal

import numpy as np
import xarray as xr
from scipy.signal import convolve2d
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def convolve_sv(da_Sv: xr.DataArray, kernel: np.ndarray) -> xr.DataArray:
    """Smooth Sv in the linear domain, preserving the original NaN mask."""
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 2:
        raise ValueError("conv_kernel must be 2D, with axes (ping_time, depth).")

    def _smooth(sv_db: np.ndarray) -> np.ndarray:
        valid = np.isfinite(sv_db)
        sv_lin = np.where(valid, 10 ** (sv_db / 10), 0.0)
        num = convolve2d(sv_lin, kernel, mode="same", boundary="fill")
        den = convolve2d(valid.astype(float), kernel, mode="same", boundary="fill")
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 10 * np.log10(num / den)
        return np.where(valid, out, np.nan)

    return xr.apply_ufunc(
        _smooth,
        da_Sv,
        input_core_dims=[["ping_time", "depth"]],
        output_core_dims=[["ping_time", "depth"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[float],
    )


def stack_data(
    da_Sv: xr.DataArray,
    *,
    mode: Literal["1D", "2D"],
) -> tuple[xr.DataArray, np.ndarray]:
    if mode == "1D":
        if da_Sv.sizes["channel"] != 2:
            raise ValueError('mode="1D" requires exactly two channels in order (ref, f1)')
        delta_Sv = da_Sv.isel(channel=1) - da_Sv.isel(channel=0)
    elif mode == "2D":
        if da_Sv.sizes["channel"] != 3:
            raise ValueError('mode="2D" requires exactly three channels in order (ref, f1, f2)')
        delta_Sv = da_Sv.drop_isel(channel=0) - da_Sv.isel(channel=0)
    else:
        raise ValueError("Invalid mode.")

    stacked = delta_Sv.stack(pixel=("ping_time", "depth"))
    if mode == "2D":
        stacked = stacked.transpose("pixel", "channel")
    return stacked, stacked.values


def fit_best_gmm(
    data: np.ndarray,
    mode: Literal["1D", "2D"],
    n_components_range: tuple[int, ...] = (1, 2, 3),
    random_state: int = 0,
) -> GaussianMixture:
    """Fit GMMs for each n_components and keep the one with the lowest BIC."""
    X = data.reshape(-1, 1) if mode == "1D" else data
    X = X[np.isfinite(X).all(axis=1)]

    best: GaussianMixture | None = None
    best_bic = np.inf
    for n in n_components_range:
        if len(X) < max(n, 2):
            continue
        gmm = GaussianMixture(n_components=n, random_state=random_state).fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best, best_bic = gmm, bic
    if best is None:
        raise ValueError("Not enough valid data to fit a GaussianMixture.")
    return best


def fit_model(
    data: np.ndarray, mode: Literal["1D", "2D"], model: KMeans | GaussianMixture
) -> KMeans | GaussianMixture:
    """Fit a clustering model on the finite pixels (GMM: n_components chosen by BIC)."""
    if isinstance(model, GaussianMixture):
        return fit_best_gmm(data, mode)
    if isinstance(model, KMeans):
        X = data.reshape(len(data), -1)
        return clone(model).fit(X[np.isfinite(X).all(axis=1)])  # type: ignore
    raise TypeError("model must be a GaussianMixture or a KMeans instance.")


def find_best_cluster(
    labels: np.ndarray, data: np.ndarray, override: int | None = None
) -> tuple[int, float]:
    """Return the cluster id and its share of labelled pixels.

    The largest cluster is selected, unless the second largest is within 30% of its
    pixel count; then the one whose mean ΔMVBS lies furthest from zero is selected.
    """
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(ids) == 0:
        raise ValueError("No labelled pixels.")
    if override is not None:
        cid = int(override)
    else:
        order = np.argsort(-counts, kind="stable")
        top = order[0]
        if len(order) > 1 and counts[top] - counts[order[1]] < 0.3 * counts[top]:
            tied = [int(ids[top]), int(ids[order[1]])]
            cid = max(tied, key=lambda k: _distance_from_zero(data, labels, k))
        else:
            cid = int(ids[top])
    share = float(counts[ids == cid].sum() / counts.sum())
    return cid, share


def _distance_from_zero(data: np.ndarray, labels: np.ndarray, cluster_id: int) -> float:
    return float(np.linalg.norm(data[labels == cluster_id].mean(axis=0)))


def get_labels(data: np.ndarray, model: KMeans | GaussianMixture) -> np.ndarray:
    """Predict labels for each pixel; -1 marks non-finite values."""
    X = data.reshape(len(data), -1)
    valid = np.isfinite(X).all(axis=1)
    labels = np.full(len(X), -1, dtype=int)
    labels[valid] = model.predict(X[valid])
    return labels


def filter_labels_by_confidence(
    data: np.ndarray, labels: np.ndarray, model: GaussianMixture, threshold: float
) -> np.ndarray:
    """Mark assignments below the GMM posterior threshold as unselected (-1)."""
    X = data.reshape(len(data), -1)
    valid = (labels >= 0) & np.isfinite(X).all(axis=1)
    filtered = labels.copy()
    if np.any(valid):
        probabilities = model.predict_proba(X[valid])
        assigned_probability = probabilities[np.arange(valid.sum()), labels[valid]]
        valid_indices = np.flatnonzero(valid)
        filtered[valid_indices[assigned_probability < threshold]] = -1
    return filtered


def labels_to_grid(stacked: xr.DataArray, labels: np.ndarray) -> xr.DataArray:
    """Map per-pixel labels back onto the (depth, ping_time) grid."""
    da = xr.DataArray(labels, dims="pixel", coords={"pixel": stacked["pixel"]})
    return da.unstack("pixel").transpose("depth", "ping_time")
