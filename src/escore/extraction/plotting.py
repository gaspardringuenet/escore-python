"""Matplotlib review reports for notebooks and scripts."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from echograms import RGB, Encoding, OKLab, echogram
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture

from ..review.diagnostics import assess_selection
from ..review.utils import get_candidate
from ._processing import fit_best_gmm
from .data import ExtractionResults


def _features(sv: xr.DataArray) -> np.ndarray:
    values = np.asarray(sv.transpose("ping_time", "depth", "channel").values)
    return (values[..., 1:] - values[..., :1]).reshape(-1, values.shape[-1] - 1)


def _echogram(ax: Axes, sv: xr.DataArray, title: str, enc: Encoding) -> None:
    sv = sv.isel(channel=slice(3))
    # Pixels with any missing channel are left blank
    sv = sv.where(sv.notnull().all("channel"))
    if sv.sizes["channel"] == 2:
        sv = sv.pad(channel=(0, 1), constant_values=np.nan)
    echogram(sv, enc, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Depth")
    ax.tick_params(axis="x", labelrotation=25)


def _overlay_gmm_1d(
    ax: Axes, gmm: GaussianMixture, X: np.ndarray, delta_range: tuple[float, float] | None
) -> None:
    lo, hi = delta_range if delta_range is not None else (X.min(), X.max())
    x = np.linspace(lo, hi, 200)
    bin_width = (hi - lo) / 40  # scales the density to counts per histogram bin
    for k, (mu, var, weight) in enumerate(  # type: ignore
        zip(gmm.means_[:, 0], gmm.covariances_.reshape(-1), gmm.weights_)  # type: ignore
    ):
        pdf = np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
        ax.plot(x, len(X) * weight * bin_width * pdf, color=f"C{k + 1}", label=f"GMM #{k + 1}")
    ax.legend(loc="upper left")


def _overlay_gmm_2d(ax: Axes, gmm: GaussianMixture) -> None:
    radius = np.sqrt(chi2.ppf(0.95, df=2))  # 95% region of a 2D Gaussian
    for k, (mu, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):  # type: ignore
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        width, height = 2 * radius * np.sqrt(vals[::-1])
        ax.add_patch(
            Ellipse(
                mu,
                width,
                height,
                angle=angle,
                fill=False,
                edgecolor=f"C{k + 1}",
                linewidth=2,
                label=f"GMM #{k + 1} 95% CI",
            )
        )
    ax.legend(loc="upper left")


def _distribution(
    ax: Axes,
    data: np.ndarray,
    features: tuple[int, ...],
    delta_range: tuple[float, float] | None,
    labels: np.ndarray | None = None,
    selected: int | None = None,
    show_gmm: bool = False,
) -> None:
    valid = np.isfinite(data).all(axis=1)
    overlay = None
    if show_gmm and labels is None and np.sum(valid) >= 10:
        X = data[valid][:, list(features)]
        overlay = (fit_best_gmm(X, "1D" if len(features) == 1 else "2D"), X)
    if labels is None and len(features) == 2:
        if np.any(valid):
            *_, quad = ax.hist2d(
                data[valid, features[0]],
                data[valid, features[1]],
                bins=40,
                range=None if delta_range is None else [delta_range, delta_range],
                cmin=1,
            )
            ax.figure.colorbar(quad, ax=ax, label="#(Pixels)")
    else:
        if labels is None:
            groups = [(None, valid)]
        else:
            groups = [(int(label), valid & (labels == label)) for label in np.unique(labels[valid])]
        for label, mask in groups:
            if not np.any(mask):
                continue
            name = (
                None
                if label is None
                else (
                    "low confidence"
                    if label < 0
                    else f"cluster {label}" + (" (selected)" if label == selected else "")
                )
            )
            color = None if label is None else ("gray" if label < 0 else f"C{label}")
            if len(features) == 1:
                ax.hist(
                    data[mask, features[0]],
                    bins=40,
                    range=delta_range,
                    alpha=0.4 if label is not None and label < 0 else 0.6,
                    label=name,
                    color=color,
                )
            else:
                ax.scatter(
                    data[mask, features[0]],
                    data[mask, features[1]],
                    s=2,
                    alpha=0.3,
                    label=name,
                    color=color,
                    rasterized=True,
                )
        if labels is not None and groups:
            ax.legend(loc="upper left", markerscale=4)
    if delta_range is not None:
        ax.set_xlim(*delta_range)
        if len(features) == 2:
            ax.set_ylim(*delta_range)
    if len(features) == 2:
        ax.set_aspect("equal")
    if overlay is not None:
        gmm, X = overlay
        if len(features) == 1:
            _overlay_gmm_1d(ax, gmm, X, delta_range)
        else:
            _overlay_gmm_2d(ax, gmm)
    ax.set_axisbelow(True)
    ax.grid(color="gray", linestyle="dashed", alpha=0.5)


def plot_review(
    results: ExtractionResults,
    roi_id: int,
    *,
    cluster_id: int,
    feature_indices: tuple[int, ...] | None = None,
    sv_range: tuple[float, float] = (-90.0, -50.0),
    delta_range: tuple[float, float] | None = (-20.0, 20.0),
    rgb_encoder: Encoding | None = None,
) -> Figure:
    """Plot seven inspection views without refitting the candidate clustering.

    The ungrouped ΔMVBS panels also show a best-BIC GMM fitted to their own pixels.
    At most two ΔMVBS features and three Sv channels are displayed. Additional
    ΔMVBS features can be inspected with ``feature_indices``.
    """
    candidate = get_candidate(results, roi_id)
    diagnostics = assess_selection(results, roi_id, cluster_id)
    if rgb_encoder is None:
        rgb_encoder = (
            OKLab()
            if len(results.config.channels) >= 3
            else RGB(vmin=sv_range[0], vmax=sv_range[1])
        )
    n_features = len(results.config.channels) - 1
    features = feature_indices if feature_indices is not None else tuple(range(min(n_features, 2)))
    if (
        len(features) not in (1, 2)
        or len(set(features)) != len(features)
        or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n_features
            for i in features
        )
    ):
        raise ValueError("feature_indices must contain one or two distinct valid feature indices.")
    if len(sv_range) != 2 or not np.isfinite(sv_range).all() or sv_range[0] >= sv_range[1]:
        raise ValueError("sv_range must contain finite increasing bounds.")
    if delta_range is not None and (
        len(delta_range) != 2
        or not np.isfinite(delta_range).all()
        or delta_range[0] >= delta_range[1]
    ):
        raise ValueError("delta_range must contain finite increasing bounds or be None.")

    raw = _features(candidate.raw_sv)
    smoothed = _features(candidate.smoothed_sv)
    selected = _features(candidate.selected_sv(cluster_id))
    labels = candidate.filtered_labels.transpose("ping_time", "depth").values.ravel()
    fig, grid = plt.subplots(2, 4, figsize=(18, 8), layout="constrained")
    axes = grid.ravel()
    _echogram(axes[0], candidate.raw_sv, "Raw echogram", rgb_encoder)
    _distribution(axes[1], raw, features, delta_range, show_gmm=True)
    axes[1].set_title("Raw ΔMVBS")
    _echogram(axes[2], candidate.smoothed_sv, "Smoothed echogram", rgb_encoder)
    _distribution(axes[3], smoothed, features, delta_range, show_gmm=True)
    axes[3].set_title("Smoothed ΔMVBS")
    _distribution(axes[4], smoothed, features, delta_range, labels=labels, selected=cluster_id)
    axes[4].set_title("Smoothed clusters")
    _distribution(axes[5], selected, features, delta_range, show_gmm=True)
    axes[5].set_title("Selected raw ΔMVBS")
    _echogram(axes[6], candidate.selected_sv(cluster_id), "Selected raw pixels", rgb_encoder)
    axes[7].axis("off")
    names = [f"{c} - {results.config.channels[0]}" for c in results.config.channels[1:]]
    for ax in (axes[1], axes[3], axes[4], axes[5]):
        ax.set_xlabel(f"ΔMVBS {names[features[0]]} [dB]")
        ax.set_ylabel(f"ΔMVBS {names[features[1]]} [dB]" if len(features) == 2 else "Pixels")
    displayed = ", ".join([str(c) for c in results.config.channels[:3]])
    fig.suptitle(
        f"ROI {roi_id}, cluster {cluster_id}: {diagnostics.retained_pixels} pixels "
        f"({diagnostics.retained_share:.1%} retained); displayed channels: {displayed}"
    )
    return fig
