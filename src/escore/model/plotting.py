"""Plots for fitted echoclasses and classified Sv."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Ellipse

from .classifier import EscoreClassifier


def plot_classes(model: EscoreClassifier, means: pd.DataFrame, *, features=(0, 1)):
    """Plot classified means and two-standard-deviation covariance contours in 2D or 3D."""
    model._check_fitted()
    if (
        len(features) not in (2, 3)
        or len(set(features)) != len(features)
        or any(
            not isinstance(i, int) or i < 0 or i >= len(model.feature_names_in_) for i in features
        )
    ):
        raise ValueError("features must contain two or three distinct valid feature indices.")
    from .classifier import _check_values

    data = _check_values(means, model.feature_names_in_)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d" if len(features) == 3 else None)
    labels = model.predict(means)
    for k in range(model.n_classes):
        ax.scatter(
            *(data[labels == k, i] for i in features), s=15, color=f"C{k}", label=f"class {k}"
        )
        cov = model.covariances_[k][np.ix_(features, features)]
        vals, vecs = np.linalg.eigh(cov)
        if len(features) == 2:
            major = vecs[:, -1]
            ax.add_patch(
                Ellipse(
                    model.means_[k, list(features)],
                    4 * np.sqrt(vals[-1]),
                    4 * np.sqrt(vals[0]),
                    angle=np.degrees(np.arctan2(major[1], major[0])),
                    facecolor="none",
                    edgecolor=f"C{k}",
                )
            )
        else:
            u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 15))
            sphere = np.stack((np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)), axis=-1)
            surface = (sphere * (2 * np.sqrt(vals))) @ vecs.T + model.means_[k, list(features)]
            ax.plot_wireframe(
                surface[..., 0],
                surface[..., 1],
                surface[..., 2],
                color=f"C{k}",
                alpha=0.25,
                linewidth=0.5,
            )
    if np.any(labels == -1):
        ax.scatter(
            *(data[labels == -1, i] for i in features), s=15, color="gray", label="unclassified"
        )
    ax.set_xlabel(model.feature_names_in_[features[0]])
    ax.set_ylabel(model.feature_names_in_[features[1]])
    if len(features) == 3:
        ax.set_zlabel(model.feature_names_in_[features[2]])
    else:
        ax.set_aspect("equal")
    ax.legend()
    return fig


def plot_responses(model: EscoreClassifier):
    """Plot class mean relative-frequency responses; fit with channels first."""
    model._check_fitted()
    if model.channels_ is None:
        raise ValueError("Fit with channels to plot frequency responses.")
    fig, ax = plt.subplots()
    for k, mean in enumerate(model.means_):
        ax.plot([str(c) for c in model.channels_], np.r_[0, mean], marker="o", label=f"class {k}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean Sv difference from reference [dB]")
    ax.legend()
    return fig


def plot_classification(labels: xr.DataArray, *, class_id: int | None = None, ax=None):
    """Plot all labels, or the cells of a single class."""
    if labels.ndim != 2:
        raise ValueError("labels must be two-dimensional.")
    if ax is None:
        _, ax = plt.subplots()
    if class_id is not None:
        labels = labels.where(labels == class_id)
    labels.plot(ax=ax, cmap="tab20", add_colorbar=True)
    return ax


def plot_sensitivity(by_class: pd.DataFrame, *, weighted: bool = False):
    """Plot class-level correct, incorrect and unclassified threshold fractions."""
    suffix = "_weight_fraction" if weighted else "_fraction"
    columns = [name + suffix for name in ("correct", "incorrect", "unclassified")]
    if not {"threshold", "expected_class"}.issubset(by_class.index.names):
        raise ValueError("Expected sensitivity output indexed by threshold and expected_class.")
    if not set(columns).issubset(by_class.columns):
        raise ValueError("Requested fractions are not present in the sensitivity output.")
    classes = by_class.index.get_level_values("expected_class").unique()
    fig, axes = plt.subplots(1, len(classes), squeeze=False, figsize=(5 * len(classes), 4))
    for ax, class_id in zip(axes.ravel(), classes):
        group = by_class.xs(class_id, level="expected_class").sort_index()
        group[columns].plot(ax=ax)
        ax.set_title(f"class {class_id}")
        ax.set_xlabel("Squared-distance threshold")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1)
    return fig


def depth_summary(
    labels: xr.DataArray, bins, *, period: xr.DataArray | None = None
) -> pd.DataFrame:
    """Count class labels per depth bin and optional aligned day/night category."""
    if "depth" not in labels.dims or "depth" not in labels.coords:
        raise ValueError("labels must have a depth coordinate.")
    arrays = {"label": labels}
    if period is not None:
        if not set(period.dims).issubset(labels.dims):
            raise ValueError("period dimensions must be a subset of label dimensions.")
        period, _ = xr.align(period, labels, join="exact")
        arrays["period"] = period.broadcast_like(labels)
    frame = xr.Dataset(arrays).to_dataframe().reset_index()
    frame["depth_bin"] = pd.cut(frame["depth"], bins=bins, include_lowest=True)
    group = ["depth_bin"] + (["period"] if period is not None else []) + ["label"]
    return frame.groupby(group, observed=True, dropna=False).size().rename("cells").reset_index()


def plot_depth_summary(summary: pd.DataFrame):
    """Plot class cell counts against depth, optionally split by period."""
    required = {"depth_bin", "label", "cells"}
    if not required.issubset(summary.columns):
        raise ValueError("summary must be returned by depth_summary.")
    fig, ax = plt.subplots()
    groups = ["label"] + (["period"] if "period" in summary else [])
    for key, group in summary.dropna(subset=["depth_bin"]).groupby(groups, sort=True):
        depths = [interval.mid for interval in group["depth_bin"]]
        ax.plot(group["cells"], depths, marker="o", label=str(key))
    ax.invert_yaxis()
    ax.set_xlabel("Cells")
    ax.set_ylabel("Depth")
    ax.legend()
    return fig
