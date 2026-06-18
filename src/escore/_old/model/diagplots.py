from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure

from .transform import EscoreTransform


def make_grid(
    x_values: np.ndarray, y_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    xx, yy = np.meshgrid(np.array(x_values), np.array(y_values))
    grid = np.stack([xx, yy], axis=-1)

    return xx, yy, grid


def show_scores_2D(
    classes: np.ndarray,
    scores: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    ncols: int,
    figsize: Tuple[int, int],
    layout: str,
    cmap: str | Colormap | None,
    norm: str | Normalize | None,
) -> Tuple[Figure, Any]:

    # Classes subplots setup
    n_classes = len(classes)
    nrows = n_classes // ncols + (n_classes % ncols != 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout=layout)
    axes = np.atleast_1d(axes).flatten()

    for i, (ax, c) in enumerate(zip(axes, classes)):
        ax: Axes = ax

        mp = ax.pcolormesh(xx, yy, scores[..., i], cmap=cmap, norm=norm)
        plt.colorbar(mp, ax=ax)
        ax.set_title(f"Echoclass {c}")

    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    return fig, axes[:n_classes]  # return Figure + used Axes


def best_to_second_plot_2D(
    best: np.ndarray,
    second_best: np.ndarray,
    ratio: np.ndarray,
    trans: EscoreTransform,
    xx: np.ndarray,
    yy: np.ndarray,
    figsize: Tuple[int, int],
    layout: str,
    scores_cmap: str | Colormap | None,
    scores_norm: str | Normalize | None,
    ratio_cmap: str | Colormap | None,
    ratio_norm: str | Normalize | None,
) -> Tuple[Figure, Any]:
    fig, axes = plt.subplots(ncols=3, figsize=figsize, layout=layout, sharey=True)

    ax: Axes = axes[0]
    mp = ax.pcolormesh(xx, yy, best, cmap=scores_cmap, norm=scores_norm)
    plt.colorbar(mp, ax=ax)
    ax.set_title("Scores for best class")

    ax = axes[1]
    mp = ax.pcolormesh(xx, yy, second_best, cmap=scores_cmap, norm=scores_norm)
    plt.colorbar(mp, ax=ax)
    ax.set_title(r"Scores for $2^{\text{nd}}$ best")

    ax = axes[2]
    if ratio_norm is None:
        mp = ax.pcolormesh(xx, yy, ratio, cmap=ratio_cmap, vmin=0, vmax=1)
    else:
        mp = ax.pcolormesh(xx, yy, ratio, cmap=ratio_cmap, norm=ratio_norm)
    plt.colorbar(mp, ax=ax)

    title = (
        r"Ratio best / $2^{\text{nd}}$"
        if (trans.ratio_schema == "best / second")
        else r"Ratio $2^{\text{nd}}$ / best"
    )
    ax.set_title(title)

    return fig, axes


def boundaries_plot_2D(
    classes: Sequence[Any],
    results: dict[str, Any],
    xx: np.ndarray,
    yy: np.ndarray,
    figsize: Tuple[int, int],
    layout: str,
    cmap: str | Colormap | None,
    norm: str | Normalize | None,
) -> Tuple[Figure, Axes]:

    preds_idx = results["predicted_class_idx"]
    absolute_mask = results["absolute_threshold_mask"]
    relative_mask = results["relative_threshold_mask"]
    absolute_threshold = results["params"]["absolute_threshold"]
    relative_threshold = results["params"]["relative_threshold"]

    # Format results as numpy masked arrays
    preds_idx = np.ma.array(preds_idx, mask=np.isnan(preds_idx))
    relative_mask = np.ma.array(relative_mask, mask=~relative_mask)
    absolute_mask = np.ma.array(absolute_mask, mask=~absolute_mask)

    # Figure
    fig, ax = plt.subplots(figsize=figsize, layout=layout)

    # Plot relative threshold mask
    ax.pcolormesh(
        xx,
        yy,
        relative_mask,
        cmap="Greys_r",
        alpha=0.5,
    )

    # Plot absolute threshold mask on top of it (more important)
    ax.pcolormesh(
        xx,
        yy,
        absolute_mask,
        cmap="Greys_r",
    )

    # Handle cmap and norm
    n_classes = len(classes)
    if cmap is None:
        cmap = "tab10"
    if isinstance(cmap, str):
        cmap = ListedColormap(plt.get_cmap(cmap).colors[:n_classes])
    norm = BoundaryNorm(boundaries=np.arange(-0.5, n_classes + 0.5, 1), ncolors=n_classes)

    # Plot classes predictions (masked array means we only plot where the two masks are not)
    mp = ax.pcolormesh(xx, yy, preds_idx, cmap=cmap, norm=norm)

    cbar = plt.colorbar(mp, ticks=np.arange(len(classes)), ax=ax)
    cbar.set_ticklabels(classes)

    abs_str = f"{absolute_threshold:.1g}" if absolute_threshold is not None else "None"
    rel_str = f"{relative_threshold:.2g}" if relative_threshold is not None else "None"

    ax.set_title(f"Absolute threshold: {abs_str} | Relative threshold: {rel_str}")

    return fig, ax
