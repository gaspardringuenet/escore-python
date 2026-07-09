from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.typing import ColorType

from ..models.escore_classifier import EscoreClassifier


def escore_boundary_plot(
    model: EscoreClassifier,
    ax: Axes,
    grid_resolution: int = 200,
    xrange: Tuple[int, int] = (-40, 40),
    yrange: Tuple[int, int] = (-40, 40),
    classes_cmap: str = "tab10",
    e_thresh_color: ColorType = "black",
    q_thresh_color: ColorType = "grey",
    **kwrgs,
):

    # Create 2D mesh grid
    x_values, y_values = (
        np.linspace(*xrange, grid_resolution),
        np.linspace(*yrange, grid_resolution),
    )
    xx, yy = np.meshgrid(np.array(x_values), np.array(y_values))
    grid = np.stack([xx, yy], axis=-1)

    # Compute escore with passed kwargs
    grid_flat = grid.reshape(grid_resolution**2, 2)
    preds = model.predict(grid_flat)  # type: ignore
    preds_grid = preds.reshape(grid_resolution, grid_resolution)

    # cmap & norm
    n_classes = len(model.classes_)
    n_cat = n_classes + 2

    classes_colors = list(plt.get_cmap(classes_cmap).colors[:n_classes])  # type: ignore
    cmap = ListedColormap([e_thresh_color, q_thresh_color] + classes_colors)
    norm = BoundaryNorm(boundaries=np.arange(-2.5, n_classes + 0.5, 1), ncolors=n_cat)

    # Override defaults
    default_kwrgs = {"norm": norm, "cmap": cmap, "alpha": 0.5}
    kwrgs = default_kwrgs | kwrgs

    # Plot
    mp = ax.pcolormesh(xx, yy, preds_grid, **kwrgs)

    # Cbar
    cbar = plt.colorbar(mp, ax=ax)
    cbar.set_ticks(list(np.arange(-2, n_classes)))
    cbar.set_ticklabels(["< E thresh", "> Q thresh"] + list(model.classes_))

    ax.set_title("Escore decision boundary plot")
