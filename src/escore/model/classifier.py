from typing import Any, Literal, Sequence, Tuple

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure

from .diagplots import best_to_second_plot_2D, boundaries_plot_2D, make_grid, show_scores_2D
from .transform import TRANSFORM_DICT, EscoreTransform
from .utils import *


class EscoreClassifier:
    def __init__(self, classes: Sequence, centers: np.ndarray, cov_matrices: np.ndarray):
        # TODO Add channels

        valid_shapes_on_init(classes, centers, cov_matrices)

        self.transforms_dict: dict[str, EscoreTransform] = TRANSFORM_DICT

        self.classes: Sequence = classes
        self.centers: np.ndarray = centers
        self.cov_matrices: np.ndarray = cov_matrices

        self.det_cov = np.array([np.linalg.det(cov) for cov in cov_matrices])

        self.results: dict[str, Any] | None = None

        self.diagnostics: DiagnosticPlotter = DiagnosticPlotter(self)

    def _create_mahalanobis_method(self, features_dim_name: str | None):
        """Factory to build xarray compatible mahalanobis distance function."""

        @multiarrays(
            in_features_dim_name=features_dim_name,
            features_dim_action="replace",
            out_features_dim_size=len(self.classes),
            out_features_dim_name="echoclass",
            out_features_coords=self.classes,
            output_var_name="score",
        )
        def _mahalanobis_squared_impl(
            X: np.ndarray,
        ) -> np.ndarray:

            n_classes = self.centers.shape[0]
            original_shape = X.shape[:-1]

            X_flat = X.reshape(-1, X.shape[-1])  # shape (n_samples, n_features)
            distances = np.zeros((X_flat.shape[0], n_classes))

            for c in range(n_classes):
                centered = X_flat - self.centers[c]
                cov_inv = np.linalg.inv(self.cov_matrices[c])
                distances[:, c] = np.sum(centered @ cov_inv * centered, axis=1)

            return distances.reshape(*original_shape, n_classes)

        return _mahalanobis_squared_impl

    def get_scores(
        self,
        X: np.ndarray | xr.DataArray,
        transform: Literal[
            "none", "mahalanobis", "log_likelihood", "log_posterior", "posterior"
        ] = "none",
        class_prior: np.ndarray | None = None,
        X_prior: np.ndarray | None = None,
        features_dim_name: str | None = None,
        compute_dask: bool = False,
    ) -> np.ndarray | xr.DataArray:

        # Compute default Escore (squared mahalanobis distance)
        scores_fn = self._create_mahalanobis_method(features_dim_name)
        scores: np.ndarray | xr.DataArray = scores_fn(X)

        # Fetch transform
        try:
            trans = self.transforms_dict[transform]
        except:
            raise ValueError(
                f"Invalid transform key. Valid keys: {list(self.transforms_dict.keys())}"
            )

        # Apply transform function and return
        transformed_scores = trans.func(scores, self.det_cov, class_prior, X_prior)

        if compute_dask and hasattr(transformed_scores, "compute"):
            return transformed_scores.compute()
        else:
            return transformed_scores

    def predict(
        self,
        X: np.ndarray | xr.DataArray,
        transform: Literal[
            "none", "mahalanobis", "log_likelihood", "log_posterior", "posterior"
        ] = "none",
        absolute_threshold: float | None = None,
        relative_threshold: float | None = None,
        class_prior: np.ndarray | None = None,
        X_prior: np.ndarray | None = None,
        features_dim_name: str | None = None,
        compute_dask: bool = False,
        store: bool = False,
    ) -> dict[str, Any]:

        # Log arguments
        args: dict = locals()
        args.pop("X")
        args["shape"] = X.shape

        # Get scores with desired transform (validates transform at the same time)
        scores: np.ndarray | xr.DataArray = self.get_scores(
            X, transform, class_prior, X_prior, features_dim_name
        )

        # Fetch transform from dict
        trans = self.transforms_dict[transform]

        # Compute interim data
        scores_sorted = sort_class_scores(scores)
        best = get_best_scores(scores_sorted, trans)
        ratio = get_best_ratio_01(scores, trans)

        # Compute absolute and relative masks
        absolute_mask = get_absolute_threshold_mask(best, absolute_threshold, trans)
        relative_mask = get_relative_threshold_mask(ratio, relative_threshold)
        combined_mask = absolute_mask | relative_mask

        # Predict
        preds_idx = get_predicted_class_idx(scores, trans)

        # Mask with np.nan (compatible with both numpy and xarray)
        preds_idx = xr.where(combined_mask, np.nan, preds_idx)

        # Compute if required + Dask
        if compute_dask and hasattr(preds_idx, "compute"):
            preds_idx = preds_idx.compute()
            absolute_mask = absolute_mask.compute()
            relative_mask = relative_mask.compute()

        # Format results dictionary
        results = {
            "predicted_class_idx": preds_idx,
            "absolute_threshold_mask": absolute_mask,
            "relative_threshold_mask": relative_mask,
            "class_labels": list(self.classes),
            "params": {"transform": trans.describe(), **args},
        }

        # Store if required
        self.results = results if store else None

        return results


class DiagnosticPlotter:
    def __init__(self, parent: EscoreClassifier):

        self.parent = parent

    def show_scores_2D(
        self,
        x_values: range | np.ndarray = range(-50, 50),
        y_values: range | np.ndarray = range(-50, 50),
        ncols: int = 4,
        figsize: Tuple[int, int] = (15, 5),
        layout: str = "tight",
        cmap: str | Colormap | None = None,
        norm: str | Normalize | None = None,
        **score_kwargs,
    ) -> Tuple[Figure, Any]:

        # Create 2D mesh grid
        xx, yy, grid = make_grid(np.array(x_values), np.array(y_values))

        # Compute escore with passed kwargs
        scores = self.parent.get_scores(grid, **score_kwargs)

        # Plot each class's score on grid
        fig, axes = show_scores_2D(
            self.parent.classes, scores, xx, yy, ncols, figsize, layout, cmap, norm
        )

        return fig, axes  # return Figure + used Axes

    def best_to_second_plot_2D(
        self,
        x_values: range | np.ndarray = range(-50, 50),
        y_values: range | np.ndarray = range(-50, 50),
        figsize: Tuple[int, int] = (15, 5),
        layout: str = "tight",
        scores_cmap: str | Colormap | None = None,
        scores_norm: str | Normalize | None = None,
        ratio_cmap: str | Colormap | None = "RdBu_r",
        ratio_norm: str | Normalize | None = None,
        **compute_scores_kwargs,
    ) -> Tuple[Figure, Any]:

        # Create 2D mesh grid
        xx, yy, grid = make_grid(np.array(x_values), np.array(y_values))

        # Compute escore with passed kwargs
        scores = self.parent.get_scores(grid, **compute_scores_kwargs)

        # Get transform argument if it was given
        transform = compute_scores_kwargs.get(
            "transform", "none"
        )  # if not given, .get_score will default to 'none'
        trans = self.parent.transforms_dict[transform]

        # Compute best and second best scores, as well as ratio of the two
        scores_sorted = sort_class_scores(scores)
        best = get_best_scores(scores_sorted, trans)
        second_best = get_second_best_scores(scores_sorted, trans)
        ratio = get_best_ratio_01(
            scores, trans
        )  # a bit inefficient but need to have np.ndarray -> np.ndarray signature for all funcs

        # Plot score of best class, score of second class, and ratio
        fig, axes = best_to_second_plot_2D(
            best,
            second_best,
            ratio,
            trans,
            xx,
            yy,
            figsize,
            layout,
            scores_cmap,
            scores_norm,
            ratio_cmap,
            ratio_norm,
        )

        return fig, axes

    def boundaries_plot_2D(
        self,
        x_values: range | np.ndarray = range(-50, 50),
        y_values: range | np.ndarray = range(-50, 50),
        figsize: Tuple[int, int] = (7, 5),
        layout: str = "tight",
        cmap: str | Colormap | None = None,
        norm: str | Normalize | None = None,
        **predict_kwargs,
    ) -> Tuple[Figure, Axes, dict[str, Any]]:

        # Create 2D mesh grid
        xx, yy, grid = make_grid(np.array(x_values), np.array(y_values))

        # Compute escore with passed kwargs
        results = self.parent.predict(grid, **predict_kwargs)

        # Plot prediction boundaries
        fig, ax = boundaries_plot_2D(
            self.parent.classes, results, xx, yy, figsize, layout, cmap, norm
        )

        return fig, ax, results
