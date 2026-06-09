import numpy as np

from .transform import EscoreTransform
from .compatibility import multiarrays


def valid_shapes_on_init(classes, centers, cov_matrices) -> None:

    if not len({len(classes), centers.shape[0], cov_matrices.shape[0]}) == 1:
        raise ValueError("Number of classes mismatch between input parameters.")
    if not cov_matrices.ndim == 3 and cov_matrices.shape[1] == cov_matrices.shape[2]:
        raise ValueError("Covariance matrices array must contains square matrices.")
    if not centers.shape[1] == cov_matrices.shape[1]:
        raise ValueError("Features space dimensions mismatch between centers and cov_matrices.")
    
    return


@multiarrays(features_dim_action='keep')
def sort_class_scores(scores: np.ndarray) -> np.ndarray:
    return np.sort(scores, axis=-1)


@multiarrays(features_dim_action='drop')
def get_best_scores(scores_sorted: np.ndarray, trans: EscoreTransform) -> np.ndarray:
    match trans.better:
        case "low":
            return scores_sorted[..., 0]
        case "high":
            return scores_sorted[..., -1]


@multiarrays(features_dim_action='drop')
def get_second_best_scores(scores_sorted: np.ndarray, trans: EscoreTransform) -> np.ndarray:
    match trans.better:
        case "low":
            return scores_sorted[..., 1]
        case "high":
            return scores_sorted[..., -2]


@multiarrays(features_dim_action='drop')
def get_best_ratio_01(scores: np.ndarray, trans: EscoreTransform) -> np.ndarray:

    scores_sorted = sort_class_scores(scores)
    best = get_best_scores(scores_sorted, trans)
    second = get_second_best_scores(scores_sorted, trans)

    numerator, denumerator = (best, second) if trans.ratio_schema == "best / second" else (second, best)
    denumerator[denumerator == 0] = np.nan

    return numerator / denumerator


@multiarrays(features_dim_action='keep', output_var_name="absolute_threshold", dtype=bool)
def get_absolute_threshold_mask(best: np.ndarray, absolute_threshold: float | None, trans: EscoreTransform) -> np.ndarray:
    
    match trans.better:
        case "low":
            absolute_mask = best > absolute_threshold if absolute_threshold else np.zeros_like(best, dtype=bool)
        case "high":
            absolute_mask = best < absolute_threshold if absolute_threshold else np.zeros_like(best, dtype=bool)

    return absolute_mask


@multiarrays(features_dim_action='keep', output_var_name="relative_threshold", dtype=bool)
def get_relative_threshold_mask(ratio: np.ndarray, relative_threshold: float | None) -> np.ndarray:

    relative_mask = ratio > relative_threshold  if relative_threshold else np.zeros_like(ratio, dtype=bool)

    return relative_mask


@multiarrays(features_dim_action='drop', output_var_name="class_idx", dtype=float)
def get_predicted_class_idx(scores: np.ndarray, trans: EscoreTransform) -> np.ndarray :

    # get points without scores
    no_scores_mask = np.all(np.isnan(scores), axis=-1)

    match trans.better:
        case "low":
            preds = np.argmin(scores, axis=-1)
        case "high":
            preds = np.argmax(scores, axis=-1)
        
    preds = preds.astype(float)
    preds[no_scores_mask] = np.nan
    return preds