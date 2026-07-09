"""Rejection safe metrics for EscoreClassifier (safe for "rejected" classes -2 and -1)"""

import numpy as np
from sklearn.metrics import accuracy_score

UNSURE = -1  # > q_thresh
OUT_OF_SCOPE = -2  # < e_thresh
REJECTED = {OUT_OF_SCOPE, UNSURE}


# ---- Consider "unsure" and "out of scope" as "rejected" ---- #


def coverage(y_true, y_pred):
    """Fraction of samples that were classified (not rejected)"""
    mask = ~np.isin(y_pred, list(REJECTED))
    return mask.mean()


def classified_accuracy(y_true, y_pred):
    """Accuracy on non-rejected samples"""
    mask = ~np.isin(y_pred, list(REJECTED))
    if mask.sum() == 0:
        return 0
    return accuracy_score(y_true[mask], y_pred[mask])


def combined_score(y_true, y_pred, alpha=0.5):
    """Trade-off between accuracy and coverage. alpha is the weighting of accuracy."""
    return alpha * classified_accuracy(y_true, y_pred) + (1 - alpha) * coverage(y_true, y_pred)


# ---- Consider "out of scope" as "rejected" and "unsure" as a semi-good prediction ---- #


def coverage2(y_true, y_pred):
    """Fraction of samples that were classified (not rejected)"""
    mask = ~(y_pred == OUT_OF_SCOPE)
    return mask.mean()


def rejection_safe_accuracy(y_true, y_pred, unsure_sample_weight=0.5):
    """Accuracy with no checks on the allowed classes."""

    out_of_scope = y_pred == OUT_OF_SCOPE
    mask = ~out_of_scope

    if mask.sum() == 0:
        return 0

    y_true, y_pred = y_true[mask], y_pred[mask]

    n = len(y_true)
    correct = (y_true == y_pred).sum()
    unsure = (y_pred == UNSURE).sum()

    return (correct + unsure_sample_weight * unsure) / n


def combined_score2(y_true, y_pred, alpha=0.5, **kwrgs):
    """Trade-off between accuracy and coverage. alpha is the weighting of accuracy."""
    return alpha * rejection_safe_accuracy(y_true, y_pred, **kwrgs) + (1 - alpha) * coverage2(
        y_true, y_pred
    )
