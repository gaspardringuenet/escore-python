import numpy as np
from sklearn.metrics import accuracy_score

REJECTED = {-1, -2}


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
