from dataclasses import dataclass, asdict
import numpy as np
from typing import Any, Literal, Callable, Sequence

from .compatibility import multiarrays


# ---- Transform functions (support xarrays) ----

@multiarrays(features_dim_action='keep')
def identity(X: np.ndarray, *_) -> np.ndarray:
    return X

@multiarrays(features_dim_action='keep')
def square_root(X: np.ndarray, *_) -> np.ndarray:
    return np.sqrt(X)

@multiarrays(features_dim_action='keep')
def log_likelihood(
    escore: np.ndarray,
    det_cov: np.ndarray,
    *_ 
) -> np.ndarray:
    """Compute log likelihood: log(P(x | Class))
    """
    
    n = det_cov.shape[-1]
    normalization = 1. / np.sqrt((2 * np.pi)**n * det_cov)      # broadcasted to (n_e,)

    return np.log(normalization) - 0.5 * escore                 # broadcasted to (..., n_e)

@multiarrays(features_dim_action='keep')
def log_posterior(
    escore: np.ndarray, 
    det_cov: np.ndarray,
    class_prior: np.ndarray | None = None,
    X_prior: np.ndarray | None = None
) -> np.ndarray:
    """Compute the log posteriors log(P(class | x)) using squared Mahalanobis distance.
    """
    
    # Default prior
    if X_prior is None:
        X_prior = np.ones(escore.shape)         # assume uniform distribution (wrong in practice but sufficient for prediction)
    if class_prior is None:
        class_prior = det_cov / det_cov.sum()   # prior = relative ellipsoid volume

    # Compute log likelihood: log(P(x | E=e))
    log_likelihood_array = log_likelihood(escore, det_cov)

    # Compute log posteriors: log(P(E=e | x)) = log(P(x | E=e)) + log(P(E=e)) - log(P(x))
    log_posterior = log_likelihood_array + np.log(class_prior) - np.log(X_prior)    

    return log_posterior

@multiarrays(features_dim_action='keep')
def posterior(
    escore: np.ndarray, 
    det_cov: np.ndarray,
    class_prior: np.ndarray | None = None,
    X_prior: np.ndarray | None = None
) -> np.ndarray:
    
    log_post = log_posterior(
        escore, 
        det_cov,
        class_prior,
        X_prior
    )

    return np.exp(log_post)



# ---- Transform Class ----

@dataclass
class EscoreTransform:
    func: Callable[[np.ndarray, Sequence[Any]], np.ndarray]
    better: Literal["high", "low"]
    negative: bool
    description: str
    default_thresholds: dict[str, float]

    @property
    def ratio_schema(self) -> str:
        if (self.better == "low" and not self.negative) or (self.better == "high" and self.negative):
            return "best / second"
        if (self.better == "high" and not self.negative) or (self.better == "low" and self.negative):
            return "second / best"
        
    def describe(self):
        return asdict(self)



# ---- Transform Dictionary (used by EscoreClassifier) ----

TRANSFORM_DICT: dict[str, EscoreTransform] = {
    "none": EscoreTransform(
        identity, 
        "low", 
        False,
        """Default Escore, equivalent to the squared Mahalanobis distance to the ellipses.
        Formula: Escore(x, class) = (x - μ)^T Σ^{-1} (x - μ) for each data point x, and class of
        covariance matrix Σ and mean μ.
        """,
        dict(absolute=30, relative=0.8)
    ), 
    "mahalanobis": EscoreTransform(
        square_root, 
        "low", 
        False,
        "Square root applied to the Escore, returning the Mahalanobis distance.",
        dict(absolute=5, relative=0.8)
    ), 
    "log_likelihood": EscoreTransform(
        log_likelihood, 
        "high", 
        True,
        """Log likelihood log(P(x | class)) for each data point x and class c.
        Each class if assumed to follow a Gaussian distribution parametrized by the
        ellipse's center and covariance matrix.
        """,
        dict(absolute=-12, relative=0.95)
    ), 
    "log_posterior": EscoreTransform(
        log_posterior, 
        "high", 
        True,
        """Log posterior log(P(class | x)) for each data point x and class c.
        Each class if assumed to follow a Gaussian distribution parametrized by the
        ellipse's center and covariance matrix.
        """,
        dict(absolute=-20, relative=0.9)
    ), 
    "posterior": EscoreTransform(
        posterior, 
        "high", 
        False,
        """Posterior P(class | x) for each data point x and class c.
        Each class if assumed to follow a Gaussian distribution parametrized by the
        ellipse's center and covariance matrix.
        """,
        dict(absolute=1e-8, relative=0.5)
    )
}