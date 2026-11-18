from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..extraction.data import ExtractionResults
from .decisions import ReviewDecision, ReviewDecisions
from .utils import check_review, get_candidate


@dataclass(frozen=True)
class SelectionDiagnostics:
    roi_id: int
    cluster_id: int
    valid_pixels: int
    retained_pixels: int
    retained_share: float
    mean_delta: pd.Series
    std_delta: pd.Series
    flags: tuple[str, ...]


def assess_selection(
    results: ExtractionResults,
    roi_id: int,
    cluster_id: int,
    *,
    min_share: float | None = None,
) -> SelectionDiagnostics:
    """Compute descriptive diagnostics from the selected raw pixels."""
    if min_share is not None and (
        isinstance(min_share, bool)
        or not isinstance(min_share, (int, float))
        or not 0 <= min_share <= 1
    ):
        raise ValueError("min_share must be between 0 and 1.")
    candidate = get_candidate(results, roi_id)
    selected = candidate.selected_sv(cluster_id).transpose("ping_time", "depth", "channel")
    values = np.asarray(selected.values).reshape(-1, selected.sizes["channel"])
    values = values[np.isfinite(values).all(axis=1)]
    delta = values[:, 1:] - values[:, :1]
    feature_names = tuple(
        f"{channel} - {results.config.channels[0]}" for channel in results.config.channels[1:]
    )
    count = len(delta)
    valid = int(np.count_nonzero(candidate.labels.values >= 0))
    share = count / valid
    means = delta.mean(axis=0) if count else np.full(len(feature_names), np.nan)
    std = delta.std(axis=0, ddof=1) if count > 1 else np.full(len(feature_names), np.nan)
    flags: list[str] = []
    if count == 0:
        flags.append("empty_selection")
    elif count == 1:
        flags.append("insufficient_pixels")
    if min_share is not None and share < min_share:
        flags.append("low_share")
    return SelectionDiagnostics(
        roi_id,
        cluster_id,
        valid,
        count,
        share,
        pd.Series(means, index=feature_names, name="mean_delta"),
        pd.Series(std, index=feature_names, name="std_delta"),
        tuple(flags),
    )


def summarize_candidates(
    results: ExtractionResults,
    decisions: ReviewDecisions | None = None,
    *,
    min_share: float | None = None,
) -> pd.DataFrame:
    """Return one row per review selection, including extraction failures."""
    if decisions is not None:
        check_review(results, decisions)
    by_roi: dict[int, list[ReviewDecision]] = {}
    if decisions is not None:
        for decision in decisions.decisions:
            by_roi.setdefault(decision.roi_id, []).append(decision)
    rows = []
    for roi_id, candidate in results.candidates.items():
        selections = by_roi.get(roi_id) or [
            ReviewDecision(roi_id, candidate.proposed_cluster, "pending")
        ]
        for decision in selections:
            cluster = decision.cluster_id
            diagnostic = (
                assess_selection(results, roi_id, cluster, min_share=min_share)
                if cluster is not None
                else None
            )
            rows.append(
                {
                    "roi_id": roi_id,
                    "cluster_id": cluster,
                    "status": decision.status,
                    "proposed_cluster": candidate.proposed_cluster,
                    "retained_pixels": diagnostic.retained_pixels if diagnostic else None,
                    "retained_share": diagnostic.retained_share if diagnostic else None,
                    "flags": diagnostic.flags if diagnostic else (),
                    "reason": decision.reason,
                    "failure": None,
                }
            )
    for roi_id, failure in results.failures.items():
        rows.append(
            {
                "roi_id": roi_id,
                "cluster_id": None,
                "status": "failed",
                "proposed_cluster": None,
                "retained_pixels": None,
                "retained_share": None,
                "flags": (),
                "reason": "",
                "failure": f"{failure.stage}: {failure.message}",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "roi_id",
            "cluster_id",
            "status",
            "proposed_cluster",
            "retained_pixels",
            "retained_share",
            "flags",
            "reason",
            "failure",
        ],
    )
