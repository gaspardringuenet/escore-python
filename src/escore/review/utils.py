from ..extraction.data import EchotypeCandidate, ExtractionResults
from .decisions import ReviewDecisions


def get_candidate(results: ExtractionResults, roi_id: int) -> EchotypeCandidate:
    if roi_id not in results.candidates:
        raise ValueError(f"ROI {roi_id} has no extracted candidate.")
    return results.candidates[roi_id]


def check_review(results: ExtractionResults, decisions: ReviewDecisions) -> None:
    if decisions.extraction_id != results.extraction_id:
        raise ValueError("Review decisions belong to a different extraction result.")
    for decision in decisions.decisions:
        candidate = get_candidate(results, decision.roi_id)
        if decision.cluster_id is not None and decision.cluster_id not in candidate.cluster_ids:
            raise ValueError(
                f"Cluster {decision.cluster_id} was not fitted for ROI {decision.roi_id}."
            )
