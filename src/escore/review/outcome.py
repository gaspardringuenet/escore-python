from dataclasses import dataclass

import pandas as pd
import xarray as xr

from ..extraction.data import ExtractionFailure, ExtractionResults
from .decisions import ReviewDecision, ReviewDecisions
from .diagnostics import assess_selection
from .utils import check_review, get_candidate


@dataclass(frozen=True)
class Echotype:
    roi_id: int
    cluster_id: int
    raw_sv: xr.DataArray
    mask: xr.DataArray
    mean_delta: pd.Series
    decision: ReviewDecision
    region_data: pd.Series


@dataclass(frozen=True)
class EchotypeCollection:
    items: tuple[Echotype, ...]
    feature_names: tuple[str, ...]

    @property
    def means(self) -> pd.DataFrame:
        index = pd.MultiIndex.from_tuples(
            [(item.roi_id, item.cluster_id) for item in self.items],
            names=["roi_id", "cluster_id"],
        )
        return pd.DataFrame(
            [item.mean_delta for item in self.items],
            index=index,
            columns=self.feature_names,
        )

    @property
    def region_data(self) -> pd.DataFrame:
        """One row per echotype with its parent ROI's Regions2D.data. Missing columns are NaN."""
        index = pd.MultiIndex.from_tuples(
            [(item.roi_id, item.cluster_id) for item in self.items],
            names=["roi_id", "cluster_id"],
        )
        return pd.DataFrame([item.region_data for item in self.items], index=index)


@dataclass(frozen=True)
class ReviewOutcome:
    echotypes: EchotypeCollection
    pending: tuple[ReviewDecision, ...]
    rejected: tuple[ReviewDecision, ...]
    failures: dict[int, ExtractionFailure]


def apply_decisions(
    results: ExtractionResults,
    decisions: ReviewDecisions,
    *,
    allow_pending: bool = False,
) -> ReviewOutcome:
    """Materialize accepted raw-pixel echotypes after validating review decisions."""
    check_review(results, decisions)
    recorded_rois = {decision.roi_id for decision in decisions.decisions}
    pending = tuple(d for d in decisions.decisions if d.status == "pending") + tuple(
        ReviewDecision(roi_id, results.candidates[roi_id].proposed_cluster, "pending")
        for roi_id in results.candidates
        if roi_id not in recorded_rois
    )
    if pending and not allow_pending:
        raise ValueError(f"{len(pending)} review decision(s) are pending or missing.")
    items: list[Echotype] = []
    for decision in decisions.decisions:
        if decision.status != "accepted":
            continue
        assert decision.cluster_id is not None
        candidate = get_candidate(results, decision.roi_id)
        diagnostic = assess_selection(results, decision.roi_id, decision.cluster_id)
        if diagnostic.retained_pixels == 0:
            raise ValueError(f"ROI {decision.roi_id}, cluster {decision.cluster_id} is empty.")
        items.append(
            Echotype(
                decision.roi_id,
                decision.cluster_id,
                candidate.selected_sv(decision.cluster_id),
                candidate.mask(decision.cluster_id),
                diagnostic.mean_delta,
                decision,
                candidate.metadata.copy(),
            )
        )
    features = tuple(
        f"{channel} - {results.config.channels[0]}" for channel in results.config.channels[1:]
    )
    return ReviewOutcome(
        EchotypeCollection(tuple(items), features),
        pending,
        tuple(d for d in decisions.decisions if d.status == "rejected"),
        results.failures.copy(),
    )
