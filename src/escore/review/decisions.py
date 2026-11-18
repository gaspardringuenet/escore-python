"""Review candidate clusters and materialize accepted echotypes."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Self

from ..extraction.data import ExtractionResults

ReviewStatus = Literal["pending", "accepted", "rejected"]


@dataclass(frozen=True)
class ReviewDecision:
    """A reviewer's decision for one cluster of an ROI, or for a whole ROI.

    Attributes
    ----------
    roi_id : int
        The region_id of the reviewed ROI.
    cluster_id : int or None
        Cluster being decided. None rejects the whole ROI.
    status : {"pending", "accepted", "rejected"}
        Current review state. Accepted decisions require a cluster_id, and
        whole-ROI decisions (cluster_id None) must be rejected.
    reason : str, default ""
        Free-text justification, typically given for rejections.
    overridden_flags : tuple of str, default ()
        Names of diagnostic flags (e.g. "low_share") the reviewer chose to override.
    """

    roi_id: int
    cluster_id: int | None
    status: ReviewStatus
    reason: str = ""
    overridden_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.roi_id, bool) or not isinstance(self.roi_id, int):
            raise TypeError("roi_id must be an integer.")
        if self.cluster_id is not None and (
            isinstance(self.cluster_id, bool) or not isinstance(self.cluster_id, int)
        ):
            raise TypeError("cluster_id must be an integer or None.")
        if self.status not in ("pending", "accepted", "rejected"):
            raise ValueError(f"Invalid review status: {self.status!r}.")
        if self.status == "accepted" and self.cluster_id is None:
            raise ValueError("An accepted decision requires a cluster_id.")
        if self.cluster_id is None and self.status != "rejected":
            raise ValueError("A whole-ROI decision must be rejected.")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string.")
        if isinstance(self.overridden_flags, str) or not all(
            isinstance(flag, str) and flag for flag in self.overridden_flags
        ):
            raise ValueError("overridden_flags must contain non-empty strings.")
        object.__setattr__(self, "overridden_flags", tuple(self.overridden_flags))


class ReviewDecisions:
    """Editable review decisions for one extraction run.

    Each ROI is either rejected as a whole or decided per cluster. Decisions start
    as pending proposals from :meth:`new`; an ROI without a decision is unresolved.
    Move them to accepted or rejected with :meth:`accept_roi`, :meth:`reject_roi`
    or :meth:`set`. Persist them between sessions with :meth:`save` and :meth:`load`.

    Attributes
    ----------
    extraction_id : str
        Identifier of the ExtractionResults being reviewed.
    """

    def __init__(self, extraction_id: str, decisions: tuple[ReviewDecision, ...] = ()) -> None:
        """Create review decisions for an extraction run.

        Parameters
        ----------
        extraction_id : str
            Identifier of the ExtractionResults being reviewed. Must be non-empty.
        decisions : tuple of ReviewDecision, default ()
            Initial decisions, usually from :meth:`load`. Each (roi_id, cluster_id)
            pair must be unique, and whole-ROI and cluster decisions cannot mix
            for the same ROI.

        Raises
        ------
        ValueError
            If ``extraction_id`` is empty, a decision is duplicated, or decisions conflict.
        """
        if not isinstance(extraction_id, str) or not extraction_id:
            raise ValueError("extraction_id must be a non-empty string.")
        self.extraction_id = extraction_id
        self._decisions: dict[tuple[int, int | None], ReviewDecision] = {}
        for decision in decisions:
            key = (decision.roi_id, decision.cluster_id)
            if key in self._decisions:
                raise ValueError(
                    f"Duplicate decision for ROI {decision.roi_id}, cluster {decision.cluster_id}."
                )
            self.set(decision)

    # --- Get all decisions --- #

    @property
    def decisions(self) -> tuple[ReviewDecision, ...]:
        """Snapshot of the current decisions.

        Each ROI is either rejected as a whole (``cluster_id`` None) or has cluster
        decisions. Change them only through ``set``, ``select``, ``remove``,
        ``reject_roi``, ``accept_roi`` and ``edit_roi``.
        """

        return tuple(self._decisions.values())

    # --- Creation & Persistence --- #

    @classmethod
    def new(cls, results: ExtractionResults) -> Self:
        """Create pending proposals for all successfully extracted ROIs.

        Each extracted ROI gets one pending decision on its proposed cluster. ROIs
        whose extraction failed get no decision.
        """
        return cls(
            results.extraction_id,
            tuple(
                ReviewDecision(roi_id, candidate.proposed_cluster, "pending")
                for roi_id, candidate in results.candidates.items()
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load review decisions saved by :meth:`save`.

        Raises
        ------
        ValueError
            If the file does not contain exactly ``extraction_id`` and ``decisions``.
        TypeError
            If ``decisions`` or one of its entries has the wrong type.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"extraction_id", "decisions"}:
            raise ValueError("Review file must contain extraction_id and decisions.")
        if not isinstance(payload["decisions"], list):
            raise TypeError("Review decisions must be a list.")
        decisions = []
        for entry in payload["decisions"]:
            if not isinstance(entry, dict):
                raise TypeError("Each review decision must be an object.")
            decisions.append(ReviewDecision(**entry))
        return cls(payload["extraction_id"], tuple(decisions))

    def save(self, path: str | Path) -> None:
        """Save decisions as JSON, replacing the file atomically."""
        payload = {
            "extraction_id": self.extraction_id,
            "decisions": [asdict(decision) for decision in self.decisions],
        }
        path = Path(path)
        text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            temp_path = Path(file.name)
            try:
                file.write(text)
            except OSError:
                temp_path.unlink(missing_ok=True)
                raise
        try:
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    # --- Make decisions --- #

    def set(self, decision: ReviewDecision) -> None:
        """Record a decision, replacing any decision with the same (roi_id, cluster_id).

        Use it to accept or reject a cluster directly, or to reject the whole ROI
        with ``cluster_id=None``. Other pending clusters of the same ROI are kept; :meth:`select` replaces them instead.

        Raises
        ------
        ValueError
            If the ROI already has a whole-ROI decision and this is a cluster
            decision, or the reverse.
        """
        if not isinstance(decision, ReviewDecision):
            raise TypeError("decision must be a ReviewDecision.")
        others = [
            item
            for key, item in self._decisions.items()
            if item.roi_id == decision.roi_id and key != (decision.roi_id, decision.cluster_id)
        ]
        if decision.cluster_id is None and others:
            raise ValueError("Remove cluster decisions before recording a whole-ROI decision.")
        if decision.cluster_id is not None and any(item.cluster_id is None for item in others):
            raise ValueError("Remove the whole-ROI decision before recording a cluster decision.")
        self._decisions[(decision.roi_id, decision.cluster_id)] = decision

    def select(
        self, roi_id: int, cluster_id: int, reason: str = "", *, append: bool = False
    ) -> None:
        """Select a cluster for an ROI as a pending decision.

        Removes the ROI's other pending decisions, including the proposal from
        :meth:`new`, and records a pending decision for ``cluster_id`` with
        ``reason``. Set ``append=True`` to retain other pending selections.
        The selection is not accepted until :meth:`accept_roi`.

        Raises
        ------
        ValueError
            If the ROI is rejected as a whole, or the cluster already has an accepted
            or rejected decision. Call :meth:`edit_roi` first to reopen it.
        """
        replacement = ReviewDecision(roi_id, cluster_id, "pending", reason)
        if (roi_id, None) in self._decisions:
            raise ValueError("Reopen the rejected ROI with edit_roi before selecting a cluster.")
        current = self._decisions.get((roi_id, cluster_id))
        if current is not None and current.status != "pending":
            raise ValueError("Remove the existing decision before selecting this cluster again.")
        if not append:
            for key, item in list(self._decisions.items()):
                if item.roi_id == roi_id and item.status == "pending":
                    del self._decisions[key]
        self.set(replacement)

    # --- Delete a decision --- #

    def remove(self, roi_id: int, cluster_id: int | None) -> None:
        """Delete the decision for ``(roi_id, cluster_id)``.

        Use ``cluster_id=None`` to delete a whole-ROI decision. Raises KeyError if
        there is no such decision.
        """
        del self._decisions[(roi_id, cluster_id)]

    # --- Changing status --- #

    def reject_roi(self, roi_id: int, reason: str = "") -> None:
        """Reject a whole ROI with ``reason``.

        Removes the ROI's pending and rejected decisions and records a rejected
        whole-ROI decision. Accepted cluster decisions must be removed first.

        Raises
        ------
        ValueError
            If the ROI has an accepted cluster decision.
        """
        replacement = ReviewDecision(roi_id, None, "rejected", reason)
        if any(d.roi_id == roi_id and d.status == "accepted" for d in self.decisions):
            raise ValueError("Remove accepted cluster decisions before rejecting an ROI.")
        for key in list(self._decisions):
            if key[0] == roi_id:
                del self._decisions[key]
        self.set(replacement)

    def accept_roi(self, roi_id: int) -> None:
        """Accept the pending cluster selections of an ROI.

        Each pending cluster decision of the ROI becomes accepted and keeps its
        reason and overridden flags. Calling it on an ROI that is already accepted
        does nothing.

        Raises
        ------
        ValueError
            If the ROI is rejected, or has neither a pending nor an accepted cluster selection.
        """
        if any(d.roi_id == roi_id and d.status == "rejected" for d in self.decisions):
            raise ValueError("Reopen the rejected ROI with edit_roi before accepting it.")
        pending = [
            d
            for d in self.decisions
            if d.roi_id == roi_id and d.cluster_id is not None and d.status == "pending"
        ]
        if not pending:
            if any(d.roi_id == roi_id and d.status == "accepted" for d in self.decisions):
                return
            raise ValueError(f"ROI {roi_id} has no pending cluster selection to accept.")
        for decision in pending:
            self._decisions[(roi_id, decision.cluster_id)] = replace(decision, status="accepted")

    def edit_roi(self, roi_id: int) -> None:
        """Reopen an ROI so its decisions can be changed again.

        A rejected ROI becomes unresolved: the rejection is removed and nothing is
        recorded in its place. Accepted clusters become pending selections, keeping
        their reasons and overridden flags. Does nothing for an ROI without accepted
        clusters or rejection.
        """
        if (roi_id, None) in self._decisions:
            del self._decisions[(roi_id, None)]
            return
        for key, item in list(self._decisions.items()):
            if item.roi_id == roi_id:
                self._decisions[key] = replace(item, status="pending")

    def summary(self, roi_id: int) -> None:
        """Print the decisions recorded for an ROI.

        Prints ``unresolved`` if the ROI has no decision, the rejection and its
        reason if it is rejected, or one line per cluster decision otherwise.
        """
        decisions = [d for d in self.decisions if d.roi_id == roi_id]
        if not decisions:
            print(f"ROI {roi_id}: unresolved")
            return
        if decisions[0].cluster_id is None:
            reason = f" - {decisions[0].reason}" if decisions[0].reason else ""
            print(f"ROI {roi_id}: rejected{reason}")
            return
        print(f"ROI {roi_id}:")
        for d in sorted(decisions, key=lambda d: d.cluster_id):  # type: ignore
            line = f"  cluster {d.cluster_id}: {d.status}"
            if d.reason:
                line += f" - {d.reason}"
            if d.overridden_flags:
                line += f" (overrides: {', '.join(d.overridden_flags)})"
            print(line)
