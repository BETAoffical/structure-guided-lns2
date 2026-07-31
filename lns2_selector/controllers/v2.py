from __future__ import annotations

from typing import Any

from experiments.closed_loop_confirmation import score_online_candidates
from lns2_selector.runtime.contracts import (
    SelectionDecision,
    SelectionRequest,
)


class PairwiseV2Selector:
    """Adapter shared by the canonical and mixed V2 bundles."""

    def __init__(self, controller_id: str, bundle: Any):
        if controller_id not in {"v2-full", "mixed-full-v2"}:
            raise ValueError("unsupported pairwise V2 controller id")
        self.controller_id = controller_id
        self.bundle = bundle

    def select(self, request: SelectionRequest) -> SelectionDecision:
        if not request.candidates:
            return SelectionDecision(
                controller_id=self.controller_id,
                candidate_index=None,
                candidate=None,
                diagnostics={"route": "official_adaptive"},
                fallback_reason="no_candidates",
            )
        model = self.bundle.main_models["realized_dynamic"]
        index, scores, margin = score_online_candidates(
            list(map(dict, request.candidate_rows)), model
        )
        return SelectionDecision(
            controller_id=self.controller_id,
            candidate_index=index,
            candidate=request.candidates[index],
            diagnostics={
                "route": "model",
                "profile": "realized_dynamic",
                "scores": scores,
                "margin": margin,
            },
        )
