from __future__ import annotations

from experiments.v3_s3 import V3S3Bundle, V3S3ControllerState
from lns2_selector.runtime.contracts import (
    SelectionDecision,
    SelectionRequest,
)


class V3S3Selector:
    controller_id = "v3-s3"

    def __init__(self, bundle: V3S3Bundle):
        self.state = V3S3ControllerState(bundle)

    def select(self, request: SelectionRequest) -> SelectionDecision:
        index, diagnostics = self.state.select(
            list(map(dict, request.candidates)),
            list(map(dict, request.candidate_rows)),
            temporal_context=dict(request.temporal_context),
            before_fingerprint=request.before_fingerprint,
            agent_count=request.agent_count,
        )
        if index is None:
            return SelectionDecision(
                controller_id=self.controller_id,
                candidate_index=None,
                candidate=None,
                diagnostics=diagnostics,
                fallback_reason=str(
                    diagnostics.get("selection_kind", "no_eligible_sequence")
                ),
            )
        return SelectionDecision(
            controller_id=self.controller_id,
            candidate_index=index,
            candidate=request.candidates[index],
            diagnostics=diagnostics,
        )
