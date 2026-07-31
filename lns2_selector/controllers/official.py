from __future__ import annotations

from lns2_selector.runtime.contracts import (
    SelectionDecision,
    SelectionRequest,
)


class OfficialAdaptiveSelector:
    controller_id = "official_adaptive"

    def select(self, request: SelectionRequest) -> SelectionDecision:
        return SelectionDecision(
            controller_id=self.controller_id,
            candidate_index=None,
            candidate=None,
            diagnostics={"route": "native-official-adaptive"},
            fallback_reason="native_policy",
        )
