from __future__ import annotations

from experiments.v3_s3 import V3S3Bundle, V3S3ControllerState
from lns2_selector.runtime.contracts import (
    SelectionDecision,
    SelectionObservation,
    SelectionRequest,
)
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


class V3S3Selector:
    controller_id = "v3-s3"

    def __init__(self, bundle: V3S3Bundle):
        self.state = V3S3ControllerState(bundle)

    def select(self, request: SelectionRequest) -> SelectionDecision:
        index, diagnostics = self.state.select(
            list(request.candidates),
            list(request.candidate_rows),
            temporal_context=dict(request.temporal_context),
            before_fingerprint=request.before_fingerprint,
            agent_count=request.agent_count,
            candidate_pool_mode=request.candidate_pool_mode,
            generation_context=dict(request.generation_context),
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

    def observe(self, observation: SelectionObservation) -> dict[str, object]:
        if (
            self.state.pending_candidate_id is None
            or self.state.pending_agents is None
        ):
            raise RuntimeError("v3-S3 observe called without a pending action")
        if str(observation.candidate_id) != str(self.state.pending_candidate_id):
            raise ValueError("v3-S3 observed candidate ID differs from pending action")
        actual_agents = tuple(sorted(map(int, observation.actual_agents)))
        if actual_agents != self.state.pending_agents:
            raise ValueError("v3-S3 observed agents differ from pending action")
        if str(observation.before_fingerprint) != str(
            self.state.pending_before_fingerprint
        ):
            raise ValueError("v3-S3 pending state fingerprint mismatch")
        repair_outcome = classify_repair_outcome(
            before_fingerprint=observation.before_fingerprint,
            after_fingerprint=observation.after_fingerprint,
            replan_success=observation.replan_success,
            conflicts_before=observation.conflicts_before,
            conflicts_after=observation.conflicts_after,
            feasible=observation.feasible,
        )
        continuation_expected = self.state.observe(
            candidate_id=observation.candidate_id,
            actual_agents=actual_agents,
            before_fingerprint=observation.before_fingerprint,
            after_fingerprint=observation.after_fingerprint,
            repair_outcome=repair_outcome,
            conflict_reduction=float(
                max(0, observation.conflicts_before - observation.conflicts_after)
            ),
            total_seconds=float(observation.total_seconds),
            feasible=observation.feasible,
            terminal=observation.terminal,
        )
        return {
            "repair_outcome": repair_outcome,
            "continuation_expected": bool(continuation_expected),
            "state_unchanged": (
                observation.before_fingerprint == observation.after_fingerprint
            ),
        }
