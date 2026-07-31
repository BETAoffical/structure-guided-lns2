from __future__ import annotations

from types import SimpleNamespace

import pytest

from lns2_selector.controllers.official import OfficialAdaptiveSelector
from lns2_selector.controllers.v2 import PairwiseV2Selector
from lns2_selector.runtime.contracts import SelectionRequest, Selector


class DirectModel:
    def score_candidates(self, rows):
        return [float(row["score"]) for row in rows]


def request(scores=(1.0, 2.0)) -> SelectionRequest:
    candidates = tuple(
        {"candidate_id": f"candidate-{index}", "agents": [index]}
        for index in range(len(scores))
    )
    rows = tuple(
        {"candidate_key": f"candidate-{index}", "score": score}
        for index, score in enumerate(scores)
    )
    return SelectionRequest(
        candidates=candidates,
        candidate_rows=rows,
        before_fingerprint="state",
        agent_count=10,
    )


def test_request_rejects_mismatched_candidate_rows() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        SelectionRequest(
            candidates=({},),
            candidate_rows=(),
            before_fingerprint="state",
        )


def test_official_selector_routes_to_native_policy() -> None:
    selector = OfficialAdaptiveSelector()
    assert isinstance(selector, Selector)
    decision = selector.select(request())
    assert decision.uses_native_adaptive
    assert decision.fallback_reason == "native_policy"


@pytest.mark.parametrize("controller_id", ["v2-full", "mixed-full-v2"])
def test_v2_contract_selects_best_candidate(controller_id: str) -> None:
    bundle = SimpleNamespace(main_models={"realized_dynamic": DirectModel()})
    decision = PairwiseV2Selector(controller_id, bundle).select(request())
    assert decision.controller_id == controller_id
    assert decision.candidate_index == 1
    assert decision.candidate["candidate_id"] == "candidate-1"
    assert decision.fallback_reason is None


@pytest.mark.parametrize("controller_id", ["v2-full", "mixed-full-v2"])
def test_v2_contract_falls_back_only_when_pool_is_empty(controller_id: str) -> None:
    bundle = SimpleNamespace(main_models={"realized_dynamic": DirectModel()})
    empty = SelectionRequest(
        candidates=(), candidate_rows=(), before_fingerprint="state"
    )
    decision = PairwiseV2Selector(controller_id, bundle).select(empty)
    assert decision.uses_native_adaptive
    assert decision.fallback_reason == "no_candidates"
