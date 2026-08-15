from __future__ import annotations

from experiments.stride_repairability_causal_audit import (
    conflict_priority_order,
    external_blocker_order,
    stable_effect,
)


def test_conflict_priority_order_is_deterministic() -> None:
    state = {
        "agents": [
            {"id": 1, "conflict_degree": 2},
            {"id": 2, "conflict_degree": 5},
            {"id": 3, "conflict_degree": 5},
        ]
    }
    assert conflict_priority_order(state, [3, 1, 2]) == [2, 3, 1]


def test_external_blockers_follow_first_observed_order_and_cap() -> None:
    diagnostics = [
        {"order_index": 2, "external_blocker_agents": [9, 7]},
        {"order_index": 0, "external_blocker_agents": [8, 9]},
        {"order_index": 3, "external_blocker_agents": [1]},
    ]
    assert external_blocker_order(diagnostics, [1, 2], maximum=2) == [8, 9]


def _rows(delta: float) -> tuple[list[dict], list[dict]]:
    reference = []
    treatment = []
    for index in range(16):
        reference.append(
            {
                "trial_index": index,
                "applicable": True,
                "normalized_conflict_reduction": 0.0,
                "no_progress": True,
            }
        )
        treatment.append(
            {
                "trial_index": index,
                "applicable": True,
                "normalized_conflict_reduction": delta,
                "no_progress": False,
            }
        )
    return reference, treatment


def test_stable_effect_requires_both_fixed_halves() -> None:
    reference, treatment = _rows(0.03)
    assert stable_effect(reference, treatment, minimum=0.02)[
        "stable_effect_passed"
    ]
    treatment[-1]["applicable"] = False
    assert not stable_effect(reference, treatment, minimum=0.02)[
        "stable_effect_passed"
    ]
