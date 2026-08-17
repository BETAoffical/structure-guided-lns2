from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from experiments.stride_structshell_seed25_replay_diagnostic import (
    OBSERVED_ROLE,
    PRIMARY_ROLES,
    _step_record,
    build_plan,
    extract_action_candidates,
    paired_pp_seed,
    select_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structshell_seed25_replay_diagnostic_v1.json"


def _event(
    index: int,
    *,
    fingerprint: str,
    agents: list[int],
    trigger: bool = False,
) -> dict:
    return {
        "event": "transition",
        "decision_index": index,
        "before_fingerprint": fingerprint,
        "action": {"mode": "explicit_neighborhood", "agents": agents},
        "controller": {"proposal": {"guardpool_triggered": trigger}},
        # Deliberately contradictory outcome-like values: selection must ignore them.
        "metrics": {"conflicts_after": -999},
        "elapsed_wall_seconds": -999.0,
    }


def test_checkpoint_rule_is_pre_action_and_mechanical() -> None:
    fingerprints = [f"{index + 1:064x}" for index in range(3)]
    dual = [
        _event(index, fingerprint=fingerprints[index], agents=[index, index + 10])
        for index in range(3)
    ]
    guard = [
        _event(
            index,
            fingerprint=fingerprints[index],
            agents=([99, 100] if index == 2 else [index, index + 10]),
            trigger=index == 1,
        )
        for index in range(3)
    ]
    selected = select_checkpoint(dual, guard)
    assert selected["decision_index"] == 2
    assert selected["selection_rule"].startswith("first_equal_pre_action")
    assert selected["outcome_fields_read"] is False

    guard[2]["action"]["agents"] = dual[2]["action"]["agents"]
    selected = select_checkpoint(dual, guard)
    assert selected["decision_index"] == 1
    assert selected["selection_rule"] == "first_registered_plateau_guard_trigger"


def test_candidate_extraction_adds_only_a_distinct_observed_dual_winner() -> None:
    def candidate(identifier: str, agents: list[int], family: str, score: float) -> dict:
        return {
            "candidate_id": identifier,
            "agents": agents,
            "selection_families": [family],
            "score": score,
        }

    pool = [
        candidate("anchor", [1, 2], "target:2", 3.0),
        candidate(
            "component", list(range(3, 19)), "structpool-conflict-component:16", 2.0
        ),
        candidate(
            "hotspot", list(range(20, 36)), "structpool-spatiotemporal-hotspot:16", 1.0
        ),
        candidate("choice-set", [7, 8], "target:8", 4.0),
    ]
    event = {
        "action": {"mode": "explicit_neighborhood", "agents": [8, 7]},
        "controller": {
            "selected_candidate_id": "choice-set",
            "proposal": {"hybridstructpool_v2_anchor_candidate_id": "anchor"},
            "candidate_pool": pool,
        },
    }
    rows = extract_action_candidates(event)
    assert tuple(row["role"] for row in rows[:3]) == PRIMARY_ROLES
    assert rows[-1]["role"] == OBSERVED_ROLE
    assert len(rows) == 4

    event["controller"]["selected_candidate_id"] = "component"
    event["action"]["agents"] = list(range(3, 19))
    assert len(extract_action_candidates(event)) == 3


def test_registered_real_trace_plan_has_13_serial_branches_and_paired_seeds() -> None:
    plan = build_plan(CONFIG)
    assert plan["state_count"] == 4
    assert plan["branch_count"] == 13
    assert plan["maximum_native_pp_calls"] == 91
    assert plan["maximum_v2_continuation_selections"] == 78
    assert plan["strict_serial"] is True
    assert plan["full_ttf_run"] is False
    assert plan["official_adaptive_included"] is False
    assert plan["per_action_time_limit_seconds"] == 5.0
    assert plan["total_process_time_limit_seconds"] == 600.0
    assert [row["decision_index"] for row in plan["states"]] == [31, 48, 30, 0]
    assert [len(row["candidates"]) for row in plan["states"]] == [3, 4, 3, 3]
    for state in plan["states"]:
        repair_fingerprint = state["repair_structure_fingerprint"]
        assert state["first_action_pp_seed"] == paired_pp_seed(repair_fingerprint, 0)
        assert state["continuation_pp_seeds"] == [
            paired_pp_seed(repair_fingerprint, step) for step in range(1, 7)
        ]


def test_replay_step_forwards_the_registered_native_pp_limit() -> None:
    class Environment:
        def __init__(self) -> None:
            self.calls: list[tuple[dict, float]] = []

        def step_with_time_limit(self, action: dict, limit: float) -> dict:
            self.calls.append((action, limit))
            return {"placeholder": True}

    environment = Environment()
    before = {"repair": "before", "num_of_colliding_pairs": 5}
    after = {"repair": "after", "num_of_colliding_pairs": 4, "done": False}
    metrics = {
        "replan_success": True,
        "pp_rolled_back": False,
        "pp_failure_reason": "",
        "native_replan_seconds": 0.25,
    }
    with (
        patch(
            "experiments.stride_structshell_seed25_replay_diagnostic."
            "repair_structure_fingerprint",
            side_effect=lambda state: state["repair"],
        ),
        patch(
            "experiments.stride_structshell_seed25_replay_diagnostic."
            "_validate_native_repair",
            return_value=(after, metrics),
        ),
    ):
        _, record = _step_record(
            environment,
            before,
            role="v2_anchor",
            step_index=0,
            candidate={
                "candidate_id": "anchor",
                "agents": [1, 2],
                "selection_families": ["target:2"],
            },
            pp_seed=123,
            pp_time_limit_seconds=5.0,
        )
    assert len(environment.calls) == 1
    action, limit = environment.calls[0]
    assert limit == 5.0
    assert action["pp_random_seed"] == 123
    assert record["requested_pp_time_limit_seconds"] == 5.0
