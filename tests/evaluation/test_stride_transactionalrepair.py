from __future__ import annotations

import json
from pathlib import Path

from experiments.stride_transactionalrepair import (
    BASELINE_POLICY,
    POLICIES,
    PRIMARY_POLICY,
    SET_POLICY,
    retry_plan,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stride_transactionalrepair_v1_registration.json"


def _state() -> dict:
    return {
        "agents": [
            {"id": 1, "conflict_degree": 1},
            {"id": 2, "conflict_degree": 4},
            {"id": 3, "conflict_degree": 2},
            {"id": 8, "conflict_degree": 0},
            {"id": 9, "conflict_degree": 0},
        ]
    }


def test_registration_freezes_three_policies_and_bounded_global_execution() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert tuple(config["policies"]["ids"]) == POLICIES
    assert config["policies"]["primary"] == PRIMARY_POLICY
    assert config["policies"]["maximum_attempts"] == {
        BASELINE_POLICY: 1,
        SET_POLICY: 2,
        PRIMARY_POLICY: 3,
    }
    assert config["policies"]["maximum_added_external_blockers"] == 8
    assert config["execution"]["worker_count"] == 16
    assert config["execution"]["per_state_trial_policy_timeout_seconds"] == 300
    assert config["execution"]["task_granularity"] == "state_x_trial_index_x_policy"
    assert config["claim_boundary"]["model_training_allowed"] is False
    assert config["claim_boundary"]["ttf_experiment_allowed"] is False


def test_retry_appends_first_observed_external_blockers_to_order_tail() -> None:
    plan = retry_plan(
        policy_id=PRIMARY_POLICY,
        state=_state(),
        current_agents=[1, 2, 3],
        applied_order=[3, 1, 2],
        diagnostic_agents=[
            {"order_index": 2, "external_blocker_agents": [9]},
            {"order_index": 0, "external_blocker_agents": [8, 9]},
        ],
        added_agents=[],
        priority_used=False,
        maximum_added_agents=8,
    )
    assert plan == (
        [1, 2, 3, 8, 9],
        [3, 1, 2, 8, 9],
        [8, 9],
        False,
        "append_observed_external_blockers",
    )


def test_joint_retry_uses_conflict_priority_only_without_new_blockers() -> None:
    plan = retry_plan(
        policy_id=PRIMARY_POLICY,
        state=_state(),
        current_agents=[1, 2, 3],
        applied_order=[1, 3, 2],
        diagnostic_agents=[],
        added_agents=[],
        priority_used=False,
        maximum_added_agents=8,
    )
    assert plan == (
        [1, 2, 3],
        [2, 3, 1],
        [],
        True,
        "conflict_priority_reorder",
    )


def test_set_only_and_baseline_do_not_invent_order_retries() -> None:
    common = {
        "state": _state(),
        "current_agents": [1, 2, 3],
        "applied_order": [1, 3, 2],
        "diagnostic_agents": [],
        "added_agents": [],
        "priority_used": False,
        "maximum_added_agents": 8,
    }
    assert retry_plan(policy_id=BASELINE_POLICY, **common) is None
    assert retry_plan(policy_id=SET_POLICY, **common) is None


def test_blocker_budget_is_global_across_attempts() -> None:
    plan = retry_plan(
        policy_id=PRIMARY_POLICY,
        state=_state(),
        current_agents=[1, 2, 3, 8],
        applied_order=[1, 2, 3, 8],
        diagnostic_agents=[
            {"order_index": 1, "external_blocker_agents": [9]},
        ],
        added_agents=[8],
        priority_used=False,
        maximum_added_agents=1,
    )
    assert plan == (
        [1, 2, 3, 8],
        [2, 3, 1, 8],
        [8],
        True,
        "conflict_priority_reorder",
    )
