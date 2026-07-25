from __future__ import annotations

import copy

import pytest

from experiments import receding_q_stability as module


def _row(
    state: str,
    candidate: str,
    trial: int,
    auc: float,
    *,
    v2: bool = False,
) -> dict:
    return {
        "state_id": state,
        "candidate_id": candidate,
        "trial_index": trial,
        "complete": True,
        "feasible": True,
        "final_conflict_ratio": auc,
        "normalized_step_auc": auc,
        "normalized_wall_auc_seconds": auc,
        "observed_total_seconds": auc,
        "is_v2_candidate": v2,
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 100,
        "low_level": {},
        "agents": [1, 2, 3, 4],
        "conflict_trajectory": [10, 5],
        "feature_names": ["x"],
        "feature_values": [1.0],
        "selection_families": ["collision:4"],
        "steps": [],
    }


def test_identify_targets_uses_exact_two_seed_winner_change() -> None:
    rows = [
        _row("stable", "a", 0, 0.1),
        _row("stable", "b", 0, 0.2),
        _row("stable", "a", 1, 0.1),
        _row("stable", "b", 1, 0.2),
        _row("unstable", "a", 0, 0.1),
        _row("unstable", "b", 0, 0.2),
        _row("unstable", "a", 1, 0.3),
        _row("unstable", "b", 1, 0.1),
    ]
    targets = module.identify_stability_targets(rows)
    assert targets["target_state_ids"] == ["unstable"]
    assert targets["stable_state_ids"] == ["stable"]


def test_followup_jobs_skip_stable_states_and_add_only_trial_2_3() -> None:
    plan = {
        "states": [
            {
                "state_id": "unstable",
                "arms": [{"candidate_id": "a"}, {"candidate_id": "b"}],
            },
            {
                "state_id": "stable",
                "arms": [{"candidate_id": "a"}],
            },
        ]
    }
    jobs = module.build_followup_jobs(
        plan=plan,
        targets={"target_state_ids": ["unstable"]},
        horizon=3,
        continuation_teacher="official_adaptive",
    )
    assert len(jobs) == 4
    assert {job["trial_index"] for job in jobs} == {2, 3}
    assert {job["state"]["state_id"] for job in jobs} == {"unstable"}


def test_merge_rejects_duplicate_followup_key() -> None:
    source = [_row("state", "a", 0, 0.1)]
    with pytest.raises(ValueError, match="duplicates"):
        module.merge_followup_rollouts(source, [copy.deepcopy(source[0])])


def test_four_seed_analysis_uses_outcome_stability_not_exact_id() -> None:
    rows = []
    for trial in range(4):
        rows.append(_row("state", "a", trial, 0.10, v2=True))
        rows.append(_row("state", "b", trial, 0.11))
    plan = {
        "states": [
            {
                "state_id": "state",
                "arms": [{"candidate_id": "a"}, {"candidate_id": "b"}],
            }
        ]
    }
    targets = {"target_state_ids": ["state"]}
    report, states, loo = module.analyze_four_seed_stability(
        rows, plan=plan, targets=targets
    )
    assert states[0]["operationally_stable"] is True
    assert states[0]["half_top3_overlap"] == 2
    assert report["loo"]["losses"] == 0
    assert len(loo) == 4
