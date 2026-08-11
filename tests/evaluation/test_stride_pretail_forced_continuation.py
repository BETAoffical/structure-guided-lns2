from __future__ import annotations

from experiments.stride_pretail_forced_continuation import (
    ARMS,
    _candidate_roles,
    _paired_first_action_seed,
    continuation_schedule,
    select_shard,
)


def test_paired_first_action_seed_is_stable_and_trial_specific() -> None:
    assert _paired_first_action_seed("case", 0) == _paired_first_action_seed("case", 0)
    assert _paired_first_action_seed("case", 0) != _paired_first_action_seed("case", 1)


def test_schedule_rotates_three_arms_inside_each_paired_trial() -> None:
    checkpoint = {
        "case_id": "c0",
        "state_id": "s0",
        "task_id": "t0",
        "solver_seed": 3,
        "challenger": "v2-plus-structpool",
        "treatment_policy": "struct-then-struct",
    }
    roles = {
        arm: {"candidate_id": f"candidate-{arm}"}
        for arm in ARMS
    }
    schedule = continuation_schedule([{"checkpoint": checkpoint, "roles": roles}])
    assert len(schedule) == 6
    assert [row["arm"] for row in schedule[:3]] == list(ARMS)
    assert [row["trial_index"] for row in schedule] == [0, 0, 0, 1, 1, 1]


def test_candidate_roles_use_oracle_and_diverse_coverage() -> None:
    checkpoint = {
        "selected_candidate_id": "selected",
        "state_fingerprint": "state",
        "candidate_pool": [
            {"candidate_id": "selected", "agents": [1, 2], "score": 9.0},
            {"candidate_id": "oracle", "agents": [3, 4], "score": 8.0},
            {"candidate_id": "coverage", "agents": [5, 6], "score": 7.0},
        ],
    }
    aggregates = {
        ("state", "selected"): {
            "features": {
                "realized.internal_conflict_coverage": 0.2,
                "realized.incident_conflict_coverage": 0.3,
            }
        },
        ("state", "oracle"): {
            "features": {
                "realized.internal_conflict_coverage": 0.4,
                "realized.incident_conflict_coverage": 0.5,
            }
        },
        ("state", "coverage"): {
            "features": {
                "realized.internal_conflict_coverage": 0.8,
                "realized.incident_conflict_coverage": 0.9,
            }
        },
    }
    roles = _candidate_roles(
        checkpoint, {"best_candidate_id": "oracle"}, aggregates
    )
    assert roles["actual_selected"]["candidate_id"] == "selected"
    assert roles["one_step_oracle"]["candidate_id"] == "oracle"
    assert roles["coverage_diverse"]["candidate_id"] == "coverage"


def test_eight_process_shards_partition_cases_without_overlap() -> None:
    cases = [{"case": value} for value in range(45)]
    shards = [select_shard(cases, shard_index=index, shard_count=8) for index in range(8)]
    assert sorted(row["case"] for shard in shards for row in shard) == list(range(45))
    assert [len(shard) for shard in shards] == [6, 6, 6, 6, 6, 5, 5, 5]
