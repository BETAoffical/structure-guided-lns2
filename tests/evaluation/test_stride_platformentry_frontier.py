from __future__ import annotations

from experiments.stride_platformentry_frontier import (
    _episode_override,
    _frontier_diagnostics,
    _initial_fingerprint_integrity,
    frontier_schedule,
)


def _case() -> dict:
    return {
        "checkpoint": {
            "case_id": "case-a",
            "state_id": "state-a",
            "task_id": "task-a",
            "solver_seed": 17,
            "map_id": "maze-a",
            "challenger": "v2-plus-structpool",
            "treatment_policy": "treatment-a",
            "state_fingerprint": "state-fingerprint-a",
            "conflict_pair_count": 3,
        },
        "actions": [
            {
                "candidate_id": "actual",
                "agents": [1, 2],
                "logical_roles": ["historical_actual"],
            },
            {
                "candidate_id": "augment",
                "agents": [1, 2, 3],
                "logical_roles": [
                    "frontier_candidate:compact-augment",
                    "deterministic_compact_augment",
                ],
                "frontierdependency_variant": "compact-augment",
            },
        ],
    }


def test_schedule_pairs_seed_across_unique_actions() -> None:
    schedule = frontier_schedule([_case()], (0, 1))
    assert len(schedule) == 4
    assert len({row["job_id"] for row in schedule}) == 4
    for trial_index in (0, 1):
        rows = [row for row in schedule if row["trial_index"] == trial_index]
        assert len({row["first_action_pp_seed"] for row in rows}) == 1
        assert {row["candidate_id"] for row in rows} == {"actual", "augment"}


def test_override_never_requests_repair_order() -> None:
    item = frontier_schedule([_case()], (0,))[1]
    restore = {
        "collection_root": "source",
        "manifest": {"status": "ok"},
        "decision_index": 2,
        "expected_fingerprint": "state-fingerprint-a",
        "repair_structure_fingerprint": "repair-a",
        "expected_conflicts": 3,
        "restore_seed": 9,
    }
    override = _episode_override(item, restore)
    assert override["forced_first_action"]["agents"] == [1, 2, 3]
    assert "repair_order" not in override["forced_first_action"]
    assert override["forced_candidate_id"] == "augment"


def test_size_diagnostic_is_explicitly_noncausal() -> None:
    rows = [
        {
            "case_id": "case-a",
            "trial_index": 0,
            "logical_roles": ["frontier_candidate:compact-augment"],
            "frontier_variant": "compact-augment",
            "candidate_size": 33,
            "entered_platform": False,
            "success": True,
            "normalized_fixed_auc": 0.2,
        },
        {
            "case_id": "case-a",
            "trial_index": 0,
            "logical_roles": ["frontier_candidate:same-size-exchange"],
            "frontier_variant": "same-size-exchange",
            "candidate_size": 24,
            "entered_platform": True,
            "success": False,
            "normalized_fixed_auc": 0.4,
        },
    ]
    report = _frontier_diagnostics(rows)
    assert report["fraction_with_any_candidate_avoiding_platform"] == 1.0
    assert report["by_size_band"]["le24"]["platform_rate"] == 1.0
    assert "posthoc" in report["claim_boundary"]


def test_initial_identity_keeps_hash_domains_separate() -> None:
    groups = [
        {
            "action-a": {
                "case_id": "case-a",
                "observed_first_action": {
                    "before_fingerprint": "runtime-a",
                    "before_repair_fingerprint": "repair-a",
                },
            },
            "action-b": {
                "case_id": "case-a",
                "observed_first_action": {
                    "before_fingerprint": "runtime-a",
                    "before_repair_fingerprint": "repair-a",
                },
            },
        }
    ]
    integrity = _initial_fingerprint_integrity(groups, {"case-a": "repair-a"})
    assert integrity == {
        "same_initial_runtime_fingerprint_across_actions": True,
        "registered_initial_repair_fingerprint": True,
    }

    groups[0]["action-b"]["observed_first_action"]["before_fingerprint"] = (
        "runtime-b"
    )
    assert not _initial_fingerprint_integrity(
        groups, {"case-a": "repair-a"}
    )["same_initial_runtime_fingerprint_across_actions"]
