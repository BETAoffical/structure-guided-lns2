from __future__ import annotations

from experiments.repair_collection import _read_jsonl
from experiments.stride_repairability_basin_audit import (
    _attempt_diagnostics,
    _outcome_class,
    load_registration,
)


CONFIG = "configs/stride_repairability_basin_audit_v1_registration.json"


def _metrics() -> dict:
    return {
        "neighborhood": [1, 2, 3, 4],
        "replan_success": False,
        "pp_failure_reason": "conflict_bound_exceeded",
        "pp_failed_agent": 4,
        "pp_failed_order_index": 3,
        "pp_attempted_agent_count": 4,
        "pp_inserted_agent_count": 3,
        "pp_agent_diagnostics": [
            {
                "agent_id": 1,
                "external_blocker_agents": [8, 9],
                "internal_blocker_agents": [],
                "new_conflict_pairs": [[1, 8]],
            },
            {
                "agent_id": 2,
                "external_blocker_agents": [9, 10],
                "internal_blocker_agents": [1],
                "new_conflict_pairs": [[2, 9]],
            },
            {
                "agent_id": 3,
                "external_blocker_agents": [],
                "internal_blocker_agents": [1],
                "new_conflict_pairs": [[1, 3]],
            },
            {
                "agent_id": 4,
                "external_blocker_agents": [10],
                "internal_blocker_agents": [2, 3],
                "new_conflict_pairs": [[2, 4], [3, 4]],
            },
        ],
    }


def _durability(*, escaped: bool, observed: bool, new_platform: bool) -> dict:
    return {
        "horizons": {
            "1": {"sustained_escape": escaped},
            "8": {
                "observed": observed,
                "new_platform_formed": new_platform,
                "sustained_escape": escaped and observed and not new_platform,
            },
        }
    }


def test_attempt_diagnostics_deduplicate_blockers_and_locate_failure() -> None:
    row = _attempt_diagnostics(_metrics())
    assert row["external_blockers"] == [8, 9, 10]
    assert row["internal_blockers"] == [1, 2, 3]
    assert row["new_conflict_pair_count"] == 5
    assert row["failed_order_fraction"] == 1.0
    assert row["failed_agent_external_blockers"] == [10]
    assert row["failed_agent_internal_blockers"] == [2, 3]


def test_outcome_partition_is_mutually_exclusive() -> None:
    assert _outcome_class(
        _durability(escaped=False, observed=True, new_platform=False)
    ) == "immediate_unresolved"
    assert _outcome_class(
        _durability(escaped=True, observed=False, new_platform=False)
    ) == "right_censored_before_h8"
    assert _outcome_class(
        _durability(escaped=True, observed=True, new_platform=True)
    ) == "new_platform_by_h8"
    assert _outcome_class(
        _durability(escaped=True, observed=True, new_platform=False)
    ) == "durable_through_h8"


def test_registration_freezes_all_81_compact_blocker_trigger_events() -> None:
    _path, _root, config, inputs = load_registration(CONFIG)
    rows = _read_jsonl(inputs["durability_rows"])
    selected = [
        row
        for row in rows
        if row["arm"] == config["population"]["arm"]
        and row["trigger_eligible"]
    ]
    assert len(selected) == 81
    assert {int(row["trial_index"]) for row in selected} == {0, 1, 2, 3}
