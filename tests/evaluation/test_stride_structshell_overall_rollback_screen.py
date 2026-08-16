from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.stride_structshell_overall_rollback_screen import (
    CONTROLLERS,
    _controller_kwargs,
    _state_guard_trace_audit,
    load_config,
    run,
    schedule,
    screen_keys,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structshell_overall_rollback_screen_v1.json"


def test_registration_reuses_exactly_ten_seen_seed16_keys() -> None:
    _path, _root, config, source = load_config(CONFIG)
    keys = screen_keys(source)
    rows = schedule(config, source)
    assert len(keys) == 10
    assert len(rows) == 40
    assert all(seed == 16 for _group, _task, seed in keys)
    assert {
        (row["group_id"], row["task_id"], row["solver_seed"]) for row in rows
    } == keys
    for offset in range(0, len(rows), 4):
        block = rows[offset : offset + 4]
        assert {row["controller"] for row in block} == set(CONTROLLERS)
        assert [row["within_key_position"] for row in block] == [0, 1, 2, 3]
        assert len(
            {
                (row["group_id"], row["task_id"], row["solver_seed"])
                for row in block
            }
        ) == 1
    first_positions = [rows[offset]["controller"] for offset in range(0, 40, 4)]
    counts = {name: first_positions.count(name) for name in CONTROLLERS}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_four_arms_change_only_registered_pool_and_guard_contracts() -> None:
    _path, root, config, _source = load_config(CONFIG)
    official = _controller_kwargs(root, config, "official_adaptive")
    v2 = _controller_kwargs(root, config, "v2_only")
    no_guard = _controller_kwargs(root, config, "structshell_routed_v1")
    v3 = _controller_kwargs(root, config, "structshell_rollback_overall_v3")
    assert official["controller"] == "official_adaptive"
    no_guard_augmentation = no_guard.pop("hybridstructpool_augmentation")
    v3_augmentation = v3.pop("hybridstructpool_augmentation")
    assert no_guard == v2
    assert v3 == v2
    assert no_guard_augmentation["source_mode"] == "routed_structshell"
    assert no_guard_augmentation["activation_gate"]["gate_id"] == (
        "stride-highstress-conflict-structure-v1"
    )
    assert "exact_rollback_guard" not in no_guard_augmentation
    assert v3_augmentation["pool_id"] == "stride-hybridstructpool-routed-v3"
    assert v3_augmentation["exact_rollback_guard"] == {
        "guard_id": "stride-exact-rollback-state-guard-v1",
        "exact_rollback_limit": 3,
        "candidate_scope": "all_pure_structshell_per_repair_fingerprint",
        "fallback": "fresh_v2_only",
        "repair_state_cache": "pre_budget_only",
        "state_history": "persistent_per_repair_fingerprint",
        "maximum_pp_calls_per_decision": 1,
    }


def test_dry_run_freezes_serial_timing_and_parallel_qualification() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result = run(CONFIG, directory, dry_run=True)
    assert result["qualification_key_count"] == 60
    assert result["paired_key_count"] == 10
    assert result["schedule_entry_count"] == 40
    assert result["timed_worker_count"] == 1
    assert result["qualification_worker_limit"] == 16
    assert result["wall_time_budget_seconds"] == 180.0
    assert result["episode_process_timeout_seconds"] == 240.0
    assert result["outer_job_timeout_seconds"] == 300.0
    assert result["promotion_claim"] is False


def _trace_row(
    index: int,
    *,
    count: int,
    selected_structshell: bool,
    newly_suppressed: bool = False,
    latched: bool = False,
) -> dict:
    selection_phase = "fresh_v2_only" if latched else "frozen_base_ranking"
    return {
        "decision_index": index,
        "before_platform_signature": "repair-a",
        "after_platform_signature": "repair-a",
        "before_conflicts": 10,
        "after_conflicts": 10,
        "actual_action": {},
        "actual_metrics": {},
        "controller": {
            "candidate_pool": [],
            "proposal": {
                "repair_state_cache_hit": False,
                "hybridstructpool_gate_evaluated": not latched,
                "hybridstructpool_state_bounded_v2_fallback": latched,
                "hybridstructpool_gate_reason": (
                    "state_exact_rollback_budget_exhausted" if latched else "passed"
                ),
            },
            "exact_rollback_state_guard": {
                "selection": {
                    "repair_fingerprint": "repair-a",
                    "exact_rollback_limit": 3,
                    "state_exact_rollbacks": (
                        count - 1 if selected_structshell and not latched else count
                    ),
                    "structshell_suppressed": latched,
                    "selection_phase": selection_phase,
                    "selected_candidate_is_pure_structshell": selected_structshell,
                    "pure_structshell_candidate_count": 0 if latched else 5,
                },
                "observation": {
                    "pure_structshell_exact_rollbacks": count,
                    "rollback_counted": selected_structshell,
                    "newly_suppressed": newly_suppressed,
                    "structshell_suppressed": newly_suppressed or latched,
                    "exact_conflict_bound_rollback": True,
                    "selected_candidate_is_pure_structshell": selected_structshell,
                },
            },
        },
    }


def test_trace_audit_proves_state_budget_fresh_v2_and_no_reopen() -> None:
    rows = [
        _trace_row(0, count=1, selected_structshell=True),
        _trace_row(1, count=2, selected_structshell=True),
        _trace_row(
            2,
            count=3,
            selected_structshell=True,
            newly_suppressed=True,
        ),
        _trace_row(3, count=3, selected_structshell=False, latched=True),
        _trace_row(4, count=3, selected_structshell=False, latched=True),
    ]
    indexed = {
        name: {}
        for name in CONTROLLERS
    }
    indexed["structshell_rollback_overall_v3"] = {
        ("map-a", "task-a", 16): {
            "status": "ok",
            "summary": {"repair_iterations": len(rows)},
        }
    }
    with tempfile.TemporaryDirectory() as directory, patch(
        "experiments.stride_structshell_overall_rollback_screen._decision_rows",
        return_value=rows,
    ):
        report = _state_guard_trace_audit(Path(directory), indexed)
    assert report["passed"] is True
    assert report["maximum_counted_rollbacks_per_repair_fingerprint"] == 3
    assert report["newly_suppressed_state_count"] == 1
    assert report["fresh_v2_decision_count"] == 2
    assert report["v2_exact_rollbacks_while_suppressed"] == 2


def test_trace_audit_rejects_structshell_after_latch() -> None:
    rows = [
        _trace_row(0, count=1, selected_structshell=True),
        _trace_row(1, count=2, selected_structshell=True),
        _trace_row(
            2,
            count=3,
            selected_structshell=True,
            newly_suppressed=True,
        ),
        _trace_row(3, count=3, selected_structshell=True, latched=True),
    ]
    indexed = {name: {} for name in CONTROLLERS}
    indexed["structshell_rollback_overall_v3"] = {
        ("map-a", "task-a", 16): {
            "status": "ok",
            "summary": {"repair_iterations": len(rows)},
        }
    }
    with tempfile.TemporaryDirectory() as directory, patch(
        "experiments.stride_structshell_overall_rollback_screen._decision_rows",
        return_value=rows,
    ):
        report = _state_guard_trace_audit(Path(directory), indexed)
    assert report["passed"] is False
    assert report["gates"]["no_structshell_selection_while_latched"] is False


def test_trace_audit_rejects_third_rollback_without_immediate_latch() -> None:
    rows = [
        _trace_row(0, count=1, selected_structshell=True),
        _trace_row(1, count=2, selected_structshell=True),
        _trace_row(2, count=3, selected_structshell=True),
    ]
    indexed = {name: {} for name in CONTROLLERS}
    indexed["structshell_rollback_overall_v3"] = {
        ("map-a", "task-a", 16): {
            "status": "ok",
            "summary": {"repair_iterations": len(rows)},
        }
    }
    with tempfile.TemporaryDirectory() as directory, patch(
        "experiments.stride_structshell_overall_rollback_screen._decision_rows",
        return_value=rows,
    ):
        report = _state_guard_trace_audit(Path(directory), indexed)
    assert report["passed"] is False
    assert (
        report["gates"]["third_counted_rollback_immediately_latches"] is False
    )


def test_trace_audit_rejects_structshell_hidden_in_fallback_pool() -> None:
    rows = [
        _trace_row(0, count=1, selected_structshell=True),
        _trace_row(1, count=2, selected_structshell=True),
        _trace_row(
            2,
            count=3,
            selected_structshell=True,
            newly_suppressed=True,
        ),
        _trace_row(3, count=3, selected_structshell=False, latched=True),
    ]
    rows[3]["controller"]["candidate_pool"] = [
        {
            "candidate_id": "hidden-struct",
            "hybridstructpool_provenance": ["structshell_equal_four_size"],
        }
    ]
    indexed = {name: {} for name in CONTROLLERS}
    indexed["structshell_rollback_overall_v3"] = {
        ("map-a", "task-a", 16): {
            "status": "ok",
            "summary": {"repair_iterations": len(rows)},
        }
    }
    with tempfile.TemporaryDirectory() as directory, patch(
        "experiments.stride_structshell_overall_rollback_screen._decision_rows",
        return_value=rows,
    ):
        report = _state_guard_trace_audit(Path(directory), indexed)
    assert report["passed"] is False
    assert report["gates"]["latched_fallback_is_fresh_v2_only"] is False
