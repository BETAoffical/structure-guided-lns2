from __future__ import annotations

import tempfile
from pathlib import Path

from experiments.stride_structshell_rollback_aware_platform_replay import (
    ARMS,
    BASELINE_ARM,
    CHALLENGER_ARM,
    _controller_kwargs,
    _exact_rollback,
    _exact_repair_platforms,
    _first_exact_repair_platform,
    _longest_exact_streak,
    _metric_manifest,
    _selection_class,
    _status,
    _successful_manifest,
    load_config,
    schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structshell_rollback_aware_platform_replay_v1.json"


def test_known_platform_replay_identity_and_schedule_are_frozen() -> None:
    _path, _root, config = load_config(CONFIG)
    assert config["cohort"]["promotion_evidence"] is False
    assert config["cohort"]["solver_seeds"] == [14]
    rows = schedule(config)
    assert len(rows) == 4
    assert [row["arm"] for row in rows] == [
        BASELINE_ARM,
        CHALLENGER_ARM,
        CHALLENGER_ARM,
        BASELINE_ARM,
    ]
    assert len({(row["task_id"], row["solver_seed"]) for row in rows}) == 2
    for offset in range(0, len(rows), 2):
        assert {row["arm"] for row in rows[offset : offset + 2]} == set(ARMS)
        assert [row["within_key_position"] for row in rows[offset : offset + 2]] == [0, 1]


def test_replay_changes_only_rollback_aware_augmentation() -> None:
    _path, root, config = load_config(CONFIG)
    baseline = _controller_kwargs(root, config, BASELINE_ARM)
    challenger = _controller_kwargs(root, config, CHALLENGER_ARM)
    baseline_augmentation = baseline.pop("hybridstructpool_augmentation")
    challenger_augmentation = challenger.pop("hybridstructpool_augmentation")
    assert baseline == challenger
    assert baseline_augmentation["pool_id"] == "stride-hybridstructpool-routed-v1"
    assert challenger_augmentation["pool_id"] == "stride-hybridstructpool-routed-v2"
    assert challenger_augmentation["exact_rollback_guard"] == {
        "guard_id": "stride-exact-rollback-candidate-guard-v1",
        "exact_rollback_limit": 3,
        "candidate_scope": "hybrid_challenger_only",
        "fallback": "v2_anchor",
        "repair_state_cache": True,
        "maximum_pp_calls_per_decision": 1,
    }


def test_replay_is_serial_bounded_and_has_no_runtime_retry() -> None:
    _path, _root, config = load_config(CONFIG)
    runtime = config["runtime"]
    assert runtime["workers_for_qualification"] == 16
    assert runtime["workers_for_timed_episodes"] == 1
    assert runtime["wall_time_budget_seconds"] == 180.0
    assert runtime["episode_process_timeout_seconds"] == 240.0
    assert runtime["outer_job_timeout_seconds"] == 300.0
    assert runtime["maximum_pp_calls_per_decision"] == 1
    assert runtime["runtime_retry_or_rescue"] is False


def test_replay_status_preserves_resumable_schedule_counter() -> None:
    _path, _root, config = load_config(CONFIG)
    items = schedule(config)
    with tempfile.TemporaryDirectory() as directory:
        status = _status(
            Path(directory),
            items,
            {
                "schema": "lns2.stride.structshell_rollback_aware_platform_replay_status.v1",
                "total_schedule_entries": len(items),
            },
        )
    assert status["completed_schedule_entries"] == 0
    assert status["total_schedule_entries"] == len(items)
    assert status["complete"] is False


def test_replay_accepts_only_successful_atomic_manifest_states() -> None:
    assert _successful_manifest({"status": "ok"})
    assert _successful_manifest({"status": "resumed"})
    assert not _successful_manifest({"status": "error"})
    assert not _successful_manifest({"status": "timeout"})
    assert not _successful_manifest({})
    resumed = {"status": "resumed", "summary": {"success": True}}
    normalized = _metric_manifest(resumed)
    assert normalized["status"] == "ok"
    assert normalized["summary"] == resumed["summary"]
    assert resumed["status"] == "resumed"
    assert _metric_manifest({"status": "error"})["status"] == "error"


def _exact_row(candidate_id: str, *, anchor_fallback: bool = False) -> dict:
    return {
        "before_platform_signature": "repair-a",
        "after_platform_signature": "repair-a",
        "actual_metrics": {
            "pp_failure_reason": "conflict_bound_exceeded",
            "replan_success": False,
            "pp_rolled_back": True,
        },
        "controller": {
            "selected_candidate_id": candidate_id,
            "proposal": {
                "hybridstructpool_selected_provenance": (
                    ["v2_base"]
                    if anchor_fallback
                    else ["structshell_equal_four_size"]
                )
            },
            "exact_rollback_candidate_guard": {
                "selection": {
                    "v2_anchor_fallback_used": anchor_fallback,
                }
            },
        },
    }


def test_replay_diagnostics_separate_structshell_and_anchor_platforms() -> None:
    rows = [
        _exact_row("struct-a"),
        _exact_row("struct-a"),
        _exact_row("struct-a"),
        _exact_row("v2-anchor", anchor_fallback=True),
    ]
    assert all(_exact_rollback(row) for row in rows)
    assert _selection_class(rows[0]) == "pure_structshell_bannable"
    assert _selection_class(rows[-1]) == "v2_anchor_fallback"
    assert _longest_exact_streak(
        rows, selection_class="pure_structshell_bannable"
    ) == 3
    assert _longest_exact_streak(rows, selection_class="v2_anchor_fallback") == 1


def test_candidate_rotation_does_not_count_as_repair_platform_escape() -> None:
    censored_rows = [
        *[_exact_row("struct-a") for _ in range(3)],
        _exact_row("v2-anchor", anchor_fallback=True),
        *[_exact_row("struct-a") for _ in range(3)],
        _exact_row("v2-anchor", anchor_fallback=True),
    ]
    censored = _first_exact_repair_platform(censored_rows)
    assert censored["formed"]
    assert not censored["escaped"]
    assert censored["escape_latency_decisions"] is None
    assert censored["longest_consecutive_exact_same_repair"] == 8

    escaped_rows = [*censored_rows[:4], _exact_row("struct-b")]
    escaped_rows[-1]["after_platform_signature"] = "repair-b"
    escaped_rows[-1]["actual_metrics"].update(
        {
            "pp_failure_reason": "none",
            "replan_success": True,
            "pp_rolled_back": False,
        }
    )
    escaped = _first_exact_repair_platform(escaped_rows)
    assert escaped["escaped"]
    assert escaped["escape_decision"] == 4
    assert escaped["escape_latency_decisions"] == 5

    transferred_rows = list(escaped_rows)
    for _ in range(3):
        row = _exact_row("v2-anchor", anchor_fallback=True)
        row["before_platform_signature"] = "repair-b"
        row["after_platform_signature"] = "repair-b"
        transferred_rows.append(row)
    platforms = _exact_repair_platforms(transferred_rows)
    assert [row["repair_fingerprint"] for row in platforms] == [
        "repair-a",
        "repair-b",
    ]
    assert platforms[-1]["start_decision"] > escaped["escape_decision"]
