from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.stride_structshell_rollback_aware_ttf import (
    CONTROLLERS,
    _cluster_bootstrap,
    _controller_kwargs,
    _episode_job,
    _failed_episode_job,
    _qualification_resume,
    _status,
    extension_keys,
    full_keys,
    load_config,
    run_extension,
    run_screen,
    schedule,
    screen_keys,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structshell_rollback_aware_ttf_v1.json"


def test_two_stage_schedule_is_a_fixed_partition_of_sixty_keys() -> None:
    _path, _root, config = load_config(CONFIG)
    screen = screen_keys(config)
    extension = extension_keys(config)
    full = full_keys(config)
    assert len(screen) == 10
    assert len(extension) == 50
    assert len(full) == 60
    assert screen.isdisjoint(extension)
    assert screen | extension == full
    assert all(seed == 16 for _group, _task, seed in screen)
    assert len(schedule(config, "screen")) == 30
    assert len(schedule(config, "extension")) == 150
    assert len(schedule(config, "full")) == 180


def test_each_key_has_strict_rotating_three_arm_serial_order() -> None:
    _path, _root, config = load_config(CONFIG)
    rows = schedule(config, "full")
    for offset in range(0, len(rows), 3):
        block = rows[offset : offset + 3]
        assert {row["controller"] for row in block} == set(CONTROLLERS)
        assert [row["within_key_position"] for row in block] == [0, 1, 2]
        assert len(
            {
                (row["group_id"], row["task_id"], row["solver_seed"])
                for row in block
            }
        ) == 1
    assert [rows[index]["controller"] for index in (0, 3, 6)] == list(
        CONTROLLERS
    )
    for stage in ("screen", "extension"):
        stage_rows = schedule(config, stage)
        first_positions = [
            row["controller"]
            for offset, row in enumerate(stage_rows)
            if offset % 3 == 0
        ]
        counts = {name: first_positions.count(name) for name in CONTROLLERS}
        assert max(counts.values()) - min(counts.values()) <= 1


def test_challenger_changes_only_the_registered_augmentation() -> None:
    _path, root, config = load_config(CONFIG)
    v2 = _controller_kwargs(root, config, "v2_only")
    challenger = _controller_kwargs(
        root, config, "structshell_rollback_aware_v2"
    )
    augmentation = challenger.pop("hybridstructpool_augmentation")
    assert challenger == v2
    assert augmentation["pool_id"] == "stride-hybridstructpool-routed-v2"
    assert augmentation["exact_rollback_guard"]["exact_rollback_limit"] == 3
    assert augmentation["exact_rollback_guard"]["maximum_pp_calls_per_decision"] == 1


def test_dry_runs_freeze_worker_and_timeout_contract() -> None:
    with tempfile.TemporaryDirectory() as directory:
        screen = run_screen(CONFIG, directory, dry_run=True)
        extension = run_extension(CONFIG, directory, dry_run=True)
    assert screen["qualification_key_count"] == 60
    assert screen["qualification_worker_limit"] == 16
    assert screen["timed_worker_count"] == 1
    assert screen["schedule_entry_count"] == 30
    assert extension["schedule_entry_count"] == 150
    _path, _root, config = load_config(CONFIG)
    assert config["runtime"] == {
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": 180.0,
        "environment_time_limit_seconds": 180.0,
        "episode_process_timeout_seconds": 240.0,
        "outer_job_timeout_seconds": 300.0,
        "native_pp_order_only": True,
        "maximum_pp_calls_per_decision": 1,
        "runtime_retry_or_rescue": False,
    }


def test_status_counts_atomic_manifests_without_claiming_active_jobs() -> None:
    _path, _root, config = load_config(CONFIG)
    items = schedule(config, "screen")
    with tempfile.TemporaryDirectory() as directory:
        status = _status(
            Path(directory),
            "screen",
            items,
            {"schema": "lns2.stride.structshell_rollback_aware_ttf_screen_status.v1"},
        )
    assert status["completed_jobs"] == 0
    assert status["completed_schedule_entries"] == 0
    assert status["total_jobs"] == 30
    assert status["active_jobs"] == 0
    assert status["complete"] is False


def _manifest(value: float) -> dict:
    return {
        "status": "ok",
        "summary": {"capped_wall_time_to_feasible": value},
    }


def test_final_bootstrap_resamples_ten_map_clusters_not_sixty_keys() -> None:
    keys = [
        (f"map-{group}", f"task-{task}", seed)
        for group in range(10)
        for task in range(2)
        for seed in (16, 17, 18)
    ]
    baseline = {key: _manifest(100.0 + int(key[0].split("-")[1])) for key in keys}
    challenger = {key: _manifest(80.0 + int(key[0].split("-")[1])) for key in keys}
    result = _cluster_bootstrap(baseline, challenger, keys, 1000)
    assert result["valid"]
    assert result["group_count"] == 10
    assert result["pair_count"] == 60
    assert result["resampling_unit"] == "map_group_with_all_task_seed_pairs"
    assert result["ci95_lower"] > 0.0


def test_extension_is_blocked_until_the_complete_screen_passes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result = run_extension(CONFIG, directory)
    assert result["blocked"] is True
    assert result["terminal_failure"] == "screen_not_passed"


def test_second_stage_reuses_the_shared_qualification_without_outer_resume() -> None:
    with tempfile.TemporaryDirectory() as directory:
        qualification = Path(directory) / "qualification" / "map-a"
        qualification.mkdir(parents=True)
        assert _qualification_resume(qualification) is False
        qualification.joinpath("run_config.json").write_text("{}", encoding="utf-8")
        assert _qualification_resume(qualification) is True


def test_episode_job_qualifies_the_complete_group_before_a_single_timed_key() -> None:
    _path, _root, config = load_config(CONFIG)
    item = next(
        row
        for row in schedule(config, "screen")
        if row["group_id"] == "den312d" and row["controller"] == "v2_only"
    )
    calls: list[dict] = []

    def record_collection(*_args, **kwargs):
        calls.append(kwargs)

    with tempfile.TemporaryDirectory() as directory, patch(
        "experiments.stride_structshell_rollback_aware_ttf.run_closed_loop_collection",
        side_effect=record_collection,
    ), patch(
        "experiments.stride_structshell_rollback_aware_ttf._manifest",
        return_value={"status": "ok"},
    ):
        result = _episode_job(
            {
                "config_path": str(CONFIG),
                "output_root": directory,
                "stage": "screen",
                "item": item,
            }
        )

    expected_qualification_keys = {
        (task, seed)
        for task in next(
            group for group in config["cohort"]["groups"] if group["id"] == "den312d"
        )["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    }
    assert result["status"] == "ok"
    assert len(calls) == 2
    assert calls[0]["job_keys"] == expected_qualification_keys
    assert calls[0]["cohort_job_keys"] == expected_qualification_keys
    assert calls[1]["job_keys"] == {(item["task_id"], item["solver_seed"])}
    assert calls[1]["cohort_job_keys"] == expected_qualification_keys


def test_outer_worker_failure_uses_the_ttf_job_schema() -> None:
    item = {
        "group_id": "den312d",
        "task_id": "den312d__random_04__agents_0300",
        "solver_seed": 16,
        "controller": "v2_only",
    }
    result = _failed_episode_job({"item": item}, "error", "boom")
    assert result["status"] == "error"
    assert result["error"] == "boom"
    assert result["state_count"] == 0
    assert result["outcome_count"] == 0


def test_platforms_are_diagnostic_not_a_promotion_gate() -> None:
    _path, _root, config = load_config(CONFIG)
    assert "no_post_escape_platform_transfer" not in config["performance_gates"]
    assert config["performance_gates"]["paired_bootstrap_replicates"] == 10000
    assert config["performance_gates"]["maximum_map_restricted_ttf_regression"] == 0.05
