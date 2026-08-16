from __future__ import annotations

import json
from pathlib import Path

from experiments.stride_hybridstructpool_routed_confirmation import (
    CONTROLLERS,
    _bounded_paired_comparison,
    _bootstrap_improvement,
    _qualification_summary,
    _runtime_config_path,
    load_config,
    schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_hybridstructpool_routed_confirmation_v1.json"
CONFIG_V2 = ROOT / "configs" / "stride_hybridstructpool_routed_confirmation_v2.json"
POOL_CONFIG = ROOT / "configs" / "stride_structshell_v2_paired_confirmation_v1.json"
BOUNDED_CONFIG = (
    ROOT / "configs" / "stride_structshell_v2_official_bounded_confirmation_v1.json"
)
BOUNDED_CONFIG_V2 = (
    ROOT / "configs" / "stride_structshell_v2_official_bounded_confirmation_v2.json"
)


def test_confirmation_config_is_map_disjoint_and_complete() -> None:
    _path, _root, config = load_config(CONFIG)
    groups = list(config["cohort"]["groups"])
    assert len(groups) == 10
    assert {group["family"] for group in groups} >= {
        "maze",
        "room",
        "warehouse",
        "game",
        "dao",
    }
    assert not {group["id"] for group in groups} & set(
        config["cohort"]["development_map_ids"]
    )


def test_confirmation_schedule_is_rotating_serial_180_episode_contract() -> None:
    _path, _root, config = load_config(CONFIG)
    rows = schedule(config)
    assert len(rows) == 180
    assert len({(row["group_id"], row["task_id"], row["solver_seed"]) for row in rows}) == 60
    for offset in range(0, len(rows), 3):
        block = rows[offset : offset + 3]
        assert {row["controller"] for row in block} == set(CONTROLLERS)
        assert [row["within_key_position"] for row in block] == [0, 1, 2]


def test_v2_replaces_ineligible_maps_without_changing_the_contract() -> None:
    _path, _root, config = load_config(CONFIG_V2)
    groups = list(config["cohort"]["groups"])
    assert len(groups) == 10
    assert {group["id"] for group in groups} >= {
        "random-32-32-10",
        "random-64-64-10",
        "random-64-64-20",
    }
    assert len(schedule(config)) == 180
    assert config["runtime"]["episode_process_timeout_seconds"] == 900.0


def test_pool_confirmation_uses_fresh_seeds_and_two_rotating_arms() -> None:
    _path, _root, config = load_config(POOL_CONFIG)
    assert config["controllers"] == ["v2_only", "structshell_only"]
    assert config["cohort"]["solver_seeds"] == [4, 5, 6]
    rows = schedule(config)
    assert len(rows) == 120
    assert len(
        {(row["group_id"], row["task_id"], row["solver_seed"]) for row in rows}
    ) == 60
    for offset in range(0, len(rows), 2):
        block = rows[offset : offset + 2]
        assert {row["controller"] for row in block} == {
            "v2_only",
            "structshell_only",
        }
        assert [row["within_key_position"] for row in block] == [0, 1]
    first = [rows[offset]["controller"] for offset in range(0, len(rows), 2)]
    assert first[:4] == [
        "v2_only",
        "structshell_only",
        "v2_only",
        "structshell_only",
    ]


def test_pool_confirmation_materializes_registered_fresh_seed_runtime(
    tmp_path: Path,
) -> None:
    _path, root, config = load_config(POOL_CONFIG)
    group = config["cohort"]["groups"][0]
    runtime = _runtime_config_path(root, tmp_path, config, group)
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["solver_seeds"] == [4, 5, 6]
    assert payload["dataset_design"]["map_count"] == 12


def test_bounded_confirmation_adds_official_lns2_with_fresh_seeds() -> None:
    _path, _root, config = load_config(BOUNDED_CONFIG)
    assert config["controllers"] == [
        "official_adaptive",
        "v2_only",
        "structshell_only",
    ]
    assert config["cohort"]["solver_seeds"] == [7, 8, 9]
    assert config["runtime"] == {
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": 180.0,
        "environment_time_limit_seconds": 180.0,
        "episode_process_timeout_seconds": 240.0,
        "outer_job_timeout_seconds": 300.0,
    }
    rows = schedule(config)
    assert len(rows) == 180
    for offset in range(0, len(rows), 3):
        assert {row["controller"] for row in rows[offset : offset + 3]} == set(
            CONTROLLERS
        )


def test_bounded_confirmation_materializes_registered_seed_runtime(
    tmp_path: Path,
) -> None:
    _path, root, config = load_config(BOUNDED_CONFIG)
    group = config["cohort"]["groups"][0]
    runtime = _runtime_config_path(root, tmp_path, config, group)
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["solver_seeds"] == [7, 8, 9]
    assert runtime.name.endswith("solver_seeds_7_8_9.json")


def test_bounded_confirmation_v2_replaces_only_the_ineligible_warehouse_group() -> None:
    _path, _root, predecessor = load_config(BOUNDED_CONFIG)
    _path, _root, replacement = load_config(BOUNDED_CONFIG_V2)
    assert replacement["cohort"]["solver_seeds"] == [10, 11, 12]
    assert replacement["cohort_repair"]["controller_outcomes_consulted"] is False
    assert replacement["cohort_repair"]["old_formal_episode_count"] == 0
    old_groups = {group["id"]: group for group in predecessor["cohort"]["groups"]}
    new_groups = {group["id"]: group for group in replacement["cohort"]["groups"]}
    old_groups.pop("warehouse-10-20-10-2-2")
    warehouse = new_groups.pop("warehouse-20-40-10-2-2-congestion")
    assert new_groups == old_groups
    assert warehouse["family"] == "warehouse"
    assert all("opposite_exchange" in task for task in warehouse["tasks"])
    assert all("agents_0600" in task for task in warehouse["tasks"])
    assert len(schedule(replacement)) == 180


def test_qualification_is_a_hard_all_map_gate(tmp_path: Path) -> None:
    _path, _root, config = load_config(CONFIG_V2)
    for group in config["cohort"]["groups"]:
        report = tmp_path / "maps" / group["id"] / "qualification"
        report.mkdir(parents=True)
        (report / "qualification_report.json").write_text(
            '{"passed":true,"valid_count":6,"incomplete_reset_count":0,'
            '"decision":"eligible_for_closed_loop","nonzero_state_count":1,'
            '"initial_feasible_count":0}',
            encoding="utf-8",
        )
    summary = _qualification_summary(tmp_path, config)
    assert summary["all_maps_passed"] is True
    failed = tmp_path / "maps" / config["cohort"]["groups"][0]["id"]
    (failed / "qualification" / "qualification_report.json").write_text(
        '{"passed":false,"valid_count":6,"incomplete_reset_count":0,'
        '"decision":"inconclusive_do_not_resample","nonzero_state_count":0,'
        '"initial_feasible_count":6}',
        encoding="utf-8",
    )
    summary = _qualification_summary(tmp_path, config)
    assert summary["all_maps_passed"] is False
    assert summary["passed_map_count"] == 9


def test_bootstrap_requires_positive_paired_gain() -> None:
    keys = [("map", f"task-{index}", 1) for index in range(12)]
    baseline = {
        key: {
            "status": "ok",
            "summary": {"success": True, "wall_time_to_feasible": 10.0},
        }
        for key in keys
    }
    challenger = {
        key: {
            "status": "ok",
            "summary": {"success": True, "wall_time_to_feasible": 8.0},
        }
        for key in keys
    }
    report = _bootstrap_improvement(baseline, challenger, keys, 200)
    assert report["relative_improvement"] == 0.2
    assert report["ci95_lower"] == 0.2


def test_bounded_comparison_keeps_right_censored_pairs() -> None:
    keys = [("map", f"task-{index}", 7) for index in range(4)]
    baseline = {
        key: {
            "status": "ok",
            "summary": {
                "success": index < 2,
                "wall_time_to_feasible": 100.0 if index < 2 else None,
                "capped_wall_time_to_feasible": 100.0 if index < 2 else 180.0,
                "normalized_wall_clock_conflict_auc": 0.5,
            },
        }
        for index, key in enumerate(keys)
    }
    challenger = {
        key: {
            "status": "ok",
            "summary": {
                "success": index < 3,
                "wall_time_to_feasible": 80.0 if index < 3 else None,
                "capped_wall_time_to_feasible": 80.0 if index < 3 else 180.0,
                "normalized_wall_clock_conflict_auc": 0.4,
            },
        }
        for index, key in enumerate(keys)
    }
    comparison = _bounded_paired_comparison(baseline, challenger, keys)
    assert comparison["valid"] is True
    assert comparison["paired_episode_count"] == 4
    assert comparison["baseline_success_count"] == 2
    assert comparison["challenger_success_count"] == 3
    assert comparison["challenger_mean_restricted_ttf"] < comparison[
        "baseline_mean_restricted_ttf"
    ]
    bootstrap = _bootstrap_improvement(
        baseline, challenger, keys, 200, bounded=True
    )
    assert bootstrap["pair_count"] == 4
    assert bootstrap["metric"] == "capped_wall_time_to_feasible"
