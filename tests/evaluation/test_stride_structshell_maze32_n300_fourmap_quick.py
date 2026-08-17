from __future__ import annotations

from pathlib import Path

import pytest

import experiments.stride_structshell_maze32_n300_fourmap_quick as subject


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_structshell_maze32_n300_fourmap_quick_v1.json"
)


def test_schedule_is_seed25_four_maps_four_arms_strict_serial() -> None:
    result = subject.plan(CONFIG)
    assert result["map_count"] == 4
    assert result["task_count"] == 4
    assert result["controller_count"] == 4
    assert result["timed_episode_count"] == 16
    assert result["solver_seed"] == 25
    assert result["controllers"] == list(subject.CONTROLLERS)
    assert result["keys"] == [
        "maze-32-32-4-n300@25",
        "random-32-32-20-high-load@25",
        "room-64-64-16@25",
        "warehouse-w1020a-opposite-exchange@25",
    ]
    assert result["first_position_rotation"] == list(subject.CONTROLLERS)
    assert result["wall_time_budget_seconds"] == 90.0
    assert result["episode_process_fuse_seconds"] == 105.0
    assert result["maximum_registered_timed_seconds"] == 1440.0
    assert result["maximum_reset_plus_timed_process_fuse_seconds"] == 2100.0
    assert result["strict_serial_timing"] is True
    assert result["reset_inclusive_ttf"] is True
    assert result["solver_or_controller_invoked"] is False

    _path, _root, config = subject.load_config(CONFIG)
    rows = subject.schedule(config)
    assert len(rows) == 16
    assert {row["solver_seed"] for row in rows} == {25}
    for start in range(0, 16, 4):
        block = rows[start : start + 4]
        assert {row["controller"] for row in block} == set(subject.CONTROLLERS)
        assert [row["within_key_position"] for row in block] == [0, 1, 2, 3]


def test_controller_kwargs_freeze_dual_and_plateau_contracts() -> None:
    _path, root, config = subject.load_config(CONFIG)
    official = subject.controller_kwargs(root, config, "official_adaptive")
    assert official["controller"] == "official_adaptive"
    assert "hybridstructpool_augmentation" not in official

    v2 = subject.controller_kwargs(root, config, "v2_only")
    assert v2["controller"] == "v2-full"
    assert "hybridstructpool_augmentation" not in v2

    dual = subject.controller_kwargs(root, config, "dual16")[
        "hybridstructpool_augmentation"
    ]
    assert dual["runtime_id"] == "stride-structshell-dual16-runtime-v1"
    assert "stall_guard" not in dual

    plateau = subject.controller_kwargs(root, config, "dual16_plateau")[
        "hybridstructpool_augmentation"
    ]
    assert plateau["pool_id"] == "stride-structshell-dual16-v1"
    assert (
        plateau["runtime_id"]
        == "stride-structshell-dual16-plateau-guard-runtime-v1"
    )
    assert plateau["runtime_filter_id"] == "dual_family_fixed16_plateau_guard_v1"
    assert plateau["stall_guard"] == {
        "guard_id": "stride-dual16-plateau-guard-v1",
        "no_progress_limit": 8,
        "counter": "consecutive_non_decreasing_conflict_decisions",
        "fallback": "fresh_v2_full",
        "release_condition": "strict_conflict_decrease",
        "wall_time_condition": None,
        "maximum_pp_calls_per_decision": 1,
        "retry_rollback_or_rescue": False,
    }
    for controller in subject.CONTROLLERS:
        kwargs = subject.controller_kwargs(root, config, controller)
        assert kwargs["wall_time_budget_seconds"] == 90.0
        assert kwargs["environment_time_limit_seconds"] == 90.0
        assert kwargs["episode_process_timeout_seconds"] == 105.0


def test_exact_dataset_identity_and_dry_run_invoke_no_solver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    task = (
        ROOT
        / "build"
        / "stride-structshell-maze32-n300-dataset-v1"
        / "balanced_wall_clock"
        / "tasks"
        / "maze-32-32-4__random_04__agents_0300.json"
    )
    assert subject.sha256_file(task) == (
        "d5138f0399d8bf94cc4ef0f251f3a1cf418c239130d8b889bd632b4ee1e12676"
    )
    _path, _root, config = subject.load_config(CONFIG)
    assert config["cohort"]["groups"][0]["task"] == (
        "maze-32-32-4__random_04__agents_0300"
    )
    assert config["solver_seed_identity_audit"]["global_freshness_scan"] is False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run invoked the solver")

    monkeypatch.setattr(subject, "run_closed_loop_collection", forbidden)
    result = subject.run(CONFIG, tmp_path, dry_run=True)
    assert result == subject.plan(CONFIG)
    assert list(tmp_path.iterdir()) == []


def test_materialized_runtime_uses_each_groups_registered_split(
    tmp_path: Path,
) -> None:
    _path, _root, config = subject.load_config(CONFIG)
    observed = {}
    for group in config["cohort"]["groups"]:
        runtime = subject._runtime_config_path(tmp_path, group, 25)
        payload = subject.read_json(runtime)
        observed[str(group["id"])] = payload["split"]
        assert payload["solver_seeds"] == [25]
    assert observed == {
        "maze-32-32-4-n300": "balanced_wall_clock",
        "random-32-32-20-high-load": "balanced_wall_clock",
        "room-64-64-16": "movingai_ood",
        "warehouse-w1020a-opposite-exchange": "balanced_wall_clock",
    }


def test_analyze_reports_timing_and_plateau_diagnostics_without_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _path, _root, config = subject.load_config(CONFIG)
    controller_scale = {
        controller: index + 1
        for index, controller in enumerate(subject.CONTROLLERS)
    }

    def manifest(_output: Path, item: dict[str, object]) -> dict[str, object]:
        scale = controller_scale[str(item["controller"])]
        return {
            "status": "ok",
            "summary": {
                "wall_time_budget_seconds": 90.0,
                "ttf_clock_schema": "lns2.ttf.reset_inclusive_wall.v1",
                "capped_wall_time_to_feasible": float(scale),
                "invalid_action_count": 0,
                "fingerprint_mismatch_count": 0,
                "initial_fingerprint": f"fp-{item['group_id']}",
                "initial_conflicts": 10,
                "success": True,
                "controller_totals": {
                    "candidate_generation_seconds": scale * 0.1,
                    "neighborhood_selection_seconds": scale * 0.2,
                    "pp_replan_seconds": scale * 0.3,
                    "hybridstructpool_stall_guard_active_decision_count": scale,
                    "hybridstructpool_stall_guard_trigger_count": scale,
                    "hybridstructpool_stall_guard_release_count": scale,
                },
            },
        }

    monkeypatch.setattr(subject, "_manifest_row", manifest)
    report = subject.analyze(CONFIG, tmp_path)
    plateau = report["controller_summaries"]["dual16_plateau"]
    assert plateau["mean_candidate_generation_seconds"] == pytest.approx(0.4)
    assert plateau["mean_neighborhood_selection_seconds"] == pytest.approx(0.8)
    assert plateau["mean_pp_replan_seconds"] == pytest.approx(1.2)
    assert plateau["hybridstructpool_stall_guard_active_decision_count"] == 16
    assert plateau["hybridstructpool_stall_guard_trigger_count"] == 16
    assert plateau["hybridstructpool_stall_guard_release_count"] == 16
    per_map = report["per_map"]["maze-32-32-4-n300"]["dual16_plateau"]
    assert per_map["candidate_generation_seconds"] == pytest.approx(0.4)
    assert per_map["neighborhood_selection_seconds"] == pytest.approx(0.8)
    assert per_map["pp_replan_seconds"] == pytest.approx(1.2)
    assert per_map["hybridstructpool_stall_guard_active_decision_count"] == 4
    assert report["selection_gate"] is None
