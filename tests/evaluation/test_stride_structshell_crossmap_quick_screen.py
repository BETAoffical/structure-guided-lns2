from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.stride_structshell_crossmap_quick_screen as subject


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structshell_crossmap_quick_screen_v1.json"


def test_plan_is_the_small_nonwarehouse_screen_and_invokes_nothing() -> None:
    result = subject.plan(CONFIG)
    assert result["map_count"] == 4
    assert result["paired_key_count"] == 4
    assert result["timed_episode_count"] == 12
    assert result["qualification_reset_count"] == 4
    assert result["solver_seed"] == 17
    assert result["controllers"] == list(subject.CONTROLLERS)
    assert result["first_position_rotation"] == [
        "v2_only",
        "component16",
        "hotspot16",
        "v2_only",
    ]
    assert result["wall_time_budget_seconds"] == 30.0
    assert result["episode_process_fuse_seconds"] == 45.0
    assert result["maximum_registered_timed_seconds"] == 360.0
    assert result["maximum_timed_process_fuse_seconds"] == 540.0
    assert result["maximum_qualification_process_fuse_seconds"] == 180.0
    assert result["maximum_qualification_plus_timed_process_fuse_seconds"] == 720.0
    assert result["strict_serial_timing"] is True
    assert result["solver_or_controller_invoked"] is False
    assert result["official_adaptive"] is False
    assert result["q0_geometry_audit"] is False
    assert result["bootstrap"] is False
    assert result["auc_gate"] is False


def test_schedule_rotates_three_controllers_once_per_key() -> None:
    _path, _root, config = subject.load_config(CONFIG)
    rows = subject.schedule(config)
    assert len(rows) == 12
    assert len(
        {
            (row["group_id"], row["task_id"], row["solver_seed"])
            for row in rows
        }
    ) == 4
    for start in range(0, len(rows), 3):
        block = rows[start : start + 3]
        assert {row["controller"] for row in block} == set(subject.CONTROLLERS)
        assert [row["within_key_position"] for row in block] == [0, 1, 2]
        assert {row["solver_seed"] for row in block} == {17}
    assert [rows[index]["controller"] for index in range(0, 12, 3)] == [
        "v2_only",
        "component16",
        "hotspot16",
        "v2_only",
    ]


def test_controller_contract_exposes_only_one_fixed16_family() -> None:
    _path, root, config = subject.load_config(CONFIG)
    baseline = subject.controller_kwargs(root, config, "v2_only")
    assert "hybridstructpool_augmentation" not in baseline
    for controller, profile in subject.PROFILES.items():
        kwargs = subject.controller_kwargs(root, config, controller)
        augmentation = kwargs["hybridstructpool_augmentation"]
        assert augmentation["source_mode"] == "structshell_single_family"
        assert augmentation["structural_profile"] == profile
        assert augmentation["nominal_size"] == 16
        assert augmentation["maximum_added_candidates"] == 1
        assert len(augmentation["runtime_structural_family_sizes"]) == 1
        assert list(augmentation["runtime_structural_family_sizes"].values()) == [
            [16]
        ]


def test_config_rejects_scope_or_time_expansion(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["runtime"]["wall_time_budget_seconds"] = 180.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime contract changed"):
        subject.load_config(changed)


def test_stage_b_selection_is_ttf_only_and_never_promotes() -> None:
    groups = {f"map-{index}": 20.0 for index in range(4)}
    summaries = {
        "v2_only": {
            "success_count": 4,
            "mean_restricted_ttf": 20.0,
            "restricted_ttf_by_group": groups,
        },
        "component16": {
            "success_count": 4,
            "mean_restricted_ttf": 18.5,
            "restricted_ttf_by_group": {
                "map-0": 18.0,
                "map-1": 19.0,
                "map-2": 18.0,
                "map-3": 19.0,
            },
        },
        "hotspot16": {
            "success_count": 4,
            "mean_restricted_ttf": 18.0,
            "restricted_ttf_by_group": {
                "map-0": 17.0,
                "map-1": 17.0,
                "map-2": 21.0,
                "map-3": 17.0,
            },
        },
    }
    result = subject.select_stage_b_candidate(summaries)
    assert result["eligible_controllers"] == ["component16", "hotspot16"]
    assert result["selected_controller"] == "hotspot16"
    assert result["metrics"]["hotspot16"] == {
        "relative_restricted_ttf_improvement": 0.1,
        "paired_non_worse_count": 3,
        "paired_key_count": 4,
    }
    assert result["decision"] == "advance_one_family_to_second_key_screen"
    assert result["promotion_or_speed_claim_allowed"] is False


def test_stage_b_requires_five_percent_and_three_non_worse_pairs() -> None:
    baseline = {f"map-{index}": 20.0 for index in range(4)}
    summaries = {
        "v2_only": {
            "success_count": 4,
            "mean_restricted_ttf": 20.0,
            "restricted_ttf_by_group": baseline,
        },
        "component16": {
            "success_count": 4,
            "mean_restricted_ttf": 19.2,
            "restricted_ttf_by_group": {key: 19.2 for key in baseline},
        },
        "hotspot16": {
            "success_count": 4,
            "mean_restricted_ttf": 19.0,
            "restricted_ttf_by_group": {
                "map-0": 17.0,
                "map-1": 17.0,
                "map-2": 21.0,
                "map-3": 21.0,
            },
        },
    }
    result = subject.select_stage_b_candidate(summaries)
    assert result["eligible_controllers"] == []
    assert result["gates"]["component16"] == {
        "success_noninferior": True,
        "overall_relative_restricted_ttf_improvement_at_least_five_percent": False,
        "paired_non_worse_at_least_three_of_four": True,
    }
    assert result["gates"]["hotspot16"] == {
        "success_noninferior": True,
        "overall_relative_restricted_ttf_improvement_at_least_five_percent": True,
        "paired_non_worse_at_least_three_of_four": False,
    }
