from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.stride_structshell_fourmap_fivearm_quick as subject


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structshell_fourmap_fivearm_quick_v1.json"


def test_plan_is_exactly_four_maps_five_arms_and_invokes_nothing() -> None:
    result = subject.plan(CONFIG)
    assert result["map_count"] == 4
    assert result["task_count"] == 4
    assert result["paired_key_count"] == 4
    assert result["controller_count"] == 5
    assert result["timed_episode_count"] == 20
    assert result["paired_reset_anchor_count"] == 4
    assert result["solver_seed"] == 23
    assert result["controllers"] == list(subject.CONTROLLERS)
    assert result["keys"] == [
        "maze-32-32-4@23",
        "random-32-32-20-high-load@23",
        "room-64-64-16@23",
        "warehouse-w1020a-opposite-exchange@23",
    ]
    assert result["first_position_rotation"] == [
        "official_adaptive",
        "v2_only",
        "component16",
        "hotspot16",
    ]
    assert result["wall_time_budget_seconds"] == 60.0
    assert result["episode_process_fuse_seconds"] == 75.0
    assert result["maximum_registered_timed_seconds"] == 1200.0
    assert result["maximum_timed_process_fuse_seconds"] == 1500.0
    assert result["maximum_reset_anchor_process_fuse_seconds"] == 300.0
    assert result["maximum_reset_plus_timed_process_fuse_seconds"] == 1800.0
    assert result["strict_serial_timing"] is True
    assert result["reset_inclusive_ttf"] is True
    assert result["solver_or_controller_invoked"] is False
    assert result["map_generation"] is False
    assert result["global_freshness_scan"] is False
    assert result["q0_geometry_audit"] is False
    assert result["q1_state_supply_audit"] is False
    assert result["bootstrap"] is False
    assert result["auc_gate"] is False


def test_schedule_rotates_five_arms_strictly_serial() -> None:
    _path, _root, config = subject.load_config(CONFIG)
    rows = subject.schedule(config)
    assert len(rows) == 20
    assert len(
        {
            (row["group_id"], row["task_id"], row["solver_seed"])
            for row in rows
        }
    ) == 4
    for start in range(0, 20, 5):
        block = rows[start : start + 5]
        assert {row["controller"] for row in block} == set(subject.CONTROLLERS)
        assert [row["within_key_position"] for row in block] == [0, 1, 2, 3, 4]
        assert {row["solver_seed"] for row in block} == {23}
    assert [rows[index]["controller"] for index in range(0, 20, 5)] == [
        "official_adaptive",
        "v2_only",
        "component16",
        "hotspot16",
    ]


def test_all_five_controller_contracts_are_exact() -> None:
    _path, root, config = subject.load_config(CONFIG)
    official = subject.controller_kwargs(root, config, "official_adaptive")
    assert official["controller"] == "official_adaptive"
    assert official["controller_runtime"] == "reference"
    assert "hybridstructpool_augmentation" not in official

    v2 = subject.controller_kwargs(root, config, "v2_only")
    assert v2["controller"] == "v2-full"
    assert v2["controller_runtime"] == "optimized"
    assert "hybridstructpool_augmentation" not in v2

    for controller, profile in subject.PROFILES.items():
        augmentation = subject.controller_kwargs(root, config, controller)[
            "hybridstructpool_augmentation"
        ]
        assert augmentation["source_mode"] == "structshell_single_family"
        assert augmentation["structural_profile"] == profile
        assert augmentation["nominal_size"] == 16
        assert augmentation["maximum_added_candidates"] == 1

    dual = subject.controller_kwargs(root, config, "dual16")[
        "hybridstructpool_augmentation"
    ]
    assert dual["pool_id"] == "stride-structshell-dual16-v1"
    assert dual["runtime_id"] == "stride-structshell-dual16-runtime-v1"
    assert dual["source_mode"] == "structshell_dual16"
    assert dual["runtime_structural_family_sizes"] == {
        "conflict_component": [16],
        "spatiotemporal_hotspot": [16],
    }
    assert dual["maximum_added_candidates"] == 2
    assert dual["maximum_total_candidates"] == 65

    for controller in subject.CONTROLLERS:
        kwargs = subject.controller_kwargs(root, config, controller)
        assert kwargs["wall_time_budget_seconds"] == 60.0
        assert kwargs["environment_time_limit_seconds"] == 60.0
        assert kwargs["episode_process_timeout_seconds"] == 75.0


def test_seed23_targeted_identity_audit_is_frozen() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    audit = payload["solver_seed_identity_audit"]
    assert audit == subject.SEED_IDENTITY_AUDIT
    assert audit["selected_solver_seed"] == 23
    assert audit["exact_task_count"] == 4
    assert audit["controller_result_match_count"] == 0
    assert audit["global_freshness_scan"] is False
    assert audit["outcome_fields_read"] is False


def test_official_arm_uses_official_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _path, root, config = subject.load_config(CONFIG)
    item = subject.schedule(config)[0]
    assert item["controller"] == "official_adaptive"
    phases: list[str] = []

    monkeypatch.setattr(
        subject,
        "_runtime_config_path",
        lambda *_args, **_kwargs: CONFIG,
    )
    monkeypatch.setattr(
        subject,
        "run_closed_loop_collection",
        lambda *_args, **kwargs: phases.append(str(kwargs["phase"])),
    )
    monkeypatch.setattr(subject, "_manifest_row", lambda *_args: {"status": "ok"})

    result = subject._run_episode(root, tmp_path, config, item, tmp_path / "anchor")
    assert result == {"status": "ok"}
    assert phases == ["qualify", "official_adaptive"]


def test_materialized_runtime_is_bounded_without_touching_source(
    tmp_path: Path,
) -> None:
    _path, _root, config = subject.load_config(CONFIG)
    for group in config["cohort"]["groups"]:
        source = Path(str(group["_runtime_path"]))
        before = subject.sha256_file(source)
        runtime = subject._runtime_config_path(tmp_path, group)
        payload = subject.read_json(runtime)
        assert payload["solver_seeds"] == [23]
        assert payload["wall_time_budget_seconds"] == 60.0
        assert payload["episode_process_timeout_seconds"] == 75.0
        assert payload["environment"]["time_limit"] == 60.0
        assert payload["workers"] == 1
        assert subject.sha256_file(source) == before


def test_dry_run_cannot_call_solver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run invoked the solver")

    monkeypatch.setattr(subject, "run_closed_loop_collection", forbidden)
    result = subject.run(CONFIG, tmp_path, dry_run=True)
    assert result == subject.plan(CONFIG)
    assert result["solver_or_controller_invoked"] is False
    assert list(tmp_path.iterdir()) == []


def test_config_rejects_time_or_scope_expansion(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["runtime"]["wall_time_budget_seconds"] = 180.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime contract changed"):
        subject.load_config(changed)

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["cohort"]["groups"].append(dict(payload["cohort"]["groups"][0]))
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cohort changed"):
        subject.load_config(changed)
