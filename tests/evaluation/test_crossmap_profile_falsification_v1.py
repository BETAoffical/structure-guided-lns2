from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest

import experiments.crossmap_profile_falsification_v1 as subject


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "crossmap_profile_falsification_v1.json"


def test_plan_is_four_fresh_keys_and_invokes_nothing() -> None:
    result = subject.plan(CONFIG)
    assert result["map_count"] == 2
    assert result["task_count"] == 2
    assert result["paired_key_count"] == 4
    assert result["timed_episode_count"] == 12
    assert result["qualification_reset_count"] == 4
    assert result["solver_seeds"] == [19, 20]
    assert result["controllers"] == list(subject.CONTROLLERS)
    assert result["keys"] == [
        "maze-32-32-4@19",
        "random-32-32-20-high-load@19",
        "maze-32-32-4@20",
        "random-32-32-20-high-load@20",
    ]
    assert result["first_position_rotation"] == [
        "v2_only",
        "component16",
        "hotspot16",
        "v2_only",
    ]
    assert result["wall_time_budget_seconds"] == 15.0
    assert result["episode_process_fuse_seconds"] == 30.0
    assert result["maximum_registered_timed_seconds"] == 180.0
    assert result["maximum_timed_process_fuse_seconds"] == 360.0
    assert result["maximum_qualification_process_fuse_seconds"] == 120.0
    assert result["maximum_qualification_plus_timed_process_fuse_seconds"] == 480.0
    assert result["strict_serial_timing"] is True
    assert result["early_stop_after_first_failed_completed_key"] is True
    assert result["solver_or_controller_invoked"] is False
    assert result["router_executed"] is False
    assert result["official_adaptive"] is False
    assert result["bootstrap"] is False
    assert result["auc_gate"] is False


def test_schedule_rotates_three_controllers_for_each_map_seed_key() -> None:
    _path, _root, config = subject.load_config(CONFIG)
    rows = subject.schedule(config)
    assert len(rows) == 12
    assert len(
        {
            (row["group_id"], row["task_id"], row["solver_seed"])
            for row in rows
        }
    ) == 4
    assert [rows[index]["key_id"] for index in range(0, 12, 3)] == [
        "maze-32-32-4@19",
        "random-32-32-20-high-load@19",
        "maze-32-32-4@20",
        "random-32-32-20-high-load@20",
    ]
    assert [rows[index]["controller"] for index in range(0, 12, 3)] == [
        "v2_only",
        "component16",
        "hotspot16",
        "v2_only",
    ]
    for start in range(0, len(rows), 3):
        block = rows[start : start + 3]
        assert {row["controller"] for row in block} == set(subject.CONTROLLERS)
        assert [row["within_key_position"] for row in block] == [0, 1, 2]
        assert len({row["solver_seed"] for row in block}) == 1


def test_controller_contract_keeps_v2_and_only_one_fixed16_family() -> None:
    _path, root, config = subject.load_config(CONFIG)
    baseline = subject.controller_kwargs(root, config, "v2_only")
    assert "hybridstructpool_augmentation" not in baseline
    assert baseline["wall_time_budget_seconds"] == 15.0
    assert baseline["episode_process_timeout_seconds"] == 30.0
    for controller, profile in subject.PROFILES.items():
        kwargs = subject.controller_kwargs(root, config, controller)
        augmentation = kwargs["hybridstructpool_augmentation"]
        assert augmentation["source_mode"] == "structshell_single_family"
        assert augmentation["structural_profile"] == profile
        assert augmentation["nominal_size"] == 16
        assert augmentation["maximum_added_candidates"] == 1
        assert list(augmentation["runtime_structural_family_sizes"].values()) == [
            [16]
        ]


def test_config_rejects_more_time_new_seed_or_threshold_tuning(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["runtime"]["wall_time_budget_seconds"] = 30.0
    changed = tmp_path / "time.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime contract changed"):
        subject.load_config(changed)

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["cohort"]["solver_seeds"] = [18, 19, 20]
    changed = tmp_path / "seeds.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cohort changed"):
        subject.load_config(changed)

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["frozen_profile_hypothesis"][
        "component16_minimum_fixed_size_to_lcc_ratio"
    ] = 0.14
    changed = tmp_path / "threshold.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hypothesis changed"):
        subject.load_config(changed)


def _summary(ttf: float, success: bool = True) -> dict[str, object]:
    return {"capped_wall_time_to_feasible": ttf, "success": success}


def test_key_gate_requires_the_registered_family_to_be_strictly_fastest() -> None:
    maze = subject.evaluate_key(
        "component16",
        {
            "v2_only": _summary(4.0),
            "component16": _summary(2.0),
            "hotspot16": _summary(5.0),
        },
    )
    assert maze["passed"] is True
    assert maze["gates"] == {
        "expected_winner_strictly_fastest": True,
        "expected_winner_success_noninferior": True,
    }

    random = subject.evaluate_key(
        "hotspot16",
        {
            "v2_only": _summary(12.0),
            "component16": _summary(15.0, False),
            "hotspot16": _summary(9.0),
        },
    )
    assert random["passed"] is True


def test_key_gate_fails_on_tie_or_success_inferiority() -> None:
    tied = subject.evaluate_key(
        "component16",
        {
            "v2_only": _summary(2.0),
            "component16": _summary(2.0),
            "hotspot16": _summary(3.0),
        },
    )
    assert tied["passed"] is False
    assert tied["gates"]["expected_winner_strictly_fastest"] is False

    inferior = subject.evaluate_key(
        "hotspot16",
        {
            "v2_only": _summary(15.0, True),
            "component16": _summary(15.0, False),
            "hotspot16": _summary(10.0, False),
        },
    )
    assert inferior["passed"] is False
    assert inferior["gates"]["expected_winner_success_noninferior"] is False


def test_seed_freshness_and_task_hashes_are_frozen() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    audit = payload["freshness_audit"]
    assert audit["selected_solver_seeds"] == [19, 20]
    assert not set(audit["selected_solver_seeds"]) & set(
        audit["maze_consumed_solver_seeds"]
    )
    assert not set(audit["selected_solver_seeds"]) & set(
        audit["random_consumed_solver_seeds"]
    )
    correction = audit["pre_controller_identity_correction"]
    assert correction["selected_solver_seeds"] == [19, 20]
    assert correction["outcome_fields_read"] is False
    assert correction["excluded"] == [
        {
            "controller_episode_match_count": 0,
            "exact_manifest_match_count": 20,
            "matches": [
                {
                    "qualification_manifest_match_count": 10,
                    "source_experiment_id": (
                        "stride-structshell-overall-rollback-screen-v1"
                    ),
                },
                {
                    "qualification_manifest_match_count": 2,
                    "source_experiment_id": (
                        "stride-structshell-rollback-aware-ttf-v1"
                    ),
                },
                {
                    "qualification_manifest_match_count": 8,
                    "source_experiment_id": (
                        "stride-structshell-rollback-aware-ttf-v1-r2"
                    ),
                },
            ],
            "reason": "qualification_only_exact_task_artifacts",
            "solver_seed": 18,
            "source_experiment_ids": [
                "stride-structshell-overall-rollback-screen-v1",
                "stride-structshell-rollback-aware-ttf-v1",
                "stride-structshell-rollback-aware-ttf-v1-r2",
            ],
        }
    ]
    assert payload["inputs"]["maze_task"]["sha256"] == (
        "c441ca5a2f7e5178bba952d80ec8769c6e123df88d772a1c1293213733c4abd7"
    )
    assert payload["inputs"]["random_task"]["sha256"] == (
        "5fc8509b1c2415a88ec021e6a225f84b79b7b027ef98ed3fad6bf1d058999b54"
    )


def test_machine_freshness_scan_finds_exact_artifact_and_excludes_output(
    tmp_path: Path,
) -> None:
    task_id = "maze-32-32-4__random_04__agents_0200"
    old = tmp_path / "build" / "old" / "qualification_manifest.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "solver_seed": 19,
                "status": "ok",
                "initial_conflicts": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "build" / "current"
    ignored = output / "qualification_manifest.jsonl"
    ignored.parent.mkdir(parents=True)
    ignored.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")

    audit = subject.scan_freshness_artifacts(
        tmp_path,
        output,
        task_ids={task_id},
        solver_seeds={19, 20},
        config_sha256="0" * 64,
    )
    assert audit["passed"] is False
    assert audit["match_count"] == 1
    assert audit["matches"][0]["path"] == (
        "build/old/qualification_manifest.jsonl"
    )
    assert audit["inventory_sha256"]


def test_inner_run_identity_recomputes_expected_augmentation_and_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _path, root, config = subject.load_config(CONFIG)
    item = subject.schedule(config)[1]
    group = subject._group(config, str(item["group_id"]))
    runtime = subject._runtime_config_path(
        tmp_path, group, int(item["solver_seed"])
    )
    configuration = subject._expected_effective_configuration(
        root, config, item, runtime
    )
    assert configuration["proposal"]["hybridstructpool"]["nominal_size"] == 16
    configuration_fingerprint = subject.json_fingerprint(configuration)
    payload = {
        "schema": subject.CLOSED_LOOP_SCHEMA,
        "formal": False,
        "dataset": str((root / str(group["dataset"])).resolve()),
        "dataset_fingerprint": "a" * 64,
        "configuration": configuration,
        "configuration_fingerprint": configuration_fingerprint,
        "frozen_models": {"id": "frozen"},
        "controller_bundle": {"id": "v2"},
        "diagnostic_shadow_bundles": {},
        "v3_s3_bundle": None,
        "controller_implementation": {"sha256": "b" * 64},
        "trace_format": "delta-gzip-v2",
        "storage_fingerprint": "c" * 64,
        "controller": "v2-full",
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
    }
    payload["run_fingerprint"] = subject.json_fingerprint(
        {
            "dataset_fingerprint": payload["dataset_fingerprint"],
            "configuration_fingerprint": configuration_fingerprint,
            "freeze_manifest": payload["frozen_models"],
            "controller_bundle_manifest": payload["controller_bundle"],
            "diagnostic_shadow_bundle_manifests": {},
            "v3_s3_bundle_manifest": None,
            "controller_implementation": payload["controller_implementation"],
        }
    )
    monkeypatch.setattr(
        subject,
        "_expected_episode_run",
        lambda *_args, **_kwargs: {
            "run_fingerprint": payload["run_fingerprint"],
            "trace_format": payload["trace_format"],
            "storage_fingerprint": payload["storage_fingerprint"],
        },
    )
    assert subject._validate_run_config_identity(
        root, tmp_path, config, item, payload, runtime_path=runtime
    ) == payload["run_fingerprint"]

    tampered = json.loads(json.dumps(payload))
    tampered["configuration"]["proposal"]["hybridstructpool"][
        "nominal_size"
    ] = 32
    with pytest.raises(ValueError, match="inner run identity changed"):
        subject._validate_run_config_identity(
            root, tmp_path, config, item, tampered, runtime_path=runtime
        )


def test_analysis_rejects_manifest_without_a_real_trace(tmp_path: Path) -> None:
    _path, _root, config = subject.load_config(CONFIG)
    block = subject.schedule(config)[:3]
    ttf = {"v2_only": 3.0, "component16": 4.0, "hotspot16": 5.0}
    for item in block:
        manifest = subject._manifest_path(tmp_path, item)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "wall_time_budget_seconds": 15.0,
            "stop_reason": "success",
            "invalid_action_count": 0,
            "fingerprint_mismatch_count": 0,
            "capped_wall_time_to_feasible": ttf[str(item["controller"])],
            "success": True,
            "initial_fingerprint": "shared-initial-state",
            "initial_conflicts": 312,
        }
        manifest.write_text(
            json.dumps(
                {
                    "task_id": item["task_id"],
                    "solver_seed": item["solver_seed"],
                    "status": "ok",
                    "summary": summary,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    qualification = (
        subject._key_root(tmp_path, block[0])
        / "qualification"
        / "qualification_manifest.jsonl"
    )
    qualification.parent.mkdir(parents=True, exist_ok=True)
    qualification.write_text(
        json.dumps(
            {
                "task_id": block[0]["task_id"],
                "solver_seed": block[0]["solver_seed"],
                "state_fingerprint": "shared-initial-state",
                "initial_conflicts": 312,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = subject.analyze(CONFIG, tmp_path)
    assert report["integrity_passed"] is False
    assert report["decision"] == "invalid_or_incomplete_profile_falsification"
    assert report["evaluated_paired_key_count"] == 1
    assert report["executed_episode_count"] == 3
    assert report["cancelled_episode_count"] == 0
    assert report["first_failed_key"] is None
    assert report["trace_integrity"] == []
    assert any("manifest trace path is missing" in error for error in report["errors"])


def test_run_cancels_remaining_nine_episodes_after_first_failed_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stored: dict[tuple[str, str], dict[str, object]] = {}
    executed: list[dict[str, object]] = []

    monkeypatch.setattr(subject, "closed_loop_producer_identity", lambda **_kwargs: {})
    monkeypatch.setattr(
        subject,
        "_prepare_freshness_audit",
        lambda *_args, **_kwargs: {
            "passed": True,
            "match_count": 0,
            "scan_errors": [],
            "inventory_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        subject,
        "_freshness_status_base",
        lambda base, *_args: {
            **dict(base),
            "freshness_audit_sha256": "4" * 64,
            "freshness_inventory_sha256": "3" * 64,
            "freshness_match_count": 0,
        },
    )
    monkeypatch.setattr(
        subject,
        "prepare_resumable_output",
        lambda *_args, **_kwargs: SimpleNamespace(
            base_status={
                "schema": subject.STATUS_SCHEMA,
                "config_sha256": "0" * 64,
                "schedule_sha256": "1" * 64,
                "total_schedule_entries": 12,
                "producer_identity": {},
                "run_fingerprint": "2" * 64,
            },
            status={"completed_schedule_entries": 0, "complete": False},
            resumed=False,
            completed_report=None,
        ),
    )
    monkeypatch.setattr(
        subject,
        "_qualify_key",
        lambda *_args, **_kwargs: tmp_path / "qualification",
    )

    def fake_manifest(_output: Path, item: Mapping[str, object]):
        return stored.get((str(item["key_id"]), str(item["controller"])))

    def fake_episode(
        _root: Path,
        _output: Path,
        _config: Mapping[str, object],
        item: Mapping[str, object],
        _qualification: Path,
    ) -> dict[str, object]:
        copied = dict(item)
        executed.append(copied)
        row = {"status": "ok"}
        stored[(str(item["key_id"]), str(item["controller"]))] = row
        return row

    monkeypatch.setattr(subject, "_manifest_row", fake_manifest)
    monkeypatch.setattr(subject, "_run_episode", fake_episode)

    def fake_analyze(
        _config: Path,
        output: Path,
        *,
        producer: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        report = {
            "schema": subject.REPORT_SCHEMA,
            "decision": "stop_profile_hypothesis_falsified",
            "errors": [],
            "producer_identity": dict(producer or {}),
        }
        subject.write_json(output / subject.REPORT_FILENAME, report)
        return report

    monkeypatch.setattr(subject, "analyze", fake_analyze)
    result = subject.run(CONFIG, tmp_path)
    assert result["decision"] == "stop_profile_hypothesis_falsified"
    assert len(executed) == 3
    assert {str(row["controller"]) for row in executed} == set(subject.CONTROLLERS)
    status = json.loads(
        (tmp_path / subject.STATUS_FILENAME).read_text(encoding="utf-8")
    )
    assert status["complete"] is True
    assert status["scientific_stop"] is True
    assert status["completed_schedule_entries"] == 12
    assert status["executed_schedule_entries"] == 3
    assert status["cancelled_schedule_entries"] == 9


def test_resume_stops_before_reset_when_completed_block_analysis_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _path, _root, config = subject.load_config(CONFIG)
    completed = {
        (str(row["key_id"]), str(row["controller"])): {"status": "ok"}
        for row in subject.schedule(config)[:3]
    }
    monkeypatch.setattr(subject, "closed_loop_producer_identity", lambda **_kwargs: {})
    monkeypatch.setattr(
        subject,
        "prepare_resumable_output",
        lambda *_args, **_kwargs: SimpleNamespace(
            base_status={
                "schema": subject.STATUS_SCHEMA,
                "config_sha256": "0" * 64,
                "schedule_sha256": "1" * 64,
                "total_schedule_entries": 12,
                "producer_identity": {},
                "run_fingerprint": "2" * 64,
            },
            status={"completed_schedule_entries": 0, "complete": False},
            resumed=False,
            completed_report=None,
        ),
    )
    monkeypatch.setattr(
        subject,
        "_prepare_freshness_audit",
        lambda *_args, **_kwargs: {
            "passed": True,
            "match_count": 0,
            "scan_errors": [],
            "inventory_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        subject,
        "_freshness_status_base",
        lambda base, *_args: {
            **dict(base),
            "freshness_audit_sha256": "4" * 64,
            "freshness_inventory_sha256": "3" * 64,
            "freshness_match_count": 0,
        },
    )
    monkeypatch.setattr(
        subject,
        "_manifest_row",
        lambda _output, item: completed.get(
            (str(item["key_id"]), str(item["controller"]))
        ),
    )
    monkeypatch.setattr(
        subject,
        "analyze",
        lambda *_args, **_kwargs: {
            "decision": "invalid_or_incomplete_profile_falsification",
            "errors": ["trace SHA mismatch"],
        },
    )

    def forbidden_reset(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("an invalid completed block must stop before reset")

    monkeypatch.setattr(subject, "_qualify_key", forbidden_reset)
    result = subject.run(CONFIG, tmp_path)
    assert result["terminal_failure"]["phase"] == "analysis"
    assert result["terminal_failure"]["terminal"] is True
    assert result["terminal_failure"]["completed_block"] == "maze-32-32-4@19"
