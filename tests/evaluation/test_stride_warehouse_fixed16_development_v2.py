from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.stride_warehouse_fixed16_development_v2 as subject
from experiments._common import sha256_file


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_warehouse_fixed16_development_v2.json"


def _qualification_report(config: dict) -> dict:
    rows = []
    for item in subject.qualification_schedule(config):
        conflicts = 16 if item["variant"] == "opposite_exchange" else 1
        rows.append(
            {
                "task_id": item["task_id"],
                "solver_seed": item["solver_seed"],
                "initial_conflicts": conflicts,
                "initial_feasible": False,
                "initial_complete": True,
                "initial_state_consistent": True,
                "state_fingerprint": f"{len(rows):064x}",
            }
        )
    return {
        "errors": [],
        "valid_count": 16,
        "incomplete_reset_count": 0,
        "inconsistent_initial_state_count": 0,
        "gates": {"all_resets_valid": True},
        "natural_distribution": {"tasks": rows},
    }


def _passed_selection(config: dict) -> dict:
    result = subject.select_qualified_cohort(config, _qualification_report(config))
    result.update(
        {
            "qualification_manifest_sha256": "a" * 64,
            "qualification_report_sha256": "b" * 64,
        }
    )
    return result


def test_config_is_a_narrow_independent_identity() -> None:
    _path, _root, config = subject.load_config(CONFIG)
    assert config["experiment_id"] == subject.EXPERIMENT_ID
    assert config["cohort"]["map"]["id"] == subject.MAP_ID
    assert config["cohort"]["agent_count"] == 600
    assert config["cohort"]["qualification"]["forbidden_agent_counts"] == [800, 1000]
    assert config["controller_contract"][
        "single_family_maximum_added_candidates_per_decision"
    ] == 1
    assert config["controller_contract"]["boundary16_static_cache"][
        "maximum_added_candidates"
    ] == 2
    assert config["design_provenance"] == subject.DESIGN_PROVENANCE
    assert config["design_provenance"]["r2_overall_qualification_passed"] is False
    assert config["design_provenance"]["r2_formal_episode_count"] == 0
    assert config["design_provenance"]["controller_results_consulted"] is False
    assert config["design_provenance"]["reset_or_episode_artifacts_imported"] is False


def test_plan_has_4_tasks_16_resets_8_pairs_and_48_episodes() -> None:
    result = subject.plan(CONFIG)
    assert result["map_count"] == 1
    assert result["dataset_task_count"] == 4
    assert result["q0_geometry_audit_count"] == 4
    assert result["qualification_reset_count"] == 16
    assert result["formal_paired_key_count_after_qualification"] == 8
    assert result["formal_episode_count_after_qualification"] == 48
    assert result["qualification_required_before_collect"] is True


def test_development_schedule_contains_no_800_or_1000_load() -> None:
    _path, _root, config = subject.load_config(CONFIG)
    tasks = subject.development_task_specs(config)
    resets = subject.qualification_schedule(config)
    assert len(tasks) == 4
    assert {row["agent_count"] for row in tasks} == {600}
    assert len(resets) == 16
    assert len({(row["task_id"], row["solver_seed"]) for row in resets}) == 16


def test_q0_regenerates_all_four_pinned_tasks_in_a_new_root(tmp_path: Path) -> None:
    summary = subject.prepare_development_dataset(CONFIG, tmp_path / "fresh-v2")
    dataset = tmp_path / "fresh-v2" / "dataset"
    audit = json.loads((dataset / subject.Q0_FILENAME).read_text(encoding="utf-8"))
    assert summary["dataset_revision"] == subject.EXPERIMENT_ID
    assert audit["passed"] is True
    assert audit["solver_or_controller_invoked"] is False
    assert audit["old_r2_artifacts_imported"] is False
    assert len(audit["checks"]) == 4
    for task_id, expected in subject.REGISTERED_DEVELOPMENT_TASKS.items():
        scenario = dataset / "balanced_wall_clock" / "scenarios" / f"{task_id}.scen"
        task = dataset / "balanced_wall_clock" / "tasks" / f"{task_id}.json"
        assert sha256_file(scenario) == expected["scenario_sha256"]
        assert sha256_file(task) == expected["task_sha256"]


def test_existing_dataset_reuse_reaudits_task_bytes(tmp_path: Path) -> None:
    output = tmp_path / "fresh-v2"
    subject.prepare_development_dataset(CONFIG, output)
    task = (
        output
        / "dataset"
        / "balanced_wall_clock"
        / "tasks"
        / "w1020a__oe__t0233__n0600.json"
    )
    task.write_text(task.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registered task bytes changed"):
        subject.prepare_development_dataset(CONFIG, output)


def test_existing_dataset_reuse_reaudits_map_and_manifest_identity(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fresh-v2"
    subject.prepare_development_dataset(CONFIG, output)
    split = output / "dataset" / "balanced_wall_clock"
    manifest = split / "manifest.jsonl"
    original_manifest = manifest.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in original_manifest.splitlines() if line]
    rows[0]["map_file"] = "maps/not-the-registered-map.map"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest row changed"):
        subject.prepare_development_dataset(CONFIG, output)

    manifest.write_text(original_manifest, encoding="utf-8")
    map_path = split / "maps" / f"{subject.MAP_ID}.map"
    map_path.write_text(map_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset map bytes changed"):
        subject.prepare_development_dataset(CONFIG, output)


def test_output_cannot_equal_or_nest_under_old_r2() -> None:
    _path, root, _config = subject.load_config(CONFIG)
    forbidden = root / "build" / "wh-f16-v1-r2"
    with pytest.raises(ValueError, match="must not reuse"):
        subject._guard_output(root, forbidden)
    with pytest.raises(ValueError, match="must not reuse"):
        subject._guard_output(root, forbidden / "nested")
    assert subject._guard_output(root, root / "build" / "wh-f16-v2-new") != forbidden


def test_q1_requires_all_16_fresh_reset_rows_and_both_thresholds() -> None:
    _path, _root, config = subject.load_config(CONFIG)
    report = _qualification_report(config)
    selection = subject.select_qualified_cohort(config, report)
    assert selection["passed"] is True
    assert selection["observed_reset_count"] == 16
    assert selection["minimum_initial_conflicts"] == {
        "opposite_exchange": 16,
        "uniform_random": 1,
    }

    report["natural_distribution"]["tasks"].pop()
    assert subject.select_qualified_cohort(config, report)["passed"] is False


def test_q1_manifest_and_report_are_cross_checked(tmp_path: Path) -> None:
    _path, _root, config = subject.load_config(CONFIG)
    report = _qualification_report(config)
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    (qualification / "qualification_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    anchors = {
        (row["task_id"], row["solver_seed"]): row
        for row in report["natural_distribution"]["tasks"]
    }
    rows = []
    for item in subject.qualification_schedule(config):
        anchor = anchors[(item["task_id"], item["solver_seed"])]
        raw_anchor = dict(anchor)
        raw_anchor.pop("initial_state_consistent")
        rows.append(
            {
                **raw_anchor,
                "status": "ok",
                "error": None,
                "map_id": subject.MAP_ID,
                "split": "balanced_wall_clock",
                "agent_count": 600,
                "task_variant": (
                    f"{item['variant']}_seed_{item['task_seed']}_agents_600"
                ),
            }
        )
    manifest = qualification / "qualification_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    subject._audit_qualification_evidence(config, qualification)
    rows[0]["initial_conflicts"] += 1
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="report/manifest mismatch"):
        subject._audit_qualification_evidence(config, qualification)
    rows[0]["initial_conflicts"] -= 1
    report["natural_distribution"]["tasks"][0]["initial_state_consistent"] = False
    (qualification / "qualification_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="inconsistent initial state"):
        subject._audit_qualification_evidence(config, qualification)
    report = _qualification_report(config)
    next(
        row
        for row in report["natural_distribution"]["tasks"]
        if "__ur__" in row["task_id"]
    )["initial_conflicts"] = 0
    assert subject.select_qualified_cohort(config, report)["passed"] is False


def test_formal_schedule_is_hash_bound_rotating_and_half_balanced() -> None:
    _path, _root, config = subject.load_config(CONFIG)
    rows = subject.formal_schedule(config, _passed_selection(config))
    assert len(rows) == 48
    keys = {(row["task_id"], row["solver_seed"]) for row in rows}
    assert len(keys) == 8
    assert sum(row["half"] == "A" for row in rows) == 24
    assert sum(row["half"] == "B" for row in rows) == 24
    assert {row["qualification_manifest_sha256"] for row in rows} == {"a" * 64}
    assert {row["qualification_report_sha256"] for row in rows} == {"b" * 64}
    for index in range(0, len(rows), 6):
        block = rows[index : index + 6]
        assert {row["controller"] for row in block} == set(subject.CONTROLLERS)
        assert [row["within_key_position"] for row in block] == list(range(6))


def test_formal_schedule_refuses_failed_or_unbound_qualification() -> None:
    _path, _root, config = subject.load_config(CONFIG)
    failed = _passed_selection(config)
    failed["passed"] = False
    with pytest.raises(ValueError, match="requires the frozen passed cohort"):
        subject.formal_schedule(config, failed)
    unbound = _passed_selection(config)
    unbound.pop("qualification_manifest_sha256")
    with pytest.raises(ValueError, match="lacks qualification hash binding"):
        subject.formal_schedule(config, unbound)


def test_collect_wires_full_inner_trace_identity_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "legacy._audit_inner_collections(" in source
    assert "require_all=prepared.completed_report is not None" in source
    assert "legacy._completed_schedule_prefix_length(output, items)" in source
    assert "_audit_formal_identity(path, formal, items, expected_producer)" in source


def test_runtime_model_registration_matches_frozen_v2() -> None:
    runtime = json.loads(
        (
            ROOT / "configs" / "stride_warehouse_fixed16_development_runtime_v2.json"
        ).read_text(encoding="utf-8")
    )
    reference = json.loads(
        (ROOT / "configs" / "stride_stage4r_high_load_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert runtime["model_registration"] == reference["model_registration"]
    assert runtime["dataset_design"]["map_count"] == 1
    assert runtime["dataset_design"]["instance_count"] == 4


def test_producer_identity_lists_all_direct_metric_and_dataset_dependencies() -> None:
    assert set(subject.PRODUCER_SOURCE_FILES) >= {
        "experiments/stride_warehouse_fixed16_development_v2.py",
        "experiments/stride_warehouse_fixed16_development.py",
        "experiments/balanced_wall_clock.py",
        "experiments/closed_loop_confirmation.py",
        "experiments/stride_hybridstructpool_routed_confirmation.py",
        "experiments/stride_structpool_ttf_quick.py",
        "lns2_selector/runtime/structshell_single_family.py",
        "lns2_selector/runtime/topology_candidates.py",
    }


def test_next_step_preserves_boundary_final_and_family_size_branches() -> None:
    assert "fresh_final_confirmation" in subject._registered_next_step(
        "boundary16_static_cache"
    )
    for controller in subject.SINGLE_FAMILY_PROFILES:
        assert "size_study_8_16_24_32" in subject._registered_next_step(controller)
    assert subject._registered_next_step(None) == (
        "stop_fixed16_narrow_slice_branch_keep_v2_default"
    )
