from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.stride_warehouse_fixed16_development as subject
from experiments.stride_warehouse_fixed16_development import (
    AGENT_COUNT_LADDER,
    BOUNDARY16_STATIC_CACHE,
    CHALLENGERS,
    CONTROLLERS,
    FINAL_NAMESPACE,
    MAP_IDS,
    REPORT_SCHEMA,
    STATUS_SCHEMA,
    _audit_formal_identity,
    _completed_schedule_prefix_length,
    _controller_report_metrics,
    _failed_episode_job,
    _fingerprint,
    _load_selection,
    _normalized_manifest_row,
    _terminal_manifest_failures,
    _validate_inner_run_config_payload,
    _validate_manifest_trace_row,
    choose_winner,
    controller_kwargs,
    development_task_specs,
    formal_schedule,
    load_config,
    performance_gate,
    plan,
    qualification_schedule,
    select_qualified_cohort,
)
from experiments._common import sha256_file
from experiments.run_output_guard import (
    RESUMABLE_RUN_IDENTITY_SCHEMA,
    RUNNER_CONFIG_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_warehouse_fixed16_development_v1.json"


def _qualification_report(config: dict, *, minimum_load: int = 600) -> dict:
    rows = []
    for item in qualification_schedule(config):
        threshold = 16 if item["variant"] == "opposite_exchange" else 1
        conflicts = threshold if item["agent_count"] >= minimum_load else threshold - 1
        rows.append(
            {
                "task_id": item["task_id"],
                "solver_seed": item["solver_seed"],
                "initial_conflicts": conflicts,
                "initial_feasible": conflicts == 0,
                "initial_complete": True,
                "initial_state_consistent": True,
                "state_fingerprint": f"{len(rows):064x}",
            }
        )
    return {
        "errors": [],
        "valid_count": len(rows),
        "incomplete_reset_count": 0,
        "inconsistent_initial_state_count": 0,
        "gates": {"all_resets_valid": True},
        "natural_distribution": {"tasks": rows},
    }


def test_config_freezes_development_and_final_namespaces() -> None:
    _path, _root, config = load_config(CONFIG)
    assert tuple(config["controllers"]) == CONTROLLERS
    assert tuple(row["id"] for row in config["cohort"]["maps"]) == MAP_IDS
    assert config["controller_contract"]["single_family_semantics"] == (
        "all_state_fixed_family_ablation"
    )
    assert config["controller_contract"]["boundary16_static_cache"] == (
        BOUNDARY16_STATIC_CACHE
    )
    assert config["final_namespace"] == FINAL_NAMESPACE
    assert config["final_namespace"]["generated"] is False
    assert config["final_namespace"]["reset"] is False


def test_plan_and_reset_schedule_have_registered_dimensions() -> None:
    _path, _root, config = load_config(CONFIG)
    tasks = development_task_specs(config)
    resets = qualification_schedule(config)
    assert len(tasks) == 64
    assert len({item["task_id"] for item in tasks}) == 64
    assert {item["agent_count"] for item in tasks} == set(AGENT_COUNT_LADDER)
    assert len(resets) == 256
    assert len({(item["task_id"], item["solver_seed"]) for item in resets}) == 256
    result = plan(CONFIG)
    assert result["formal_paired_key_count_after_qualification"] == 32
    assert result["formal_episode_count_after_qualification"] == 192
    assert result["final_namespace_touched"] is False


def test_registered_short_task_ids_keep_windows_trace_paths_below_240() -> None:
    _path, _root, config = load_config(CONFIG)
    task_id = max(
        (row["task_id"] for row in development_task_specs(config)), key=len
    )
    trace = (
        ROOT
        / "build"
        / "stride-warehouse-fixed16-development-v1"
        / "formal"
        / "controllers"
        / "b16"
        / "episodes"
        / "balanced_wall_clock"
        / "realized_dynamic"
        / f"{task_id}__seed_0022__realized_dynamic.jsonl.gz"
    )
    assert len(str(trace.resolve())) < 240


def test_qualification_selects_lowest_common_both_variant_load() -> None:
    _path, _root, config = load_config(CONFIG)
    selection = select_qualified_cohort(config, _qualification_report(config))
    assert selection["passed"] is True
    assert selection["selected_loads"] == {map_id: 600 for map_id in MAP_IDS}
    assert selection["final_namespace_touched"] is False


def test_qualification_rejects_missing_or_zero_random_evidence() -> None:
    _path, _root, config = load_config(CONFIG)
    report = _qualification_report(config)
    report["natural_distribution"]["tasks"].pop()
    selection = select_qualified_cohort(config, report)
    assert selection["passed"] is False
    assert any("coverage" in error for error in selection["errors"])

    report = _qualification_report(config)
    for row in report["natural_distribution"]["tasks"]:
        if (
            row["task_id"].startswith("w1020a__")
            and "__ur__" in row["task_id"]
        ):
            row["initial_conflicts"] = 0
            row["initial_feasible"] = True
    selection = select_qualified_cohort(config, report)
    assert selection["passed"] is False
    assert MAP_IDS[0] not in selection["selected_loads"]


def test_formal_schedule_is_checkerboard_half_and_rotating_serial() -> None:
    _path, _root, config = load_config(CONFIG)
    selection = select_qualified_cohort(config, _qualification_report(config))
    rows = formal_schedule(config, selection)
    assert len(rows) == 192
    keys = {
        (row["map_id"], row["task_id"], row["solver_seed"]) for row in rows
    }
    assert len(keys) == 32
    assert sum(row["half"] == "A" for row in rows) == 96
    assert sum(row["half"] == "B" for row in rows) == 96
    for index in range(0, len(rows), 6):
        block = rows[index : index + 6]
        assert {row["controller"] for row in block} == set(CONTROLLERS)
        assert [row["within_key_position"] for row in block] == list(range(6))
    first = [rows[index]["controller"] for index in range(0, len(rows), 6)]
    assert first[:7] == list(CONTROLLERS) + [CONTROLLERS[0]]


def test_boundary_uses_legacy_topology_and_families_are_all_state_fixed16() -> None:
    _path, root, config = load_config(CONFIG)
    boundary = controller_kwargs(root, config, "boundary16_static_cache")
    assert boundary["topology_boundary_augmentation"] == BOUNDARY16_STATIC_CACHE
    assert "hybridstructpool_augmentation" not in boundary
    for controller in CHALLENGERS[1:]:
        kwargs = controller_kwargs(root, config, controller)
        augmentation = kwargs["hybridstructpool_augmentation"]
        assert "topology_boundary_augmentation" not in kwargs
        assert augmentation["nominal_size"] == 16
        assert augmentation["maximum_added_candidates"] == 1
        assert augmentation["activation_gate"] == {
            "gate_id": "single_family_ablation_all_states"
        }


def test_only_registered_hard_gates_affect_eligibility() -> None:
    base = {"success_count": 30}
    challenger = {"success_count": 30}
    better = {"valid": True, "mean_restricted_ttf_delta_seconds": -0.01}
    result = performance_gate(
        integrity_passed=True,
        v2_summary=base,
        challenger_summary=challenger,
        overall=better,
        half_a=better,
        half_b=better,
    )
    assert set(result) == {
        "integrity_and_zero_execution_errors",
        "success_count_not_lower_than_v2",
        "mean_restricted_ttf_strictly_lower_than_v2",
        "half_a_mean_restricted_ttf_strictly_lower_than_v2",
        "half_b_mean_restricted_ttf_strictly_lower_than_v2",
    }
    assert all(result.values())
    result["half_b_mean_restricted_ttf_strictly_lower_than_v2"] = False
    assert not all(result.values())


def test_winner_uses_registered_under_one_percent_tie_breakers() -> None:
    gates = {name: {"passed": name in {"bottleneck16", "hotspot16"}} for name in CHALLENGERS}
    comparisons = {
        name: {
            "challenger_mean_restricted_ttf": (
                100.0 if name == "bottleneck16" else 100.5 if name == "hotspot16" else 120.0
            )
        }
        for name in CHALLENGERS
    }
    metrics = {
        name: {
            "mean_added_candidate_count_per_decision": (
                1.0 if name == "bottleneck16" else 0.5
            ),
            "mean_neighborhood_selection_seconds": 1.0,
        }
        for name in CHALLENGERS
    }
    winner = choose_winner(gates, comparisons, metrics)
    assert winner["near_tie_controllers"] == ["bottleneck16", "hotspot16"]
    assert winner["selected_controller"] == "hotspot16"


def test_near_tie_candidate_metric_counts_boundary_sources_before_dedup() -> None:
    row = {
        "status": "ok",
        "summary": {
            "repair_iterations": 1,
            "controller_totals": {
                "model_decision_count": 2,
                "topology_boundary_generated_count": 4,
                "topology_boundary_added_candidate_count": 1,
            },
        },
    }
    metrics = _controller_report_metrics(
        [row], "boundary16_static_cache"
    )
    assert metrics["total_added_candidate_count"] == 4
    assert metrics["mean_added_candidate_count_per_decision"] == 2.0


def test_failure_builder_is_local_schema_safe_and_resume_rows_normalize(
    tmp_path: Path,
) -> None:
    job = {
        "job_id": "job-1",
        "config_path": str(CONFIG),
        "output_root": str(tmp_path),
        "item": {
            "controller": "hotspot16",
            "task_id": "task",
            "solver_seed": 19,
        },
    }
    failure = _failed_episode_job(job, "timeout", "outer fuse")
    assert failure["status"] == "timeout"
    assert failure["error"] == "outer fuse"
    assert failure["collection_path"].endswith("hs16")
    resumed = _normalized_manifest_row({"status": "resumed", "summary": {}})
    assert resumed["status"] == "ok"
    assert resumed["source_manifest_status"] == "resumed"
    failed = _normalized_manifest_row({"status": "error"})
    assert failed["status"] == "error"


def test_selection_is_recomputed_from_qualification_before_formal_use(
    tmp_path: Path,
) -> None:
    path, _root, config = load_config(CONFIG)
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    report_path = qualification / "qualification_report.json"
    report_path.write_text(
        json.dumps(_qualification_report(config), sort_keys=True), encoding="utf-8"
    )
    selection = select_qualified_cohort(config, _qualification_report(config))
    selection.update(
        {
            "config_sha256": sha256_file(path),
            "qualification_report_sha256": sha256_file(report_path),
        }
    )
    selection_path = tmp_path / "selected_cohort.json"
    selection_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    assert _load_selection(path, tmp_path, config)["selected_loads"][MAP_IDS[0]] == 600
    selection["selected_loads"][MAP_IDS[0]] = 800
    selection_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="recomputed lowest common"):
        _load_selection(path, tmp_path, config)


def test_formal_identity_audit_rejects_stale_producer(tmp_path: Path) -> None:
    path, _root, config = load_config(CONFIG)
    selection = select_qualified_cohort(config, _qualification_report(config))
    items = formal_schedule(config, selection)
    formal = tmp_path / "formal"
    formal.mkdir()
    schedule_sha = _fingerprint(items)
    producer = {"schema": "test-producer", "source_sha256": {"runner": "0" * 64}}
    identity = {
        "schema": RESUMABLE_RUN_IDENTITY_SCHEMA,
        "status_schema": STATUS_SCHEMA,
        "config_sha256": sha256_file(path),
        "schedule_sha256": schedule_sha,
        "producer_identity": producer,
        "report_schema": REPORT_SCHEMA,
    }
    run_fingerprint = _fingerprint(identity)
    (formal / "execution_schedule.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in items),
        encoding="utf-8",
    )
    (formal / "runner_config.json").write_text(
        json.dumps(
            {
                "schema": RUNNER_CONFIG_SCHEMA,
                "schema_version": 1,
                "identity_fingerprint": run_fingerprint,
                "identity": identity,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (formal / "collection_status.json").write_text(
        json.dumps(
            {
                "schema": STATUS_SCHEMA,
                "config_sha256": sha256_file(path),
                "schedule_sha256": schedule_sha,
                "producer_identity": producer,
                "run_fingerprint": run_fingerprint,
                "total_schedule_entries": len(items),
                "completed_schedule_entries": 0,
                "complete": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assert _audit_formal_identity(path, formal, items, producer)["complete"] is False
    with pytest.raises(ValueError, match="producer/run identity"):
        _audit_formal_identity(path, formal, items, {**producer, "extra": True})


def test_existing_terminal_manifest_is_detected_before_resume(tmp_path: Path) -> None:
    manifest = (
        tmp_path
        / "formal"
        / "controllers"
        / "cc16"
        / "realized_dynamic_manifest.jsonl"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "task_id": "task",
                "solver_seed": 21,
                "status": "timeout",
                "error": "process fuse",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _terminal_manifest_failures(tmp_path) == [
        {
            "controller": "component16",
            "task_id": "task",
            "solver_seed": 21,
            "status": "timeout",
            "error": "process fuse",
        }
    ]


def test_resume_requires_existing_manifests_to_be_a_schedule_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [
        {
            "controller": "v2_only",
            "task_id": f"task-{index}",
            "solver_seed": 19,
        }
        for index in range(3)
    ]
    present = {"task-0", "task-2"}
    monkeypatch.setattr(
        subject,
        "_manifest",
        lambda _output, _controller, task_id, _seed: (
            {"status": "ok"} if task_id in present else None
        ),
    )
    with pytest.raises(ValueError, match="strict prefix"):
        _completed_schedule_prefix_length(tmp_path, items)
    present.remove("task-2")
    assert _completed_schedule_prefix_length(tmp_path, items) == 1


def test_inner_run_config_binds_controller_augmentation_and_cohort() -> None:
    cohort = {("task-a", 19), ("task-b", 20)}
    payload = {
        "run_fingerprint": "expected",
        "configuration": {
            "controller": "v2-full",
            "feature_backend": "native",
            "controller_runtime": "optimized",
            "verification_profile": "deployment",
            "stopping_rule": "wall-clock",
            "repair_seed_policy": "episode_stream",
            "deterministic_pp_replay": False,
            "wall_time_budget_seconds": 180.0,
            "episode_process_timeout_seconds": 240.0,
            "environment": {"time_limit": 180.0},
            "cohort_job_keys_override": [["task-a", 19], ["task-b", 20]],
            "proposal": {"topology_boundary": BOUNDARY16_STATIC_CACHE},
        },
    }
    _validate_inner_run_config_payload(
        payload,
        expected_run_fingerprint="expected",
        controller="boundary16_static_cache",
        cohort_keys=cohort,
    )
    payload["configuration"]["proposal"]["topology_boundary"] = {
        **BOUNDARY16_STATIC_CACHE,
        "maximum_added_candidates": 1,
    }
    with pytest.raises(ValueError, match="inner run identity"):
        _validate_inner_run_config_payload(
            payload,
            expected_run_fingerprint="expected",
            controller="boundary16_static_cache",
            cohort_keys=cohort,
        )


def test_trace_audit_rejects_hash_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "episode.jsonl.gz"
    trace.write_bytes(b"trace")
    summary = {"success": True}
    row = {
        "status": "resumed",
        "task_id": "task-a",
        "solver_seed": 19,
        "episode_id": "task-a__seed_0019__realized_dynamic",
        "policy": "realized_dynamic",
        "split": "balanced_wall_clock",
        "map_id": "warehouse",
        "task_variant": "uniform_random_seed_1_agents_400",
        "agent_count": 400,
        "trace_file": trace.name,
        "trace_sha256": sha256_file(trace),
        "trace_event_count": 2,
        "summary": summary,
    }
    monkeypatch.setattr(
        subject,
        "validate_closed_loop_trace",
        lambda *_args, **_kwargs: {"summary": summary, "event_count": 2},
    )
    _validate_manifest_trace_row(
        tmp_path,
        row,
        run_fingerprint="inner-run",
        expected_item={
            "task_id": "task-a",
            "solver_seed": 19,
            "map_id": "warehouse",
            "agent_count": 400,
        },
        expected_dataset_row={
            "task_id": "task-a",
            "map_id": "warehouse",
            "task_variant": "uniform_random_seed_1_agents_400",
            "agent_count": 400,
        },
        metric_iteration_budget=None,
    )
    row["trace_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="trace hash changed"):
        _validate_manifest_trace_row(
            tmp_path,
            row,
            run_fingerprint="inner-run",
            expected_item={
                "task_id": "task-a",
                "solver_seed": 19,
                "map_id": "warehouse",
                "agent_count": 400,
            },
            expected_dataset_row={
                "task_id": "task-a",
                "map_id": "warehouse",
                "task_variant": "uniform_random_seed_1_agents_400",
                "agent_count": 400,
            },
            metric_iteration_budget=None,
        )


def test_unknown_controller_is_rejected() -> None:
    _path, root, config = load_config(CONFIG)
    with pytest.raises(ValueError, match="unknown Warehouse fixed16 controller"):
        controller_kwargs(root, config, "official_adaptive")
