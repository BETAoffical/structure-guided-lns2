from __future__ import annotations

import json
import hashlib
import shutil
import contextlib
from pathlib import Path

import pytest

import experiments.stride_warehouse_crossaisle_paired as subject
from experiments._common import sha256_file
from experiments._common import producer_identity
from experiments.balanced_wall_clock import _movingai_passable_cells


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_warehouse_crossaisle_paired_v1.json"
RUNTIME = ROOT / "configs" / "stride_warehouse_crossaisle_paired_runtime_v1.json"
REGISTRY = ROOT / "configs" / "stride_warehouse_crossaisle_paired_tasks_v1.json"


@pytest.fixture(scope="module")
def q0_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("crossaisle-q0") / "output"
    result = subject.prepare_q0(CONFIG, output)
    assert result["q0_passed"] is True
    return output


def _passing_report(config: dict) -> dict:
    tasks = []
    for row in subject.qualification_schedule(config):
        tasks.append(
            {
                **row,
                "initial_conflicts": (
                    16 if row["variant"] == subject.STRUCTURED_VARIANT else 1
                ),
                "active_conflict_agent_count": (
                    32 if row["variant"] == subject.STRUCTURED_VARIANT else 2
                ),
                "largest_conflict_component_size": (
                    16 if row["variant"] == subject.STRUCTURED_VARIANT else 2
                ),
                "initial_complete": True,
                "initial_feasible": False,
                "reported_initial_feasible": False,
                "initial_state_consistent": True,
                "state_fingerprint": f"{len(tasks) + 1:064x}",
            }
        )
    return {
        "schema": subject.QUALIFICATION_REPORT_SCHEMA,
        "experiment_id": subject.EXPERIMENT_ID,
        "phase": "q1_reset_only",
        "expected_reset_count": 128,
        "observed_result_count": 128,
        "valid_reset_count": 128,
        "execution_error_count": 0,
        "process_timeout_count": 0,
        "duplicate_keys": [],
        "unexpected_keys": [],
        "duplicate_solver_seed_trajectories": [],
        "all_expected_results_present": True,
        "controller_step_invoked": False,
        "policy_episode_invoked": False,
        "formal_ttf_invoked": False,
        "tasks": tasks,
        "execution_errors": [],
    }


def _copy_registered_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for relative in (
        "configs/stride_warehouse_crossaisle_paired_runtime_v1.json",
        "configs/stride_warehouse_crossaisle_paired_tasks_v1.json",
        "experiments/warehouse_crossaisle_tasks.py",
        "build/movingai-dev/maps/warehouse-10-20-10-2-1.map",
        "build/movingai-ood-dev/maps/warehouse-10-20-10-2-2.map",
        "build/movingai-dev/maps/warehouse-20-40-10-2-1.map",
        "build/movingai-ood-dev/maps/warehouse-20-40-10-2-2.map",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return root


def _fake_producer() -> dict:
    return producer_identity(
        project_root=ROOT,
        source_files=("experiments/stride_warehouse_crossaisle_paired.py",),
        native_required=False,
    )


def _fake_dataset_rows(config: dict) -> dict[str, dict]:
    return {
        row["task_id"]: {
            "split": "balanced_wall_clock",
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": "warehouse",
            "task_variant": row["variant"],
            "agent_count": row["agent_count"],
        }
        for row in subject.benchmark_task_specs(config)
    }


def _fake_ok_result(job: dict) -> dict:
    row = job["row"]
    structured = row["task_variant"] == subject.STRUCTURED_VARIANT
    material = f"{row['task_id']}::{job['solver_seed']}".encode()
    return {
        "split": row["split"],
        "map_id": row["map_id"],
        "task_id": row["task_id"],
        "layout_mode": row["layout_mode"],
        "task_variant": row["task_variant"],
        "agent_count": row["agent_count"],
        "solver_seed": job["solver_seed"],
        "initial_conflicts": 16 if structured else 1,
        "initial_feasible": False,
        "initial_complete": True,
        "state_fingerprint": hashlib.sha256(material).hexdigest(),
        "initial_complexity": {
            "agent_count": row["agent_count"],
            "conflict_pair_count": 16 if structured else 1,
            "active_conflict_agent_count": 32 if structured else 2,
            "largest_conflict_component_size": 16 if structured else 2,
        },
        "status": "ok",
        "error": None,
    }


def test_config_is_hash_frozen_reset_only_identity() -> None:
    _path, _root, config, registry = subject.load_config(
        CONFIG, require_frozen_registry=True
    )
    assert registry is not None
    assert config["registration"]["q1_allowed"] is True
    assert len(registry["tasks"]) == 32
    assert registry["q0_manifest_sha256"] == (
        "fa084691b492655a9e70ab7f2b7bbe6e049fa78c66986a43de755ff3022d0ee8"
    )
    assert config["phase_contract"]["controller_step_allowed"] is False
    assert config["phase_contract"]["policy_episode_allowed"] is False
    assert config["phase_contract"]["formal_ttf_allowed"] is False
    assert sha256_file(REGISTRY) == config["registration"]["task_registry_sha256"]


@pytest.mark.parametrize(
    "section,field,value",
    (
        ("q0_gates", "required_chosen_cross_aisle_group_count_per_map", 11),
        ("q1_gates", "all_four_maps_must_select_a_q", False),
        ("result_interpretation", "ttf_claim_allowed", True),
        ("cohort", "claim_boundary", "broader_claim"),
    ),
)
def test_config_rejects_gate_or_claim_mutation(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    root = _copy_registered_root(tmp_path)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config[section][field] = value
    path = root / "configs" / "stride_warehouse_crossaisle_paired_v1.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="gates changed|interpretation changed|cohort changed"):
        subject.load_config(path)


def test_pending_registration_mechanically_blocks_q1(tmp_path: Path) -> None:
    root = _copy_registered_root(tmp_path)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["scientific_status"] = (
        "preregistered_q0_generation_pending_task_hash_registration_reset_only"
    )
    config["registration"] = {
        "status": "pending_q0_task_hash_registration",
        "task_registry_path": "configs/stride_warehouse_crossaisle_paired_tasks_v1.json",
        "task_registry_sha256": None,
        "q1_allowed": False,
    }
    config["inputs"]["task_generator"]["sha256"] = None
    path = root / "configs" / "stride_warehouse_crossaisle_paired_v1.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Q1 is blocked"):
        subject.qualify(path, tmp_path / "never-reset", dry_run=True)
    assert not (tmp_path / "never-reset").exists()


def test_plan_and_schedule_are_32_tasks_128_resets_without_formal() -> None:
    result = subject.plan(CONFIG)
    assert result["map_count"] == 4
    assert result["task_count"] == 32
    assert result["qualification_reset_count"] == 128
    assert result["controller_count"] == 0
    assert result["policy_episode_count"] == 0
    assert result["formal_ttf_episode_count"] == 0
    _path, _root, config, _registry = subject.load_config(CONFIG)
    schedule = subject.qualification_schedule(config)
    assert len(schedule) == 128
    assert len({(row["task_id"], row["solver_seed"]) for row in schedule}) == 128
    assert {row["solver_seed"] for row in schedule} == {61, 62, 63, 64}
    assert {row["q"] for row in schedule} == {16, 20}
    assert {row["variant"] for row in schedule} == set(subject.TASK_VARIANTS)


def test_q0_regenerates_registered_tasks_and_reaudits_every_byte(
    tmp_path: Path, q0_base: Path,
) -> None:
    output = tmp_path / "crossaisle-q0"
    shutil.copytree(q0_base, output)
    dataset = output / "dataset"
    q0 = json.loads((dataset / subject.Q0_FILENAME).read_text(encoding="utf-8"))
    assert q0["passed"] is True
    assert q0["solver_or_controller_invoked"] is False
    assert len(q0["pair_audits"]) == 16
    for map_spec in subject.MAPS:
        geometry = q0["map_geometry"][map_spec["id"]]
        assert geometry["internal_group_count"] == map_spec[
            "expected_internal_group_count"
        ]
        assert geometry["aisle_width"] == map_spec["expected_aisle_width"]
        assert geometry["group_count"] == 12
    for audit in q0["pair_audits"]:
        paired = audit["metrics"]["paired_distance"]
        assert paired["mean_relative_difference"] <= 0.05
        assert paired["p95_relative_difference"] <= 0.10
    proposal = json.loads(
        (dataset / subject.Q0_REGISTRATION_PROPOSAL_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    tracked = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert proposal == tracked
    subject.prepare_q0(CONFIG, output)

    first = next(iter(tracked["tasks"]))
    task = dataset / "balanced_wall_clock" / "tasks" / f"{first}.json"
    task.write_text(task.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Q0 row changed|registered task bytes changed"):
        subject.prepare_q0(CONFIG, output)


@pytest.mark.parametrize("mutation", ("row_count", "coordinate", "distance"))
def test_scenario_semantics_reject_row_coordinate_and_distance_tamper(
    tmp_path: Path, q0_base: Path, mutation: str
) -> None:
    output = tmp_path / mutation
    shutil.copytree(q0_base, output)
    split = output / "dataset" / "balanced_wall_clock"
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    task_id = next(iter(registry["tasks"]))
    entry = registry["tasks"][task_id]
    scenario = split / "scenarios" / f"{task_id}.scen"
    task_path = split / "tasks" / f"{task_id}.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    lines = scenario.read_text(encoding="utf-8").splitlines()
    if mutation == "row_count":
        lines.pop()
    else:
        fields = lines[1].split("\t")
        fields[4 if mutation == "coordinate" else 8] = str(
            int(fields[4 if mutation == "coordinate" else 8]) + 1
        )
        lines[1] = "\t".join(fields)
    scenario.write_text("\n".join(lines) + "\n", encoding="utf-8")
    map_path = split / "maps" / f"{entry['map_id']}.map"
    rows, cols, _grid, passable = _movingai_passable_cells(map_path)
    with pytest.raises(ValueError, match="scenario"):
        subject._parse_and_audit_scenario(
            scenario,
            map_name=f"{entry['map_id']}.map",
            rows=rows,
            cols=cols,
            passable=passable,
            task=task,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("map_file", "maps/warehouse-10-20-10-2-2.map"),
        ("scenario_file", "../../outside.scen"),
    ),
)
def test_q0_rejects_wrong_or_escaping_manifest_paths(
    tmp_path: Path, q0_base: Path, field: str, value: str
) -> None:
    output = tmp_path / field
    shutil.copytree(q0_base, output)
    manifest = output / "dataset" / "balanced_wall_clock" / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = value
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest row changed"):
        subject.prepare_q0(CONFIG, output)


def test_q0_rejects_manifest_metadata_and_registry_generator_tamper(
    tmp_path: Path, q0_base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "metadata"
    shutil.copytree(q0_base, output)
    q0_manifest = output / "dataset" / subject.Q0_MANIFEST_FILENAME
    rows = [
        json.loads(line)
        for line in q0_manifest.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["q"] = 999
    q0_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Q0 dataset identity changed|Q0 row changed"):
        subject.prepare_q0(CONFIG, output)

    _path, root, config, _registry = subject.load_config(CONFIG)
    bad_registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    bad_registry["task_generator_sha256"] = "0" * 64
    bad_path = tmp_path / "bad-registry.json"
    bad_path.write_text(json.dumps(bad_registry), encoding="utf-8")
    original = subject._resolve_registered_input

    def redirected(root_arg: Path, specification: dict, label: str) -> Path:
        if label == "Warehouse cross-aisle task registry":
            return bad_path
        return original(root_arg, specification, label)

    monkeypatch.setattr(subject, "_resolve_registered_input", redirected)
    with pytest.raises(ValueError, match="task registry identity changed"):
        subject._load_registry(root, config)


def test_q1_selects_lowest_common_passing_q_per_map() -> None:
    _path, _root, config, _registry = subject.load_config(CONFIG)
    report = _passing_report(config)
    selection = subject.select_qualified_benchmark(config, report)
    assert selection["benchmark_ready"] is True
    assert set(selection["selected_q_by_map"].values()) == {16}

    map_id = subject.MAPS[0]["id"]
    failed_q16 = next(
        row
        for row in report["tasks"]
        if row["map_id"] == map_id
        and row["q"] == 16
        and row["variant"] == subject.MATCHED_VARIANT
    )
    failed_q16["initial_conflicts"] = 0
    selection = subject.select_qualified_benchmark(config, report)
    assert selection["benchmark_ready"] is True
    assert selection["selected_q_by_map"][map_id] == 20

    failed_q20 = next(
        row
        for row in report["tasks"]
        if row["map_id"] == map_id
        and row["q"] == 20
        and row["variant"] == subject.STRUCTURED_VARIANT
    )
    failed_q20["largest_conflict_component_size"] = 15
    selection = subject.select_qualified_benchmark(config, report)
    assert selection["benchmark_ready"] is False
    assert selection["selected_map_count"] == 3
    assert selection["decision"] == "stop_do_not_resample_or_replace"


def test_q1_structured_gate_uses_conflicts_active_agents_and_component() -> None:
    _path, _root, config, _registry = subject.load_config(CONFIG)
    for field, failing_value in (
        ("initial_conflicts", 15),
        ("active_conflict_agent_count", 31),
        ("largest_conflict_component_size", 15),
    ):
        report = _passing_report(config)
        row = next(
            task
            for task in report["tasks"]
            if task["map_id"] == subject.MAPS[0]["id"]
            and task["q"] == 16
            and task["variant"] == subject.STRUCTURED_VARIANT
        )
        row[field] = failing_value
        result = subject.select_qualified_benchmark(config, report)
        assert result["map_results"][subject.MAPS[0]["id"]]["q_results"]["16"][
            "both_variants_passed"
        ] is False


def test_runtime_is_full_validated_schema_but_runner_is_reset_only() -> None:
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    reference = json.loads(
        (
            ROOT / "configs" / "stride_warehouse_fixed16_development_runtime_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert runtime["schema_version"] == 1
    assert runtime["formal"] is False
    assert runtime["solver_seeds"] == [61, 62, 63, 64]
    assert runtime["model_registration"] == reference["model_registration"]
    assert runtime["dataset_design"]["map_count"] == 4
    assert runtime["dataset_design"]["instance_count"] == 32
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "_qualification_worker" in source
    assert 'phase="qualify"' in source
    assert 'controller="official_adaptive"' in source
    assert ".step(" not in source


def test_qualify_dry_run_stops_before_native_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "prepare_q0", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        subject, "_audit_registered_dataset", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        subject,
        "_runtime_preflight",
        lambda *_args, **_kwargs: {"run_fingerprint": "d" * 64},
    )
    monkeypatch.setattr(
        subject,
        "_producer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native producer must not be loaded on dry-run")
        ),
    )
    monkeypatch.setattr(
        subject,
        "_run_jobs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reset worker must not run on dry-run")
        ),
    )
    output = tmp_path / "dry"
    result = subject.qualify(CONFIG, output, dry_run=True)
    assert result["dry_run"] is True
    assert result["native_reset_invoked"] is False
    assert not (output / "qualification").exists()


def _mock_q1_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    _path, _root, config, _registry = subject.load_config(CONFIG)
    monkeypatch.setattr(subject, "prepare_q0", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        subject, "_audit_registered_dataset", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        subject,
        "_runtime_preflight",
        lambda *_args, **_kwargs: {"run_fingerprint": "e" * 64},
    )
    producer = _fake_producer()
    monkeypatch.setattr(subject, "_producer", lambda *_args, **_kwargs: producer)
    original_sha256 = subject.sha256_file
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    def q0_aware_sha256(path: Path) -> str:
        path = Path(path)
        if path.name == subject.Q0_MANIFEST_FILENAME and not path.is_file():
            return registry["q0_manifest_sha256"]
        return original_sha256(path)

    monkeypatch.setattr(subject, "sha256_file", q0_aware_sha256)
    rows = _fake_dataset_rows(config)
    monkeypatch.setattr(subject, "_dataset_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(
        subject,
        "_CollectionRunLock",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )


@pytest.mark.parametrize("failure_status", ("error", "timeout"))
def test_first_q1_worker_failure_is_terminal_and_resume_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_status: str,
) -> None:
    _mock_q1_dependencies(monkeypatch)

    def fail_first(_worker, jobs, _workers, **kwargs):
        result = kwargs["failure_result"](
            jobs[0], failure_status, f"synthetic {failure_status}"
        )
        kwargs["on_result"](result)
        return [result]

    monkeypatch.setattr(subject, "_run_jobs", fail_first)
    output = tmp_path / failure_status
    result = subject.qualify(CONFIG, output)
    assert result["benchmark_ready"] is False
    assert result["terminal_error"] is True
    status = json.loads(
        (
            output
            / "qualification"
            / subject.QUALIFICATION_STATUS_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert status["terminal_error"] is True
    assert status["complete"] is False
    assert status["error_jobs"] + status["timeout_jobs"] == 1
    with pytest.raises(ValueError, match="output is terminal"):
        subject.qualify(CONFIG, output, resume=True)


def test_same_fingerprint_partial_q1_resume_runs_only_missing_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_q1_dependencies(monkeypatch)
    first_seen: list[tuple[str, int]] = []

    def interrupt_after_three(_worker, jobs, _workers, **kwargs):
        for job in jobs[:3]:
            result = _fake_ok_result(job)
            first_seen.append((result["task_id"], result["solver_seed"]))
            kwargs["on_result"](result)
        raise KeyboardInterrupt("synthetic interruption")

    monkeypatch.setattr(subject, "_run_jobs", interrupt_after_three)
    output = tmp_path / "resume"
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        subject.qualify(CONFIG, output)
    manifest = output / "qualification" / subject.QUALIFICATION_MANIFEST_FILENAME
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 3

    resumed_jobs: list[tuple[str, int]] = []

    def finish_missing(_worker, jobs, _workers, **kwargs):
        results = []
        for job in jobs:
            result = _fake_ok_result(job)
            resumed_jobs.append((result["task_id"], result["solver_seed"]))
            kwargs["on_result"](result)
            results.append(result)
        return results

    monkeypatch.setattr(subject, "_run_jobs", finish_missing)
    result = subject.qualify(CONFIG, output, resume=True)
    assert result["benchmark_ready"] is True
    assert len(resumed_jobs) == 125
    assert not set(first_seen) & set(resumed_jobs)
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 128


def test_q1_evidence_rejects_unknown_and_registered_identity_tamper(
    tmp_path: Path,
) -> None:
    _path, _root, config, registry = subject.load_config(CONFIG)
    assert registry is not None
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    manifest = qualification / subject.QUALIFICATION_MANIFEST_FILENAME
    manifest.write_text(
        json.dumps({"task_id": "unregistered", "solver_seed": 61}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown key"):
        subject._audit_qualification_evidence(
            config,
            qualification,
            require_complete=False,
            registry=registry,
            runtime_preflight_run_fingerprint="f" * 64,
        )

    item = subject.qualification_schedule(config)[0]
    dataset_row = _fake_dataset_rows(config)[item["task_id"]]
    raw = _fake_ok_result(
        {"row": dataset_row, "solver_seed": item["solver_seed"]}
    )
    task_registration = registry["tasks"][item["task_id"]]
    raw.update(
        {
            "registered_task_identity": {
                "map_id": item["map_id"],
                "variant": item["variant"],
                "task_seed": item["task_seed"],
                "q": item["q"],
                "agent_count": item["agent_count"],
            },
            "registered_scenario_sha256": task_registration["scenario_sha256"],
            "registered_task_sha256": task_registration["task_sha256"],
            "registered_q0_manifest_sha256": registry["q0_manifest_sha256"],
            "runtime_preflight_run_fingerprint": "f" * 64,
        }
    )
    for mutation in ("q", "task_sha"):
        changed = json.loads(json.dumps(raw))
        if mutation == "q":
            changed["registered_task_identity"]["q"] = 999
        else:
            changed["registered_task_sha256"] = "0" * 64
        manifest.write_text(json.dumps(changed) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="row identity changed"):
            subject._audit_qualification_evidence(
                config,
                qualification,
                require_complete=False,
                registry=registry,
                runtime_preflight_run_fingerprint="f" * 64,
            )


def test_cli_exposes_no_collect_or_analyze_command() -> None:
    source = (
        ROOT / "scripts" / "run_stride_warehouse_crossaisle_paired.py"
    ).read_text(encoding="utf-8")
    assert 'choices=("plan", "prepare-q0", "qualify", "run")' in source
    assert '"collect"' not in source
    assert '"analyze"' not in source
