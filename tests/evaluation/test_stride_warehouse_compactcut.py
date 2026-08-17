from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import experiments.stride_warehouse_compactcut as subject
from experiments._common import producer_identity, sha256_file
from experiments.balanced_wall_clock import _movingai_passable_cells


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_warehouse_compactcut_v1.json"
RUNTIME = ROOT / "configs" / "stride_warehouse_compactcut_runtime_v1.json"
GENERATOR = ROOT / "experiments" / "warehouse_compactcut_tasks.py"
REGISTRY = ROOT / "configs" / "stride_warehouse_compactcut_tasks_v1.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_pending_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    for source, relative in (
        (RUNTIME, "configs/stride_warehouse_compactcut_runtime_v1.json"),
        (GENERATOR, "experiments/warehouse_compactcut_tasks.py"),
        (REGISTRY, "configs/stride_warehouse_compactcut_tasks_v1.json"),
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    config_path = root / "configs" / "stride_warehouse_compactcut_v1.json"
    shutil.copy2(CONFIG, config_path)
    return root, config_path


def _fake_registry() -> dict:
    tasks = {
        row["task_id"]: {
            "map_id": row["map_id"],
            "map_role": row["map_role"],
            "variant": row["variant"],
            "task_seed": row["task_seed"],
            "agent_count": subject.AGENT_COUNT,
            "scenario_sha256": hashlib.sha256(f"scenario:{row['task_id']}".encode()).hexdigest(),
            "task_sha256": hashlib.sha256(f"task:{row['task_id']}".encode()).hexdigest(),
            "endpoint_payload_sha256": hashlib.sha256(
                f"endpoints:{row['task_id']}".encode()
            ).hexdigest(),
            "pair_payload_sha256": subject.Q0_BYTE_REPRODUCIBILITY[
                "cohort_payload_hash_parity"
            ]["pair_payload_sha256"][f"{row['map_id']}/t{row['task_seed']:04d}"],
            "joint_feasibility_witness_sha256": hashlib.sha256(
                f"witness:{row['task_id']}".encode()
            ).hexdigest(),
        }
        for row in subject.benchmark_task_specs({})
    }
    maps = {
        row["id"]: {
            "template": row["template"],
            "map_seed": row["map_seed"],
            "map_role": row["map_role"],
            "map_sha256": hashlib.sha256(f"map:{row['id']}".encode()).hexdigest(),
            "map_metadata_sha256": hashlib.sha256(f"metadata:{row['id']}".encode()).hexdigest(),
            "geometry_sha256": hashlib.sha256(f"geometry:{row['id']}".encode()).hexdigest(),
        }
        for row in subject.MAPS
    }
    return {
        "schema": subject.REGISTRY_SCHEMA,
        "experiment_id": subject.EXPERIMENT_ID,
        "map_count": 8,
        "task_count": 32,
        "q0_manifest_sha256": "a" * 64,
        "task_generator_sha256": sha256_file(GENERATOR),
        "q0_byte_reproducibility": subject.Q0_BYTE_REPRODUCIBILITY,
        "q0_generation_toolchain": subject._require_q0_runtime_identity(),
        "static_pre_registration_record": subject.STATIC_PRE_REGISTRATION_RECORD,
        "maps": maps,
        "tasks": tasks,
    }


def _frozen_root(tmp_path: Path) -> tuple[Path, Path, dict]:
    root, config_path = _copy_pending_root(tmp_path)
    registry = _fake_registry()
    registry["task_generator_sha256"] = sha256_file(
        root / "experiments" / "warehouse_compactcut_tasks.py"
    )
    registry_path = root / "configs" / "stride_warehouse_compactcut_tasks_v1.json"
    _write_json(registry_path, registry)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["scientific_status"] = (
        "preregistered_q0_task_hash_frozen_q1_reset_only_benchmark_qualification"
    )
    config["registration"] = {
        "status": "frozen_q0_task_hash_registration",
        "task_registry_path": "configs/stride_warehouse_compactcut_tasks_v1.json",
        "task_registry_sha256": sha256_file(registry_path),
        "q1_allowed": True,
    }
    config["inputs"]["task_generator"]["sha256"] = sha256_file(
        root / "experiments" / "warehouse_compactcut_tasks.py"
    )
    _write_json(config_path, config)
    return root, config_path, registry


def _passing_report(config: dict) -> dict:
    tasks = []
    for item in subject.qualification_schedule(config):
        structured = item["variant"] == subject.STRUCTURED_VARIANT
        tasks.append(
            {
                **item,
                "initial_conflicts": 16 if structured else 0,
                "active_conflict_agent_count": 32 if structured else 0,
                "largest_conflict_component_size": 16 if structured else 0,
                "initial_complete": True,
                "initial_feasible": not structured,
                "reported_initial_feasible": not structured,
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


def _fake_dataset_rows(config: dict) -> dict[str, dict]:
    return {
        row["task_id"]: {
            "split": subject.SPLIT,
            "map_id": row["map_id"],
            "map_role": row["map_role"],
            "task_id": row["task_id"],
            "layout_mode": "warehouse",
            "task_variant": row["variant"],
            "agent_count": subject.AGENT_COUNT,
        }
        for row in subject.benchmark_task_specs(config)
    }


def _fake_ok_result(job: dict) -> dict:
    row = job["row"]
    structured = row["task_variant"] == subject.STRUCTURED_VARIANT
    return {
        "split": row["split"],
        "map_id": row["map_id"],
        "map_role": row["map_role"],
        "task_id": row["task_id"],
        "layout_mode": row["layout_mode"],
        "task_variant": row["task_variant"],
        "agent_count": row["agent_count"],
        "solver_seed": job["solver_seed"],
        "initial_conflicts": 16 if structured else 0,
        "initial_feasible": not structured,
        "initial_complete": True,
        "state_fingerprint": hashlib.sha256(
            f"{row['task_id']}::{job['solver_seed']}".encode()
        ).hexdigest(),
        "initial_complexity": {
            "agent_count": subject.AGENT_COUNT,
            "conflict_pair_count": 16 if structured else 0,
            "active_conflict_agent_count": 32 if structured else 0,
            "largest_conflict_component_size": 16 if structured else 0,
        },
        "status": "ok",
        "error": None,
    }


def test_frozen_config_is_exact_reset_only_identity() -> None:
    _path, _root, config, registry = subject.load_config(CONFIG)
    assert registry is not None
    assert config["registration"] == {
        "status": "frozen_q0_task_hash_registration",
        "task_registry_path": "configs/stride_warehouse_compactcut_tasks_v1.json",
        "task_registry_sha256": "368ed483fec21875f34f0acbf888bb28562b0bce0f7a00fd4356813284aab7b9",
        "q1_allowed": True,
    }
    assert config["phase_contract"]["controller_step_allowed"] is False
    assert config["phase_contract"]["formal_ttf_allowed"] is False
    assert config["q0_gates"]["maximum_relative_mean_shortest_distance_difference"] == 0.05
    assert config["q0_gates"]["pairing_semantics"] == (
        "independent_registered_endpoints_distance_near_matched_secondary_control"
    )
    assert config["q0_gates"]["causal_control_claim_allowed"] is False
    assert config["q0_gates"]["all_8_map_grid_sha256_unique_required"] is True
    assert config["inputs"]["task_generator"]["sha256"] == subject.TASK_GENERATOR_SHA256
    assert config["q0_gates"][
        "deterministic_collision_free_prioritized_joint_mapf_witness_required"
    ] is True
    assert {
        row["id"]: row["map_role"] for row in config["cohort"]["maps"]
    } == {
        "whcc_cfg_01": "development",
        "whcc_cfg_02": "controller_held_out",
        "whcc_cfg_03": "development",
        "whcc_cfg_04": "controller_held_out",
        "whcc_dh_01": "development",
        "whcc_dh_02": "controller_held_out",
        "whcc_dh_03": "development",
        "whcc_dh_04": "controller_held_out",
    }


@pytest.mark.parametrize(
    "section,field,value",
    (
        ("q0_gates", "maximum_relative_p95_shortest_distance_difference", 0.11),
        ("q1_gates", "no_map_task_variant_or_seed_replacement", False),
        ("result_interpretation", "ttf_claim_allowed", True),
        ("cohort", "claim_boundary", "broader_claim"),
        ("q0_byte_reproducibility", "assignment_backend", "other"),
        ("static_pre_registration_record", "selected_map_seed", 2026081714),
    ),
)
def test_config_rejects_gate_or_claim_mutation(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    _root, config_path = _copy_pending_root(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[section][field] = value
    _write_json(config_path, config)
    with pytest.raises(
        ValueError,
        match=(
            "gates changed|interpretation changed|cohort changed|"
            "byte reproducibility changed|static preregistration changed"
        ),
    ):
        subject.load_config(config_path)


def test_config_rejects_preregistered_map_role_mutation(tmp_path: Path) -> None:
    _root, config_path = _copy_pending_root(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["cohort"]["maps"][0]["map_role"] = "controller_held_out"
    _write_json(config_path, config)
    with pytest.raises(ValueError, match="cohort changed"):
        subject.load_config(config_path)


def test_plan_has_exact_8_map_32_task_128_reset_cohort_without_formal() -> None:
    result = subject.plan(CONFIG)
    assert result["map_count"] == 8
    assert result["task_count"] == 32
    assert result["qualification_reset_count"] == 128
    assert result["structured_reset_count"] == 64
    assert result["diagnostic_control_reset_count"] == 64
    assert result["controller_count"] == 0
    assert result["formal_ttf_episode_count"] == 0
    _path, _root, config, _registry = subject.load_config(CONFIG)
    schedule = subject.qualification_schedule(config)
    assert len({(row["task_id"], row["solver_seed"]) for row in schedule}) == 128
    assert {row["solver_seed"] for row in schedule} == {71, 72, 73, 74}
    assert {row["task_seed"] for row in schedule} == {521, 557}
    assert {row["agent_count"] for row in schedule} == {120}
    assert sum(row["map_role"] == "development" for row in schedule) == 64
    assert sum(row["map_role"] == "controller_held_out" for row in schedule) == 64


def test_frozen_registry_rejects_duplicate_map_grid_sha256(tmp_path: Path) -> None:
    root, config_path, registry = _frozen_root(tmp_path)
    map_ids = sorted(registry["maps"])
    registry["maps"][map_ids[1]]["map_sha256"] = registry["maps"][map_ids[0]][
        "map_sha256"
    ]
    registry_path = root / "configs" / "stride_warehouse_compactcut_tasks_v1.json"
    _write_json(registry_path, registry)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["registration"]["task_registry_sha256"] = sha256_file(registry_path)
    _write_json(config_path, config)
    with pytest.raises(ValueError, match="task registry identity changed"):
        subject.load_config(config_path, require_frozen_registry=True)


def test_pending_registration_mechanically_blocks_q1_before_output(tmp_path: Path) -> None:
    _root, config_path = _copy_pending_root(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["scientific_status"] = (
        "preregistered_q0_generation_pending_task_hash_registration_reset_only"
    )
    config["registration"] = {
        "status": "pending_q0_task_hash_registration",
        "task_registry_path": "configs/stride_warehouse_compactcut_tasks_v1.json",
        "task_registry_sha256": None,
        "q1_allowed": False,
    }
    _write_json(config_path, config)
    output = tmp_path / "never-reset"
    with pytest.raises(ValueError, match="Q1 is blocked"):
        subject.qualify(config_path, output, dry_run=True)
    assert not output.exists()


@pytest.fixture(scope="module")
def q0_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("compactcut-q0") / "output"
    result = subject.prepare_q0(CONFIG, output)
    assert result["q0_passed"] is True
    return output


def test_q0_generates_and_reaudits_all_maps_tasks_distances_and_witnesses(
    tmp_path: Path, q0_base: Path
) -> None:
    output = tmp_path / "copy"
    shutil.copytree(q0_base, output)
    dataset = output / "dataset"
    q0 = json.loads((dataset / subject.Q0_FILENAME).read_text(encoding="utf-8"))
    assert q0["passed"] is True
    assert q0["map_count"] == 8
    assert q0["task_count"] == 32
    assert q0["solver_or_controller_invoked"] is False
    assert len(q0["task_audits"]) == 16
    for item in q0["task_audits"]:
        assert item["audit"]["passed"] is True
        assert item["distance_metrics"]["mean_relative_difference"] <= 0.05
        assert item["distance_metrics"]["p95_relative_difference"] <= 0.10
    proposal = json.loads(
        (dataset / subject.Q0_REGISTRATION_PROPOSAL_FILENAME).read_text(encoding="utf-8")
    )
    assert len(proposal["maps"]) == 8
    assert len(proposal["tasks"]) == 32
    assert len({row["map_sha256"] for row in proposal["maps"].values()}) == 8
    assert {
        map_id: row["map_role"] for map_id, row in proposal["maps"].items()
    } == {row["id"]: row["map_role"] for row in subject.MAPS}
    assert all(
        row["map_role"] == subject.MAP_BY_ID[row["map_id"]]["map_role"]
        for row in proposal["tasks"].values()
    )
    assert all(
        len(row["joint_feasibility_witness_sha256"]) == 64
        for row in proposal["tasks"].values()
    )
    assert all(
        len(row["endpoint_payload_sha256"]) == 64
        and len(row["task_sha256"]) == 64
        for row in proposal["tasks"].values()
    )
    subject.prepare_q0(CONFIG, output)


@pytest.mark.parametrize("mutation", ("row_count", "coordinate", "distance"))
def test_scenario_semantics_reject_row_coordinate_and_distance_tamper(
    tmp_path: Path, q0_base: Path, mutation: str
) -> None:
    output = tmp_path / mutation
    shutil.copytree(q0_base, output)
    split = output / "dataset" / subject.SPLIT
    manifest = [
        json.loads(line)
        for line in (split / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    row = manifest[0]
    scenario = split / row["scenario_file"]
    task = json.loads((split / row["task_file"]).read_text(encoding="utf-8"))
    lines = scenario.read_text(encoding="utf-8").splitlines()
    if mutation == "row_count":
        lines.pop()
    else:
        fields = lines[1].split("\t")
        field = 4 if mutation == "coordinate" else 8
        fields[field] = str(int(fields[field]) + 1)
        lines[1] = "\t".join(fields)
    scenario.write_text("\n".join(lines) + "\n", encoding="utf-8")
    map_path = split / row["map_file"]
    rows, cols, _grid, passable = _movingai_passable_cells(map_path)
    with pytest.raises(ValueError, match="scenario"):
        subject._parse_and_audit_scenario(
            scenario,
            map_name=map_path.name,
            rows=rows,
            cols=cols,
            passable=passable,
            task=task,
        )


@pytest.mark.parametrize(
    "field,value", (("map_file", "maps/whcc_cfg_02.map"), ("task_file", "../../outside.json"))
)
def test_q0_rejects_wrong_or_escaping_manifest_paths(
    tmp_path: Path, q0_base: Path, field: str, value: str
) -> None:
    output = tmp_path / field
    shutil.copytree(q0_base, output)
    manifest = output / "dataset" / subject.SPLIT / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = value
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest row changed"):
        subject.prepare_q0(CONFIG, output)


def test_q1_structured_gates_all_three_metrics_but_control_is_diagnostic_only() -> None:
    _path, _root, config, _registry = subject.load_config(CONFIG)
    report = _passing_report(config)
    result = subject.select_qualified_benchmark(config, report)
    assert result["benchmark_ready"] is True
    assert result["diagnostic_control_structure_thresholds_gate_benchmark_ready"] is False
    assert result["diagnostic_control_observed_reset_count"] == 64
    for field, value in (
        ("initial_conflicts", 15),
        ("active_conflict_agent_count", 31),
        ("largest_conflict_component_size", 15),
    ):
        changed = json.loads(json.dumps(report))
        row = next(item for item in changed["tasks"] if item["variant"] == subject.STRUCTURED_VARIANT)
        row[field] = value
        failed = subject.select_qualified_benchmark(config, changed)
        assert failed["benchmark_ready"] is False
        assert failed["decision"] == "stop_do_not_resample_replace_or_run_controller"


def test_frozen_registry_and_q1_evidence_reject_unknown_or_tampered_identity(
    tmp_path: Path,
) -> None:
    _root, config_path, registry = _frozen_root(tmp_path)
    _path, _project, config, loaded = subject.load_config(
        config_path, require_frozen_registry=True
    )
    assert loaded == registry
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    manifest = qualification / subject.QUALIFICATION_MANIFEST_FILENAME
    manifest.write_text(
        json.dumps({"task_id": "unregistered", "solver_seed": 71}) + "\n", encoding="utf-8"
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
    raw = _fake_ok_result(
        {"row": _fake_dataset_rows(config)[item["task_id"]], "solver_seed": item["solver_seed"]}
    )
    registered_task = registry["tasks"][item["task_id"]]
    raw.update(
        {
            "registered_task_identity": {
                "map_id": item["map_id"],
                "map_role": item["map_role"],
                "variant": item["variant"],
                "task_seed": item["task_seed"],
                "agent_count": subject.AGENT_COUNT,
            },
            "registered_scenario_sha256": registered_task["scenario_sha256"],
            "registered_task_sha256": registered_task["task_sha256"],
            "registered_map_sha256": registry["maps"][item["map_id"]]["map_sha256"],
            "registered_q0_manifest_sha256": registry["q0_manifest_sha256"],
            "runtime_preflight_run_fingerprint": "f" * 64,
        }
    )
    raw["registered_task_identity"]["task_seed"] = 999
    manifest.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row identity changed"):
        subject._audit_qualification_evidence(
            config,
            qualification,
            require_complete=False,
            registry=registry,
            runtime_preflight_run_fingerprint="f" * 64,
        )


def test_qualify_dry_run_stops_before_native_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, config_path, _registry = _frozen_root(tmp_path)
    monkeypatch.setattr(subject, "prepare_q0", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(subject, "_audit_registered_dataset", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(subject, "_runtime_preflight", lambda *_args, **_kwargs: {"run_fingerprint": "d" * 64})
    monkeypatch.setattr(
        subject,
        "_producer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("producer must not load")),
    )
    monkeypatch.setattr(
        subject,
        "_run_jobs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("worker must not run")),
    )
    output = tmp_path / "dry"
    result = subject.qualify(config_path, output, dry_run=True)
    assert result["native_reset_invoked"] is False
    assert not (output / "qualification").exists()


def _mock_q1_dependencies(
    config_path: Path,
    registry: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _path, _root, config, _loaded = subject.load_config(config_path)
    monkeypatch.setattr(subject, "prepare_q0", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(subject, "_audit_registered_dataset", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(subject, "_runtime_preflight", lambda *_args, **_kwargs: {"run_fingerprint": "e" * 64})
    monkeypatch.setattr(
        subject,
        "_producer",
        lambda *_args, **_kwargs: producer_identity(
            project_root=ROOT,
            source_files=("experiments/stride_warehouse_compactcut.py",),
            native_required=False,
        ),
    )
    original_sha = subject.sha256_file

    def q0_aware_sha(path: Path) -> str:
        path = Path(path)
        if path.name == subject.Q0_MANIFEST_FILENAME and not path.is_file():
            return registry["q0_manifest_sha256"]
        return original_sha(path)

    monkeypatch.setattr(subject, "sha256_file", q0_aware_sha)
    monkeypatch.setattr(subject, "_dataset_rows", lambda *_args, **_kwargs: _fake_dataset_rows(config))
    monkeypatch.setattr(subject, "_CollectionRunLock", lambda *_args, **_kwargs: contextlib.nullcontext())


@pytest.mark.parametrize("failure_status", ("error", "timeout"))
def test_first_q1_failure_is_terminal_and_resume_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_status: str
) -> None:
    _root, config_path, registry = _frozen_root(tmp_path)
    _mock_q1_dependencies(config_path, registry, monkeypatch)

    def fail_first(_worker, jobs, _workers, **kwargs):
        result = kwargs["failure_result"](jobs[0], failure_status, f"synthetic {failure_status}")
        kwargs["on_result"](result)
        return [result]

    monkeypatch.setattr(subject, "_run_jobs", fail_first)
    output = tmp_path / failure_status
    result = subject.qualify(config_path, output)
    assert result["benchmark_ready"] is False
    assert result["terminal_error"] is True
    with pytest.raises(ValueError, match="output is terminal"):
        subject.qualify(config_path, output, resume=True)


def test_same_fingerprint_partial_resume_runs_only_missing_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, config_path, registry = _frozen_root(tmp_path)
    _mock_q1_dependencies(config_path, registry, monkeypatch)
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
        subject.qualify(config_path, output)

    resumed: list[tuple[str, int]] = []

    def finish(_worker, jobs, _workers, **kwargs):
        for job in jobs:
            result = _fake_ok_result(job)
            resumed.append((result["task_id"], result["solver_seed"]))
            kwargs["on_result"](result)
        return []

    monkeypatch.setattr(subject, "_run_jobs", finish)
    result = subject.qualify(config_path, output, resume=True)
    assert result["benchmark_ready"] is True
    assert len(resumed) == 125
    assert not set(first_seen) & set(resumed)


def test_runtime_is_full_schema_but_runner_and_cli_are_reset_only() -> None:
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    reference = json.loads(
        (ROOT / "configs" / "stride_warehouse_fixed16_development_runtime_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert runtime["formal"] is False
    assert runtime["solver_seeds"] == [71, 72, 73, 74]
    assert runtime["max_decisions"] == 0
    assert runtime["model_registration"] == reference["model_registration"]
    assert runtime["dataset_design"]["map_count"] == 8
    assert runtime["dataset_design"]["instance_count"] == 32
    runner_source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "_qualification_worker" in runner_source
    assert 'phase="qualify"' in runner_source
    assert ".step(" not in runner_source
    cli_source = (ROOT / "scripts" / "run_stride_warehouse_compactcut.py").read_text(
        encoding="utf-8"
    )
    assert 'choices=("plan", "prepare-q0", "qualify", "run")' in cli_source
    assert '"collect"' not in cli_source
    assert '"analyze"' not in cli_source
