from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_compact_flow_source_v1 as source


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_source_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


@pytest.fixture()
def compact_flow_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    output = tmp_path / "compact-flow"
    dataset = output / "dataset"
    source_config_path = tmp_path / "source-config.json"
    registration_path = tmp_path / "registration.json"
    registration_report_path = tmp_path / "registration-report.json"
    registered_task_path = tmp_path / "registered-tasks.jsonl"
    materializer_path = tmp_path / "materializer.py"
    base_config_path = tmp_path / "base-config.json"
    controller_root = tmp_path / "controller"
    controller_root.mkdir()
    controller_manifest = controller_root / "controller_manifest.json"
    _write_json(controller_manifest, {"controller": "v2-full"})
    _write_json(source_config_path, {"fixture": True})
    _write_json(registration_path, {"fixture": True})
    _write_json(registration_report_path, {"fixture": True})
    materializer_path.write_text("# fixture\n", encoding="utf-8")
    _write_json(base_config_path, {"environment": {"time_limit": 200.0}})

    map_splits = {
        "train": [f"train-map-{index}" for index in range(8)],
        "development": [f"development-map-{index}" for index in range(8)],
    }
    strata = ("ultra", "small", "medium", "large")
    od_variants = ("opposite_exchange", "uniform_random")
    per_map_loads = {
        map_id: (10, 20, 30)
        for maps in map_splits.values()
        for map_id in maps
    }
    per_map_task_seeds = {
        map_id: (11, 17)
        for maps in map_splits.values()
        for map_id in maps
    }
    map_capacity_strata = {
        map_id: strata[index % len(strata)]
        for maps in map_splits.values()
        for index, map_id in enumerate(maps)
    }
    manifests: dict[str, list[dict]] = {}
    registered: list[dict] = []
    for split, maps in map_splits.items():
        split_root = dataset / split
        rows: list[dict] = []
        for map_index, map_id in enumerate(maps):
            map_file = Path("maps") / f"{map_id}.map"
            metadata_file = Path("maps") / f"{map_id}.json"
            (split_root / map_file).parent.mkdir(parents=True, exist_ok=True)
            (split_root / map_file).write_text("type octile\nheight 1\nwidth 1\nmap\n.\n")
            _write_json(split_root / metadata_file, {"map_id": map_id})
            for od_index, od_variant in enumerate(od_variants):
                for task_seed_index, task_seed in enumerate((11, 17)):
                    scenario_index = od_index * 2 + task_seed_index
                    for agent_count in (10, 20, 30):
                        task_id = (
                            f"{map_id}__{od_variant}__seed_{task_seed}"
                            f"__agents_{agent_count}"
                        )
                        scenario_file = Path("scenarios") / f"{task_id}.scen"
                        task_file = Path("tasks") / f"{task_id}.json"
                        (split_root / scenario_file).parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        (split_root / scenario_file).write_text(
                            "version 1\n", encoding="utf-8"
                        )
                        _write_json(
                            split_root / task_file,
                            {
                                "task_id": task_id,
                                "agent_count": agent_count,
                                "od_variant": od_variant,
                                "task_seed": task_seed,
                            },
                        )
                        row = {
                            "split": split,
                            "source_group": "movingai",
                            "map_id": map_id,
                            "task_id": task_id,
                            "map_file": map_file.as_posix(),
                            "map_sha256": _sha256(split_root / map_file),
                            "map_metadata_file": metadata_file.as_posix(),
                            "map_metadata_sha256": _sha256(
                                split_root / metadata_file
                            ),
                            "scenario_file": scenario_file.as_posix(),
                            "scenario_sha256": _sha256(
                                split_root / scenario_file
                            ),
                            "task_file": task_file.as_posix(),
                            "task_sha256": _sha256(split_root / task_file),
                            "layout_mode": f"fixture-family-{map_index % 2}",
                            "layout_variant": map_id,
                            "scenario_type": f"compact_flow_{scenario_index}",
                            "scenario_index": scenario_index,
                            "task_variant": (
                                f"{od_variant}_seed_{task_seed}_agents_{agent_count}"
                            ),
                            "od_variant": od_variant,
                            "task_seed": task_seed,
                            "agent_count": agent_count,
                            "capacity_stratum": map_capacity_strata[map_id],
                            "training_authorized": False,
                        }
                        rows.append(row)
                        registered.append(
                            {
                                "schema": "fixture.registered_task.v1",
                                "split": split,
                                "task_id": task_id,
                            }
                        )
        rows.sort(key=lambda row: str(row["task_id"]))
        _write_jsonl(split_root / "manifest.jsonl", rows)
        manifests[split] = rows
    _write_jsonl(registered_task_path, registered)
    registration_sha = _sha256(registration_path)
    registration_report_sha = _sha256(registration_report_path)
    split_manifests = {
        split: {
            "path": str(dataset / split / "manifest.jsonl"),
            "sha256": _sha256(dataset / split / "manifest.jsonl"),
            "task_count": 96,
        }
        for split in source.ALLOWED_SPLITS
    }
    materializer_report_path = output / "compact_flow_materialization_report.json"
    _write_json(
        materializer_report_path,
        {
            "schema": source.TASK_MATERIALIZER_SCHEMA,
            "experiment_id": source.EXPERIMENT_ID,
            "status": "MATERIALIZED",
            "task_count": 192,
            "dataset_root": str(dataset),
            "split_manifests": split_manifests,
            "training_authorized": False,
        },
    )
    trust = {
        "schema": source.MATERIALIZATION_SCHEMA,
        "experiment_id": source.EXPERIMENT_ID,
        "status": "MATERIALIZED",
        "registration_sha256": registration_sha,
        "registration_report_sha256": registration_report_sha,
        "dataset_root": str(dataset),
        "task_count": 192,
        "materializer_report_sha256": _sha256(materializer_report_path),
        "split_manifests": split_manifests,
        "training_authorized": False,
    }
    _write_json(output / "materialization_trust_report.json", trust)
    source_config = {
        "collection": {
            "workers": 16,
            "maximum_workers": 20,
            "solver_seeds": [41, 42],
            "policies_in_config": ["official_adaptive", "realized_dynamic"],
            "executed_policy": "realized_dynamic",
            "max_decisions": 12,
            "environment_time_limit_seconds": 200.0,
            "wall_time_budget_seconds": 200.0,
            "episode_process_timeout_seconds": 240.0,
            "qualification_process_timeout_seconds": 240.0,
            "trace_format": "delta-gzip-v2",
        }
    }
    ctx = {
        "project_root": tmp_path,
        "source_config_path": source_config_path,
        "source_config": source_config,
        "registration_path": registration_path,
        "registration": {"legacy_h1_exclusion": {"map_ids": []}},
        "registration_sha256": registration_sha,
        "registration_report_path": registration_report_path,
        "registration_report": {"fixture": True},
        "registration_report_sha256": registration_report_sha,
        "materializer_path": materializer_path,
        "registered_task_path": registered_task_path,
        "registered_task_sha256": _sha256(registered_task_path),
        "registered_tasks_by_split": {
            split: [row for row in registered if row["split"] == split]
            for split in source.ALLOWED_SPLITS
        },
        "base_config_path": base_config_path,
        "base_config": _read_json(base_config_path),
        "controller_root": controller_root,
        "map_splits": map_splits,
        "allowed_ids": {map_id for maps in map_splits.values() for map_id in maps},
        "od_variants": od_variants,
        "capacity_strata": strata,
        "per_map_loads": per_map_loads,
        "per_map_task_seeds": per_map_task_seeds,
        "map_capacity_strata": map_capacity_strata,
    }
    monkeypatch.setattr(
        source, "load_registered_source_context", lambda *_args, **_kwargs: ctx
    )
    return {"ctx": ctx, "output": output, "dataset": dataset, "manifests": manifests}


def _write_qualification(
    fixture: dict,
    *,
    output: Path,
    split: str,
    jobs: set[tuple[str, int]],
    conflicts: int,
) -> None:
    tasks = {
        str(row["task_id"]): row for row in fixture["manifests"][split]
    }
    rows = []
    for task_id, solver_seed in sorted(jobs):
        task = tasks[task_id]
        rows.append(
            {
                "task_id": task_id,
                "solver_seed": solver_seed,
                "map_id": task["map_id"],
                "split": split,
                "layout_mode": task["layout_mode"],
                "task_variant": task["task_variant"],
                "agent_count": task["agent_count"],
                "status": "ok",
                "error": None,
                "initial_complete": True,
                "initial_feasible": conflicts == 0,
                "initial_conflicts": conflicts,
                "state_fingerprint": f"{task_id}/{solver_seed}",
            }
        )
    _write_json(output / "run_config.json", {"run_fingerprint": f"fixture/{split}"})
    _write_jsonl(output / "qualification_manifest.jsonl", rows)
    _write_json(output / "qualification_report.json", {"passed": True})


def test_repository_context_pins_exact_fresh_product_and_dry_run_is_pure(
    tmp_path: Path,
) -> None:
    ctx = source.load_registered_source_context(CONFIG)
    assert len(ctx["allowed_ids"]) == 16
    assert sum(map(len, ctx["registered_tasks_by_split"].values())) == 192
    assert ctx["od_variants"] == ("opposite_exchange", "uniform_random")
    assert ctx["capacity_strata"] == ("ultra", "small", "medium", "large")
    assert ctx["source_config"]["collection"]["solver_seeds"] == [41, 42]
    assert ctx["source_config"]["collection"]["workers"] == 16
    assert ctx["source_config"]["collection"]["maximum_workers"] == 20
    assert (
        ctx["registration"]["parent_source_v2_terminal_failure"]
        ["completed_v2_reset_rows_imported"]
        == 0
    )
    output = tmp_path / "dry-run"
    report = source.materialize_source_dataset(CONFIG, output, dry_run=True)
    assert report["status"] == "DRY_RUN"
    assert report["expected_task_count"] == 192
    assert report["source_v2_qualification_rows_imported"] == 0
    assert not output.exists()


@pytest.fixture()
def planned(compact_flow_fixture: dict) -> dict:
    report = source.plan_source_collection(
        "ignored.json", compact_flow_fixture["output"]
    )
    assert report["status"] == "PLANNED"
    return compact_flow_fixture


def test_exact_384_job_plan_and_worker_fingerprint(
    compact_flow_fixture: dict,
) -> None:
    output = compact_flow_fixture["output"]
    report = source.plan_source_collection("ignored.json", output)
    schedule = _read_jsonl(output / "source_schedule.jsonl")
    assert report["episode_count"] == 384
    assert report["workers"] == 16
    assert len(schedule) == 384
    assert len(
        {
            (row["split"], row["task_id"], row["solver_seed"])
            for row in schedule
        }
    ) == 384
    assert {row["workers"] for row in schedule} == {16}
    assert {row["solver_seed"] for row in schedule} == {41, 42}
    assert source.plan_source_collection(
        "ignored.json", output, dry_run=True, workers=20
    )["workers"] == 20
    with pytest.raises(ValueError, match="between 1 and 20"):
        source.plan_source_collection(
            "ignored.json", output, dry_run=True, workers=21
        )


def test_dry_collect_is_pure_and_never_invokes_runner(planned: dict) -> None:
    calls: list[dict] = []

    def forbidden_runner(**kwargs: object) -> dict:
        calls.append(dict(kwargs))
        raise AssertionError("dry-run invoked runner")

    trust_path = planned["output"] / "collection_trust_report__all.json"
    report = source.collect_source_episodes(
        "ignored.json", planned["output"], dry_run=True, runner=forbidden_runner
    )
    assert report["status"] == "DRY_RUN_PREVIEW"
    assert report["planned_episode_count"] == 384
    assert report["qualification_job_count"] == 384
    assert report["source_v2_qualification_rows_imported"] == 0
    assert report["runner_invoked"] is False
    assert calls == []
    assert not trust_path.exists()


def test_both_qualifications_finish_before_fail_closed_and_no_policy(
    planned: dict,
) -> None:
    calls: list[tuple[str, str, object]] = []

    def fake_runner(**kwargs: object) -> dict:
        split = Path(str(kwargs["config_path"])).stem
        phase = str(kwargs["phase"])
        calls.append((split, phase, kwargs.get("qualification_source")))
        assert phase == "qualify"
        _write_qualification(
            planned,
            output=Path(str(kwargs["output"])),
            split=split,
            jobs=kwargs["job_keys"],  # type: ignore[arg-type]
            conflicts=16 if split == "train" else 0,
        )
        return {"status": "ok"}

    report = source.collect_source_episodes(
        "ignored.json", planned["output"], runner=fake_runner
    )
    assert calls == [
        ("train", "qualify", None),
        ("development", "qualify", None),
    ]
    assert report["status"] == "STATE_SUPPLY_FAIL_NO_BACKFILL"
    assert report["realized_episode_count"] == 0
    assert report["qualification_passed"] == {
        "train": True,
        "development": False,
    }
    assert report["source_v2_qualification_rows_imported"] == 0
    assert not list(
        (planned["output"] / "collection").rglob(
            "realized_dynamic_manifest.jsonl"
        )
    )
    audit = _read_json(
        planned["output"]
        / "collection"
        / "development"
        / "compact_flow_qualification_audit.json"
    )
    assert audit["gates"]["minimum_nonzero_states"] is False
    assert audit["gates"]["minimum_nonzero_resets_per_map_od_variant"] is False
    assert audit["gates"]["minimum_nonzero_high_load_resets_per_map"] is False
    assert audit["gates"]["minimum_maps_with_a_reset_at_least_16_conflicts"] is False
    assert audit["gates"]["all_capacity_strata_active"] is False


def test_resume_reuses_both_exact_qualifications_then_runs_only_policy(
    planned: dict,
) -> None:
    for split in source.ALLOWED_SPLITS:
        jobs = {
            (str(row["task_id"]), solver_seed)
            for row in planned["manifests"][split]
            for solver_seed in source.SOLVER_SEEDS
        }
        _write_qualification(
            planned,
            output=planned["output"] / "collection" / split,
            split=split,
            jobs=jobs,
            conflicts=16,
        )
    calls: list[tuple[str, str, int]] = []

    def fake_runner(**kwargs: object) -> dict:
        split = Path(str(kwargs["config_path"])).stem
        phase = str(kwargs["phase"])
        calls.append((split, phase, int(kwargs["workers"])))
        assert phase == "realized_dynamic"
        rows = [
            {"task_id": task_id, "solver_seed": solver_seed}
            for task_id, solver_seed in sorted(kwargs["job_keys"])  # type: ignore[arg-type]
        ]
        _write_jsonl(
            Path(str(kwargs["output"])) / "realized_dynamic_manifest.jsonl", rows
        )
        return {"status": "ok"}

    report = source.collect_source_episodes(
        "ignored.json", planned["output"], resume=True, runner=fake_runner
    )
    assert calls == [
        ("train", "realized_dynamic", 16),
        ("development", "realized_dynamic", 16),
    ]
    assert report["status"] == "COLLECTED_EXACT_PRODUCT"
    assert report["realized_episode_count"] == 384
    assert report["qualification_reused"] == {
        "train": True,
        "development": True,
    }
    assert report["qualification_passed"] == {
        "train": True,
        "development": True,
    }
    assert report["source_v2_qualification_rows_imported"] == 0


def test_ge16_conflict_gate_is_independent_of_other_supply_gates(planned: dict) -> None:
    split = "train"
    jobs = {
        (str(row["task_id"]), solver_seed)
        for row in planned["manifests"][split]
        for solver_seed in source.SOLVER_SEEDS
    }
    output = planned["output"] / "collection" / split
    _write_qualification(
        planned, output=output, split=split, jobs=jobs, conflicts=4
    )
    audit = source._qualification_audit(
        planned["ctx"], split, output, jobs, planned["manifests"][split]
    )
    assert audit is not None
    assert audit["gates"]["minimum_nonzero_states"] is True
    assert audit["gates"]["all_capacity_strata_active"] is True
    assert audit["gates"]["minimum_resets_with_at_least_4_conflicts"] is True
    assert audit["gates"]["minimum_resets_with_at_least_16_conflicts"] is False
    assert audit["passed"] is False
