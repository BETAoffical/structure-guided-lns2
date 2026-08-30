from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

import experiments.stride_hierarchical_ch_compact_flow_tasks_v1 as tasks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_source_v1_registration.json"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _project_relative(project: Path, path: Path) -> str:
    return path.relative_to(project).as_posix()


def _finalize_registration(project: Path, config: Path) -> None:
    registration = _json(config)
    task_manifest = project / registration["registered_task_manifest"]["path"]
    registration["registered_task_manifest"]["sha256"] = hashlib.sha256(
        task_manifest.read_bytes()
    ).hexdigest()
    _write_json(config, registration)
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    _write_json(
        task_manifest.parent / "registration_report.json",
        {
            "schema": "lns2.stride.hierarchical_ch_compact_flow_registration_report.v1",
            "experiment_id": tasks.EXPERIMENT_ID,
            "status": "REGISTERED",
            "passed": True,
            "config_sha256": config_sha,
            "registered_task_count": tasks.EXPECTED_TASK_COUNT,
            "training_authorized": False,
        },
    )


def _fixture_registration(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    config = project / "configs" / "registration.json"
    split_maps = {
        "train": [f"train_{index:02d}" for index in range(8)],
        "development": [f"development_{index:02d}" for index in range(8)],
    }
    source_revisions: dict[str, dict[str, Any]] = {}
    source_manifest_rows: dict[str, list[dict[str, Any]]] = {"v4": [], "v5": []}
    registered_rows: list[dict[str, Any]] = []
    map_registration: dict[str, dict[str, Any]] = {}
    variants = list(tasks.OD_VARIANTS)

    for split, map_ids in split_maps.items():
        for index, map_id in enumerate(map_ids):
            source_id = "v4" if index < 4 else "v5"
            source_root = project / "sources" / source_id
            seeds = [11, 17] if source_id == "v4" else [23, 29]
            loads = [1, 2, 3]
            map_registration[map_id] = {
                "split": split,
                "source_revision": source_id,
                "capacity_free_cells": 20,
                "capacity_stratum": "small",
                "loads": loads,
                "task_seeds": seeds,
            }
            map_path = source_root / "maps" / f"{map_id}.map"
            metadata_path = source_root / "maps" / f"{map_id}.json"
            map_path.parent.mkdir(parents=True, exist_ok=True)
            map_path.write_bytes(f"map:{map_id}\n".encode())
            _write_json(
                metadata_path,
                {
                    "benchmark_id": map_id,
                    "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
                    "topology_metrics": {"free_cell_count": 20},
                },
            )
            for variant in variants:
                for seed in seeds:
                    for load in loads:
                        task_id = (
                            f"{map_id}__derived_{variant}__task_seed_{seed:04d}"
                            f"__agents_{load:04d}"
                        )
                        scenario_path = source_root / "scenarios" / f"{task_id}.scen"
                        task_path = source_root / "tasks" / f"{task_id}.json"
                        scenario_path.parent.mkdir(parents=True, exist_ok=True)
                        scenario_path.write_bytes(
                            f"version 1\n{map_id} {variant} {seed} {load}\n".encode()
                        )
                        _write_json(
                            task_path,
                            {
                                "benchmark_id": map_id,
                                "agent_count": load,
                                "od_variant": variant,
                                "task_seed": seed,
                                "scenario_sha256": hashlib.sha256(
                                    scenario_path.read_bytes()
                                ).hexdigest(),
                            },
                        )
                        source_row = {
                            "split": "balanced_wall_clock",
                            "source_group": "movingai",
                            "map_id": map_id,
                            "task_id": task_id,
                            "map_file": f"maps/{map_id}.map",
                            "map_metadata_file": f"maps/{map_id}.json",
                            "scenario_file": f"scenarios/{task_id}.scen",
                            "task_file": f"tasks/{task_id}.json",
                            "layout_mode": "dao_fixture",
                            "layout_variant": map_id,
                            "scenario_type": f"movingai_map_derived_{variant}",
                            "task_variant": f"{variant}_seed_{seed}_agents_{load}",
                            "agent_count": load,
                            "topology_metrics": {"free_cell_count": 20},
                            "dominant_flow_ratio": 0.0,
                            "hotspot_skew": 0.0,
                            "required_bottleneck_crossing_ratio": 0.0,
                            "mean_shortest_distance": float(load),
                        }
                        source_manifest_rows[source_id].append(source_row)
                        registered_rows.append(
                            {
                                "schema": tasks.REGISTERED_TASK_SCHEMA,
                                "split": split,
                                "source_revision": source_id,
                                "map_id": map_id,
                                "task_id": task_id,
                                "agent_count": load,
                                "od_variant": variant,
                                "task_seed": seed,
                                "map": {
                                    "path": _project_relative(project, map_path),
                                    "sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
                                },
                                "map_metadata": {
                                    "path": _project_relative(project, metadata_path),
                                    "sha256": hashlib.sha256(
                                        metadata_path.read_bytes()
                                    ).hexdigest(),
                                },
                                "scenario": {
                                    "path": _project_relative(project, scenario_path),
                                    "sha256": hashlib.sha256(
                                        scenario_path.read_bytes()
                                    ).hexdigest(),
                                },
                                "task": {
                                    "path": _project_relative(project, task_path),
                                    "sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
                                },
                            }
                        )

    for source_id, rows in source_manifest_rows.items():
        source_root = project / "sources" / source_id
        manifest = source_root / "manifest.jsonl"
        _write_jsonl(manifest, sorted(rows, key=lambda row: row["task_id"]))
        source_revisions[source_id] = {
            "dataset_root": _project_relative(project, source_root),
            "candidate_manifest": {
                "path": _project_relative(project, manifest),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
        }
    registration_root = project / "build" / "registration"
    task_manifest = registration_root / "registered_tasks.jsonl"
    _write_jsonl(
        task_manifest,
        sorted(registered_rows, key=lambda row: (row["split"], row["task_id"])),
    )
    _write_json(
        config,
        {
            "schema": tasks.REGISTRATION_SCHEMA,
            "experiment_id": tasks.EXPERIMENT_ID,
            "status": "REGISTERED",
            "map_splits": split_maps,
            "map_registration": map_registration,
            "task_product": {
                "od_variants": variants,
                "loads_per_map": 3,
                "task_seeds_per_map": 2,
                "tasks_per_map": 12,
                "tasks_per_split": 96,
                "registered_task_count": 192,
            },
            "source_revisions": source_revisions,
            "registered_task_manifest": {
                "path": _project_relative(project, task_manifest),
                "schema": tasks.REGISTERED_TASK_SCHEMA,
                "sha256": "0" * 64,
                "row_count": 192,
                "unique_map_file_count": 16,
                "unique_map_metadata_file_count": 16,
                "unique_scenario_file_count": 192,
                "unique_task_file_count": 192,
            },
            "legacy_h1_exclusion": {"map_ids": ["legacy_map"], "overlap_allowed": False},
            "claim_boundary": {
                "historical_rows_imported_as_new_source_rows": 0,
                "completed_v2_reset_rows_imported": 0,
                "sealed_final_semantic_access": False,
                "sealed_final_source_registered": False,
                "training_authorized": False,
            },
        },
    )
    _finalize_registration(project, config)
    return project, config


def test_materializes_exact_192_task_closed_loop_compatible_product(
    tmp_path: Path,
) -> None:
    project, config = _fixture_registration(tmp_path)
    output = tmp_path / "materialized"
    report = tasks.materialize_compact_flow_tasks(
        config, output, project_root=project
    )
    assert report == _json(output / tasks.REPORT_FILENAME)
    assert report["status"] == "MATERIALIZED"
    assert report["task_count"] == 192
    assert report["training_authorized"] is False
    assert report["sealed_final_semantic_access"] is False
    assert (output / tasks.IDENTITY_FILENAME).is_file()
    dataset = output / "dataset"
    assert _json(dataset / "dataset_summary.json")["task_count"] == 192
    assert len(_jsonl(dataset / "source_manifest.jsonl")) == 192

    map_counts: Counter[str] = Counter()
    products: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    for split in tasks.ALLOWED_SPLITS:
        rows = _jsonl(dataset / split / "manifest.jsonl")
        assert len(rows) == 96
        assert report["split_manifests"][split]["task_count"] == 96
        assert hashlib.sha256(
            (dataset / split / "manifest.jsonl").read_bytes()
        ).hexdigest() == report["split_manifests"][split]["sha256"]
        for row in rows:
            assert row["split"] == split
            assert row["training_authorized"] is False
            assert row["layout_mode"] == "dao_fixture"
            assert row["scenario_type"] == f"compact_flow_{row['scenario_index']}"
            assert row["scenario_index"] in {0, 1, 2, 3}
            assert row["source_scenario_type"].startswith("movingai_map_derived_")
            assert row["map_metadata_sha256"]
            map_counts[row["map_id"]] += 1
            products[row["map_id"]].add(
                (row["od_variant_id"], row["task_seed"], row["agent_count"])
            )
            for path_field, hash_field in (
                ("map_file", "map_sha256"),
                ("map_metadata_file", "map_metadata_sha256"),
                ("scenario_file", "scenario_sha256"),
                ("task_file", "task_sha256"),
            ):
                materialized = dataset / split / row[path_field]
                assert hashlib.sha256(materialized.read_bytes()).hexdigest() == row[hash_field]
    assert set(map_counts.values()) == {12}
    assert all(len(product) == 12 for product in products.values())


def test_dry_run_validates_sources_but_writes_nothing(tmp_path: Path) -> None:
    project, config = _fixture_registration(tmp_path)
    output = tmp_path / "dry-run"
    report = tasks.materialize_compact_flow_tasks(
        config, output, project_root=project, dry_run=True
    )
    assert report["status"] == "DRY_RUN"
    assert report["task_count"] == 192
    assert not output.exists()


def test_resume_is_exact_and_nonempty_foreign_output_fails_closed(
    tmp_path: Path,
) -> None:
    project, config = _fixture_registration(tmp_path)
    output = tmp_path / "materialized"
    expected = tasks.materialize_compact_flow_tasks(
        config, output, project_root=project
    )
    with pytest.raises(FileExistsError, match="resume"):
        tasks.materialize_compact_flow_tasks(config, output, project_root=project)
    resumed = tasks.materialize_compact_flow_tasks(
        config, output, project_root=project, resume=True
    )
    assert resumed == expected

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "unrelated.txt").write_text("foreign\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no materialization identity"):
        tasks.materialize_compact_flow_tasks(
            config, foreign, project_root=project, resume=True
        )

    identity = _json(output / tasks.IDENTITY_FILENAME)
    identity["materialization_fingerprint"] = "0" * 64
    _write_json(output / tasks.IDENTITY_FILENAME, identity)
    with pytest.raises(ValueError, match="different fingerprint"):
        tasks.materialize_compact_flow_tasks(
            config, output, project_root=project, resume=True
        )


def test_registered_source_sha_tamper_fails_before_output(tmp_path: Path) -> None:
    project, config = _fixture_registration(tmp_path)
    registration = _json(config)
    registered_manifest = project / registration["registered_task_manifest"]["path"]
    first = _jsonl(registered_manifest)[0]
    source_task = project / first["task"]["path"]
    source_task.write_bytes(source_task.read_bytes() + b"tampered\n")
    output = tmp_path / "tampered-output"
    with pytest.raises(ValueError, match="task.*SHA256 mismatch"):
        tasks.materialize_compact_flow_tasks(
            config, output, project_root=project, dry_run=True
        )
    assert not output.exists()


def test_sealed_final_raw_path_is_rejected_before_file_access(tmp_path: Path) -> None:
    project, config = _fixture_registration(tmp_path)
    registration = _json(config)
    registered_manifest = project / registration["registered_task_manifest"]["path"]
    rows = _jsonl(registered_manifest)
    rows[0]["scenario"] = {
        "path": "sealed_final/raw/never_read.scen",
        "sha256": "1" * 64,
    }
    _write_jsonl(registered_manifest, rows)
    _finalize_registration(project, config)
    with pytest.raises(ValueError, match="sealed-final registered scenario raw path"):
        tasks.materialize_compact_flow_tasks(
            config,
            tmp_path / "sealed-output",
            project_root=project,
            dry_run=True,
        )
    assert not (project / "sealed_final").exists()


def test_repository_registration_dry_run_when_frozen(tmp_path: Path) -> None:
    registration = _json(REAL_CONFIG)
    manifest_pin = dict(registration.get("registered_task_manifest") or {})
    if not tasks._is_sha256(manifest_pin.get("sha256")) or str(
        manifest_pin.get("sha256")
    ).startswith("__"):
        pytest.skip("compact-flow registered task manifest has not been frozen yet")
    report = tasks.materialize_compact_flow_tasks(
        REAL_CONFIG,
        tmp_path / "real-dry-run",
        project_root=PROJECT_ROOT,
        dry_run=True,
    )
    assert report["status"] == "DRY_RUN"
    assert report["task_count"] == 192
    assert not (tmp_path / "real-dry-run").exists()
