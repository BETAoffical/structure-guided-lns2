#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import sha256_file  # noqa: E402
from experiments.balanced_wall_clock import _write_jsonl_atomic  # noqa: E402
from experiments.repair_collection import _read_json, _read_jsonl, _write_json  # noqa: E402


SCHEMA = "lns2.stride.structpool_revised_six_map_materialization.v1"
SPLIT = "balanced_wall_clock"
ARTIFACT_FIELDS = (
    "map_file",
    "scenario_file",
    "map_metadata_file",
    "task_file",
    "legacy_instance_file",
)


def _registered(path_value: str, sha256: str) -> Path:
    path = (PROJECT_ROOT / path_value).resolve()
    if not path.is_file() or sha256_file(path) != sha256:
        raise ValueError(f"registered revised-cohort input changed: {path}")
    return path


def materialize(config_path: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    if (
        config.get("schema") != SCHEMA
        or config.get("scientific_status")
        != "qualification_conditioned_exact_task_reuse_before_revised_reset_qualification"
        or config.get("experiment_id") != "stride-structpool-revised-six-map-reset-v1"
        or config.get("pre_registration_parent_commit") != "75e39c9"
        or config.get("split") != SPLIT
        or int(config.get("expected_map_count", -1)) != 6
        or int(config.get("expected_task_count", -1)) != 12
        or config.get("task_generation_changed") is not False
        or config.get("task_selection_uses_reset_outcomes") is not True
        or config.get("repair_outcomes_read") is not False
        or config.get("controller_outcomes_read") is not False
        or config.get("ttf_outcomes_read") is not False
        or config.get("claim_boundary")
        != "qualification_conditioned_mixed_task_semantics_not_fresh_ood_or_speed_evidence"
    ):
        raise ValueError("revised six-map materialization contract changed")
    output_root = Path(output or config["output"]).resolve()
    if output_root == PROJECT_ROOT or output_root == config_path.parent:
        raise ValueError("invalid revised-cohort output")

    selected_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    artifact_registry: dict[str, str] = {}
    source_counts: collections.Counter[str] = collections.Counter()
    for source_spec in config.get("sources") or ():
        source = dict(source_spec)
        dataset_root = (PROJECT_ROOT / str(source["dataset"])).resolve()
        split_root = dataset_root / SPLIT
        manifest_path = _registered(
            str(Path(source["dataset"]) / SPLIT / "manifest.jsonl"),
            str(source["manifest_sha256"]),
        )
        _registered(str(source["evidence"]), str(source["evidence_sha256"]))
        indexed = {str(row["task_id"]): dict(row) for row in _read_jsonl(manifest_path)}
        requested = list(map(str, source.get("task_ids") or ()))
        if not requested or len(requested) != len(set(requested)):
            raise ValueError(f"{source['id']}: invalid registered task list")
        missing = sorted(set(requested) - set(indexed))
        if missing:
            raise ValueError(f"{source['id']}: missing registered tasks: {missing}")
        for task_id in requested:
            if task_id in selected_ids:
                raise ValueError(f"duplicate revised-cohort task: {task_id}")
            selected_ids.add(task_id)
            row = indexed[task_id]
            source_counts[str(source["id"])] += 1
            for field in ARTIFACT_FIELDS:
                if not row.get(field):
                    continue
                relative = Path(str(row[field]))
                source_path = split_root / relative
                if not source_path.is_file():
                    raise ValueError(f"revised-cohort artifact missing: {source_path}")
                artifact_sha = sha256_file(source_path)
                previous = artifact_registry.get(relative.as_posix())
                if previous is not None and previous != artifact_sha:
                    raise ValueError(f"revised-cohort artifact collision: {relative}")
                artifact_registry[relative.as_posix()] = artifact_sha
                destination = output_root / SPLIT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file() and sha256_file(destination) != artifact_sha:
                    raise ValueError(f"revised-cohort output collision: {relative}")
                if not destination.is_file():
                    shutil.copy2(source_path, destination)
            selected_rows.append(row)

    expected_maps = {
        str(row["map_id"]): (str(row["layout_mode"]), int(row["agent_count"]))
        for row in config.get("expected_maps") or ()
    }
    observed_maps: dict[str, tuple[str, int]] = {}
    by_map: collections.Counter[str] = collections.Counter()
    for row in selected_rows:
        map_id = str(row["map_id"])
        value = (str(row["layout_mode"]), int(row["agent_count"]))
        if map_id in observed_maps and observed_maps[map_id] != value:
            raise ValueError(f"revised-cohort map dimensions drifted: {map_id}")
        observed_maps[map_id] = value
        by_map[map_id] += 1
    if (
        len(selected_rows) != 12
        or observed_maps != expected_maps
        or set(by_map.values()) != {2}
    ):
        raise ValueError("revised six-map dimensions differ from registration")

    selected_rows.sort(key=lambda row: str(row["task_id"]))
    manifest_path = output_root / SPLIT / "manifest.jsonl"
    _write_jsonl_atomic(manifest_path, selected_rows)
    artifact_rows = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(artifact_registry.items())
    ]
    artifact_path = output_root / "artifact_registry.jsonl"
    _write_jsonl_atomic(artifact_path, artifact_rows)
    summary = {
        "schema": "lns2.stride.structpool_revised_six_map_dataset.v1",
        "materialization_config_sha256": sha256_file(config_path),
        "map_count": len(observed_maps),
        "task_count": len(selected_rows),
        "task_ids": [str(row["task_id"]) for row in selected_rows],
        "source_counts": dict(sorted(source_counts.items())),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_registry_sha256": sha256_file(artifact_path),
        "task_generation_changed": False,
        "ttf_outcomes_read": False,
    }
    _write_json(output_root / "dataset_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the revised six-map reset cohort.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    report = materialize(arguments.config, arguments.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
