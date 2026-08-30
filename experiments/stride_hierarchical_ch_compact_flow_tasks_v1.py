from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_source_v1_registration.json"
)
REGISTRATION_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_source_registration.v1"
)
REGISTERED_TASK_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_registered_task.v1"
)
MATERIALIZATION_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_task_materialization.v1"
)
IDENTITY_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_task_materialization_identity.v1"
)
DATASET_SUMMARY_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_dataset_summary.v1"
)
SOURCE_MANIFEST_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_materialized_source.v1"
)
EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_source_v1"
ALLOWED_SPLITS = ("train", "development")
OD_VARIANTS = ("opposite_exchange", "uniform_random")
EXPECTED_MAPS_PER_SPLIT = 8
EXPECTED_TASKS_PER_MAP = 12
EXPECTED_TASKS_PER_SPLIT = 96
EXPECTED_TASK_COUNT = 192
EXPECTED_UNIQUE_MAP_FILES = 16
EXPECTED_UNIQUE_MAP_METADATA_FILES = 16
EXPECTED_UNIQUE_SCENARIO_FILES = 192
EXPECTED_UNIQUE_TASK_FILES = 192
REPORT_FILENAME = "compact_flow_materialization_report.json"
IDENTITY_FILENAME = "compact_flow_materialization_identity.json"

_PIN_FIELDS = {
    "map": ("map_file", "map_sha256"),
    "map_metadata": ("map_metadata_file", "map_metadata_sha256"),
    "scenario": ("scenario_file", "scenario_sha256"),
    "task": ("task_file", "task_sha256"),
}
_TASK_ID = re.compile(
    r"^(?P<map>.+)__derived_(?P<variant>opposite_exchange|uniform_random)"
    r"__task_seed_(?P<seed>\d+)__agents_(?P<load>\d+)$"
)
_SEALED_TOKENS = {"sealed_final", "sealed-final", "sealedfinal"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


def _write_exact(path: Path, payload: bytes, *, resume: bool) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing output differs from registered materialization: {path}")
        if not resume:
            raise FileExistsError(f"output already exists; use --resume: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _resolve_project_path(project_root: Path, registered: str, *, label: str) -> Path:
    relative = Path(registered)
    if not registered or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a non-empty project-relative path: {registered}")
    resolved = (project_root / relative).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the project: {registered}") from error
    return resolved


def _contained_path(root: Path, relative_value: Any, *, label: str) -> Path:
    relative = Path(str(relative_value))
    if not str(relative_value) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a contained relative path: {relative_value}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its registered dataset: {relative_value}") from error
    return resolved


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _reject_sealed(value: Any, *, label: str) -> None:
    text = str(value).replace("\\", "/").lower()
    tokens = {token for token in re.split(r"[/_.]+", text) if token}
    if (
        any(marker in text for marker in _SEALED_TOKENS)
        or {"sealed", "final"} <= tokens
    ):
        raise ValueError(f"sealed-final {label} is not authorized: {value}")


def _pinned_project_file(
    project_root: Path, pin: dict[str, Any], *, label: str
) -> tuple[Path, str]:
    registered = str(pin.get("path", ""))
    _reject_sealed(registered, label=label)
    expected = str(pin.get("sha256", "")).lower()
    if not _is_sha256(expected):
        raise ValueError(f"{label} has no valid registered SHA256")
    path = _resolve_project_path(project_root, registered, label=label)
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return path, expected


def _task_identity(row: dict[str, Any]) -> tuple[str, str, int, int]:
    task_id = str(row.get("task_id", ""))
    match = _TASK_ID.fullmatch(task_id)
    if match is None:
        raise ValueError(f"registered task ID has unsupported compact-flow form: {task_id}")
    parsed = (
        str(match.group("map")),
        str(match.group("variant")),
        int(match.group("seed")),
        int(match.group("load")),
    )
    variant = str(
        row.get("od_variant_id")
        or row.get("od_variant")
        or row.get("task_variant_id")
        or parsed[1]
    )
    seed = int(row.get("task_seed", row.get("od_seed", parsed[2])))
    load = int(row.get("agent_count", parsed[3]))
    map_id = str(row.get("map_id", parsed[0]))
    observed = (map_id, variant, seed, load)
    if observed != parsed:
        raise ValueError(f"registered task metadata differs from task ID: {task_id}")
    return observed


def _registered_pin(row: dict[str, Any], kind: str) -> dict[str, Any]:
    value = row.get(kind)
    if isinstance(value, dict):
        return dict(value)
    path_field, hash_field = _PIN_FIELDS[kind]
    return {"path": row.get(path_field), "sha256": row.get(hash_field)}


def _registration_report(
    *,
    project_root: Path,
    registration: dict[str, Any],
    config_sha256: str,
    registered_manifest_path: Path,
) -> tuple[Path, str, dict[str, Any]]:
    pin = dict(registration.get("registration_report") or {})
    if pin.get("path"):
        path = _resolve_project_path(
            project_root, str(pin["path"]), label="registration report"
        )
        expected = str(pin.get("sha256", "")).lower()
        if expected and (not _is_sha256(expected) or not path.is_file() or _sha256(path) != expected):
            raise ValueError("registration report SHA256 mismatch")
    else:
        path = registered_manifest_path.parent / "registration_report.json"
    if not path.is_file():
        raise ValueError(f"registration report is missing: {path}")
    report = _read_json(path)
    report_sha = _sha256(path)
    if (
        report.get("status") != "REGISTERED"
        or report.get("passed") is not True
        or str(report.get("config_sha256", "")).lower() != config_sha256
        or int(
            report.get(
                "registered_task_count",
                report.get(
                    "expected_task_count",
                    dict(report.get("registered_task_manifest") or {}).get(
                        "row_count", -1
                    ),
                ),
            )
        )
        != EXPECTED_TASK_COUNT
    ):
        raise ValueError("registration report does not pin the exact registered config/product")
    return path, report_sha, report


def _validate_registration_structure(registration: dict[str, Any]) -> dict[str, Any]:
    if (
        registration.get("schema") != REGISTRATION_SCHEMA
        or registration.get("experiment_id") != EXPERIMENT_ID
        or registration.get("status") != "REGISTERED"
    ):
        raise ValueError("unsupported compact-flow task registration")
    claim = dict(registration.get("claim_boundary") or {})
    if (
        claim.get("training_authorized") is not False
        or claim.get("sealed_final_semantic_access") is not False
        or claim.get("sealed_final_source_registered") is not False
        or int(claim.get("historical_rows_imported_as_new_source_rows", -1)) != 0
        or int(claim.get("completed_v2_reset_rows_imported", -1)) != 0
    ):
        raise ValueError("compact-flow registration must remain no-train and final-sealed")
    raw_splits = dict(registration.get("map_splits") or {})
    if set(raw_splits) != set(ALLOWED_SPLITS):
        raise ValueError("only train/development map splits may be registered")
    map_splits = {
        split: tuple(map(str, raw_splits.get(split) or ())) for split in ALLOWED_SPLITS
    }
    if (
        any(len(values) != EXPECTED_MAPS_PER_SPLIT for values in map_splits.values())
        or any(len(set(values)) != len(values) for values in map_splits.values())
        or set(map_splits["train"]) & set(map_splits["development"])
    ):
        raise ValueError("compact-flow registration requires 8+8 disjoint maps")
    allowed_ids = {map_id for values in map_splits.values() for map_id in values}
    for map_id in allowed_ids:
        _reject_sealed(map_id, label="map ID")

    task_product = dict(registration.get("task_product") or {})
    variants = tuple(map(str, task_product.get("od_variants") or ()))
    if (
        variants != OD_VARIANTS
        or int(task_product.get("loads_per_map", -1)) != 3
        or int(task_product.get("task_seeds_per_map", -1)) != 2
        or int(task_product.get("tasks_per_map", -1)) != EXPECTED_TASKS_PER_MAP
        or int(task_product.get("tasks_per_split", -1)) != EXPECTED_TASKS_PER_SPLIT
        or int(task_product.get("registered_task_count", -1)) != EXPECTED_TASK_COUNT
    ):
        raise ValueError("compact-flow task product is not exact 3x2x2 per map")

    map_registration = {
        str(map_id): dict(value)
        for map_id, value in dict(registration.get("map_registration") or {}).items()
    }
    if set(map_registration) != allowed_ids:
        raise ValueError("map_registration differs from train/development map splits")
    for split in ALLOWED_SPLITS:
        for map_id in map_splits[split]:
            row = map_registration[map_id]
            loads = tuple(map(int, row.get("loads") or ()))
            seeds = tuple(map(int, row.get("task_seeds") or ()))
            if (
                str(row.get("split")) != split
                or len(loads) != 3
                or len(set(loads)) != 3
                or min(loads, default=0) <= 0
                or len(seeds) != 2
                or len(set(seeds)) != 2
                or not str(row.get("source_revision", ""))
            ):
                raise ValueError(f"invalid registered static product for map: {map_id}")

    legacy = dict(registration.get("legacy_h1_exclusion") or {})
    if (
        legacy.get("overlap_allowed") is not False
        or allowed_ids & set(map(str, legacy.get("map_ids") or ()))
    ):
        raise ValueError("compact-flow maps overlap the legacy H1 map cohort")
    return {
        "map_splits": map_splits,
        "allowed_ids": allowed_ids,
        "task_product": task_product,
        "map_registration": map_registration,
    }


def load_registered_compact_flow_context(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = _resolve_project_path(root, str(path), label="registration config")
    else:
        path = path.resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("registration config must remain inside project_root") from error
    config_payload = path.read_bytes()
    registration = json.loads(config_payload.decode("utf-8"))
    if not isinstance(registration, dict):
        raise ValueError(f"expected a JSON object: {path}")
    config_sha = _sha256_bytes(config_payload)
    structure = _validate_registration_structure(registration)

    registered_pin = dict(registration.get("registered_task_manifest") or {})
    if (
        registered_pin.get("schema") != REGISTERED_TASK_SCHEMA
        or int(registered_pin.get("row_count", -1)) != EXPECTED_TASK_COUNT
        or int(registered_pin.get("unique_map_file_count", -1))
        != EXPECTED_UNIQUE_MAP_FILES
        or int(registered_pin.get("unique_map_metadata_file_count", -1))
        != EXPECTED_UNIQUE_MAP_METADATA_FILES
        or int(registered_pin.get("unique_scenario_file_count", -1))
        != EXPECTED_UNIQUE_SCENARIO_FILES
        or int(registered_pin.get("unique_task_file_count", -1))
        != EXPECTED_UNIQUE_TASK_FILES
    ):
        raise ValueError("registered task manifest count/schema contract changed")
    registered_manifest_path, registered_manifest_sha = _pinned_project_file(
        root, registered_pin, label="registered task manifest"
    )
    registered_rows = _read_jsonl(registered_manifest_path)
    if len(registered_rows) != EXPECTED_TASK_COUNT:
        raise ValueError("registered task manifest is not exactly 192 rows")

    # Reject every split, map, and raw path before opening any registered raw file.
    for row in registered_rows:
        split = str(row.get("split", ""))
        map_id = str(row.get("map_id", ""))
        if split not in ALLOWED_SPLITS or map_id not in structure["map_splits"].get(split, ()):
            raise ValueError(f"registered task crosses train/development boundary: {row.get('task_id')}")
        _reject_sealed(map_id, label="registered task map ID")
        _reject_sealed(row.get("task_id", ""), label="registered task ID")
        for kind in _PIN_FIELDS:
            _reject_sealed(
                _registered_pin(row, kind).get("path", ""),
                label=f"registered {kind} raw path",
            )

    source_revisions: dict[str, dict[str, Any]] = {}
    candidate_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for source_id, raw_revision in dict(registration.get("source_revisions") or {}).items():
        revision = dict(raw_revision)
        registered_dataset_root = str(revision.get("dataset_root", ""))
        _reject_sealed(registered_dataset_root, label=f"{source_id} dataset root")
        dataset_root = _resolve_project_path(
            root, registered_dataset_root, label=f"{source_id} dataset root"
        )
        manifest_path, manifest_sha = _pinned_project_file(
            root,
            dict(revision.get("candidate_manifest") or {}),
            label=f"{source_id} candidate manifest",
        )
        if manifest_path.parent.resolve() != dataset_root.resolve():
            raise ValueError(f"{source_id} candidate manifest is outside its dataset root")
        rows = _read_jsonl(manifest_path)
        by_task: dict[str, dict[str, Any]] = {}
        for row in rows:
            task_id = str(row.get("task_id", ""))
            if not task_id or task_id in by_task:
                raise ValueError(f"{source_id} candidate manifest has duplicate/empty task ID")
            by_task[task_id] = row
        source_revisions[str(source_id)] = {
            "dataset_root": dataset_root,
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha,
        }
        candidate_rows[str(source_id)] = by_task

    referenced_revisions = {
        str(row["source_revision"]) for row in structure["map_registration"].values()
    }
    if referenced_revisions != set(source_revisions):
        raise ValueError("source revisions differ from the exact registered map product")

    report_path, report_sha, report = _registration_report(
        project_root=root,
        registration=registration,
        config_sha256=config_sha,
        registered_manifest_path=registered_manifest_path,
    )
    return {
        "project_root": root,
        "config_path": path,
        "config_sha256": config_sha,
        "registration": registration,
        "registration_report_path": report_path,
        "registration_report_sha256": report_sha,
        "registration_report": report,
        "registered_manifest_path": registered_manifest_path,
        "registered_manifest_sha256": registered_manifest_sha,
        "registered_rows": registered_rows,
        "source_revisions": source_revisions,
        "candidate_rows": candidate_rows,
        **structure,
    }


def _source_pin_path(
    *,
    project_root: Path,
    dataset_root: Path,
    candidate_path: Path,
    pin: dict[str, Any],
    label: str,
) -> tuple[Path, str]:
    registered = str(pin.get("path", ""))
    _reject_sealed(registered, label=label)
    expected_sha = str(pin.get("sha256", "")).lower()
    if not _is_sha256(expected_sha):
        raise ValueError(f"{label} has no valid registered SHA256")
    relative = Path(registered)
    if not registered or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must use a contained registered path")
    project_candidate = (project_root / relative).resolve()
    dataset_candidate = (dataset_root / relative).resolve()
    if project_candidate == candidate_path:
        source = project_candidate
    elif dataset_candidate == candidate_path:
        source = dataset_candidate
    else:
        raise ValueError(f"{label} path differs from its hash-pinned candidate manifest row")
    if not source.is_file():
        raise ValueError(f"{label} is missing: {source}")
    payload = source.read_bytes()
    if _sha256_bytes(payload) != expected_sha:
        raise ValueError(f"{label} SHA256 mismatch")
    return source, expected_sha


def _preflight_plan(ctx: dict[str, Any], output_root: Path) -> dict[str, Any]:
    dataset_root = output_root / "dataset"
    for revision in ctx["source_revisions"].values():
        source_root = Path(revision["dataset_root"]).resolve()
        if (
            dataset_root.resolve() == source_root
            or source_root in dataset_root.resolve().parents
            or dataset_root.resolve() in source_root.parents
        ):
            raise ValueError("compact-flow output must not overlap a registered source dataset")

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in ALLOWED_SPLITS}
    copy_payloads: dict[Path, bytes] = {}
    copy_hashes: dict[Path, str] = {}
    provenance_rows: list[dict[str, Any]] = []
    observed_product: dict[str, set[tuple[str, int, int]]] = {
        map_id: set() for map_id in ctx["allowed_ids"]
    }
    unique_sources: dict[str, set[Path]] = {kind: set() for kind in _PIN_FIELDS}

    for registered_row in ctx["registered_rows"]:
        map_id, variant, task_seed, load = _task_identity(registered_row)
        split = str(registered_row["split"])
        map_spec = ctx["map_registration"][map_id]
        source_id = str(registered_row.get("source_revision", ""))
        if source_id != str(map_spec["source_revision"]):
            raise ValueError(f"registered task source revision differs for map: {map_id}")
        expected_product = {
            (od_variant, int(seed), int(agent_count))
            for od_variant in OD_VARIANTS
            for seed in map_spec["task_seeds"]
            for agent_count in map_spec["loads"]
        }
        if (variant, task_seed, load) not in expected_product:
            raise ValueError(f"registered task is outside the fixed product: {registered_row['task_id']}")
        if (variant, task_seed, load) in observed_product[map_id]:
            raise ValueError(f"duplicate registered task product cell: {registered_row['task_id']}")
        observed_product[map_id].add((variant, task_seed, load))
        scenario_index = OD_VARIANTS.index(variant) * 2 + list(
            map(int, map_spec["task_seeds"])
        ).index(task_seed)

        revision = ctx["source_revisions"].get(source_id)
        source_rows = ctx["candidate_rows"].get(source_id)
        if revision is None or source_rows is None:
            raise ValueError(f"unknown registered source revision: {source_id}")
        task_id = str(registered_row["task_id"])
        candidate = source_rows.get(task_id)
        if candidate is None:
            raise ValueError(f"registered task is absent from candidate manifest: {task_id}")
        if (
            str(candidate.get("map_id")) != map_id
            or int(candidate.get("agent_count", -1)) != load
            or _task_identity(candidate) != (map_id, variant, task_seed, load)
        ):
            raise ValueError(f"candidate manifest metadata differs for registered task: {task_id}")

        output_row = dict(candidate)
        output_row.update(
            {
                "schema": REGISTERED_TASK_SCHEMA,
                "split": split,
                "source_revision": source_id,
                "source_candidate_manifest": str(revision["manifest_path"]),
                "source_candidate_manifest_sha256": revision["manifest_sha256"],
                "od_variant_id": variant,
                "od_variant": variant,
                "task_seed": task_seed,
                "source_scenario_type": candidate.get("scenario_type"),
                "scenario_type": f"compact_flow_{scenario_index}",
                "scenario_index": scenario_index,
                "capacity_stratum": str(map_spec["capacity_stratum"]),
                "capacity_free_cells": int(map_spec["capacity_free_cells"]),
                "training_authorized": False,
                "sealed_final_semantic_access": False,
            }
        )
        provenance = {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "split": split,
            "map_id": map_id,
            "task_id": task_id,
            "source_revision": source_id,
            "source_candidate_manifest": str(revision["manifest_path"]),
            "source_candidate_manifest_sha256": revision["manifest_sha256"],
        }
        for kind, (path_field, hash_field) in _PIN_FIELDS.items():
            candidate_path = _contained_path(
                Path(revision["dataset_root"]),
                candidate.get(path_field, ""),
                label=f"candidate {task_id}/{path_field}",
            )
            source, expected_sha = _source_pin_path(
                project_root=ctx["project_root"],
                dataset_root=Path(revision["dataset_root"]),
                candidate_path=candidate_path,
                pin=_registered_pin(registered_row, kind),
                label=f"registered {task_id}/{kind}",
            )
            relative = Path(str(candidate[path_field]))
            destination = _contained_path(
                dataset_root / split,
                relative.as_posix(),
                label=f"materialized {task_id}/{path_field}",
            )
            payload = source.read_bytes()
            if destination in copy_payloads and copy_payloads[destination] != payload:
                raise ValueError(f"registered sources collide at output path: {destination}")
            copy_payloads[destination] = payload
            copy_hashes[destination] = expected_sha
            unique_sources[kind].add(source)
            output_row[path_field] = relative.as_posix()
            output_row[hash_field] = expected_sha
            provenance[path_field] = relative.as_posix()
            provenance[hash_field] = expected_sha
            provenance[f"source_{path_field}"] = str(source)
        split_rows[split].append(output_row)
        provenance_rows.append(provenance)

    for map_id, cells in observed_product.items():
        if len(cells) != EXPECTED_TASKS_PER_MAP:
            raise ValueError(f"map does not contain the exact 12-task product: {map_id}")
    if {kind: len(values) for kind, values in unique_sources.items()} != {
        "map": EXPECTED_UNIQUE_MAP_FILES,
        "map_metadata": EXPECTED_UNIQUE_MAP_METADATA_FILES,
        "scenario": EXPECTED_UNIQUE_SCENARIO_FILES,
        "task": EXPECTED_UNIQUE_TASK_FILES,
    }:
        raise ValueError("registered task raw-file cardinalities differ from 16/16/192/192")
    for split in ALLOWED_SPLITS:
        split_rows[split].sort(key=lambda row: str(row["task_id"]))
        if (
            len(split_rows[split]) != EXPECTED_TASKS_PER_SPLIT
            or len({str(row["task_id"]) for row in split_rows[split]})
            != EXPECTED_TASKS_PER_SPLIT
        ):
            raise ValueError(f"{split} does not contain exactly 96 unique tasks")
    if sum(map(len, split_rows.values())) != EXPECTED_TASK_COUNT:
        raise ValueError("materialization plan is not exactly 192 tasks")

    provenance_rows.sort(key=lambda row: (str(row["split"]), str(row["task_id"])))
    manifest_payloads = {
        split: _jsonl_bytes(split_rows[split]) for split in ALLOWED_SPLITS
    }
    summary = {
        "schema": DATASET_SUMMARY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "dataset_revision": "stride-hierarchical-ch-compact-flow-tasks-v1",
        "splits": list(ALLOWED_SPLITS),
        "map_count": len(ctx["allowed_ids"]),
        "task_count": EXPECTED_TASK_COUNT,
        "split_map_counts": {
            split: len(ctx["map_splits"][split]) for split in ALLOWED_SPLITS
        },
        "split_task_counts": {
            split: len(split_rows[split]) for split in ALLOWED_SPLITS
        },
        "tasks_per_map": EXPECTED_TASKS_PER_MAP,
        "od_variants": list(OD_VARIANTS),
        "config_sha256": ctx["config_sha256"],
        "registered_task_manifest_sha256": ctx["registered_manifest_sha256"],
        "registration_report_sha256": ctx["registration_report_sha256"],
        "sealed_final_semantic_access": False,
        "training_authorized": False,
    }
    fingerprint_payload = {
        "schema": IDENTITY_SCHEMA,
        "config_sha256": ctx["config_sha256"],
        "registered_task_manifest_sha256": ctx["registered_manifest_sha256"],
        "registration_report_sha256": ctx["registration_report_sha256"],
        "copies": [
            {
                "path": destination.relative_to(output_root).as_posix(),
                "sha256": copy_hashes[destination],
            }
            for destination in sorted(copy_payloads)
        ],
        "split_manifest_sha256": {
            split: _sha256_bytes(manifest_payloads[split]) for split in ALLOWED_SPLITS
        },
        "dataset_summary_sha256": _sha256_bytes(_json_bytes(summary)),
        "source_manifest_sha256": _sha256_bytes(_jsonl_bytes(provenance_rows)),
    }
    fingerprint = _sha256_bytes(_json_bytes(fingerprint_payload))
    identity = {
        **fingerprint_payload,
        "materialization_fingerprint": fingerprint,
    }
    return {
        "dataset_root": dataset_root,
        "split_rows": split_rows,
        "copy_payloads": copy_payloads,
        "copy_hashes": copy_hashes,
        "provenance_rows": provenance_rows,
        "manifest_payloads": manifest_payloads,
        "summary": summary,
        "identity": identity,
        "fingerprint": fingerprint,
    }


def _check_output_identity(
    output_root: Path, expected_identity: dict[str, Any], *, resume: bool
) -> None:
    if not output_root.exists():
        return
    if not output_root.is_dir():
        raise ValueError(f"compact-flow output root is not a directory: {output_root}")
    if not any(output_root.iterdir()):
        return
    identity_path = output_root / IDENTITY_FILENAME
    if not identity_path.is_file():
        raise ValueError("non-empty compact-flow output has no materialization identity")
    observed = _read_json(identity_path)
    if (
        observed.get("schema") != IDENTITY_SCHEMA
        or observed.get("materialization_fingerprint")
        != expected_identity["materialization_fingerprint"]
        or observed != expected_identity
    ):
        raise ValueError("non-empty compact-flow output has a different fingerprint")
    if not resume:
        raise FileExistsError("matching compact-flow output already exists; use --resume")


def materialize_compact_flow_tasks(
    config_path: str | Path = DEFAULT_CONFIG,
    output_root: str | Path = PROJECT_ROOT
    / "build"
    / "stride-hierarchical-ch-compact-flow-source-v1",
    *,
    dry_run: bool = False,
    resume: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Copy the exact registered v4/v5 task product into a train/dev dataset.

    This is an outcome-blind copier. It reads only registration, candidate manifests,
    and the four explicitly hash-pinned raw inputs for each registered task.
    """

    if dry_run and resume:
        raise ValueError("dry-run and resume are mutually exclusive")
    ctx = load_registered_compact_flow_context(
        config_path, project_root=project_root
    )
    output = Path(output_root).resolve()
    plan = _preflight_plan(ctx, output)
    _check_output_identity(output, plan["identity"], resume=resume)

    split_manifests = {
        split: {
            "path": str(plan["dataset_root"] / split / "manifest.jsonl"),
            "sha256": _sha256_bytes(plan["manifest_payloads"][split]),
            "task_count": len(plan["split_rows"][split]),
        }
        for split in ALLOWED_SPLITS
    }
    report = {
        "schema": MATERIALIZATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "DRY_RUN" if dry_run else "MATERIALIZED",
        "config_path": str(ctx["config_path"]),
        "config_sha256": ctx["config_sha256"],
        "registration_report_path": str(ctx["registration_report_path"]),
        "registration_report_sha256": ctx["registration_report_sha256"],
        "registered_task_manifest": str(ctx["registered_manifest_path"]),
        "registered_task_manifest_sha256": ctx["registered_manifest_sha256"],
        "dataset_root": str(plan["dataset_root"]),
        "task_count": EXPECTED_TASK_COUNT,
        "split_manifests": split_manifests,
        "materialization_fingerprint": plan["fingerprint"],
        "sealed_final_semantic_access": False,
        "training_authorized": False,
    }
    if dry_run:
        return report

    output.mkdir(parents=True, exist_ok=True)
    _write_exact(
        output / IDENTITY_FILENAME,
        _json_bytes(plan["identity"]),
        resume=resume,
    )
    for destination in sorted(plan["copy_payloads"]):
        _write_exact(
            destination,
            plan["copy_payloads"][destination],
            resume=resume,
        )
    for split in ALLOWED_SPLITS:
        _write_exact(
            plan["dataset_root"] / split / "manifest.jsonl",
            plan["manifest_payloads"][split],
            resume=resume,
        )
    _write_exact(
        plan["dataset_root"] / "source_manifest.jsonl",
        _jsonl_bytes(plan["provenance_rows"]),
        resume=resume,
    )
    _write_exact(
        plan["dataset_root"] / "dataset_summary.json",
        _json_bytes(plan["summary"]),
        resume=resume,
    )

    for destination, expected_sha in plan["copy_hashes"].items():
        if _sha256(destination) != expected_sha:
            raise ValueError(f"materialized target SHA256 mismatch: {destination}")
    for split in ALLOWED_SPLITS:
        manifest_path = plan["dataset_root"] / split / "manifest.jsonl"
        if _sha256(manifest_path) != split_manifests[split]["sha256"]:
            raise ValueError(f"materialized {split} manifest SHA256 mismatch")
    _write_exact(output / REPORT_FILENAME, _json_bytes(report), resume=resume)
    return report


__all__ = [
    "ALLOWED_SPLITS",
    "DEFAULT_CONFIG",
    "EXPECTED_TASK_COUNT",
    "EXPECTED_TASKS_PER_MAP",
    "EXPECTED_TASKS_PER_SPLIT",
    "IDENTITY_FILENAME",
    "MATERIALIZATION_SCHEMA",
    "REPORT_FILENAME",
    "load_registered_compact_flow_context",
    "materialize_compact_flow_tasks",
]
