"""Outcome-blind corrected-native reselection from the complete task pool.

The legacy v6 schedule remains the source of the 36 experimental *slots*:
each slot fixes a conflict/load cell, source quota, controller order, and
schedule group.  Fresh corrected-native reset rows may replace the task,
map, or seed occupying a slot, but may not change those registered design
constraints.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    NATIVE_SEMANTICS_SCHEMA,
    PRODUCER_IDENTITY_SCHEMA,
    config_producer_fingerprint,
    producer_identity,
    sha256_file,
    validate_producer_identity,
)
from experiments.repair_collection import _dataset_fingerprint


SPLIT = "balanced_wall_clock"
CONTROLLERS = ("official_adaptive", "v2-full", "mixed-full-v2")
SCHEDULE_SCHEMA = (
    "lns2.controller_execution_schedule.corrected_native_full_pool.v3"
)
REPORT_SCHEMA = "lns2.corrected_native_full_pool_selection.v3"
SELECTOR_SCHEMA = "lns2.corrected_native_full_pool_min_cost_flow.v3"
SELECTION_SALT = "corrected-native-full-pool-selection-v3"
SELECTOR_SOURCE_FILES = (
    "experiments/corrected_native_full_pool.py",
    "experiments/_common.py",
    "experiments/balanced_wall_clock.py",
    "experiments/repair_collection.py",
)
EVIDENCE_SCHEMA = "lns2.corrected_native_full_pool_evidence.v1"
EVIDENCE_MANIFEST_PATH = "selection_evidence/manifest.json"
SAFETY_MAX_DECISIONS = 100_000
PRODUCER_REQUIRED_PACKAGES = ("numpy",)
PRODUCER_OPTIONAL_PACKAGES = ("joblib", "scikit-learn")
SOURCE_V6_REGISTRATION_SHA256 = (
    "2e966a54b642a2966292a325135e51a0f11e44240d299326bba39bd4bcd87e90"
)
ZERO_COST = (0, 0, 0, Fraction(0), 0)
Cost = tuple[int, int, int, Fraction, int]


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid JSONL artifact: {path}") from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at {path}:{number}")
        try:
            value = json.loads(
                line,
                parse_constant=_reject_constant,
                object_pairs_hook=_object,
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL row at {path}:{number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selector_identity() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    implementation = producer_identity(
        project_root=project_root,
        source_files=SELECTOR_SOURCE_FILES,
        native_required=False,
    )
    return {
        "schema": SELECTOR_SCHEMA,
        "producer_identity": implementation,
        "producer_identity_fingerprint": _fingerprint(implementation),
        "salt": SELECTION_SALT,
        "salt_sha256": hashlib.sha256(
            SELECTION_SALT.encode("utf-8")
        ).hexdigest(),
    }


def _project_relative(project_root: Path, path: Path, *, role: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"selection evidence {role} is outside the project root"
        ) from error
    if not resolved.is_file():
        raise ValueError(f"selection evidence {role} is missing")
    return relative.as_posix()


def _evidence_reference(relative: str) -> str:
    return f"selection_evidence/project/{relative}"


def _seal_selection_evidence(
    output_root: Path,
    *,
    project_root: Path,
    artifacts: dict[str, Path],
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    evidence_root = output_root / "selection_evidence"
    project_copy = evidence_root / "project"
    if evidence_root.exists():
        raise ValueError("selection evidence output already exists")
    roles: dict[str, str] = {}
    files: dict[str, str] = {}
    sources_by_relative: dict[str, Path] = {}
    for role, source in sorted(artifacts.items()):
        if not isinstance(role, str) or not role or role in roles:
            raise ValueError("selection evidence has an invalid role")
        relative = _project_relative(
            project_root,
            Path(source),
            role=role,
        )
        digest = sha256_file(Path(source))
        previous = files.setdefault(relative, digest)
        if previous != digest:
            raise ValueError(
                f"selection evidence path has conflicting bytes: {relative}"
            )
        roles[role] = relative
        sources_by_relative.setdefault(relative, Path(source))
    if len(set(roles.values())) > len(files):
        raise AssertionError("selection evidence role accounting failed")
    for relative, source in sorted(sources_by_relative.items()):
        destination = project_copy / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source, partial)
        if sha256_file(partial) != files[relative]:
            raise ValueError(
                f"sealed selection evidence hash differs: {relative}"
            )
        partial.replace(destination)
    manifest = {
        "schema": EVIDENCE_SCHEMA,
        "selector_identity": _selector_identity(),
        "roles": dict(sorted(roles.items())),
        "files": dict(sorted(files.items())),
        "descriptor": descriptor,
    }
    manifest_path = evidence_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "schema": EVIDENCE_SCHEMA,
        "manifest": EVIDENCE_MANIFEST_PATH,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _validated_selection_evidence_bundle(
    schedule_root: Path,
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    registration = provenance.get("selection_evidence")
    if (
        not isinstance(registration, dict)
        or registration.get("schema") != EVIDENCE_SCHEMA
        or registration.get("manifest") != EVIDENCE_MANIFEST_PATH
    ):
        raise ValueError("corrected-native selection evidence is missing")
    expected_manifest_sha = _sha(
        registration.get("manifest_sha256"),
        field="selection_evidence.manifest_sha256",
    )
    evidence_root = (schedule_root / "selection_evidence").resolve()
    manifest_path = (schedule_root / EVIDENCE_MANIFEST_PATH).resolve()
    try:
        manifest_path.relative_to(schedule_root.resolve())
    except ValueError as error:
        raise ValueError("selection evidence manifest escapes schedule root") from error
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or sha256_file(manifest_path) != expected_manifest_sha
    ):
        raise ValueError("selection evidence manifest hash differs")
    manifest = _read_json(manifest_path)
    roles = manifest.get("roles")
    files = manifest.get("files")
    if (
        manifest.get("schema") != EVIDENCE_SCHEMA
        or manifest.get("selector_identity") != _selector_identity()
        or not isinstance(roles, dict)
        or not roles
        or not isinstance(files, dict)
        or not files
        or not isinstance(manifest.get("descriptor"), dict)
    ):
        raise ValueError("selection evidence manifest identity is invalid")
    project_copy = evidence_root / "project"
    observed_files: dict[str, str] = {}
    if not project_copy.is_dir() or project_copy.is_symlink():
        raise ValueError("selection evidence project snapshot is missing")
    for path in project_copy.rglob("*"):
        if path.is_symlink():
            raise ValueError("selection evidence must not contain symbolic links")
        if path.is_file():
            relative = path.relative_to(project_copy).as_posix()
            observed_files[relative] = sha256_file(path)
    normalized_files: dict[str, str] = {}
    for relative, digest in files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("selection evidence contains an invalid file path")
        normalized_files[relative] = _sha(
            digest,
            field=f"selection_evidence.files[{relative}]",
        )
    if observed_files != normalized_files:
        raise ValueError("selection evidence project file set or hash differs")
    resolved_roles: dict[str, Path] = {}
    for role, relative in roles.items():
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(relative, str)
            or relative not in normalized_files
        ):
            raise ValueError("selection evidence contains an invalid role")
        path = (project_copy / Path(relative)).resolve()
        try:
            path.relative_to(project_copy.resolve())
        except ValueError as error:
            raise ValueError("selection evidence role escapes snapshot") from error
        resolved_roles[role] = path
    return manifest, resolved_roles


def _sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is not a lowercase SHA-256")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{field} must be a finite number >= {minimum}")
    return float(value)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _resolve_reference(project_root: Path, value: Any, *, field: str) -> Path:
    text = _text(value, field=field)
    candidate = Path(text)
    path = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    if not path.is_file():
        raise ValueError(f"{field} does not exist: {path}")
    return path


def _registered_file(
    payload: dict[str, Any],
    project_root: Path,
    path_field: str,
    sha_field: str,
) -> Path:
    path = _resolve_reference(project_root, payload.get(path_field), field=path_field)
    expected = _sha(payload.get(sha_field), field=sha_field)
    if sha256_file(path) != expected:
        raise ValueError(f"{path_field} differs from its registered SHA")
    return path


def conflict_stratum(conflicts: int) -> str | None:
    if 1 <= conflicts <= 10:
        return "low"
    if 11 <= conflicts <= 100:
        return "medium"
    if 101 <= conflicts <= 500:
        return "high"
    return None


def load_stratum(generated: int) -> str:
    if generated <= 100_000:
        return "low"
    if generated <= 1_000_000:
        return "medium"
    return "high"


def _agent_band(count: int) -> str:
    return "small" if count <= 200 else "medium" if count <= 400 else "large"


def _validate_config(
    config_path: Path,
    dataset_manifest: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    config = _read_json(config_path)
    environment = config.get("environment")
    design = config.get("dataset_design")
    if (
        config.get("schema_version") != 1
        or config.get("formal") is not True
        or config.get("experiment_revision")
        != "balanced-wall-clock-corrected-native-pool-v1"
        or config.get("split") != SPLIT
        or config.get("solver_seeds") != [1, 2, 3]
        or config.get("policies") != ["official_adaptive", "realized_dynamic"]
        or not isinstance(environment, dict)
        or environment.get("max_repair_iterations") != 0
        or type(environment.get("max_repair_iterations")) is not int
        or _number(environment.get("time_limit"), field="environment.time_limit")
        != 600.0
        or config.get("max_decisions") != 0
        or type(config.get("max_decisions")) is not int
        or config.get("metric_iteration_budget") != 100
        or type(config.get("metric_iteration_budget")) is not int
        or _number(
            config.get("wall_time_budget_seconds"),
            field="wall_time_budget_seconds",
        )
        != 600.0
        or _number(
            config.get("episode_process_timeout_seconds"),
            field="episode_process_timeout_seconds",
        )
        != 660.0
        or not isinstance(design, dict)
        or design.get("mode") != SPLIT
        or design.get("dataset_revision")
        != "balanced-wall-clock-qualified-compute-load-pool-v6"
        or design.get("map_count") != 52
        or type(design.get("map_count")) is not int
        or design.get("instance_count") != 688
        or type(design.get("instance_count")) is not int
        or design.get("source_counts")
        != {"generated": 360, "movingai": 328}
    ):
        raise ValueError(
            "corrected full-pool config is not the registered uncapped "
            "0/0/600/600/660, metric-100 contract"
        )
    if (
        _sha(
            design.get("qualification_dataset_manifest_sha256"),
            field="qualification_dataset_manifest_sha256",
        )
        != sha256_file(dataset_manifest)
    ):
        raise ValueError("full-pool dataset manifest differs from corrected config")
    project_root = config_path.parent.parent
    registry_path = _registered_file(
        design,
        project_root,
        "source_pool_registration",
        "source_pool_registration_sha256",
    )
    registry = _read_json(registry_path)
    expected = registry.get("expected")
    sources = registry.get("sources")
    if (
        registry.get("schema_version") != 1
        or registry.get("role")
        != "qualification_only_merged_compute_load_candidate_pool"
        or registry.get("selection_blind_to_controller_outcomes") is not True
        or not isinstance(expected, dict)
        or expected.get("map_count") != 52
        or expected.get("task_count") != 688
        or expected.get("qualification_count") != 2064
        or expected.get("source_task_counts")
        != {"generated": 360, "movingai": 328}
        or expected.get("source_qualification_counts")
        != {"generated": 1080, "movingai": 984}
        or not isinstance(sources, list)
        or sum(
            _integer(item.get("task_count"), field="source.task_count")
            for item in sources
            if isinstance(item, dict)
        )
        != 688
        or sum(
            _integer(
                item.get("qualification_count"),
                field="source.qualification_count",
            )
            for item in sources
            if isinstance(item, dict)
        )
        != 2064
        or any(not isinstance(item, dict) for item in sources)
    ):
        raise ValueError("corrected full-pool source registry is inconsistent")
    return config, registry_path, registry


def _validate_dataset(
    dataset_root: Path,
    manifest_path: Path,
    registry: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    rows = _validate_dataset_snapshot(
        manifest_path,
        dataset_root / "dataset_summary.json",
        registry,
    )
    return rows, _dataset_fingerprint(dataset_root)


def _validate_dataset_snapshot(
    manifest_path: Path,
    summary_path: Path,
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = _read_jsonl(manifest_path)
    if len(rows) != 688:
        raise ValueError("full-pool dataset must contain exactly 688 tasks")
    tasks: set[str] = set()
    maps: dict[str, tuple[str, str]] = {}
    source_counts: collections.Counter[str] = collections.Counter()
    layout_counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        task_id = _text(row.get("task_id"), field="dataset.task_id")
        map_id = _text(row.get("map_id"), field=f"{task_id}.map_id")
        source = _text(row.get("source_group"), field=f"{task_id}.source_group")
        layout = _text(row.get("layout_mode"), field=f"{task_id}.layout_mode")
        if task_id in tasks:
            raise ValueError(f"full-pool dataset repeats task: {task_id}")
        if source not in {"generated", "movingai"}:
            raise ValueError(f"unsupported source group for {task_id}")
        _integer(row.get("agent_count"), field=f"{task_id}.agent_count", minimum=1)
        if row.get("split") != SPLIT:
            raise ValueError(f"dataset task crosses split: {task_id}")
        map_file = _text(row.get("map_file"), field=f"{task_id}.map_file")
        previous = maps.setdefault(map_id, (layout, map_file))
        if previous != (layout, map_file):
            raise ValueError(f"map metadata differs across tasks: {map_id}")
        tasks.add(task_id)
        source_counts[source] += 1
        layout_counts[layout] += 1
    if (
        len(maps) != 52
        or dict(source_counts) != {"generated": 360, "movingai": 328}
    ):
        raise ValueError("full-pool dataset aggregates differ from registration")
    summary = _read_json(summary_path)
    if (
        summary.get("schema_version") != 1
        or summary.get("dataset_revision")
        != "balanced-wall-clock-qualified-compute-load-pool-v6"
        or summary.get("selection_blind_to_controller_outcomes") is not True
        or summary.get("configuration_fingerprint") != _fingerprint(registry)
    ):
        raise ValueError("full-pool dataset summary differs from registry")
    registered_layouts = (
        # The config carries task counts by layout, not map counts.
        collections.Counter(
            {
                str(name): _integer(value, field=f"layout_counts.{name}")
                for name, value in dict(
                    summary.get("splits", {})
                    .get(SPLIT, {})
                    .get("layout_counts", dict(layout_counts))
                ).items()
            }
        )
    )
    if registered_layouts and registered_layouts != layout_counts:
        raise ValueError("full-pool layout aggregates differ")
    return rows


def _native_identity(producer: Any, *, label: str) -> dict[str, str]:
    try:
        identity = validate_producer_identity(
            producer,
            native_required=True,
            package_names=PRODUCER_REQUIRED_PACKAGES,
            optional_package_names=PRODUCER_OPTIONAL_PACKAGES,
        )
    except ValueError as error:
        raise ValueError(f"{label} producer identity is invalid") from error
    native = identity["native"]
    assert isinstance(native, dict)
    return {
        "sha256": str(native["sha256"]),
        "native_semantics_schema": str(native["native_semantics_schema"]),
        "repair_timing_schema": str(native["repair_timing_schema"]),
    }


def _validated_producer_fields(
    config: dict[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, str], str]:
    if "controller_implementation" in config:
        raise ValueError(
            f"{label} uses legacy controller_implementation; "
            f"{PRODUCER_IDENTITY_SCHEMA} is required"
        )
    producer = config.get("producer_identity")
    native = _native_identity(producer, label=label)
    try:
        fingerprint = config_producer_fingerprint(
            config,
            label=label,
            native_required=True,
            package_names=PRODUCER_REQUIRED_PACKAGES,
            optional_package_names=PRODUCER_OPTIONAL_PACKAGES,
        )
    except ValueError as error:
        raise ValueError(f"{label} producer fingerprint is invalid") from error
    assert isinstance(producer, dict)
    return dict(producer), native, fingerprint


def _validate_run_config(
    run_config: dict[str, Any],
    config: dict[str, Any],
    dataset_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    effective = run_config.get("configuration")
    if not isinstance(effective, dict):
        raise ValueError("qualification effective config is absent")
    if any(effective.get(key) != value for key, value in config.items()):
        raise ValueError("qualification effective config differs from corrected config")
    if (
        run_config.get("schema") != "lns2.closed_loop_confirmation.v1"
        or run_config.get("schema_version") != 1
        or run_config.get("formal") is not True
        or run_config.get("controller") != "v2-full"
        or effective.get("stopping_rule") != "wall-clock-fixed-metric"
        or effective.get("task_ids_override") is not None
        or effective.get("cohort_job_keys_override") is not None
        or run_config.get("dataset_fingerprint") != dataset_fingerprint
        or run_config.get("configuration_fingerprint") != _fingerprint(effective)
    ):
        raise ValueError("qualification run config identity is inconsistent")
    producer, native, producer_fingerprint = _validated_producer_fields(
        run_config,
        label="full-pool qualification",
    )
    run_payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "configuration_fingerprint": run_config["configuration_fingerprint"],
        "freeze_manifest": run_config.get("frozen_models"),
        "controller_bundle_manifest": run_config.get("controller_bundle"),
        "stall_shadow_config": run_config.get("stall_shadow_config"),
        "repair_aware_bundle_manifest": None,
        "critical_seed_config": run_config.get("critical_seed_config"),
        "cost_top3_config": run_config.get("cost_top3_config"),
        "v3_bundle_manifest": run_config.get("v3_bundle"),
        "v3_s3_bundle_manifest": run_config.get("v3_s3_bundle"),
        "producer_identity": producer,
        "producer_identity_fingerprint": producer_fingerprint,
    }
    if run_config.get("run_fingerprint") != _fingerprint(run_payload):
        raise ValueError("qualification run fingerprint is inconsistent")
    return producer, native


def _validate_original_design(
    registration_path: Path,
    source_schedule: Path,
    dataset_manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if sha256_file(registration_path) != SOURCE_V6_REGISTRATION_SHA256:
        raise ValueError("source registration is not the frozen v6 registration")
    registration = _read_json(registration_path)
    project_root = registration_path.parent.parent
    if (
        registration.get("schema")
        != "lns2.compute_load_balanced_wall_clock_registration.v1"
        or registration.get("status") != "preregistered_before_controller_collection"
        or registration.get("selection_blind_to_controller_outcomes") is not True
    ):
        raise ValueError("original v6 registration is not a blind preregistration")
    paths = {
        name: _registered_file(registration, project_root, name, f"{name}_sha256")
        for name in (
            "pool_registry",
            "selection_config",
            "merged_dataset_manifest",
            "merged_qualification_manifest",
            "cohort",
            "execution_schedule",
            "cohort_report",
            "formal_dataset_manifest",
        )
    }
    if paths["execution_schedule"] != source_schedule.resolve():
        raise ValueError("provided source schedule differs from v6 registration")
    if paths["merged_dataset_manifest"] != dataset_manifest.resolve():
        raise ValueError("provided full dataset differs from v6 registration")
    schedule = _read_json(source_schedule)
    entries = schedule.get("entries")
    cohort = _read_jsonl(paths["cohort"])
    report = _read_json(paths["cohort_report"])
    if (
        schedule.get("selection_blind_to_controller_outcomes") is not True
        or not isinstance(entries, list)
        or len(entries) != 36
        or len(cohort) != 36
        or report.get("passed") is not True
        or report.get("formal_collection_allowed") is not True
        or report.get("selected_count") != 36
        or report.get("qualification_count") != 2064
    ):
        raise ValueError("original v6 schedule/report is invalid")
    slots: list[dict[str, Any]] = []
    cohort_keys: set[tuple[str, int]] = set()
    task_ids: set[str] = set()
    map_ids: set[str] = set()
    groups: set[int] = set()
    orders: collections.Counter[tuple[str, ...]] = collections.Counter()
    cell_sources: collections.Counter[tuple[str, str, str]] = collections.Counter()
    for slot_index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError("original v6 schedule entry is not an object")
        row = dict(raw)
        task = _text(row.get("task_id"), field="source.task_id")
        map_id = _text(row.get("map_id"), field=f"{task}.map_id")
        seed = _integer(row.get("solver_seed"), field=f"{task}.solver_seed")
        conflicts = _integer(
            row.get("initial_conflicts"), field=f"{task}.initial_conflicts"
        )
        generated = _integer(
            row.get("initial_low_level_generated"),
            field=f"{task}.initial_low_level_generated",
        )
        source = _text(row.get("source_group"), field=f"{task}.source_group")
        conflict_cell = _text(
            row.get("conflict_stratum"), field=f"{task}.conflict_stratum"
        )
        load_cell = _text(
            row.get("initial_pp_load_stratum"),
            field=f"{task}.initial_pp_load_stratum",
        )
        group = _integer(row.get("schedule_group"), field=f"{task}.schedule_group")
        order = row.get("controller_order")
        if (
            conflict_stratum(conflicts) != conflict_cell
            or load_stratum(generated) != load_cell
            or source not in {"generated", "movingai"}
            or not isinstance(order, list)
            or len(order) != 3
            or set(order) != set(CONTROLLERS)
            or task in task_ids
            or map_id in map_ids
        ):
            raise ValueError("original v6 schedule violates its registered design")
        task_ids.add(task)
        map_ids.add(map_id)
        groups.add(group)
        orders[tuple(order)] += 1
        cohort_keys.add((task, seed))
        cell_sources[(conflict_cell, load_cell, source)] += 1
        row["_source_slot_index"] = slot_index
        slots.append(row)
    observed_cohort_keys = {
        (
            _text(row.get("task_id"), field="cohort.task_id"),
            _integer(row.get("solver_seed"), field="cohort.solver_seed"),
        )
        for row in cohort
    }
    if (
        groups != set(range(6))
        or any(
            sum(int(row["schedule_group"]) == group for row in slots) != 6
            for group in range(6)
        )
        or any(orders[order] != 6 for order in __import__("itertools").permutations(CONTROLLERS))
        or observed_cohort_keys != cohort_keys
        or len(observed_cohort_keys) != 36
        or sum(cell_sources.values()) != 36
        or any(
            sum(cell_sources[(conflict, load, source)] for source in ("generated", "movingai"))
            != 4
            for conflict in ("low", "medium", "high")
            for load in ("low", "medium", "high")
        )
        or sum(
            count
            for (conflict, load, source), count in cell_sources.items()
            if source == "generated"
        )
        != 18
    ):
        raise ValueError("original v6 schedule aggregates are inconsistent")
    registered_counts = registration.get("counts")
    if (
        not isinstance(registered_counts, dict)
        or registered_counts.get("jobs") != 36
        or registered_counts.get("tasks") != 36
        or registered_counts.get("maps") != 36
        or registered_counts.get("jobs_per_conflict_load_cell") != 4
        or registered_counts.get("source")
        != {"generated": 18, "movingai": 18}
    ):
        raise ValueError("original v6 registration counts are inconsistent")
    evidence: dict[str, Any] = {
        "source_registration_sha256": sha256_file(registration_path),
        **{
            f"source_{name}_sha256": sha256_file(path)
            for name, path in paths.items()
        },
        "_artifact_paths": {
            "source_registration": registration_path,
            **{
                f"source_{name}": path
                for name, path in paths.items()
            },
        },
    }
    return slots, registration, evidence


def _candidate(
    result: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    task_id = _text(result.get("task_id"), field="qualification.task_id")
    seed = _integer(result.get("solver_seed"), field=f"{task_id}.solver_seed")
    if (
        result.get("schema") != "lns2.repair_collection.v2"
        or result.get("schema_version") != 2
        or result.get("status") != "ok"
        or result.get("error") is not None
        or result.get("initial_complete") is not True
        or type(result.get("initial_feasible")) is not bool
        or type(result.get("repairable")) is not bool
    ):
        raise ValueError(f"invalid corrected-native reset: {(task_id, seed)}")
    for field in ("map_id", "layout_mode", "agent_count"):
        if result.get(field) != task.get(field):
            raise ValueError(f"qualification {field} differs for {(task_id, seed)}")
    if (
        result.get("split") != SPLIT
        or (
            "source_group" in result
            and result.get("source_group") != task.get("source_group")
        )
        or (
            "task_variant" in result
            and result.get("task_variant") != task.get("task_variant")
        )
    ):
        raise ValueError(f"qualification task identity differs for {(task_id, seed)}")
    conflicts = _integer(
        result.get("initial_conflicts"), field=f"{task_id}.initial_conflicts"
    )
    if (
        result["initial_feasible"] is not (conflicts == 0)
        or result["repairable"] is not (conflicts > 0)
    ):
        raise ValueError(f"qualification feasibility differs for {(task_id, seed)}")
    complexity = result.get("initial_complexity")
    if not isinstance(complexity, dict):
        raise ValueError(f"qualification complexity is absent for {(task_id, seed)}")
    if complexity.get("conflict_pair_count") != conflicts:
        raise ValueError(f"qualification conflict aggregate differs for {(task_id, seed)}")
    generated = _integer(
        complexity.get("initial_low_level_generated"),
        field=f"{task_id}.initial_low_level_generated",
    )
    expanded = _integer(
        complexity.get("initial_low_level_expanded"),
        field=f"{task_id}.initial_low_level_expanded",
    )
    events = _integer(
        complexity.get("conflict_event_count"),
        field=f"{task_id}.conflict_event_count",
    )
    path_cost = _integer(
        complexity.get("total_path_cost"), field=f"{task_id}.total_path_cost"
    )
    active = _number(
        complexity.get("active_conflict_agent_ratio"),
        field=f"{task_id}.active_conflict_agent_ratio",
    )
    largest = _number(
        complexity.get("largest_conflict_component_ratio"),
        field=f"{task_id}.largest_conflict_component_ratio",
    )
    if active > 1.0 or largest > 1.0:
        raise ValueError(f"qualification conflict ratio exceeds one for {task_id}")
    state = _sha(result.get("state_fingerprint"), field=f"{task_id}.state_fingerprint")
    forbidden = {
        "transitions",
        "baseline_transition",
        "final_conflicts",
        "time_to_feasible",
        "policy",
        "controller",
    }
    if forbidden & set(result):
        raise ValueError(f"qualification row contains controller outcomes: {task_id}")
    return {
        "task_id": task_id,
        "solver_seed": seed,
        "map_id": str(task["map_id"]),
        "layout_mode": str(task["layout_mode"]),
        "source_group": str(task["source_group"]),
        "agent_count": int(task["agent_count"]),
        "agent_band": _agent_band(int(task["agent_count"])),
        "initial_conflicts": conflicts,
        "conflict_stratum": conflict_stratum(conflicts),
        "initial_pp_load_stratum": load_stratum(generated),
        "initial_low_level_generated": generated,
        "initial_low_level_expanded": expanded,
        "conflict_event_count": events,
        "total_path_cost": path_cost,
        "active_conflict_agent_ratio": active,
        "largest_conflict_component_ratio": largest,
        "state_fingerprint": state,
        "_initial_feasible": result["initial_feasible"],
    }


def _validate_qualification(
    qualification_root: Path,
    tasks: dict[str, dict[str, Any]],
    config: dict[str, Any],
    dataset_fingerprint: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
]:
    manifest_path = qualification_root / "qualification_manifest.jsonl"
    report_path = qualification_root / "qualification_report.json"
    run_path = qualification_root / "run_config.json"
    if not all(path.is_file() for path in (manifest_path, report_path, run_path)):
        raise ValueError("full-pool qualification is incomplete")
    run_config = _read_json(run_path)
    producer, native = _validate_run_config(run_config, config, dataset_fingerprint)
    rows = _read_jsonl(manifest_path)
    expected_keys = {
        (task_id, seed) for task_id in tasks for seed in (1, 2, 3)
    }
    expected_count = len(expected_keys)
    if len(rows) != expected_count:
        raise ValueError(
            "qualification row count differs from its task-by-seed Cartesian product"
        )
    keys: set[tuple[str, int]] = set()
    candidates: list[dict[str, Any]] = []
    for result in rows:
        task_id = _text(result.get("task_id"), field="qualification.task_id")
        seed = _integer(result.get("solver_seed"), field=f"{task_id}.solver_seed")
        key = (task_id, seed)
        if key in keys or task_id not in tasks or seed not in (1, 2, 3):
            raise ValueError(f"unknown or repeated qualification key: {key}")
        keys.add(key)
        candidates.append(_candidate(result, tasks[task_id]))
    if keys != expected_keys:
        raise ValueError("full-pool qualification Cartesian coverage differs")
    report = _read_json(report_path)
    feasible = sum(row["_initial_feasible"] is True for row in candidates)
    repairable = {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in candidates
        if row["_initial_feasible"] is False and row["initial_conflicts"] > 0
    }
    reported_keys = report.get("repairable_episode_keys")
    if (
        report.get("schema") != "lns2.closed_loop_confirmation.v1"
        or report.get("schema_version") != 1
        or report.get("formal") is not True
        or report.get("passed") is not True
        or report.get("errors") != []
        or report.get("valid_count") != expected_count
        or type(report.get("valid_count")) is not int
        or report.get("expected_reset_count") != expected_count
        or type(report.get("expected_reset_count")) is not int
        or report.get("initial_feasible_count") != feasible
        or type(report.get("initial_feasible_count")) is not int
        or report.get("nonzero_state_count") != len(repairable)
        or type(report.get("nonzero_state_count")) is not int
        or report.get("incomplete_reset_count") != 0
        or report.get("inconsistent_initial_state_count") != 0
        or report.get("registered_solver_seeds") != [1, 2, 3]
        or not isinstance(report.get("gates"), dict)
        or any(value is not True for value in report["gates"].values())
        or not isinstance(reported_keys, list)
        or len(reported_keys) != len(repairable)
        or {
            (
                _text(value[0], field="repairable key task"),
                _integer(value[1], field="repairable key seed"),
            )
            for value in reported_keys
            if isinstance(value, list) and len(value) == 2
        }
        != repairable
    ):
        raise ValueError("full-pool qualification report aggregates differ")
    evidence: dict[str, Any] = {
        "qualification_manifest_sha256": sha256_file(manifest_path),
        "qualification_report_sha256": sha256_file(report_path),
        "qualification_run_config_sha256": sha256_file(run_path),
        "qualification_run_fingerprint": _sha(
            run_config.get("run_fingerprint"), field="qualification.run_fingerprint"
        ),
        "qualification_dataset_fingerprint": dataset_fingerprint,
        "qualification_producer_identity": producer,
        "qualification_producer_identity_fingerprint": str(
            run_config["producer_identity_fingerprint"]
        ),
        "_artifact_paths": {
            "qualification_manifest": manifest_path,
            "qualification_report": report_path,
            "qualification_run_config": run_path,
        },
    }
    return candidates, report, native, evidence


_IDENTITY_FILE_FIELDS = (
    "map_file",
    "scenario_file",
    "map_metadata_file",
    "task_file",
    "legacy_instance_file",
)


def _dataset_file_identities(
    dataset_root: Path,
    rows: Iterable[dict[str, Any]],
) -> set[tuple[str, str]]:
    split_root = dataset_root / SPLIT
    identities: set[tuple[str, str]] = set()
    for row in rows:
        task_id = str(row["task_id"])
        for field in _IDENTITY_FILE_FIELDS:
            raw = row.get(field)
            if raw is None:
                continue
            relative = Path(_text(raw, field=f"{task_id}.{field}"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{task_id}.{field} must be a contained path")
            path = (split_root / relative).resolve()
            try:
                path.relative_to(dataset_root)
            except ValueError as error:
                raise ValueError(f"{task_id}.{field} escapes its dataset") from error
            if not path.is_file():
                raise ValueError(f"dataset file is missing: {path}")
            identities.add((field, sha256_file(path)))
    return identities


def _validate_extension_source(
    raw_source: dict[str, Any],
    expected_native: dict[str, str],
) -> tuple[
    str,
    Path,
    list[dict[str, Any]],
    list[dict[str, Any]],
    set[tuple[str, str]],
    dict[str, Any],
]:
    source_id = _text(raw_source.get("id"), field="extension.id")
    if any(
        not (character.isascii() and (character.isalnum() or character in "._-"))
        for character in source_id
    ):
        raise ValueError("extension.id must be a path-safe ASCII identifier")
    dataset_root = Path(
        _text(raw_source.get("dataset"), field=f"{source_id}.dataset")
    ).resolve()
    qualification_root = Path(
        _text(
            raw_source.get("qualification"),
            field=f"{source_id}.qualification",
        )
    ).resolve()
    config_path = Path(
        _text(raw_source.get("config"), field=f"{source_id}.config")
    ).resolve()
    config = _read_json(config_path)
    design = config.get("dataset_design")
    environment = config.get("environment")
    if (
        config.get("schema_version") != 1
        or config.get("formal") is not True
        or not isinstance(config.get("experiment_revision"), str)
        or not config["experiment_revision"].startswith(
            "balanced-wall-clock-corrected-native-extension-"
        )
        or config.get("split") != SPLIT
        or config.get("solver_seeds") != [1, 2, 3]
        or config.get("policies") != ["official_adaptive", "realized_dynamic"]
        or not isinstance(environment, dict)
        or type(environment.get("max_repair_iterations")) is not int
        or environment.get("max_repair_iterations") != 0
        or _number(
            environment.get("time_limit"),
            field=f"{source_id}.environment.time_limit",
        )
        != 600.0
        or type(config.get("max_decisions")) is not int
        or config.get("max_decisions") != 0
        or type(config.get("metric_iteration_budget")) is not int
        or config.get("metric_iteration_budget") != 100
        or _number(
            config.get("wall_time_budget_seconds"),
            field=f"{source_id}.wall_time_budget_seconds",
        )
        != 600.0
        or _number(
            config.get("episode_process_timeout_seconds"),
            field=f"{source_id}.episode_process_timeout_seconds",
        )
        != 660.0
        or not isinstance(design, dict)
        or design.get("mode") != SPLIT
    ):
        raise ValueError(f"extension {source_id} has an invalid uncapped contract")
    task_count = _integer(
        design.get("instance_count"),
        field=f"{source_id}.dataset_design.instance_count",
        minimum=1,
    )
    map_count = _integer(
        design.get("map_count"),
        field=f"{source_id}.dataset_design.map_count",
        minimum=1,
    )
    source_counts = design.get("source_counts")
    layout_counts = design.get("layout_counts")
    if (
        not isinstance(source_counts, dict)
        or set(source_counts) - {"generated", "movingai"}
        or sum(
            _integer(value, field=f"{source_id}.source_counts.{name}")
            for name, value in source_counts.items()
        )
        != task_count
        or not isinstance(layout_counts, dict)
        or sum(
            _integer(value, field=f"{source_id}.layout_counts.{name}")
            for name, value in layout_counts.items()
        )
        != task_count
    ):
        raise ValueError(f"extension {source_id} dataset aggregates are invalid")
    manifest_path = dataset_root / SPLIT / "manifest.jsonl"
    if (
        _sha(
            design.get("qualification_dataset_manifest_sha256"),
            field=f"{source_id}.qualification_dataset_manifest_sha256",
        )
        != sha256_file(manifest_path)
    ):
        raise ValueError(f"extension {source_id} manifest SHA differs")
    source_config_path = _registered_file(
        design,
        config_path.parent.parent,
        "source_generator_config",
        "source_generator_config_sha256",
    )
    source_config = _read_json(source_config_path)
    rows = _read_jsonl(manifest_path)
    if len(rows) != task_count:
        raise ValueError(f"extension {source_id} task count differs")
    tasks: dict[str, dict[str, Any]] = {}
    maps: set[str] = set()
    observed_sources: collections.Counter[str] = collections.Counter()
    observed_layouts: collections.Counter[str] = collections.Counter()
    sole_source = next(iter(source_counts)) if len(source_counts) == 1 else None
    for raw_row in rows:
        row = dict(raw_row)
        task_id = _text(row.get("task_id"), field=f"{source_id}.task_id")
        map_id = _text(row.get("map_id"), field=f"{task_id}.map_id")
        source = row.get("source_group", sole_source)
        if source is None:
            raise ValueError(
                f"extension {source_id} task {task_id} lacks source_group"
            )
        if source not in source_counts:
            raise ValueError(f"extension {source_id} has an unknown source group")
        row["source_group"] = source
        if (
            task_id in tasks
            or row.get("split") != SPLIT
            or not isinstance(row.get("layout_mode"), str)
            or not row["layout_mode"]
        ):
            raise ValueError(f"extension {source_id} contains an invalid task")
        _integer(row.get("agent_count"), field=f"{task_id}.agent_count", minimum=1)
        tasks[task_id] = row
        maps.add(map_id)
        observed_sources[str(source)] += 1
        observed_layouts[str(row["layout_mode"])] += 1
    if (
        len(maps) != map_count
        or dict(observed_sources) != source_counts
        or dict(observed_layouts) != layout_counts
    ):
        raise ValueError(f"extension {source_id} manifest aggregates differ")
    summary = _read_json(dataset_root / "dataset_summary.json")
    split_summary = summary.get("splits", {}).get(SPLIT, {})
    if (
        summary.get("configuration_fingerprint") != _fingerprint(source_config)
        or not isinstance(split_summary, dict)
        or split_summary.get("map_count") != map_count
        or split_summary.get("instance_count") != task_count
    ):
        raise ValueError(f"extension {source_id} dataset summary differs")
    dataset_fingerprint = _dataset_fingerprint(dataset_root)
    candidates, _report, native, qualification_evidence = _validate_qualification(
        qualification_root,
        tasks,
        config,
        dataset_fingerprint,
    )
    if native != expected_native:
        raise ValueError(
            f"extension {source_id} uses a different native binary or schema"
        )
    identities = _dataset_file_identities(dataset_root, tasks.values())
    qualification_artifacts = dict(
        qualification_evidence.pop("_artifact_paths")
    )
    evidence = {
        "id": source_id,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_summary_sha256": sha256_file(
            dataset_root / "dataset_summary.json"
        ),
        "dataset_fingerprint": dataset_fingerprint,
        "config_sha256": sha256_file(config_path),
        "source_generator_config_sha256": sha256_file(source_config_path),
        "task_count": task_count,
        "map_count": map_count,
        "qualification_count": task_count * 3,
        "native": native,
        **qualification_evidence,
        "_artifact_paths": {
            "config": config_path,
            "source_generator_config": source_config_path,
            "dataset_manifest": manifest_path,
            "dataset_summary": dataset_root / "dataset_summary.json",
            **qualification_artifacts,
        },
    }
    return (
        source_id,
        dataset_root,
        list(tasks.values()),
        candidates,
        identities,
        evidence,
    )


def _validate_sealed_extension_source(
    *,
    source_id: str,
    config_path: Path,
    manifest_path: Path,
    summary_path: Path,
    qualification_root: Path,
    expected_native: dict[str, str],
    dataset_fingerprint: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
    dict[str, Any],
    Path,
]:
    """Reopen an extension snapshot without depending on external data files."""

    config = _read_json(config_path)
    design = config.get("dataset_design")
    environment = config.get("environment")
    if (
        config.get("schema_version") != 1
        or config.get("formal") is not True
        or not isinstance(config.get("experiment_revision"), str)
        or not config["experiment_revision"].startswith(
            "balanced-wall-clock-corrected-native-extension-"
        )
        or config.get("split") != SPLIT
        or config.get("solver_seeds") != [1, 2, 3]
        or config.get("policies")
        != ["official_adaptive", "realized_dynamic"]
        or not isinstance(environment, dict)
        or type(environment.get("max_repair_iterations")) is not int
        or environment["max_repair_iterations"] != 0
        or _number(
            environment.get("time_limit"),
            field=f"{source_id}.environment.time_limit",
        )
        != 600.0
        or type(config.get("max_decisions")) is not int
        or config["max_decisions"] != 0
        or type(config.get("metric_iteration_budget")) is not int
        or config["metric_iteration_budget"] != 100
        or _number(
            config.get("wall_time_budget_seconds"),
            field=f"{source_id}.wall_time_budget_seconds",
        )
        != 600.0
        or _number(
            config.get("episode_process_timeout_seconds"),
            field=f"{source_id}.episode_process_timeout_seconds",
        )
        != 660.0
        or not isinstance(design, dict)
        or design.get("mode") != SPLIT
    ):
        raise ValueError(
            f"sealed extension {source_id} has an invalid uncapped contract"
        )
    task_count = _integer(
        design.get("instance_count"),
        field=f"{source_id}.dataset_design.instance_count",
        minimum=1,
    )
    map_count = _integer(
        design.get("map_count"),
        field=f"{source_id}.dataset_design.map_count",
        minimum=1,
    )
    source_counts = design.get("source_counts")
    layout_counts = design.get("layout_counts")
    if (
        not isinstance(source_counts, dict)
        or set(source_counts) - {"generated", "movingai"}
        or sum(
            _integer(value, field=f"{source_id}.source_counts.{name}")
            for name, value in source_counts.items()
        )
        != task_count
        or not isinstance(layout_counts, dict)
        or sum(
            _integer(value, field=f"{source_id}.layout_counts.{name}")
            for name, value in layout_counts.items()
        )
        != task_count
        or _sha(
            design.get("qualification_dataset_manifest_sha256"),
            field=f"{source_id}.qualification_dataset_manifest_sha256",
        )
        != sha256_file(manifest_path)
    ):
        raise ValueError(
            f"sealed extension {source_id} dataset design is invalid"
        )
    source_config_path = _registered_file(
        design,
        config_path.parent.parent,
        "source_generator_config",
        "source_generator_config_sha256",
    )
    source_config = _read_json(source_config_path)
    rows = _read_jsonl(manifest_path)
    if len(rows) != task_count:
        raise ValueError(f"sealed extension {source_id} task count differs")
    tasks: dict[str, dict[str, Any]] = {}
    maps: set[str] = set()
    observed_sources: collections.Counter[str] = collections.Counter()
    observed_layouts: collections.Counter[str] = collections.Counter()
    sole_source = next(iter(source_counts)) if len(source_counts) == 1 else None
    for raw_row in rows:
        row = dict(raw_row)
        task_id = _text(row.get("task_id"), field=f"{source_id}.task_id")
        map_id = _text(row.get("map_id"), field=f"{task_id}.map_id")
        source = row.get("source_group", sole_source)
        if source is None or source not in source_counts:
            raise ValueError(
                f"sealed extension {source_id} has an invalid source group"
            )
        row["source_group"] = source
        if (
            task_id in tasks
            or row.get("split") != SPLIT
            or not isinstance(row.get("layout_mode"), str)
            or not row["layout_mode"]
        ):
            raise ValueError(
                f"sealed extension {source_id} contains an invalid task"
            )
        _integer(
            row.get("agent_count"),
            field=f"{task_id}.agent_count",
            minimum=1,
        )
        tasks[task_id] = row
        maps.add(map_id)
        observed_sources[str(source)] += 1
        observed_layouts[str(row["layout_mode"])] += 1
    if (
        len(maps) != map_count
        or dict(observed_sources) != source_counts
        or dict(observed_layouts) != layout_counts
    ):
        raise ValueError(
            f"sealed extension {source_id} manifest aggregates differ"
        )
    summary = _read_json(summary_path)
    split_summary = summary.get("splits", {}).get(SPLIT, {})
    if (
        summary.get("configuration_fingerprint")
        != _fingerprint(source_config)
        or not isinstance(split_summary, dict)
        or split_summary.get("map_count") != map_count
        or split_summary.get("instance_count") != task_count
    ):
        raise ValueError(
            f"sealed extension {source_id} dataset summary differs"
        )
    candidates, _report, native, qualification_evidence = (
        _validate_qualification(
            qualification_root,
            tasks,
            config,
            dataset_fingerprint,
        )
    )
    qualification_evidence.pop("_artifact_paths", None)
    if native != expected_native:
        raise ValueError(
            f"sealed extension {source_id} native identity differs"
        )
    return (
        list(tasks.values()),
        candidates,
        native,
        qualification_evidence,
        source_config_path,
    )


def _assert_extension_disjoint(
    extension_id: str,
    *,
    existing_task_ids: set[str],
    existing_map_ids: set[str],
    existing_file_identities: set[tuple[str, str]],
    extension_task_ids: set[str],
    extension_map_ids: set[str],
    extension_file_identities: set[tuple[str, str]],
) -> None:
    if existing_task_ids & extension_task_ids:
        raise ValueError(
            f"extension {extension_id} repeats an existing task identity"
        )
    if existing_map_ids & extension_map_ids:
        raise ValueError(
            f"extension {extension_id} repeats an existing map identity"
        )
    if existing_file_identities & extension_file_identities:
        raise ValueError(
            f"extension {extension_id} repeats an existing file identity"
        )


def _add_cost(left: Cost, right: Cost) -> Cost:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def _negate_cost(value: Cost) -> Cost:
    return tuple(-item for item in value)  # type: ignore[return-value]


@dataclass
class _Edge:
    target: int
    reverse: int
    capacity: int
    cost: Cost
    candidate: dict[str, Any] | None = None
    slot_index: int | None = None
    forward: bool = True


def _add_edge(
    graph: list[list[_Edge]],
    source: int,
    target: int,
    capacity: int,
    cost: Cost,
    *,
    candidate: dict[str, Any] | None = None,
    slot_index: int | None = None,
) -> _Edge:
    forward = _Edge(
        target,
        len(graph[target]),
        capacity,
        cost,
        candidate,
        slot_index,
        True,
    )
    reverse = _Edge(
        source,
        len(graph[source]),
        0,
        _negate_cost(cost),
        None,
        None,
        False,
    )
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _edge_choice(
    candidate: dict[str, Any],
    slot: dict[str, Any],
) -> tuple[Cost, str, Fraction]:
    drift = (
        Fraction(
            abs(int(candidate["initial_conflicts"]) - int(slot["initial_conflicts"])),
            max(int(slot["initial_conflicts"]), 1),
        )
        + Fraction(
            abs(
                int(candidate["initial_low_level_generated"])
                - int(slot["initial_low_level_generated"])
            ),
            max(int(slot["initial_low_level_generated"]), 1),
        )
    )
    tie = _fingerprint(
        [
            SELECTION_SALT,
            int(slot["schedule_group"]),
            slot["task_id"],
            slot["map_id"],
            int(slot["solver_seed"]),
            candidate["task_id"],
            candidate["map_id"],
            int(candidate["solver_seed"]),
            candidate["state_fingerprint"],
        ]
    )
    cost: Cost = (
        -int(candidate["task_id"] == slot["task_id"]),
        -int(candidate["map_id"] == slot["map_id"]),
        -int(int(candidate["solver_seed"]) == int(slot["solver_seed"])),
        drift,
        int(tie, 16),
    )
    return cost, tie, drift


def _min_cost_assignment(
    slots: list[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str, Fraction]], Cost] | None:
    """Fill every registered slot with at most one task from each map."""

    eligible = [
        dict(row)
        for row in candidates
        if row.get("_initial_feasible") is False
        and row.get("conflict_stratum") in {"low", "medium", "high"}
    ]
    map_ids = sorted({str(row["map_id"]) for row in eligible})
    by_map: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in eligible:
        by_map[str(row["map_id"])].append(row)
    source = 0
    map_offset = 1
    slot_offset = map_offset + len(map_ids)
    sink = slot_offset + len(slots)
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    selection_edges: list[_Edge] = []
    for map_position, map_id in enumerate(map_ids):
        map_node = map_offset + map_position
        _add_edge(graph, source, map_node, 1, ZERO_COST)
        for slot_index, slot in enumerate(slots):
            choices = [
                row
                for row in by_map[map_id]
                if row["source_group"] == slot["source_group"]
                and row["conflict_stratum"] == slot["conflict_stratum"]
                and row["initial_pp_load_stratum"]
                == slot["initial_pp_load_stratum"]
            ]
            if not choices:
                continue
            ranked = [
                (*_edge_choice(row, slot), str(row["task_id"]), int(row["solver_seed"]), row)
                for row in choices
            ]
            cost, _tie, _drift, _task, _seed, best = min(ranked)
            selection_edges.append(
                _add_edge(
                    graph,
                    map_node,
                    slot_offset + slot_index,
                    1,
                    cost,
                    candidate=best,
                    slot_index=slot_index,
                )
            )
    for slot_index in range(len(slots)):
        _add_edge(graph, slot_offset + slot_index, sink, 1, ZERO_COST)

    total_cost = ZERO_COST
    flow = 0
    while flow < len(slots):
        distance: list[Cost | None] = [None] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = ZERO_COST
        for _iteration in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distance[node] is None:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    proposed = _add_cost(distance[node], edge.cost)
                    if distance[edge.target] is None or proposed < distance[edge.target]:
                        distance[edge.target] = proposed
                        previous[edge.target] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if distance[sink] is None:
            return None
        node = sink
        while node != source:
            link = previous[node]
            if link is None:
                raise AssertionError("min-cost predecessor chain is incomplete")
            parent, edge_index = link
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
        total_cost = _add_cost(total_cost, distance[sink])
        flow += 1

    assignments = []
    for edge in selection_edges:
        if edge.capacity != 0:
            continue
        assert edge.candidate is not None and edge.slot_index is not None
        slot = slots[edge.slot_index]
        _cost, tie, drift = _edge_choice(edge.candidate, slot)
        assignments.append((slot, edge.candidate, tie, drift))
    if len(assignments) != len(slots):
        raise AssertionError("min-cost flow assignment extraction failed")
    assignments.sort(key=lambda value: int(value[0].get("_source_slot_index", 0)))
    return assignments, total_cost


def _selection_products(
    slots: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    selected = _min_cost_assignment(slots, candidates)
    if selected is None:
        return None
    assignments, total_cost = selected
    entries: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for slot, raw_candidate, tie, drift in assignments:
        candidate = {
            key: value
            for key, value in raw_candidate.items()
            if not key.startswith("_")
        }
        candidate["schedule_group"] = int(slot["schedule_group"])
        candidate["controller_order"] = list(slot["controller_order"])
        entries.append(candidate)
        audit_rows.append(
            {
                "schedule_group": int(slot["schedule_group"]),
                "source_task_id": str(slot["task_id"]),
                "selected_task_id": str(candidate["task_id"]),
                "source_map_id": str(slot["map_id"]),
                "selected_map_id": str(candidate["map_id"]),
                "source_solver_seed": int(slot["solver_seed"]),
                "selected_solver_seed": int(candidate["solver_seed"]),
                "original_task_retained": (
                    candidate["task_id"] == slot["task_id"]
                ),
                "original_map_retained": (
                    candidate["map_id"] == slot["map_id"]
                ),
                "original_seed_retained": (
                    int(candidate["solver_seed"])
                    == int(slot["solver_seed"])
                ),
                "relative_covariate_drift": float(drift),
                "relative_covariate_drift_exact": {
                    "numerator": drift.numerator,
                    "denominator": drift.denominator,
                },
                "salted_tie_sha256": tie,
            }
        )
    cell_source_counts = collections.Counter(
        (
            str(row["conflict_stratum"]),
            str(row["initial_pp_load_stratum"]),
            str(row["source_group"]),
        )
        for row in entries
    )
    required = collections.Counter(
        (
            str(row["conflict_stratum"]),
            str(row["initial_pp_load_stratum"]),
            str(row["source_group"]),
        )
        for row in slots
    )
    gates = {
        "full_cartesian_qualification_valid": True,
        "registered_slots_satisfied": len(entries) == 36,
        "registered_cell_source_quotas_preserved": (
            cell_source_counts == required
        ),
        "one_task_per_map": (
            len({row["task_id"] for row in entries}) == 36
            and len({row["map_id"] for row in entries}) == 36
        ),
        "exact_source_balance": collections.Counter(
            row["source_group"] for row in entries
        )
        == {"generated": 18, "movingai": 18},
        "controller_order_and_groups_preserved": all(
            row["schedule_group"] == slot["schedule_group"]
            and row["controller_order"] == slot["controller_order"]
            for row, slot in zip(entries, slots)
        ),
    }
    if not all(gates.values()):
        raise AssertionError(
            "full-pool selector violated a registered constraint"
        )
    return {
        "entries": entries,
        "audit_rows": audit_rows,
        "total_cost": total_cost,
        "cell_source_counts": cell_source_counts,
        "gates": gates,
    }


def _bound_evidence_artifact(
    manifest: dict[str, Any],
    resolved_roles: dict[str, Path],
    payload: dict[str, Any],
    *,
    role: str,
    path_field: str,
    sha_field: str,
) -> Path:
    relative = dict(manifest["roles"]).get(role)
    if (
        not isinstance(relative, str)
        or role not in resolved_roles
        or payload.get(path_field) != _evidence_reference(relative)
        or payload.get(sha_field)
        != dict(manifest["files"]).get(relative)
    ):
        raise ValueError(
            f"selection evidence binding differs for {role}"
        )
    return resolved_roles[role]


def _validate_sealed_selection_replay(
    schedule_root: Path,
    provenance: dict[str, Any],
    entries: list[dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, Any]:
    manifest, roles = _validated_selection_evidence_bundle(
        schedule_root,
        provenance,
    )
    required_roles: set[str] = set()

    def bound(
        payload: dict[str, Any],
        role: str,
        path_field: str,
        sha_field: str,
    ) -> Path:
        required_roles.add(role)
        return _bound_evidence_artifact(
            manifest,
            roles,
            payload,
            role=role,
            path_field=path_field,
            sha_field=sha_field,
        )

    source_registration = bound(
        provenance,
        "source_registration",
        "source_registration_evidence_path",
        "source_registration_sha256",
    )
    source_names = (
        "pool_registry",
        "selection_config",
        "merged_dataset_manifest",
        "merged_qualification_manifest",
        "cohort",
        "execution_schedule",
        "cohort_report",
        "formal_dataset_manifest",
    )
    source_paths = {
        name: bound(
            provenance,
            f"source_{name}",
            f"source_{name}_evidence_path",
            f"source_{name}_sha256",
        )
        for name in source_names
    }
    base_manifest = bound(
        provenance,
        "base_dataset_manifest",
        "dataset_manifest_evidence_path",
        "dataset_manifest_sha256",
    )
    slots, _registration, source_replay = _validate_original_design(
        source_registration,
        source_paths["execution_schedule"],
        base_manifest,
    )
    source_replay.pop("_artifact_paths", None)
    for field, value in source_replay.items():
        if provenance.get(field) != value:
            raise ValueError(
                f"sealed original-design evidence differs: {field}"
            )

    base_config_path = bound(
        provenance,
        "base_config",
        "config_evidence_path",
        "config_sha256",
    )
    base_registry_path = bound(
        provenance,
        "base_pool_registry",
        "pool_registry_evidence_path",
        "pool_registry_sha256",
    )
    base_summary_path = bound(
        provenance,
        "base_dataset_summary",
        "dataset_summary_evidence_path",
        "dataset_summary_sha256",
    )
    base_qualification_manifest = bound(
        provenance,
        "base_qualification_manifest",
        "qualification_manifest_evidence_path",
        "qualification_manifest_sha256",
    )
    base_qualification_report = bound(
        provenance,
        "base_qualification_report",
        "qualification_report_evidence_path",
        "qualification_report_sha256",
    )
    base_qualification_run = bound(
        provenance,
        "base_qualification_run_config",
        "qualification_run_config_evidence_path",
        "qualification_run_config_sha256",
    )
    config, registry_path, registry = _validate_config(
        base_config_path,
        base_manifest,
    )
    if registry_path != base_registry_path:
        raise ValueError("sealed base registry path differs from config")
    base_rows = _validate_dataset_snapshot(
        base_manifest,
        base_summary_path,
        registry,
    )
    base_tasks = {
        str(row["task_id"]): dict(row) for row in base_rows
    }
    base_dataset_fingerprint = _sha(
        provenance.get("dataset_fingerprint"),
        field="dataset_fingerprint",
    )
    base_qualification_root = base_qualification_manifest.parent
    if (
        base_qualification_report.parent != base_qualification_root
        or base_qualification_run.parent != base_qualification_root
        or base_qualification_manifest.name != "qualification_manifest.jsonl"
        or base_qualification_report.name != "qualification_report.json"
        or base_qualification_run.name != "run_config.json"
    ):
        raise ValueError("sealed base qualification layout is invalid")
    (
        candidates,
        _base_report,
        native,
        base_qualification_evidence,
    ) = _validate_qualification(
        base_qualification_root,
        base_tasks,
        config,
        base_dataset_fingerprint,
    )
    base_qualification_evidence.pop("_artifact_paths", None)
    for field, value in base_qualification_evidence.items():
        if provenance.get(field) != value:
            raise ValueError(
                f"sealed base qualification evidence differs: {field}"
            )

    all_tasks = dict(base_tasks)
    all_map_ids = {str(row["map_id"]) for row in base_rows}
    extension_payloads = provenance.get("extensions")
    if (
        not isinstance(extension_payloads, list)
        or type(provenance.get("extension_count")) is not int
        or provenance["extension_count"] != len(extension_payloads)
    ):
        raise ValueError("sealed extension evidence list is invalid")
    extension_ids: set[str] = set()
    extension_qualification_count = 0
    producer_rows = [
        {
            "id": "base-full-pool",
            "fingerprint": base_qualification_evidence[
                "qualification_producer_identity_fingerprint"
            ],
            "identity": base_qualification_evidence[
                "qualification_producer_identity"
            ],
            "native": native,
        }
    ]
    for extension in extension_payloads:
        if not isinstance(extension, dict):
            raise ValueError("sealed extension evidence is not an object")
        extension_id = _text(extension.get("id"), field="extension.id")
        if extension_id in extension_ids:
            raise ValueError("sealed extension evidence repeats an id")
        extension_ids.add(extension_id)
        role_prefix = f"extension_{extension_id}_"
        config_path = bound(
            extension,
            role_prefix + "config",
            "config_evidence_path",
            "config_sha256",
        )
        source_config_path = bound(
            extension,
            role_prefix + "source_generator_config",
            "source_generator_config_evidence_path",
            "source_generator_config_sha256",
        )
        dataset_manifest = bound(
            extension,
            role_prefix + "dataset_manifest",
            "dataset_manifest_evidence_path",
            "dataset_manifest_sha256",
        )
        dataset_summary = bound(
            extension,
            role_prefix + "dataset_summary",
            "dataset_summary_evidence_path",
            "dataset_summary_sha256",
        )
        qualification_manifest = bound(
            extension,
            role_prefix + "qualification_manifest",
            "qualification_manifest_evidence_path",
            "qualification_manifest_sha256",
        )
        qualification_report = bound(
            extension,
            role_prefix + "qualification_report",
            "qualification_report_evidence_path",
            "qualification_report_sha256",
        )
        qualification_run = bound(
            extension,
            role_prefix + "qualification_run_config",
            "qualification_run_config_evidence_path",
            "qualification_run_config_sha256",
        )
        qualification_root = qualification_manifest.parent
        if (
            qualification_report.parent != qualification_root
            or qualification_run.parent != qualification_root
            or qualification_manifest.name != "qualification_manifest.jsonl"
            or qualification_report.name != "qualification_report.json"
            or qualification_run.name != "run_config.json"
        ):
            raise ValueError(
                f"sealed extension {extension_id} qualification layout is invalid"
            )
        extension_dataset_fingerprint = _sha(
            extension.get("dataset_fingerprint"),
            field=f"{extension_id}.dataset_fingerprint",
        )
        (
            extension_rows,
            extension_candidates,
            extension_native,
            extension_qualification_evidence,
            replayed_source_config,
        ) = _validate_sealed_extension_source(
            source_id=extension_id,
            config_path=config_path,
            manifest_path=dataset_manifest,
            summary_path=dataset_summary,
            qualification_root=qualification_root,
            expected_native=native,
            dataset_fingerprint=extension_dataset_fingerprint,
        )
        if replayed_source_config != source_config_path:
            raise ValueError(
                f"sealed extension {extension_id} source config path differs"
            )
        for field, value in extension_qualification_evidence.items():
            if extension.get(field) != value:
                raise ValueError(
                    f"sealed extension {extension_id} qualification "
                    f"evidence differs: {field}"
                )
        if (
            extension_native != native
            or extension.get("native") != native
            or extension.get("task_count") != len(extension_rows)
            or type(extension.get("task_count")) is not int
            or extension.get("map_count")
            != len({str(row["map_id"]) for row in extension_rows})
            or type(extension.get("map_count")) is not int
            or extension.get("qualification_count")
            != len(extension_candidates)
            or type(extension.get("qualification_count")) is not int
        ):
            raise ValueError(
                f"sealed extension {extension_id} aggregates differ"
            )
        extension_task_ids = {
            str(row["task_id"]) for row in extension_rows
        }
        extension_map_ids = {
            str(row["map_id"]) for row in extension_rows
        }
        if (
            set(all_tasks) & extension_task_ids
            or all_map_ids & extension_map_ids
        ):
            raise ValueError(
                f"sealed extension {extension_id} repeats task or map identity"
            )
        all_tasks.update(
            {str(row["task_id"]): row for row in extension_rows}
        )
        all_map_ids.update(extension_map_ids)
        candidates.extend(extension_candidates)
        extension_qualification_count += len(extension_candidates)
        producer_rows.append(
            {
                "id": extension_id,
                "fingerprint": extension_qualification_evidence[
                    "qualification_producer_identity_fingerprint"
                ],
                "identity": extension_qualification_evidence[
                    "qualification_producer_identity"
                ],
                "native": extension_native,
            }
        )

    if set(manifest["roles"]) != required_roles:
        raise ValueError("selection evidence contains missing or extra roles")
    descriptor = manifest.get("descriptor")
    expected_descriptor = {
        "base_dataset_fingerprint": base_dataset_fingerprint,
        "base_task_count": len(base_rows),
        "base_qualification_count": len(base_rows) * 3,
        "extension_ids": sorted(extension_ids),
        "merged_task_count": len(all_tasks),
        "merged_qualification_count": len(candidates),
    }
    if descriptor != expected_descriptor:
        raise ValueError("selection evidence descriptor differs")
    if (
        provenance.get("merged_task_count") != len(all_tasks)
        or provenance.get("merged_map_count") != len(all_map_ids)
        or provenance.get("merged_qualification_count") != len(candidates)
        or provenance.get("qualification_producer_identities")
        != producer_rows
        or provenance.get("qualification_producer_identity")
        != producer_rows[0]["identity"]
        or provenance.get("qualification_producer_identity_fingerprint")
        != producer_rows[0]["fingerprint"]
        or any(row["identity"] != producer_rows[0]["identity"] for row in producer_rows)
    ):
        raise ValueError("sealed merged producer or coverage evidence differs")
    products = _selection_products(slots, candidates)
    if products is None:
        raise ValueError("sealed candidate pool cannot fill registered slots")
    if entries != products["entries"]:
        raise ValueError(
            "schedule entries differ from sealed reset-row replay"
        )
    total_cost = products["total_cost"]
    drift = total_cost[3]
    cell_source_counts = products["cell_source_counts"]
    expected_counts = {
        "cell_source": {
            f"{conflict}__{load}__{source}": cell_source_counts[
                (conflict, load, source)
            ]
            for conflict in ("low", "medium", "high")
            for load in ("low", "medium", "high")
            for source in ("generated", "movingai")
        },
        "source": dict(
            sorted(
                collections.Counter(
                    row["source_group"] for row in products["entries"]
                ).items()
            )
        ),
        "tasks": len({row["task_id"] for row in products["entries"]}),
        "maps": len({row["map_id"] for row in products["entries"]}),
    }
    expected_selection = {
        "schema": SELECTOR_SCHEMA,
        "objective": [
            "maximize_original_task_retention",
            "maximize_original_map_retention",
            "maximize_original_seed_retention",
            "minimize_total_relative_conflict_and_pp_load_drift",
            f"minimize_additive_fixed_salt_sha256:{SELECTION_SALT}",
        ],
        "constraints": {
            "jobs_per_conflict_load_cell": 4,
            "cell_source_quotas": "exactly_preserve_v6",
            "one_task_per_map": True,
            "map_count": 36,
            "source_counts": {"generated": 18, "movingai": 18},
        },
        "controller_outcomes_used": False,
    }
    if provenance.get("selection") != expected_selection:
        raise ValueError("sealed selection objective differs")
    expected_report_fields = {
        "qualification_count": len(candidates),
        "base_qualification_count": len(base_rows) * 3,
        "extension_qualification_count": extension_qualification_count,
        "selected_count": 36,
        "original_task_retained_count": -total_cost[0],
        "original_map_retained_count": -total_cost[1],
        "original_seed_retained_count": -total_cost[2],
        "total_relative_covariate_drift": float(drift),
        "total_relative_covariate_drift_exact": {
            "numerator": drift.numerator,
            "denominator": drift.denominator,
        },
        "salted_sha256_additive_cost": str(total_cost[4]),
        "slot_assignments": products["audit_rows"],
        "counts": expected_counts,
        "gates": products["gates"],
    }
    for field, value in expected_report_fields.items():
        if report.get(field) != value:
            raise ValueError(
                f"selection report differs from sealed replay: {field}"
            )
    return {
        "slots": slots,
        "candidates": candidates,
        "products": products,
        "native": native,
        "base_rows": base_rows,
        "extension_ids": sorted(extension_ids),
    }


def select_corrected_native_full_pool_schedule(
    *,
    dataset: str | Path,
    qualification: str | Path,
    source_registration: str | Path,
    source_schedule_root: str | Path,
    config: str | Path,
    output: str | Path,
    extensions: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Validate all evidence and freeze a corrected-native 36-slot schedule."""

    output_root = Path(output).resolve()
    if output_root.is_dir() and any(output_root.iterdir()):
        raise ValueError("full-pool selection output must be a fresh directory")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("full-pool selection output is not a directory")
    dataset_root = Path(dataset).resolve()
    manifest_path = dataset_root / SPLIT / "manifest.jsonl"
    config_path = Path(config).resolve()
    config_value, pool_registry_path, pool_registry = _validate_config(
        config_path, manifest_path
    )
    dataset_rows, dataset_fingerprint = _validate_dataset(
        dataset_root, manifest_path, pool_registry
    )
    tasks = {str(row["task_id"]): dict(row) for row in dataset_rows}
    schedule_value = Path(source_schedule_root).resolve()
    source_schedule = (
        schedule_value / "execution_schedule.json"
        if schedule_value.is_dir()
        else schedule_value
    )
    registration_path = Path(source_registration).resolve()
    slots, _registration, source_evidence = _validate_original_design(
        registration_path, source_schedule, manifest_path
    )
    source_artifacts = dict(source_evidence.pop("_artifact_paths"))
    candidates, _qualification_report, native, qualification_evidence = (
        _validate_qualification(
            Path(qualification).resolve(),
            tasks,
            config_value,
            dataset_fingerprint,
        )
    )
    qualification_artifacts = dict(
        qualification_evidence.pop("_artifact_paths")
    )
    project_root = Path(__file__).resolve().parents[1]
    evidence_artifacts: dict[str, Path] = {
        **source_artifacts,
        "base_config": config_path,
        "base_pool_registry": pool_registry_path,
        "base_dataset_manifest": manifest_path,
        "base_dataset_summary": dataset_root / "dataset_summary.json",
        **{
            f"base_{role}": path
            for role, path in qualification_artifacts.items()
        },
    }
    for role, path in source_artifacts.items():
        source_evidence[f"{role}_evidence_path"] = _evidence_reference(
            _project_relative(project_root, path, role=role)
        )
    qualification_evidence.update(
        {
            f"{role}_evidence_path": _evidence_reference(
                _project_relative(project_root, path, role=f"base_{role}")
            )
            for role, path in qualification_artifacts.items()
        }
    )
    base_path_fields = {
        "config_evidence_path": config_path,
        "pool_registry_evidence_path": pool_registry_path,
        "dataset_manifest_evidence_path": manifest_path,
        "dataset_summary_evidence_path": dataset_root / "dataset_summary.json",
    }
    base_evidence_paths = {
        field: _evidence_reference(
            _project_relative(project_root, path, role=field)
        )
        for field, path in base_path_fields.items()
    }
    extension_evidence: list[dict[str, Any]] = []
    all_task_ids = set(tasks)
    all_map_ids = {str(row["map_id"]) for row in dataset_rows}
    all_file_identities = _dataset_file_identities(dataset_root, dataset_rows)
    extension_ids: set[str] = set()
    for raw_extension in extensions:
        if not isinstance(raw_extension, dict):
            raise ValueError("extension source specification is not an object")
        (
            extension_id,
            _extension_root,
            extension_rows,
            extension_candidates,
            extension_file_identities,
            evidence,
        ) = _validate_extension_source(raw_extension, native)
        extension_artifacts = dict(evidence.pop("_artifact_paths"))
        for role, path in extension_artifacts.items():
            evidence_artifacts[f"extension_{extension_id}_{role}"] = path
            evidence[f"{role}_evidence_path"] = _evidence_reference(
                _project_relative(
                    project_root,
                    path,
                    role=f"extension_{extension_id}_{role}",
                )
            )
        extension_task_ids = {str(row["task_id"]) for row in extension_rows}
        extension_map_ids = {str(row["map_id"]) for row in extension_rows}
        if extension_id in extension_ids:
            raise ValueError(f"duplicate extension id: {extension_id}")
        _assert_extension_disjoint(
            extension_id,
            existing_task_ids=all_task_ids,
            existing_map_ids=all_map_ids,
            existing_file_identities=all_file_identities,
            extension_task_ids=extension_task_ids,
            extension_map_ids=extension_map_ids,
            extension_file_identities=extension_file_identities,
        )
        extension_ids.add(extension_id)
        all_task_ids.update(extension_task_ids)
        all_map_ids.update(extension_map_ids)
        all_file_identities.update(extension_file_identities)
        tasks.update({str(row["task_id"]): row for row in extension_rows})
        candidates.extend(extension_candidates)
        extension_evidence.append(evidence)
    qualification_count = len(candidates)
    qualification_producers = [
        {
            "id": "base-full-pool",
            "fingerprint": qualification_evidence[
                "qualification_producer_identity_fingerprint"
            ],
            "identity": qualification_evidence[
                "qualification_producer_identity"
            ],
            "native": native,
        },
        *[
            {
                "id": value["id"],
                "fingerprint": value[
                    "qualification_producer_identity_fingerprint"
                ],
                "identity": value["qualification_producer_identity"],
                "native": value["native"],
            }
            for value in extension_evidence
        ],
    ]
    producer_fingerprints = {
        str(value["fingerprint"]) for value in qualification_producers
    }
    if len(producer_fingerprints) != 1:
        raise ValueError(
            "base and extension qualifications use different producer identities"
        )
    common_producer_fingerprint = next(iter(producer_fingerprints))
    common_producer_identity = qualification_producers[0]["identity"]
    if any(
        value["identity"] != common_producer_identity
        for value in qualification_producers
    ):
        raise ValueError(
            "base and extension qualifications have mismatched producer evidence"
        )
    provenance = {
        "schema": "lns2.corrected_native_full_pool_provenance.v3",
        "config_sha256": sha256_file(config_path),
        "pool_registry_sha256": sha256_file(pool_registry_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_summary_sha256": sha256_file(
            dataset_root / "dataset_summary.json"
        ),
        "dataset_fingerprint": dataset_fingerprint,
        **base_evidence_paths,
        **source_evidence,
        **qualification_evidence,
        "extensions": extension_evidence,
        "extension_count": len(extension_evidence),
        "merged_task_count": len(tasks),
        "merged_map_count": len(all_map_ids),
        "merged_qualification_count": qualification_count,
        "native": native,
        "native_semantics_schema": native["native_semantics_schema"],
        "native_identity_consistent": True,
        "qualification_producer_identity": common_producer_identity,
        "qualification_producer_identity_fingerprint": (
            common_producer_fingerprint
        ),
        "qualification_producer_identities": qualification_producers,
        "selector_identity": _selector_identity(),
        "protocol_amendment": {
            "schema": "lns2.corrected_native_full_pool_amendment.v3",
            "reason": (
                "the frozen-task corrected-native seed pool could not satisfy "
                "the registered cell/source quotas"
            ),
            "controller_outcomes_used": False,
            "input_evidence_sha256": {
                "config": sha256_file(config_path),
                "dataset_manifest": sha256_file(manifest_path),
                "qualification_manifest": qualification_evidence[
                    "qualification_manifest_sha256"
                ],
                "qualification_report": qualification_evidence[
                    "qualification_report_sha256"
                ],
                "qualification_run_config": qualification_evidence[
                    "qualification_run_config_sha256"
                ],
                "source_registration": source_evidence[
                    "source_registration_sha256"
                ],
                "source_schedule": source_evidence[
                    "source_execution_schedule_sha256"
                ],
            },
            "extension_evidence_sha256": [
                {
                    "id": value["id"],
                    "config": value["config_sha256"],
                    "dataset_manifest": value["dataset_manifest_sha256"],
                    "qualification_manifest": value[
                        "qualification_manifest_sha256"
                    ],
                    "qualification_report": value[
                        "qualification_report_sha256"
                    ],
                    "qualification_run_config": value[
                        "qualification_run_config_sha256"
                    ],
                }
                for value in extension_evidence
            ],
        },
        "stopping_contract": {
            "stopping_rule": "wall-clock-fixed-metric",
            "max_decisions": 0,
            "max_repair_iterations": 0,
            "metric_iteration_budget": 100,
            "wall_time_budget_seconds": 600.0,
            "environment_time_limit_seconds": 600.0,
            "episode_process_timeout_seconds": 660.0,
            "safety_max_decisions": SAFETY_MAX_DECISIONS,
            "safety_limit_is_not_metric_cap": True,
        },
    }
    products = _selection_products(slots, candidates)
    if products is None:
        required_counts = collections.Counter(
            (
                str(row["conflict_stratum"]),
                str(row["initial_pp_load_stratum"]),
                str(row["source_group"]),
            )
            for row in slots
        )
        eligible_map_counts = {
            f"{conflict}__{load}__{source}": len(
                {
                    str(row["map_id"])
                    for row in candidates
                    if row["_initial_feasible"] is False
                    and row["conflict_stratum"] == conflict
                    and row["initial_pp_load_stratum"] == load
                    and row["source_group"] == source
                }
            )
            for conflict in ("low", "medium", "high")
            for load in ("low", "medium", "high")
            for source in ("generated", "movingai")
        }
        report = {
            "schema": REPORT_SCHEMA,
            "selection_blind_to_controller_outcomes": True,
            "selection_uses_initial_reset_metrics_only": True,
            "qualification_count": qualification_count,
            "base_qualification_count": len(dataset_rows) * 3,
            "extension_qualification_count": (
                qualification_count - len(dataset_rows) * 3
            ),
            "selected_count": 0,
            "passed": False,
            "formal_collection_allowed": False,
            "decision": "corrected_native_full_pool_constraints_infeasible",
            "execution_schedule_sha256": None,
            "required_cell_source_counts": {
                f"{conflict}__{load}__{source}": required_counts[
                    (conflict, load, source)
                ]
                for conflict in ("low", "medium", "high")
                for load in ("low", "medium", "high")
                for source in ("generated", "movingai")
            },
            "eligible_distinct_map_counts_by_cell_source": eligible_map_counts,
            "locally_insufficient_cell_sources": sorted(
                name
                for name, count in eligible_map_counts.items()
                if count
                < required_counts[
                    tuple(name.split("__"))  # type: ignore[index]
                ]
            ),
            "gates": {
                "full_cartesian_qualification_valid": True,
                "registered_slots_satisfied": False,
                "one_task_per_map": False,
                "exact_source_balance": False,
            },
            "provenance": provenance,
        }
        _write_json(output_root / "cohort_report.json", report)
        return report

    provenance["selection_evidence"] = _seal_selection_evidence(
        output_root,
        project_root=project_root,
        artifacts=evidence_artifacts,
        descriptor={
            "base_dataset_fingerprint": dataset_fingerprint,
            "base_task_count": len(dataset_rows),
            "base_qualification_count": len(dataset_rows) * 3,
            "extension_ids": sorted(extension_ids),
            "merged_task_count": len(tasks),
            "merged_qualification_count": qualification_count,
        },
    )

    entries = products["entries"]
    audit_rows = products["audit_rows"]
    total_cost = products["total_cost"]
    cell_source_counts = products["cell_source_counts"]
    gates = products["gates"]
    provenance["selection"] = {
        "schema": SELECTOR_SCHEMA,
        "objective": [
            "maximize_original_task_retention",
            "maximize_original_map_retention",
            "maximize_original_seed_retention",
            "minimize_total_relative_conflict_and_pp_load_drift",
            f"minimize_additive_fixed_salt_sha256:{SELECTION_SALT}",
        ],
        "constraints": {
            "jobs_per_conflict_load_cell": 4,
            "cell_source_quotas": "exactly_preserve_v6",
            "one_task_per_map": True,
            "map_count": 36,
            "source_counts": {"generated": 18, "movingai": 18},
        },
        "controller_outcomes_used": False,
    }
    schedule = {
        "schema": SCHEDULE_SCHEMA,
        "selection_blind_to_controller_outcomes": True,
        "selection_uses_initial_reset_metrics_only": True,
        "provenance": provenance,
        "entries": entries,
    }
    schedule_path = output_root / "execution_schedule.json"
    _write_json(schedule_path, schedule)
    drift = total_cost[3]
    report = {
        "schema": REPORT_SCHEMA,
        "selection_blind_to_controller_outcomes": True,
        "selection_uses_initial_reset_metrics_only": True,
        "qualification_count": qualification_count,
        "base_qualification_count": len(dataset_rows) * 3,
        "extension_qualification_count": (
            qualification_count - len(dataset_rows) * 3
        ),
        "selected_count": 36,
        "original_task_retained_count": -total_cost[0],
        "original_map_retained_count": -total_cost[1],
        "original_seed_retained_count": -total_cost[2],
        "total_relative_covariate_drift": float(drift),
        "total_relative_covariate_drift_exact": {
            "numerator": drift.numerator,
            "denominator": drift.denominator,
        },
        "salted_sha256_additive_cost": str(total_cost[4]),
        "slot_assignments": audit_rows,
        "counts": {
            "cell_source": {
                f"{conflict}__{load}__{source}": cell_source_counts[
                    (conflict, load, source)
                ]
                for conflict in ("low", "medium", "high")
                for load in ("low", "medium", "high")
                for source in ("generated", "movingai")
            },
            "source": dict(
                sorted(collections.Counter(row["source_group"] for row in entries).items())
            ),
            "tasks": len({row["task_id"] for row in entries}),
            "maps": len({row["map_id"] for row in entries}),
        },
        "gates": gates,
        "passed": True,
        "formal_collection_allowed": True,
        "decision": "eligible_for_corrected_native_formal_collection",
        "execution_schedule_sha256": sha256_file(schedule_path),
        "provenance": provenance,
    }
    _write_json(output_root / "cohort_report.json", report)
    return report


def _validate_schedule_entry(row: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"schedule entry {index} is not an object")
    value = dict(row)
    task_id = _text(value.get("task_id"), field=f"entries[{index}].task_id")
    _text(value.get("map_id"), field=f"{task_id}.map_id")
    _text(value.get("layout_mode"), field=f"{task_id}.layout_mode")
    source = _text(value.get("source_group"), field=f"{task_id}.source_group")
    if source not in {"generated", "movingai"}:
        raise ValueError(f"{task_id} has an invalid source group")
    seed = _integer(value.get("solver_seed"), field=f"{task_id}.solver_seed")
    agents = _integer(
        value.get("agent_count"), field=f"{task_id}.agent_count", minimum=1
    )
    if value.get("agent_band") != _agent_band(agents):
        raise ValueError(f"{task_id} has an inconsistent agent band")
    conflicts = _integer(
        value.get("initial_conflicts"), field=f"{task_id}.initial_conflicts"
    )
    generated = _integer(
        value.get("initial_low_level_generated"),
        field=f"{task_id}.initial_low_level_generated",
    )
    for field in (
        "initial_low_level_expanded",
        "conflict_event_count",
        "total_path_cost",
    ):
        _integer(value.get(field), field=f"{task_id}.{field}")
    for field in (
        "active_conflict_agent_ratio",
        "largest_conflict_component_ratio",
    ):
        ratio = _number(value.get(field), field=f"{task_id}.{field}")
        if ratio > 1.0:
            raise ValueError(f"{task_id}.{field} exceeds one")
    if (
        conflict_stratum(conflicts) is None
        or value.get("conflict_stratum") != conflict_stratum(conflicts)
        or value.get("initial_pp_load_stratum") != load_stratum(generated)
    ):
        raise ValueError(f"{task_id} has inconsistent difficulty strata")
    _sha(value.get("state_fingerprint"), field=f"{task_id}.state_fingerprint")
    group = _integer(
        value.get("schedule_group"), field=f"{task_id}.schedule_group"
    )
    if group >= 6:
        raise ValueError(f"{task_id} has an invalid schedule group")
    order = value.get("controller_order")
    if (
        not isinstance(order, list)
        or len(order) != len(CONTROLLERS)
        or set(order) != set(CONTROLLERS)
    ):
        raise ValueError(f"{task_id} has an invalid controller order")
    return value


def validate_corrected_native_full_pool_schedule(
    schedule_root: str | Path,
    *,
    dataset: str | Path | None = None,
) -> dict[str, Any]:
    """Strictly validate a successful full-pool schedule/report pair."""

    root = Path(schedule_root).resolve()
    schedule_path = root / "execution_schedule.json"
    report_path = root / "cohort_report.json"
    if not schedule_path.is_file() or not report_path.is_file():
        raise ValueError("corrected-native schedule/report pair is incomplete")
    schedule = _read_json(schedule_path)
    report = _read_json(report_path)
    provenance = schedule.get("provenance")
    entries_value = schedule.get("entries")
    if (
        schedule.get("schema") != SCHEDULE_SCHEMA
        or schedule.get("selection_blind_to_controller_outcomes") is not True
        or schedule.get("selection_uses_initial_reset_metrics_only") is not True
        or not isinstance(provenance, dict)
        or provenance.get("schema")
        != "lns2.corrected_native_full_pool_provenance.v3"
        or not isinstance(entries_value, list)
        or len(entries_value) != 36
        or report.get("schema") != REPORT_SCHEMA
        or report.get("passed") is not True
        or report.get("formal_collection_allowed") is not True
        or report.get("selected_count") != 36
        or type(report.get("selected_count")) is not int
        or report.get("execution_schedule_sha256") != sha256_file(schedule_path)
        or report.get("provenance") != provenance
    ):
        raise ValueError("corrected-native schedule/report identity is invalid")
    gates = report.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or any(type(value) is not bool or value is not True for value in gates.values())
    ):
        raise ValueError("corrected-native schedule formal gates are invalid")
    producer_identity = provenance.get("qualification_producer_identity")
    native_value = _native_identity(
        producer_identity,
        label="corrected-native schedule",
    )
    producer_fingerprint = _sha(
        provenance.get("qualification_producer_identity_fingerprint"),
        field="qualification_producer_identity_fingerprint",
    )
    try:
        validated_fingerprint = config_producer_fingerprint(
            {
                "producer_identity": producer_identity,
                "producer_identity_fingerprint": producer_fingerprint,
            },
            label="corrected-native schedule",
            native_required=True,
            package_names=PRODUCER_REQUIRED_PACKAGES,
            optional_package_names=PRODUCER_OPTIONAL_PACKAGES,
        )
    except ValueError as error:
        raise ValueError(
            "corrected-native schedule producer fingerprint is inconsistent"
        ) from error
    if validated_fingerprint != producer_fingerprint:
        raise ValueError(
            "corrected-native schedule producer fingerprint is inconsistent"
        )
    native = provenance.get("native")
    if native != native_value:
        raise ValueError(
            "corrected-native schedule native summary differs from producer identity"
        )
    producer_rows = provenance.get("qualification_producer_identities")
    if (
        provenance.get("native_semantics_schema") != NATIVE_SEMANTICS_SCHEMA
        or native_value["native_semantics_schema"] != NATIVE_SEMANTICS_SCHEMA
        or not isinstance(producer_rows, list)
        or not producer_rows
        or any(
            not isinstance(value, dict)
            or not isinstance(value.get("id"), str)
            or not value["id"]
            or not isinstance(value.get("fingerprint"), str)
            or _sha(
                value["fingerprint"],
                field="qualification producer fingerprint",
            )
            != value["fingerprint"]
            or value.get("identity") != producer_identity
            or _fingerprint(value["identity"]) != value["fingerprint"]
            or value.get("native") != native_value
            for value in producer_rows
        )
    ):
        raise ValueError("corrected-native schedule producer aliases are invalid")
    if (
        len({str(value["fingerprint"]) for value in producer_rows}) != 1
        or producer_fingerprint != producer_rows[0]["fingerprint"]
    ):
        raise ValueError("qualification producer identities are not identical")
    selector = provenance.get("selector_identity")
    if selector != _selector_identity():
        raise ValueError("corrected-native selector identity differs")
    stopping = provenance.get("stopping_contract")
    if stopping != {
        "stopping_rule": "wall-clock-fixed-metric",
        "max_decisions": 0,
        "max_repair_iterations": 0,
        "metric_iteration_budget": 100,
        "wall_time_budget_seconds": 600.0,
        "environment_time_limit_seconds": 600.0,
        "episode_process_timeout_seconds": 660.0,
        "safety_max_decisions": SAFETY_MAX_DECISIONS,
        "safety_limit_is_not_metric_cap": True,
    }:
        raise ValueError("corrected-native schedule stopping contract differs")
    entries = [
        _validate_schedule_entry(row, index=index)
        for index, row in enumerate(entries_value)
    ]
    tasks = [str(row["task_id"]) for row in entries]
    maps = [str(row["map_id"]) for row in entries]
    states = [str(row["state_fingerprint"]) for row in entries]
    groups = collections.Counter(int(row["schedule_group"]) for row in entries)
    orders = collections.Counter(tuple(row["controller_order"]) for row in entries)
    cells = collections.Counter(
        (
            str(row["conflict_stratum"]),
            str(row["initial_pp_load_stratum"]),
            str(row["source_group"]),
        )
        for row in entries
    )
    source_counts = collections.Counter(str(row["source_group"]) for row in entries)
    report_counts = report.get("counts")
    expected_cell_report = {
        f"{conflict}__{load}__{source}": cells[(conflict, load, source)]
        for conflict in ("low", "medium", "high")
        for load in ("low", "medium", "high")
        for source in ("generated", "movingai")
    }
    if (
        len(set(tasks)) != 36
        or len(set(maps)) != 36
        or len(set(states)) != 36
        or groups != {group: 6 for group in range(6)}
        or any(
            orders[order] != 6
            for order in __import__("itertools").permutations(CONTROLLERS)
        )
        or any(
            sum(cells[(conflict, load, source)] for source in ("generated", "movingai"))
            != 4
            for conflict in ("low", "medium", "high")
            for load in ("low", "medium", "high")
        )
        or source_counts != {"generated": 18, "movingai": 18}
        or not isinstance(report_counts, dict)
        or report_counts.get("cell_source") != expected_cell_report
        or report_counts.get("source") != dict(sorted(source_counts.items()))
        or report_counts.get("tasks") != 36
        or report_counts.get("maps") != 36
    ):
        raise ValueError("corrected-native schedule aggregates are invalid")
    assignments = report.get("slot_assignments")
    if not isinstance(assignments, list) or len(assignments) != 36:
        raise ValueError("corrected-native slot assignments are incomplete")
    by_group_and_task = {
        (int(row["schedule_group"]), str(row["selected_task_id"])): row
        for row in assignments
        if isinstance(row, dict)
        and type(row.get("schedule_group")) is int
        and isinstance(row.get("selected_task_id"), str)
    }
    if len(by_group_and_task) != 36:
        raise ValueError("corrected-native slot assignments repeat identities")
    for entry in entries:
        key = (int(entry["schedule_group"]), str(entry["task_id"]))
        assignment = by_group_and_task.get(key)
        if (
            assignment is None
            or assignment.get("selected_map_id") != entry["map_id"]
            or assignment.get("selected_solver_seed") != entry["solver_seed"]
            or type(assignment.get("original_task_retained")) is not bool
            or type(assignment.get("original_map_retained")) is not bool
            or type(assignment.get("original_seed_retained")) is not bool
            or not isinstance(assignment.get("relative_covariate_drift_exact"), dict)
            or _integer(
                assignment["relative_covariate_drift_exact"].get("numerator"),
                field="relative drift numerator",
            )
            < 0
            or _integer(
                assignment["relative_covariate_drift_exact"].get("denominator"),
                field="relative drift denominator",
                minimum=1,
            )
            < 1
            or _sha(
                assignment.get("salted_tie_sha256"),
                field="slot salted tie SHA",
            )
            != assignment["salted_tie_sha256"]
        ):
            raise ValueError("corrected-native slot assignment differs from schedule")
    selection_replay = _validate_sealed_selection_replay(
        root,
        provenance,
        entries,
        report,
    )
    formal_fields = (
        "formal_dataset_fingerprint",
        "formal_dataset_manifest_sha256",
        "formal_dataset_summary_sha256",
        "formal_materialization_report_sha256",
    )
    present_formal_fields = {
        field for field in formal_fields if field in provenance
    }
    formal_dataset: Path | None = None
    if present_formal_fields:
        if present_formal_fields != set(formal_fields):
            raise ValueError("formal dataset provenance is incomplete")
        formal_dataset = (
            Path(dataset).resolve()
            if dataset is not None
            else (root / "dataset").resolve()
        )
        manifest_path = formal_dataset / SPLIT / "manifest.jsonl"
        summary_path = formal_dataset / "dataset_summary.json"
        materialization_path = root / "materialization_report.json"
        if (
            not manifest_path.is_file()
            or not summary_path.is_file()
            or not materialization_path.is_file()
            or provenance.get("formal_dataset_manifest_sha256")
            != sha256_file(manifest_path)
            or provenance.get("formal_dataset_summary_sha256")
            != sha256_file(summary_path)
            or provenance.get("formal_dataset_fingerprint")
            != _dataset_fingerprint(formal_dataset)
            or provenance.get("formal_materialization_report_sha256")
            != sha256_file(materialization_path)
        ):
            raise ValueError("formal dataset hashes differ from schedule provenance")
        materialization = _read_json(materialization_path)
        if (
            materialization.get("schema")
            != "lns2.corrected_native_dataset_materialization.v3"
            or materialization.get("passed") is not True
            or materialization.get("manifest_sha256")
            != provenance["formal_dataset_manifest_sha256"]
            or materialization.get("dataset_summary_sha256")
            != provenance["formal_dataset_summary_sha256"]
            or materialization.get("dataset_fingerprint")
            != provenance["formal_dataset_fingerprint"]
            or materialization.get("execution_schedule_sha256")
            != provenance.get("selection_execution_schedule_sha256")
            or materialization.get("cohort_report_sha256")
            != provenance.get("selection_cohort_report_sha256")
            or materialization.get(
                "selection_evidence_manifest_sha256"
            )
            != provenance.get("selection_evidence", {}).get(
                "manifest_sha256"
            )
            or materialization.get("producer")
            != {
                "schema": "lns2.corrected_native_dataset_materializer.v3",
                "identity": _selector_identity(),
            }
        ):
            raise ValueError("formal materialization report differs from schedule")
        manifest_rows = _read_jsonl(manifest_path)
        manifest_by_task: dict[str, dict[str, Any]] = {}
        for raw_row in manifest_rows:
            task_id = _text(
                raw_row.get("task_id"),
                field="formal manifest task_id",
            )
            if task_id in manifest_by_task:
                raise ValueError("formal dataset repeats a scheduled task")
            manifest_by_task[task_id] = raw_row
        if set(manifest_by_task) != set(tasks):
            raise ValueError(
                "formal dataset task coverage differs from the schedule"
            )
        for entry in entries:
            task_id = str(entry["task_id"])
            row = manifest_by_task[task_id]
            for field in (
                "map_id",
                "layout_mode",
                "source_group",
                "agent_count",
            ):
                if row.get(field) != entry.get(field):
                    raise ValueError(
                        "formal dataset metadata differs from the schedule: "
                        f"{task_id}.{field}"
                    )
        from experiments.closed_loop_confirmation import closed_loop_dataset_design

        design = closed_loop_dataset_design(
            manifest_rows,
            SPLIT,
            {
                "mode": SPLIT,
                "map_count": 36,
                "instance_count": 36,
                "source_counts": {"generated": 18, "movingai": 18},
                "layout_counts": dict(
                    sorted(
                        collections.Counter(
                            str(row["layout_mode"]) for row in manifest_rows
                        ).items()
                    )
                ),
            },
        )
        if design.get("passed") is not True:
            raise ValueError("formal dataset fails closed-loop dataset design")
    return {
        "schema": SCHEDULE_SCHEMA,
        "schedule": schedule,
        "report": report,
        "entries": entries,
        "provenance": provenance,
        "execution_schedule_sha256": sha256_file(schedule_path),
        "cohort_report_sha256": sha256_file(report_path),
        "formal_dataset": str(formal_dataset) if formal_dataset is not None else None,
        "formal_dataset_fingerprint": (
            provenance.get("formal_dataset_fingerprint")
            if formal_dataset is not None
            else None
        ),
        "passed": True,
        "selection_replay": {
            "slot_count": len(selection_replay["slots"]),
            "candidate_count": len(selection_replay["candidates"]),
            "extension_ids": selection_replay["extension_ids"],
            "passed": True,
        },
    }


def _materialization_source_rows(
    dataset_root: Path,
    *,
    default_source: str | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    manifest = dataset_root / SPLIT / "manifest.jsonl"
    rows = _read_jsonl(manifest)
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if "source_group" not in row and default_source is not None:
            row["source_group"] = default_source
        _text(row.get("task_id"), field="materialization.task_id")
        _text(row.get("map_id"), field=f"{row['task_id']}.map_id")
        _text(row.get("source_group"), field=f"{row['task_id']}.source_group")
        normalized.append(row)
    task_ids = [str(row["task_id"]) for row in normalized]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("materialization source repeats task identities")
    return normalized, sha256_file(manifest), _dataset_fingerprint(dataset_root)


def materialize_corrected_native_selected_dataset(
    *,
    schedule_root: str | Path,
    base_dataset: str | Path,
    output: str | Path,
    extensions: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Atomically materialize the exact 36 tasks in a validated schedule."""

    selection_root = Path(schedule_root).resolve()
    validated = validate_corrected_native_full_pool_schedule(selection_root)
    provenance = validated["provenance"]
    output_root = Path(output).resolve()
    if output_root.exists():
        raise ValueError("materialized dataset output must not already exist")
    base_root = Path(base_dataset).resolve()
    base_rows, base_manifest_sha, base_fingerprint = _materialization_source_rows(
        base_root
    )
    if (
        base_manifest_sha != provenance.get("dataset_manifest_sha256")
        or base_fingerprint != provenance.get("dataset_fingerprint")
    ):
        raise ValueError("base dataset differs from schedule provenance")
    source_sets: list[tuple[str, Path, list[dict[str, Any]]]] = [
        ("base-full-pool", base_root, base_rows)
    ]
    seen_task_ids = {str(row["task_id"]) for row in base_rows}
    seen_map_ids = {str(row["map_id"]) for row in base_rows}
    seen_file_identities = _dataset_file_identities(base_root, base_rows)
    registered_extensions = provenance.get("extensions")
    if not isinstance(registered_extensions, list):
        raise ValueError("schedule extension provenance is invalid")
    registered_by_id = {
        str(value.get("id")): value
        for value in registered_extensions
        if isinstance(value, dict) and isinstance(value.get("id"), str)
    }
    supplied_extensions = list(extensions)
    supplied_ids = [
        _text(value.get("id"), field="extension.id")
        for value in supplied_extensions
        if isinstance(value, dict)
    ]
    if (
        len(supplied_ids) != len(supplied_extensions)
        or len(set(supplied_ids)) != len(supplied_ids)
        or set(supplied_ids) != set(registered_by_id)
    ):
        raise ValueError("materialization extension set differs from schedule")
    native = provenance.get("native")
    if not isinstance(native, dict):
        raise ValueError("schedule native provenance is absent")
    for spec in supplied_extensions:
        (
            extension_id,
            dataset_root,
            rows,
            _candidates,
            identities,
            evidence,
        ) = (
            _validate_extension_source(spec, dict(native))
        )
        registered = registered_by_id[extension_id]
        for field in (
            "dataset_manifest_sha256",
            "dataset_summary_sha256",
            "dataset_fingerprint",
            "config_sha256",
            "source_generator_config_sha256",
            "qualification_manifest_sha256",
            "qualification_report_sha256",
            "qualification_run_config_sha256",
            "qualification_run_fingerprint",
            "qualification_producer_identity",
            "qualification_producer_identity_fingerprint",
        ):
            if evidence.get(field) != registered.get(field):
                raise ValueError(
                    f"extension {extension_id} {field} differs from schedule"
                )
        extension_task_ids = {str(row["task_id"]) for row in rows}
        extension_map_ids = {str(row["map_id"]) for row in rows}
        _assert_extension_disjoint(
            extension_id,
            existing_task_ids=seen_task_ids,
            existing_map_ids=seen_map_ids,
            existing_file_identities=seen_file_identities,
            extension_task_ids=extension_task_ids,
            extension_map_ids=extension_map_ids,
            extension_file_identities=identities,
        )
        seen_task_ids.update(extension_task_ids)
        seen_map_ids.update(extension_map_ids)
        seen_file_identities.update(identities)
        source_sets.append((extension_id, dataset_root, rows))

    indexed: dict[str, list[tuple[str, Path, dict[str, Any]]]] = (
        collections.defaultdict(list)
    )
    for source_id, root, rows in source_sets:
        for row in rows:
            indexed[str(row["task_id"])].append((source_id, root, row))
    selected: list[tuple[dict[str, Any], str, Path, dict[str, Any]]] = []
    for entry in validated["entries"]:
        matches = indexed.get(str(entry["task_id"]), [])
        if len(matches) != 1:
            raise ValueError(
                f"selected task has {len(matches)} source matches: {entry['task_id']}"
            )
        source_id, root, row = matches[0]
        if (
            row.get("map_id") != entry["map_id"]
            or row.get("source_group") != entry["source_group"]
            or row.get("layout_mode") != entry["layout_mode"]
            or row.get("agent_count") != entry["agent_count"]
        ):
            raise ValueError(
                f"selected task metadata differs from schedule: {entry['task_id']}"
            )
        selected.append((entry, source_id, root, row))
    if (
        len(selected) != 36
        or len({str(value[3]["task_id"]) for value in selected}) != 36
        or len({str(value[3]["map_id"]) for value in selected}) != 36
    ):
        raise ValueError("materialized selection is not exactly 36 unique tasks/maps")

    copy_fields = (*_IDENTITY_FILE_FIELDS, "instance_file")
    destinations: dict[str, str] = {}
    planned_files: dict[str, tuple[Path, str]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for _entry, source_id, root, raw_row in selected:
        row = dict(raw_row)
        split_root = root / SPLIT
        for field in copy_fields:
            raw_path = row.get(field)
            if raw_path is None:
                continue
            relative = Path(_text(raw_path, field=f"{row['task_id']}.{field}"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{row['task_id']}.{field} is not contained")
            probe = split_root
            for part in relative.parts:
                probe /= part
                if probe.is_symlink():
                    raise ValueError(
                        f"{row['task_id']}.{field} traverses a symbolic link"
                    )
            source_path = probe.resolve()
            try:
                source_path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"{row['task_id']}.{field} escapes its source dataset"
                ) from error
            if not source_path.is_file():
                raise ValueError(f"selected source file is missing: {source_path}")
            digest = sha256_file(source_path)
            destination = relative.as_posix()
            previous = destinations.get(destination)
            if previous is not None and previous != digest:
                destination = (
                    Path("sources") / source_id / relative
                ).as_posix()
                previous = destinations.get(destination)
            if previous is not None and previous != digest:
                raise ValueError(
                    f"selected files collide after deterministic rewrite: {destination}"
                )
            destinations[destination] = digest
            planned_files.setdefault(destination, (source_path, digest))
            row[field] = destination
        manifest_rows.append(row)
    manifest_rows.sort(key=lambda row: str(row["task_id"]))
    source_counts = dict(
        sorted(collections.Counter(row["source_group"] for row in manifest_rows).items())
    )
    layout_counts = dict(
        sorted(collections.Counter(row["layout_mode"] for row in manifest_rows).items())
    )
    observed = {
        "map_count": len({str(row["map_id"]) for row in manifest_rows}),
        "instance_count": len(manifest_rows),
        "source_counts": source_counts,
        "layout_counts": layout_counts,
    }
    if (
        observed["map_count"] != 36
        or observed["instance_count"] != 36
        or source_counts != {"generated": 18, "movingai": 18}
    ):
        raise ValueError("materialized dataset design would not pass its formal gate")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.partial-",
            dir=output_root.parent,
        )
    )
    try:
        source_evidence_root = selection_root / "selection_evidence"
        if (
            not source_evidence_root.is_dir()
            or (temporary / "selection_evidence").exists()
        ):
            raise ValueError("sealed selection evidence is unavailable")
        shutil.copytree(
            source_evidence_root,
            temporary / "selection_evidence",
        )
        copied_evidence_manifest = (
            temporary / EVIDENCE_MANIFEST_PATH
        )
        if (
            sha256_file(copied_evidence_manifest)
            != provenance["selection_evidence"]["manifest_sha256"]
        ):
            raise ValueError("copied selection evidence hash differs")
        dataset_output = temporary / "dataset"
        split_output = dataset_output / SPLIT
        for relative, (source_path, digest) in sorted(planned_files.items()):
            destination = split_output / Path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            shutil.copyfile(source_path, partial)
            if sha256_file(partial) != digest:
                raise ValueError(f"copied file hash differs: {relative}")
            partial.replace(destination)
        manifest_path = split_output / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_partial = manifest_path.with_name("manifest.jsonl.partial")
        with manifest_partial.open("w", encoding="utf-8", newline="\n") as stream:
            for row in manifest_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        manifest_partial.replace(manifest_path)
        materialization_identity = {
            "execution_schedule_sha256": validated[
                "execution_schedule_sha256"
            ],
            "cohort_report_sha256": validated["cohort_report_sha256"],
            "source_dataset_fingerprints": {
                "base-full-pool": base_fingerprint,
                **{
                    str(value["id"]): str(value["dataset_fingerprint"])
                    for value in registered_extensions
                },
            },
        }
        summary = {
            "schema_version": 1,
            "dataset_revision": (
                "balanced-wall-clock-corrected-native-selected-v3"
            ),
            "selection_blind_to_controller_outcomes": True,
            "configuration_fingerprint": _fingerprint(
                materialization_identity
            ),
            "execution_schedule_sha256": validated[
                "execution_schedule_sha256"
            ],
            "splits": {SPLIT: observed},
        }
        _write_json(dataset_output / "dataset_summary.json", summary)
        dataset_fingerprint = _dataset_fingerprint(dataset_output)
        report = {
            "schema": "lns2.corrected_native_dataset_materialization.v3",
            "passed": True,
            "selection_blind_to_controller_outcomes": True,
            "execution_schedule_sha256": validated[
                "execution_schedule_sha256"
            ],
            "cohort_report_sha256": validated["cohort_report_sha256"],
            "source_identity": materialization_identity,
            "manifest_sha256": sha256_file(manifest_path),
            "dataset_summary_sha256": sha256_file(
                dataset_output / "dataset_summary.json"
            ),
            "dataset_fingerprint": dataset_fingerprint,
            "copied_file_count": len(planned_files),
            "counts": observed,
            "producer": {
                "schema": (
                    "lns2.corrected_native_dataset_materializer.v3"
                ),
                "identity": _selector_identity(),
            },
            "selection_evidence_manifest_sha256": sha256_file(
                copied_evidence_manifest
            ),
            "gates": {
                "schedule_and_report_valid": True,
                "source_hashes_match_provenance": True,
                "exact_selected_task_coverage": True,
                "unique_tasks_and_maps": True,
                "source_balance_18_18": True,
                "all_files_hash_verified": True,
                "sealed_selection_evidence_copied": True,
            },
        }
        _write_json(temporary / "materialization_report.json", report)
        formal_provenance = json.loads(
            json.dumps(validated["provenance"])
        )
        formal_provenance.update(
            {
                "selection_execution_schedule_sha256": validated[
                    "execution_schedule_sha256"
                ],
                "selection_cohort_report_sha256": validated[
                    "cohort_report_sha256"
                ],
                "formal_dataset_relative_path": "dataset",
                "formal_dataset_manifest_sha256": report[
                    "manifest_sha256"
                ],
                "formal_dataset_summary_sha256": report[
                    "dataset_summary_sha256"
                ],
                "formal_dataset_fingerprint": dataset_fingerprint,
                "formal_materialization_report_sha256": sha256_file(
                    temporary / "materialization_report.json"
                ),
            }
        )
        formal_schedule = json.loads(json.dumps(validated["schedule"]))
        formal_schedule["provenance"] = formal_provenance
        formal_schedule_path = temporary / "execution_schedule.json"
        _write_json(formal_schedule_path, formal_schedule)
        formal_report = json.loads(json.dumps(validated["report"]))
        formal_report["provenance"] = formal_provenance
        formal_report["execution_schedule_sha256"] = sha256_file(
            formal_schedule_path
        )
        _write_json(temporary / "cohort_report.json", formal_report)
        formal_schedule_sha = sha256_file(formal_schedule_path)
        formal_report_sha = sha256_file(temporary / "cohort_report.json")
        temporary.replace(output_root)
        return {
            **report,
            "formal_schedule_root": str(output_root),
            "formal_dataset": str(output_root / "dataset"),
            "formal_execution_schedule_sha256": formal_schedule_sha,
            "formal_cohort_report_sha256": formal_report_sha,
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "REPORT_SCHEMA",
    "SCHEDULE_SCHEMA",
    "SELECTION_SALT",
    "SELECTOR_SCHEMA",
    "materialize_corrected_native_selected_dataset",
    "select_corrected_native_full_pool_schedule",
    "validate_corrected_native_full_pool_schedule",
]
