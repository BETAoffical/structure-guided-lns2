from __future__ import annotations

import collections
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from experiments.closed_loop_confirmation import run_closed_loop_collection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "lns2.stride.hierarchical_ch_source_config.v2"
TASK_REGISTRATION_SCHEMA = "lns2.stride.hierarchical_ch_source_v2_registration.v1"
MATERIALIZATION_SCHEMA = "lns2.stride.hierarchical_ch_source_materialization.v2"
PLAN_SCHEMA = "lns2.stride.hierarchical_ch_source_plan.v2"
COLLECTION_TRUST_SCHEMA = "lns2.stride.hierarchical_ch_source_collection_trust.v2"
EXPERIMENT_ID = "stride_hierarchical_ch_source_v2"
ALLOWED_SPLITS = ("train", "development")
VARIANT_IDS = ("bucket_mix_00", "bucket_mix_01")
EXPECTED_MAP_COUNT = 16
EXPECTED_TASK_COUNT = 64
EXPECTED_EPISODE_COUNT = 128


TaskMaterializer = Callable[..., list[dict[str, Any]]]


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
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{number}")
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
            raise ValueError(f"existing output differs from deterministic source-v2 product: {path}")
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


def _resolve_project_path(project_root: Path, registered: str) -> Path:
    path = Path(registered)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"registered path must be project-relative: {registered}")
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"registered path escapes the project: {registered}") from error
    return resolved


def _pinned_path(project_root: Path, pin: dict[str, Any], label: str) -> Path:
    path = _resolve_project_path(project_root, str(pin.get("path", "")))
    expected = str(pin.get("sha256", "")).lower()
    if len(expected) != 64 or not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return path


def _reject_sealed(value: str, sealed_ids: set[str], label: str) -> None:
    tokens = {value, value.split("__", 1)[0], Path(value).stem}
    if tokens & sealed_ids or value == "sealed_final":
        raise ValueError(f"sealed-final {label} is not authorized: {value}")


def _pin_pair(value: dict[str, Any]) -> tuple[str, str]:
    return (str(value.get("path", "")), str(value.get("sha256", "")).lower())


def _registration_qualification(
    registration: dict[str, Any], split: str
) -> dict[str, Any]:
    raw = dict(registration.get("reset_only_qualification") or {})
    required = {
        "minimum_nonzero_states",
        "minimum_active_maps",
        "required_layout_families",
    }
    if not required <= set(raw):
        raise ValueError(f"source-v2 {split} reset-only qualification gates are incomplete")
    def split_value(name: str) -> Any:
        value = raw[name]
        return dict(value)[split] if isinstance(value, dict) else value

    result = {
        "mode": "movingai_ood",
        "minimum_nonzero_states": int(split_value("minimum_nonzero_states")),
        "minimum_active_maps": int(split_value("minimum_active_maps")),
        "required_layout_families": sorted(
            map(str, split_value("required_layout_families"))
        ),
    }
    if result["minimum_nonzero_states"] <= 0 or result["minimum_active_maps"] <= 0:
        raise ValueError("source-v2 qualification thresholds must be positive")
    return result


def _split_gate(registration: dict[str, Any], name: str, split: str) -> Any:
    gates = dict(registration.get("reset_only_qualification") or {})
    value = gates[name]
    return dict(value)[split] if isinstance(value, dict) else value


def _validate_registered_gate_contract(registration: dict[str, Any]) -> None:
    gates = dict(registration.get("reset_only_qualification") or {})
    required = {
        "expected_reset_count",
        "all_resets_complete",
        "all_resets_valid",
        "all_initial_states_consistent",
        "minimum_nonzero_states",
        "minimum_active_maps",
        "minimum_nonzero_resets_per_map",
        "minimum_nonzero_resets_per_map_variant",
        "minimum_nonzero_high_load_resets_per_map",
        "minimum_resets_with_at_least_4_conflicts",
        "minimum_resets_with_at_least_16_conflicts",
        "minimum_maps_with_a_reset_at_least_4_conflicts",
        "minimum_maps_with_a_reset_at_least_16_conflicts",
        "required_layout_families",
        "outcome_based_subset_selection_allowed",
        "per_task_or_per_map_load_adaptation_allowed",
        "reserve_or_replacement_backfill_allowed",
        "failure_action",
    }
    if not required <= set(gates):
        raise ValueError("source-v2 reset-only registration omits a required gate")
    if (
        gates.get("atomic_whole_cohort_gate") is not True
        or gates["all_resets_complete"] is not True
        or gates["all_resets_valid"] is not True
        or gates["all_initial_states_consistent"] is not True
        or gates["outcome_based_subset_selection_allowed"] is not False
        or gates["per_task_or_per_map_load_adaptation_allowed"] is not False
        or gates["reserve_or_replacement_backfill_allowed"] is not False
        or gates["failure_action"] != "STATE_SUPPLY_FAIL_NO_BACKFILL"
    ):
        raise ValueError("source-v2 atomic no-backfill qualification contract changed")
    expected = dict(gates["expected_reset_count"])
    if (
        {split: int(expected.get(split, -1)) for split in ALLOWED_SPLITS}
        != {"train": 64, "development": 64}
        or int(expected.get("total", -1)) != EXPECTED_EPISODE_COUNT
    ):
        raise ValueError("source-v2 registered reset counts are not the exact 64/64 product")
    for split in ALLOWED_SPLITS:
        for name in (
            "minimum_nonzero_states",
            "minimum_active_maps",
            "minimum_nonzero_resets_per_map",
            "minimum_nonzero_resets_per_map_variant",
            "minimum_nonzero_high_load_resets_per_map",
            "minimum_resets_with_at_least_4_conflicts",
            "minimum_resets_with_at_least_16_conflicts",
            "minimum_maps_with_a_reset_at_least_4_conflicts",
            "minimum_maps_with_a_reset_at_least_16_conflicts",
        ):
            if int(_split_gate(registration, name, split)) <= 0:
                raise ValueError(f"source-v2 registered gate must be positive: {split}/{name}")


def load_registered_source_context(
    config_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source_config_path = Path(config_path)
    if not source_config_path.is_absolute():
        source_config_path = (root / source_config_path).resolve()
    source_config = _read_json(source_config_path)
    if (
        source_config.get("schema") != CONFIG_SCHEMA
        or source_config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("unsupported hierarchical C/H source-v2 config or experiment identity")
    claim = dict(source_config.get("claim_boundary") or {})
    if (
        claim.get("training_authorized") is not False
        or claim.get("outcome_filtering_allowed") is not False
        or claim.get("sealed_final_semantic_access_authorized") is not False
    ):
        raise ValueError("source-v2 must remain no-train, outcome-blind, and final-sealed")

    h1_pin = dict(source_config.get("registered_h1_v2") or {})
    design_path = _pinned_path(root, dict(h1_pin.get("config") or {}), "H1-v2 config")
    report_path = _pinned_path(
        root, dict(h1_pin.get("registration_report") or {}), "H1-v2 registration report"
    )
    design = _read_json(design_path)
    report = _read_json(report_path)
    design_sha = _sha256(design_path)
    report_sha = _sha256(report_path)
    if (
        report.get("schema") != h1_pin.get("required_report_schema")
        or report.get("status") != "REGISTERED"
        or report.get("passed") is not True
        or report.get("config_sha256") != design_sha
        or report.get("training_authorized") is not False
        or design.get("claim_boundary", {}).get("training_authorized") is not False
    ):
        raise ValueError("H1-v2 registration is not a frozen REGISTERED no-train design")

    map_splits = {name: list(values) for name, values in design["map_splits"].items()}
    allowed_ids = {
        str(map_id) for split in ALLOWED_SPLITS for map_id in map_splits[split]
    }
    sealed_ids = set(map(str, map_splits["sealed_final"]))
    if len(allowed_ids) != EXPECTED_MAP_COUNT or allowed_ids & sealed_ids:
        raise ValueError("H1-v2 map split integrity failed")

    task_pin = dict(source_config.get("task_registration") or {})
    task_registration_path = _pinned_path(root, task_pin, "source-v2 task registration")
    task_registration = _read_json(task_registration_path)
    task_registration_sha = _sha256(task_registration_path)
    task_registration_report_path = _pinned_path(
        root,
        dict(task_pin.get("report") or {}),
        "source-v2 task registration report",
    )
    task_registration_report = _read_json(task_registration_report_path)
    task_registration_report_sha = _sha256(task_registration_report_path)
    if (
        task_registration.get("schema")
        != task_pin.get("required_schema", TASK_REGISTRATION_SCHEMA)
        or task_registration.get("status") != task_pin.get("required_status", "REGISTERED")
        or task_registration.get("training_authorized") is not False
        or tuple(task_registration.get("allowed_splits") or ()) != ALLOWED_SPLITS
    ):
        raise ValueError("source-v2 task registration is not an executable no-train registration")
    if (
        task_registration_report.get("schema") != task_pin.get("required_report_schema")
        or task_registration_report.get("status") != "REGISTERED"
        or task_registration_report.get("passed") is not True
        or task_registration_report.get("config_sha256") != task_registration_sha
        or task_registration_report.get("training_authorized") is not False
        or int(task_registration_report.get("expected_task_count", -1))
        != EXPECTED_TASK_COUNT
        or int(task_registration_report.get("expected_episode_count", -1))
        != EXPECTED_EPISODE_COUNT
    ):
        raise ValueError("source-v2 task registration report is invalid or unpinned")
    registered_h1 = dict(
        task_registration.get("registered_h1_v2")
        or task_registration.get("parent_h1_registration")
        or {}
    )
    if not registered_h1:
        raise ValueError("task registration does not pin its parent H1-v2 identity")
    if (
            _pin_pair(dict(registered_h1.get("config") or {}))
            != _pin_pair(dict(h1_pin.get("config") or {}))
            or _pin_pair(
                dict(
                    registered_h1.get("registration_report")
                    or registered_h1.get("report")
                    or {}
                )
            )
            != _pin_pair(dict(h1_pin.get("registration_report") or {}))
    ):
        raise ValueError("task registration pins a different H1-v2 identity")

    registered_splits = {
        split: list(map(str, dict(task_registration["map_splits"])[split]))
        for split in ALLOWED_SPLITS
    }
    if any(registered_splits[split] != list(map(str, map_splits[split])) for split in ALLOWED_SPLITS):
        raise ValueError("source-v2 task map splits differ from H1-v2 registration")
    if any(name not in ALLOWED_SPLITS for name in dict(task_registration["map_splits"])):
        raise ValueError("source-v2 task registration must not contain final task details")
    if dict(task_registration.get("map_groups") or {}) != {
        map_id: str(design["map_groups"][map_id]) for map_id in sorted(allowed_ids)
    }:
        raise ValueError("source-v2 task map groups differ from H1-v2 registration")
    registered_sources = dict(task_registration.get("registered_sources") or {})
    if set(registered_sources) != allowed_ids or any(
        registered_sources[map_id] != design["registered_sources"][map_id]
        for map_id in allowed_ids
    ):
        raise ValueError("source-v2 task source pins differ from H1-v2 registration")

    variant_ids = tuple(map(str, task_registration.get("variant_ids") or ()))
    if variant_ids != VARIANT_IDS:
        raise ValueError("source-v2 bucket-mix variants differ from the exact registration")
    load_grid = {
        str(map_id): [int(value) for value in values]
        for map_id, values in dict(task_registration.get("per_map_load_grid") or {}).items()
    }
    if (
        set(load_grid) != allowed_ids
        or any(len(values) != 2 or len(set(values)) != 2 or min(values) <= 0 for values in load_grid.values())
    ):
        raise ValueError("source-v2 load grid is not two positive static loads per registered map")
    if (
        int(task_registration.get("expected_task_count", -1)) != EXPECTED_TASK_COUNT
        or int(task_registration.get("expected_episode_count", -1)) != EXPECTED_EPISODE_COUNT
    ):
        raise ValueError("source-v2 task/episode product is not the exact 64/128 registration")
    _validate_registered_gate_contract(task_registration)

    encoded_execution = json.dumps(
        {"source_config": source_config, "task_registration": task_registration},
        sort_keys=True,
    )
    if any(map_id in encoded_execution for map_id in sealed_ids):
        raise ValueError("source-v2 execution registration must not name sealed-final map IDs")

    source = dict(source_config.get("source") or {})
    if (
        tuple(source.get("allowed_splits") or ()) != ALLOWED_SPLITS
        or int(source.get("expected_map_count", -1)) != EXPECTED_MAP_COUNT
        or int(source.get("expected_task_count", -1)) != EXPECTED_TASK_COUNT
    ):
        raise ValueError("source-v2 train/development materialization contract changed")
    materializer_pin = dict(source_config.get("task_materializer") or {})
    implementation_path = _pinned_path(
        root, dict(materializer_pin.get("implementation") or {}), "bucket-mix materializer"
    )
    if (
        materializer_pin.get("module")
        != "experiments.stride_hierarchical_ch_bucket_mix_v2"
        or materializer_pin.get("function")
        != "materialize_registered_bucket_mix_map"
        or materializer_pin.get("algorithm_schema")
        != task_registration.get("task_materialization", {}).get("algorithm_schema")
        or tuple(materializer_pin.get("variant_ids") or ()) != VARIANT_IDS
    ):
        raise ValueError("source-v2 deterministic task materializer identity changed")
    collection = dict(source_config.get("collection") or {})
    base_path = _pinned_path(root, dict(collection.get("base_config") or {}), "collection base")
    controller_pin = dict(collection.get("controller_bundle") or {})
    controller_root = _resolve_project_path(root, str(controller_pin.get("path", "")))
    controller_manifest = controller_root / "controller_manifest.json"
    if (
        not controller_manifest.is_file()
        or _sha256(controller_manifest)
        != str(controller_pin.get("manifest_sha256", "")).lower()
    ):
        raise ValueError("v2-full controller bundle manifest SHA256 mismatch")
    workers = int(collection.get("workers", 0))
    if (
        collection.get("controller") != "v2-full"
        or collection.get("executed_policy") != "realized_dynamic"
        or list(map(int, collection.get("solver_seeds") or ())) != [41, 42]
        or collection.get("deterministic_pp_replay") is not True
        or collection.get("feature_backend") != "native"
        or int(collection.get("expected_episode_count", -1)) != EXPECTED_EPISODE_COUNT
        or workers < 16
        or workers > 64
    ):
        raise ValueError("source-v2 v2-full collection contract changed")
    for split in ALLOWED_SPLITS:
        _registration_qualification(task_registration, split)
    return {
        "project_root": root,
        "source_config_path": source_config_path,
        "source_config": source_config,
        "design_path": design_path,
        "design": design,
        "design_sha256": design_sha,
        "report_path": report_path,
        "report": report,
        "report_sha256": report_sha,
        "task_registration_path": task_registration_path,
        "task_registration": task_registration,
        "task_registration_sha256": task_registration_sha,
        "task_registration_report_path": task_registration_report_path,
        "task_registration_report": task_registration_report,
        "task_registration_report_sha256": task_registration_report_sha,
        "base_config_path": base_path,
        "base_config": _read_json(base_path),
        "materializer_implementation_path": implementation_path,
        "controller_root": controller_root,
        "map_splits": registered_splits,
        "allowed_ids": allowed_ids,
        "sealed_ids": sealed_ids,
        "load_grid": load_grid,
        "variant_ids": variant_ids,
    }


def _source_file(ctx: dict[str, Any], map_id: str, kind: str) -> tuple[Path, str]:
    _reject_sealed(map_id, ctx["sealed_ids"], "map ID")
    if map_id not in ctx["allowed_ids"]:
        raise ValueError(f"unregistered train/development map ID: {map_id}")
    row = dict(ctx["design"]["registered_sources"][map_id][kind])
    path = _resolve_project_path(ctx["project_root"], str(row["path"]))
    source_root = _resolve_project_path(ctx["project_root"], str(ctx["design"]["source_root"]))
    try:
        path.relative_to(source_root)
    except ValueError as error:
        raise ValueError(f"registered {kind} escapes source root: {map_id}") from error
    return path, str(row["sha256"]).lower()


def _default_materializer(**kwargs: Any) -> list[dict[str, Any]]:
    from experiments.stride_hierarchical_ch_bucket_mix_v2 import (
        VARIANT_IDS as MATERIALIZER_VARIANT_IDS,
        materialize_registered_bucket_mix_map,
    )

    if tuple(kwargs["variant_ids"]) != tuple(MATERIALIZER_VARIANT_IDS):
        raise ValueError("bucket-mix materializer variant identity differs from source-v2")
    root = Path(kwargs["project_root"])
    design = dict(kwargs["h1_design"])
    map_id = str(kwargs["map_id"])
    source = dict(design["registered_sources"][map_id])
    scenario_row = dict(source["scenario"])
    map_row = dict(source["map"])
    scenario_path = _resolve_project_path(root, str(scenario_row["path"]))
    allowed = {
        str(value)
        for split in ALLOWED_SPLITS
        for value in dict(kwargs["task_registration"]["map_splits"])[split]
    }
    return materialize_registered_bucket_mix_map(
        scenario_path,
        expected_source_sha256=str(scenario_row["sha256"]),
        map_id=map_id,
        registered_loads=tuple(kwargs["agent_counts"]),
        split=str(kwargs["split"]),
        output_split_root=Path(kwargs["split_root"]),
        allowed_map_ids=allowed,
        sealed_map_ids=set(map(str, design["map_splits"]["sealed_final"])),
        map_file=f"maps/{map_id}.map",
        map_sha256=str(map_row["sha256"]),
        layout_family=str(design["map_groups"][map_id]),
        dry_run=False,
        resume=bool(kwargs["resume"]),
    )


def _relative_materialized_path(split_root: Path, value: Any, label: str) -> tuple[str, Path]:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (split_root / path).resolve()
    try:
        relative = resolved.relative_to(split_root.resolve())
    except ValueError as error:
        raise ValueError(f"materializer {label} escapes split root: {value}") from error
    return str(relative).replace("\\", "/"), resolved


def materialize_source_dataset(
    config_path: str | Path,
    output: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    project_root: str | Path = PROJECT_ROOT,
    materializer: TaskMaterializer | None = None,
) -> dict[str, Any]:
    if dry_run and resume:
        raise ValueError("dry-run and resume are mutually exclusive")
    ctx = load_registered_source_context(config_path, project_root=project_root)
    output_root = Path(output).resolve()
    dataset_root = output_root / str(ctx["source_config"]["source"]["dataset_directory"])
    prospective = {
        "schema": MATERIALIZATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "DRY_RUN" if dry_run else "MATERIALIZED",
        "h1_v2_config_sha256": ctx["design_sha256"],
        "h1_v2_registration_report_sha256": ctx["report_sha256"],
        "task_registration_sha256": ctx["task_registration_sha256"],
        "task_registration_report_sha256": ctx["task_registration_report_sha256"],
        "split_map_counts": {split: len(ctx["map_splits"][split]) for split in ALLOWED_SPLITS},
        "expected_task_count": EXPECTED_TASK_COUNT,
        "expected_raw_map_count": EXPECTED_MAP_COUNT,
        "sealed_final_semantic_access": False,
        "training_authorized": False,
    }
    if dry_run:
        return prospective
    materializer = materializer or _default_materializer
    expected_files: set[Path] = set()
    source_manifest: list[dict[str, Any]] = []
    manifests: dict[str, list[dict[str, Any]]] = {}
    map_groups = dict(ctx["design"]["map_groups"])
    for split in ALLOWED_SPLITS:
        split_root = dataset_root / split
        manifest: list[dict[str, Any]] = []
        for map_id in ctx["map_splits"][split]:
            map_path, map_hash = _source_file(ctx, map_id, "map")
            source_scenario_path, source_scenario_hash = _source_file(ctx, map_id, "scenario")
            map_payload = map_path.read_bytes()
            if hashlib.sha256(map_payload).hexdigest() != map_hash:
                raise ValueError(f"registered map SHA256 mismatch: {map_id}")
            if _sha256(source_scenario_path) != source_scenario_hash:
                raise ValueError(f"registered source scenario SHA256 mismatch: {map_id}")
            map_output = split_root / "maps" / f"{map_id}.map"
            metadata_output = split_root / "maps" / f"{map_id}.json"
            _write_exact(map_output, map_payload, resume=resume)
            _write_exact(
                metadata_output,
                _json_bytes(
                    {
                        "schema_version": 2,
                        "benchmark_id": map_id,
                        "split": split,
                        "source": "H1-v2 hash-pinned MovingAI map",
                        "map_sha256": map_hash,
                        "source_scenario_sha256": source_scenario_hash,
                        "training_authorized": False,
                    }
                ),
                resume=resume,
            )
            expected_files.update({map_output, metadata_output})
            produced = materializer(
                project_root=ctx["project_root"],
                h1_design=ctx["design"],
                task_registration=ctx["task_registration"],
                split=split,
                map_id=map_id,
                agent_counts=tuple(ctx["load_grid"][map_id]),
                variant_ids=ctx["variant_ids"],
                split_root=split_root,
                resume=resume,
            )
            if not isinstance(produced, list) or any(not isinstance(row, dict) for row in produced):
                raise ValueError("source-v2 task materializer must return a list of manifest rows")
            observed_product: set[tuple[str, int]] = set()
            for raw in produced:
                row = dict(raw)
                if str(row.get("map_id", map_id)) != map_id:
                    raise ValueError(f"materializer returned a different map ID for {map_id}")
                variant_id = str(row.get("variant_id", row.get("bucket_mix_variant", "")))
                agent_count = int(row.get("agent_count", -1))
                observed_product.add((variant_id, agent_count))
                scenario_relative, scenario_path = _relative_materialized_path(
                    split_root, row.get("scenario_file", ""), "scenario path"
                )
                task_relative, task_path = _relative_materialized_path(
                    split_root, row.get("task_file", ""), "task path"
                )
                scenario_hash = str(row.get("scenario_sha256", "")).lower()
                if len(scenario_hash) != 64 or _sha256(scenario_path) != scenario_hash:
                    raise ValueError(f"materialized scenario SHA256 mismatch: {map_id}/{variant_id}")
                if not task_path.is_file():
                    raise ValueError(f"materialized task metadata is missing: {task_path}")
                task = _read_json(task_path)
                if (
                    int(task.get("agent_count", -1)) != agent_count
                    or int(task.get("unique_start_count", -1)) != agent_count
                    or int(task.get("unique_goal_count", -1)) != agent_count
                ):
                    raise ValueError(f"materialized task metadata mismatch: {row.get('task_id')}")
                expected_files.update({scenario_path, task_path})
                manifest.append(
                    {
                        **row,
                        "split": split,
                        "source_group": "movingai_registered_bucket_mix_v2",
                        "map_id": map_id,
                        "task_id": str(row["task_id"]),
                        "variant_id": variant_id,
                        "map_file": f"maps/{map_id}.map",
                        "scenario_file": scenario_relative,
                        "map_metadata_file": f"maps/{map_id}.json",
                        "task_file": task_relative,
                        "map_sha256": map_hash,
                        "scenario_sha256": scenario_hash,
                        "layout_mode": str(map_groups[map_id]),
                        "layout_variant": map_id,
                        "scenario_type": f"movingai_{variant_id}",
                        "task_variant": str(row.get("task_variant", f"{variant_id}_agents_{agent_count}")),
                        "agent_count": agent_count,
                        "training_authorized": False,
                    }
                )
            expected_product = {
                (variant_id, int(agent_count))
                for variant_id in ctx["variant_ids"]
                for agent_count in ctx["load_grid"][map_id]
            }
            if observed_product != expected_product or len(produced) != len(expected_product):
                raise ValueError(f"materializer product differs from registration: {map_id}")
            source_manifest.append(
                {
                    "split": split,
                    "map_id": map_id,
                    "map_source_path": str(map_path.relative_to(ctx["project_root"])).replace("\\", "/"),
                    "map_sha256": map_hash,
                    "scenario_source_path": str(source_scenario_path.relative_to(ctx["project_root"])).replace("\\", "/"),
                    "source_scenario_sha256": source_scenario_hash,
                }
            )
        manifest.sort(key=lambda row: str(row["task_id"]))
        if len(manifest) != 32 or len({str(row["task_id"]) for row in manifest}) != 32:
            raise ValueError(f"source-v2 {split} manifest is not the exact 32-task product")
        manifest_path = split_root / "manifest.jsonl"
        _write_exact(manifest_path, _jsonl_bytes(manifest), resume=resume)
        expected_files.add(manifest_path)
        manifests[split] = manifest
    source_manifest.sort(key=lambda row: (str(row["split"]), str(row["map_id"])))
    source_manifest_path = dataset_root / "source_manifest.jsonl"
    _write_exact(source_manifest_path, _jsonl_bytes(source_manifest), resume=resume)
    expected_files.add(source_manifest_path)
    summary_path = dataset_root / "dataset_summary.json"
    _write_exact(
        summary_path,
        _json_bytes(
            {
                "schema_version": 2,
                "schema": MATERIALIZATION_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "task_registration_sha256": ctx["task_registration_sha256"],
                "splits": {
                    split: {
                        "map_count": len(ctx["map_splits"][split]),
                        "task_count": len(manifests[split]),
                    }
                    for split in ALLOWED_SPLITS
                },
                "sealed_final_semantic_access": False,
                "training_authorized": False,
            }
        ),
        resume=resume,
    )
    expected_files.add(summary_path)
    observed_files = {path for path in dataset_root.rglob("*") if path.is_file()}
    if observed_files != expected_files:
        extras = sorted(str(path.relative_to(dataset_root)) for path in observed_files - expected_files)
        missing = sorted(str(path.relative_to(dataset_root)) for path in expected_files - observed_files)
        raise ValueError(f"materialized source-v2 file set differs: extras={extras}, missing={missing}")
    prospective.update(
        {
            "dataset_root": str(dataset_root),
            "task_count": sum(map(len, manifests.values())),
            "raw_map_count": len(source_manifest),
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "split_manifests": {
                split: {
                    "path": str(dataset_root / split / "manifest.jsonl"),
                    "sha256": _sha256(dataset_root / split / "manifest.jsonl"),
                    "task_count": len(manifests[split]),
                }
                for split in ALLOWED_SPLITS
            },
        }
    )
    _write_exact(output_root / "materialization_trust_report.json", _json_bytes(prospective), resume=resume)
    return prospective


def _expected_task_product(ctx: dict[str, Any], split: str) -> set[tuple[str, str, int]]:
    return {
        (map_id, variant_id, int(agent_count))
        for map_id in ctx["map_splits"][split]
        for variant_id in ctx["variant_ids"]
        for agent_count in ctx["load_grid"][map_id]
    }


def _validate_materialized(
    ctx: dict[str, Any], output_root: Path
) -> dict[str, list[dict[str, Any]]]:
    dataset_root = output_root / str(ctx["source_config"]["source"]["dataset_directory"])
    trust = _read_json(output_root / "materialization_trust_report.json")
    if (
        trust.get("schema") != MATERIALIZATION_SCHEMA
        or trust.get("experiment_id") != EXPERIMENT_ID
        or trust.get("status") != "MATERIALIZED"
        or trust.get("training_authorized") is not False
        or trust.get("h1_v2_config_sha256") != ctx["design_sha256"]
        or trust.get("h1_v2_registration_report_sha256") != ctx["report_sha256"]
        or trust.get("task_registration_sha256") != ctx["task_registration_sha256"]
        or trust.get("task_registration_report_sha256")
        != ctx["task_registration_report_sha256"]
    ):
        raise ValueError("source-v2 materialization trust report is not pinned to registrations")
    manifests: dict[str, list[dict[str, Any]]] = {}
    for split in ALLOWED_SPLITS:
        manifest_path = dataset_root / split / "manifest.jsonl"
        if _sha256(manifest_path) != trust["split_manifests"][split]["sha256"]:
            raise ValueError(f"materialized source-v2 {split} manifest SHA256 mismatch")
        rows = _read_jsonl(manifest_path)
        observed = {
            (str(row["map_id"]), str(row["variant_id"]), int(row["agent_count"]))
            for row in rows
        }
        expected = _expected_task_product(ctx, split)
        if observed != expected or len(rows) != len(expected):
            raise ValueError(f"materialized source-v2 {split} task product differs from registration")
        if len({str(row["task_id"]) for row in rows}) != len(rows):
            raise ValueError(f"materialized source-v2 {split} task IDs are not unique")
        split_root = dataset_root / split
        for row in rows:
            map_id = str(row["map_id"])
            _reject_sealed(map_id, ctx["sealed_ids"], "manifest map ID")
            if map_id not in ctx["map_splits"][split] or row.get("training_authorized") is not False:
                raise ValueError(f"materialized source-v2 {split} violates split/no-train contract")
            for field, hash_field in (("map_file", "map_sha256"), ("scenario_file", "scenario_sha256")):
                _, path = _relative_materialized_path(split_root, row[field], field)
                if _sha256(path) != str(row[hash_field]).lower():
                    raise ValueError(f"materialized source-v2 source hash mismatch: {row['task_id']}")
            _, task_path = _relative_materialized_path(split_root, row["task_file"], "task_file")
            task = _read_json(task_path)
            if (
                int(task.get("agent_count", -1)) != int(row["agent_count"])
                or int(task.get("unique_start_count", -1)) != int(row["agent_count"])
                or int(task.get("unique_goal_count", -1)) != int(row["agent_count"])
            ):
                raise ValueError(f"materialized source-v2 task metadata mismatch: {row['task_id']}")
        manifests[split] = rows
    return manifests


def _collection_config(
    ctx: dict[str, Any], split: str, rows: list[dict[str, Any]], *, workers: int
) -> dict[str, Any]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported source-v2 split: {split}")
    collection = dict(ctx["source_config"]["collection"])
    config = json.loads(json.dumps(ctx["base_config"]))
    map_ids = list(ctx["map_splits"][split])
    # Family replication is over unique maps. Four task variants per map must
    # never multiply the registered family count.
    family_counts = collections.Counter(
        str(ctx["design"]["map_groups"][map_id]) for map_id in map_ids
    )
    qualification = _registration_qualification(ctx["task_registration"], split)
    config.update(
        {
            "formal": True,
            "split": split,
            "solver_seeds": list(map(int, collection["solver_seeds"])),
            "policies": list(collection["policies_in_config"]),
            "dataset_design": {
                "mode": "movingai_ood",
                "map_count": len(map_ids),
                "task_count": len(rows),
                "scenario_indices": [int(value.rsplit("_", 1)[-1]) for value in ctx["variant_ids"]],
                "layout_family_counts": dict(sorted(family_counts.items())),
                "maps": [
                    {
                        "map_id": map_id,
                        "layout_family": str(ctx["design"]["map_groups"][map_id]),
                        "agent_counts": list(ctx["load_grid"][map_id]),
                    }
                    for map_id in map_ids
                ],
                "historical_map_ids": list(ctx["design"]["legacy_current_h1"]["map_ids"]),
            },
            "environment": {
                **dict(config["environment"]),
                "time_limit": float(collection["environment_time_limit_seconds"]),
            },
            "qualification": qualification,
            "max_decisions": int(collection["max_decisions"]),
            "metric_iteration_budget": int(collection["max_decisions"]),
            "wall_time_budget_seconds": float(collection["wall_time_budget_seconds"]),
            "episode_process_timeout_seconds": float(collection["episode_process_timeout_seconds"]),
            "workers": workers,
            "deterministic_pp_replay": True,
            "reference_datasets": [],
            "scientific_status": "registered_source_v2_collection_training_unauthorized",
            "training_authorized": False,
        }
    )
    return config


def _effective_workers(ctx: dict[str, Any], workers: int | None) -> int:
    value = int(ctx["source_config"]["collection"]["workers"] if workers is None else workers)
    if value <= 0 or value > 64:
        raise ValueError("workers must be between 1 and 64")
    return value


def plan_source_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    workers: int | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if dry_run and resume:
        raise ValueError("dry-run and resume are mutually exclusive")
    ctx = load_registered_source_context(config_path, project_root=project_root)
    output_root = Path(output).resolve()
    manifests = _validate_materialized(ctx, output_root)
    seeds = list(map(int, ctx["source_config"]["collection"]["solver_seeds"]))
    effective_workers = _effective_workers(ctx, workers)
    schedule = [
        {
            "split": split,
            "task_id": str(row["task_id"]),
            "map_id": str(row["map_id"]),
            "variant_id": str(row["variant_id"]),
            "agent_count": int(row["agent_count"]),
            "solver_seed": seed,
            "controller": "v2-full",
            "policy": "realized_dynamic",
            "qualification_required": True,
            "training_authorized": False,
        }
        for split in ALLOWED_SPLITS
        for row in manifests[split]
        for seed in seeds
    ]
    schedule.sort(key=lambda row: (str(row["split"]), str(row["task_id"]), int(row["solver_seed"])))
    keys = {(row["split"], row["task_id"], row["solver_seed"]) for row in schedule}
    if len(schedule) != EXPECTED_EPISODE_COUNT or len(keys) != EXPECTED_EPISODE_COUNT:
        raise ValueError("source-v2 schedule is not the exact 128-job product")
    report = {
        "schema": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "DRY_RUN" if dry_run else "PLANNED",
        "h1_v2_config_sha256": ctx["design_sha256"],
        "h1_v2_registration_report_sha256": ctx["report_sha256"],
        "task_registration_sha256": ctx["task_registration_sha256"],
        "task_registration_report_sha256": ctx["task_registration_report_sha256"],
        "controller": "v2-full",
        "executed_policy": "realized_dynamic",
        "solver_seeds": seeds,
        "workers": effective_workers,
        "task_count": EXPECTED_TASK_COUNT,
        "episode_count": EXPECTED_EPISODE_COUNT,
        "split_episode_counts": {split: len(manifests[split]) * len(seeds) for split in ALLOWED_SPLITS},
        "outcome_filtering": False,
        "sealed_final_semantic_access": False,
        "training_authorized": False,
    }
    if dry_run:
        return {**report, "schedule": schedule}
    for split in ALLOWED_SPLITS:
        _write_exact(
            output_root / "collection_configs" / f"{split}.json",
            _json_bytes(_collection_config(ctx, split, manifests[split], workers=effective_workers)),
            resume=resume,
        )
    schedule_path = output_root / "source_schedule.jsonl"
    _write_exact(schedule_path, _jsonl_bytes(schedule), resume=resume)
    report.update({"schedule_path": str(schedule_path), "schedule_sha256": _sha256(schedule_path)})
    _write_exact(output_root / "source_plan_report.json", _json_bytes(report), resume=resume)
    return report


def _manifest_job_product(path: Path, expected: set[tuple[str, int]], label: str) -> None:
    rows = _read_jsonl(path)
    observed = {(str(row["task_id"]), int(row["solver_seed"])) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"{label} manifest differs from the exact task/seed product")


def _qualification_audit(
    ctx: dict[str, Any],
    split: str,
    output: Path,
    jobs: set[tuple[str, int]],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    manifest = output / "qualification_manifest.jsonl"
    report = output / "qualification_report.json"
    run_config = output / "run_config.json"
    if not (manifest.is_file() and report.is_file() and run_config.is_file()):
        return None
    _manifest_job_product(manifest, jobs, "qualification")
    rows = _read_jsonl(manifest)
    closed_loop_report = _read_json(report)
    if type(closed_loop_report.get("passed")) is not bool:
        raise ValueError("qualification report lacks passed=true/false")
    task_by_id = {str(row["task_id"]): row for row in manifest_rows}
    metadata_consistent = all(
        str(row.get("task_id")) in task_by_id
        and str(row.get("map_id"))
        == str(task_by_id[str(row.get("task_id"))]["map_id"])
        and int(row.get("agent_count", -1))
        == int(task_by_id[str(row.get("task_id"))]["agent_count"])
        and str(row.get("split")) == split
        for row in rows
    )
    valid_rows = [
        row
        for row in rows
        if str(row.get("status")) == "ok" and row.get("error") in (None, "")
    ]
    complete_rows = [row for row in valid_rows if bool(row.get("initial_complete", False))]
    consistent_rows = [
        row
        for row in valid_rows
        if bool(row.get("initial_feasible", False))
        == (bool(row.get("initial_complete", False)) and int(row.get("initial_conflicts", -1)) == 0)
    ]
    nonzero = [row for row in valid_rows if int(row.get("initial_conflicts", 0)) > 0]
    active_maps = {str(row["map_id"]) for row in nonzero}
    active_families = {str(row["layout_mode"]) for row in nonzero}
    per_map_nonzero = collections.Counter(str(row["map_id"]) for row in nonzero)
    per_map_variant_nonzero: collections.Counter[tuple[str, str]] = collections.Counter()
    per_map_high_nonzero: collections.Counter[str] = collections.Counter()
    for row in nonzero:
        map_id = str(row["map_id"])
        task_id = str(row["task_id"])
        per_map_variant_nonzero[
            (map_id, str(task_by_id[task_id]["variant_id"]))
        ] += 1
        if int(row["agent_count"]) == max(ctx["load_grid"][map_id]):
            per_map_high_nonzero[map_id] += 1
    registration = ctx["task_registration"]
    map_ids = set(ctx["map_splits"][split])
    expected_resets = int(_split_gate(registration, "expected_reset_count", split))
    minimum_nonzero = int(_split_gate(registration, "minimum_nonzero_states", split))
    minimum_active_maps = int(_split_gate(registration, "minimum_active_maps", split))
    minimum_per_map = int(_split_gate(registration, "minimum_nonzero_resets_per_map", split))
    minimum_per_map_variant = int(
        _split_gate(registration, "minimum_nonzero_resets_per_map_variant", split)
    )
    minimum_high_per_map = int(
        _split_gate(registration, "minimum_nonzero_high_load_resets_per_map", split)
    )
    minimum_ge4 = int(
        _split_gate(registration, "minimum_resets_with_at_least_4_conflicts", split)
    )
    minimum_ge16 = int(
        _split_gate(registration, "minimum_resets_with_at_least_16_conflicts", split)
    )
    minimum_maps_ge4 = int(
        _split_gate(registration, "minimum_maps_with_a_reset_at_least_4_conflicts", split)
    )
    minimum_maps_ge16 = int(
        _split_gate(registration, "minimum_maps_with_a_reset_at_least_16_conflicts", split)
    )
    required_families = set(
        map(str, _split_gate(registration, "required_layout_families", split))
    )
    gates = {
        "expected_reset_count": len(rows) == expected_resets,
        "registered_task_metadata": metadata_consistent,
        "all_resets_valid": len(valid_rows) == expected_resets,
        "all_resets_complete": len(complete_rows) == expected_resets,
        "all_initial_states_consistent": len(consistent_rows) == expected_resets,
        "closed_loop_registered_gates": bool(closed_loop_report["passed"]),
        "minimum_nonzero_states": len(nonzero) >= minimum_nonzero,
        "minimum_active_maps": len(active_maps) >= minimum_active_maps,
        "required_layout_families_active": required_families <= active_families,
        "minimum_nonzero_resets_per_map": all(
            per_map_nonzero[map_id] >= minimum_per_map for map_id in map_ids
        ),
        "minimum_nonzero_resets_per_map_variant": all(
            per_map_variant_nonzero[(map_id, variant_id)] >= minimum_per_map_variant
            for map_id in map_ids
            for variant_id in ctx["variant_ids"]
        ),
        "minimum_nonzero_high_load_resets_per_map": all(
            per_map_high_nonzero[map_id] >= minimum_high_per_map for map_id in map_ids
        ),
        "minimum_resets_with_at_least_4_conflicts": sum(
            int(row.get("initial_conflicts", 0)) >= 4 for row in valid_rows
        )
        >= minimum_ge4,
        "minimum_resets_with_at_least_16_conflicts": sum(
            int(row.get("initial_conflicts", 0)) >= 16 for row in valid_rows
        )
        >= minimum_ge16,
        "minimum_maps_with_a_reset_at_least_4_conflicts": len(
            {
                str(row["map_id"])
                for row in valid_rows
                if int(row.get("initial_conflicts", 0)) >= 4
            }
        )
        >= minimum_maps_ge4,
        "minimum_maps_with_a_reset_at_least_16_conflicts": len(
            {
                str(row["map_id"])
                for row in valid_rows
                if int(row.get("initial_conflicts", 0)) >= 16
            }
        )
        >= minimum_maps_ge16,
    }
    audit = {
        "schema": "lns2.stride.hierarchical_ch_source_v2_qualification_audit.v1",
        "experiment_id": EXPERIMENT_ID,
        "split": split,
        "task_registration_sha256": ctx["task_registration_sha256"],
        "task_registration_report_sha256": ctx["task_registration_report_sha256"],
        "qualification_manifest_sha256": _sha256(manifest),
        "qualification_report_sha256": _sha256(report),
        "expected_reset_count": expected_resets,
        "observed": {
            "reset_count": len(rows),
            "valid_reset_count": len(valid_rows),
            "complete_reset_count": len(complete_rows),
            "consistent_initial_state_count": len(consistent_rows),
            "nonzero_reset_count": len(nonzero),
            "active_map_count": len(active_maps),
            "active_maps": sorted(active_maps),
            "active_layout_families": sorted(active_families),
            "nonzero_resets_by_map": dict(sorted(per_map_nonzero.items())),
            "nonzero_resets_by_map_variant": {
                f"{map_id}/{variant_id}": per_map_variant_nonzero[(map_id, variant_id)]
                for map_id in sorted(map_ids)
                for variant_id in ctx["variant_ids"]
            },
            "nonzero_high_load_resets_by_map": dict(sorted(per_map_high_nonzero.items())),
            "resets_with_at_least_4_conflicts": sum(
                int(row.get("initial_conflicts", 0)) >= 4 for row in valid_rows
            ),
            "resets_with_at_least_16_conflicts": sum(
                int(row.get("initial_conflicts", 0)) >= 16 for row in valid_rows
            ),
            "maps_with_a_reset_at_least_4_conflicts": sorted(
                {
                    str(row["map_id"])
                    for row in valid_rows
                    if int(row.get("initial_conflicts", 0)) >= 4
                }
            ),
            "maps_with_a_reset_at_least_16_conflicts": sorted(
                {
                    str(row["map_id"])
                    for row in valid_rows
                    if int(row.get("initial_conflicts", 0)) >= 16
                }
            ),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "failure_action": "STATE_SUPPLY_FAIL_NO_BACKFILL",
        "outcome_filtering": False,
        "training_authorized": False,
    }
    return audit


def collect_source_episodes(
    config_path: str | Path,
    output: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    workers: int | None = None,
    project_root: str | Path = PROJECT_ROOT,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if dry_run and resume:
        raise ValueError("dry-run and resume are mutually exclusive")
    runner = runner or run_closed_loop_collection
    ctx = load_registered_source_context(config_path, project_root=project_root)
    output_root = Path(output).resolve()
    manifests = _validate_materialized(ctx, output_root)
    plan = _read_json(output_root / "source_plan_report.json")
    schedule_path = output_root / "source_schedule.jsonl"
    effective_workers = _effective_workers(ctx, workers)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("experiment_id") != EXPERIMENT_ID
        or plan.get("status") != "PLANNED"
        or plan.get("training_authorized") is not False
        or int(plan.get("episode_count", -1)) != EXPECTED_EPISODE_COUNT
        or int(plan.get("workers", -1)) != effective_workers
        or plan.get("task_registration_sha256") != ctx["task_registration_sha256"]
        or _sha256(schedule_path) != plan.get("schedule_sha256")
    ):
        raise ValueError("source-v2 collection plan is missing, changed, or uses different workers")

    collection = dict(ctx["source_config"]["collection"])
    executions: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        task_ids = [str(row["task_id"]) for row in manifests[split]]
        jobs = {(task_id, int(seed)) for task_id in task_ids for seed in collection["solver_seeds"]}
        if len(jobs) != 64:
            raise ValueError(f"source-v2 {split} collection is not the exact 32-task x 2-seed product")
        collection_output = output_root / "collection" / split
        executions.append(
            {
                "split": split,
                "jobs": jobs,
                "output": collection_output,
                "common": {
                    "dataset": output_root / str(ctx["source_config"]["source"]["dataset_directory"]),
                    "config_path": output_root / "collection_configs" / f"{split}.json",
                    "output": collection_output,
                    "workers": effective_workers,
                    "trace_format": str(collection["trace_format"]),
                    "controller": "v2-full",
                    "feature_backend": "native",
                    "controller_bundle": ctx["controller_root"],
                    "job_keys": jobs,
                    "cohort_job_keys": jobs,
                    "wall_time_budget_seconds": float(collection["wall_time_budget_seconds"]),
                    "episode_process_timeout_seconds": float(collection["episode_process_timeout_seconds"]),
                    "environment_time_limit_seconds": float(collection["environment_time_limit_seconds"]),
                    "qualification_process_timeout_seconds": float(collection["qualification_process_timeout_seconds"]),
                    "deterministic_pp_replay": True,
                },
            }
        )

    dispatch = [
        {"split": execution["split"], "phase": phase, "workers": effective_workers}
        for phase in ("qualify", "realized_dynamic")
        for execution in executions
    ]
    if dry_run:
        # A source-v2 dry run is a pure preview: it neither invokes the runner
        # nor reserves any file beneath the production output root.
        return {
            "schema": COLLECTION_TRUST_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "DRY_RUN_PREVIEW",
            "task_registration_sha256": ctx["task_registration_sha256"],
            "controller": "v2-full",
            "policy": "realized_dynamic",
            "workers": effective_workers,
            "planned_episode_count": EXPECTED_EPISODE_COUNT,
            "realized_episode_count": 0,
            "qualification_job_count": EXPECTED_EPISODE_COUNT,
            "runner_invoked": False,
            "dispatch": dispatch,
            "outcome_filtering": False,
            "sealed_final_semantic_access": False,
            "training_authorized": False,
        }

    calls: list[dict[str, Any]] = []
    qualification_passed: dict[str, bool] = {}
    qualification_reused: dict[str, bool] = {}
    # This loop must finish for both splits before the first policy call below.
    for execution in executions:
        audit = (
            _qualification_audit(
                ctx,
                execution["split"],
                execution["output"],
                execution["jobs"],
                manifests[execution["split"]],
            )
            if resume
            else None
        )
        if audit is None:
            runner(
                phase="qualify",
                resume=resume,
                qualification_source=None,
                dry_run=False,
                **execution["common"],
            )
            calls.append({"split": execution["split"], "phase": "qualify", "reused": False})
            audit = _qualification_audit(
                ctx,
                execution["split"],
                execution["output"],
                execution["jobs"],
                manifests[execution["split"]],
            )
            if audit is None:
                raise ValueError(f"{execution['split']} qualification did not produce complete artifacts")
            qualification_reused[execution["split"]] = False
        else:
            calls.append({"split": execution["split"], "phase": "qualify", "reused": True})
            qualification_reused[execution["split"]] = True
        _write_exact(
            execution["output"] / "source_v2_qualification_audit.json",
            _json_bytes(audit),
            resume=resume,
        )
        qualification_passed[execution["split"]] = bool(audit["passed"])

    common_report = {
        "schema": COLLECTION_TRUST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "h1_v2_config_sha256": ctx["design_sha256"],
        "h1_v2_registration_report_sha256": ctx["report_sha256"],
        "task_registration_sha256": ctx["task_registration_sha256"],
        "splits": list(ALLOWED_SPLITS),
        "controller": "v2-full",
        "policy": "realized_dynamic",
        "workers": effective_workers,
        "planned_episode_count": EXPECTED_EPISODE_COUNT,
        "qualification_job_count": EXPECTED_EPISODE_COUNT,
        "qualification_passed": qualification_passed,
        "qualification_reused": qualification_reused,
        "outcome_filtering": False,
        "sealed_final_semantic_access": False,
        "training_authorized": False,
    }
    trust_path = output_root / "collection_trust_report__all.json"
    if not all(qualification_passed.values()):
        report = {
            **common_report,
            "status": "STATE_SUPPLY_FAIL_NO_BACKFILL",
            "realized_episode_count": 0,
            "calls": calls,
        }
        _write_exact(trust_path, _json_bytes(report), resume=resume)
        return report

    for execution in executions:
        runner(
            phase="realized_dynamic",
            resume=True,
            dry_run=False,
            **execution["common"],
        )
        calls.append({"split": execution["split"], "phase": "realized_dynamic", "reused": False})
        _manifest_job_product(
            execution["output"] / "realized_dynamic_manifest.jsonl",
            execution["jobs"],
            "realized_dynamic",
        )
    report = {
        **common_report,
        "status": "COLLECTED_EXACT_PRODUCT",
        "realized_episode_count": EXPECTED_EPISODE_COUNT,
        "calls": calls,
    }
    _write_exact(trust_path, _json_bytes(report), resume=resume)
    return report


__all__ = [
    "ALLOWED_SPLITS",
    "COLLECTION_TRUST_SCHEMA",
    "CONFIG_SCHEMA",
    "EXPECTED_EPISODE_COUNT",
    "EXPECTED_TASK_COUNT",
    "EXPERIMENT_ID",
    "MATERIALIZATION_SCHEMA",
    "PLAN_SCHEMA",
    "TASK_REGISTRATION_SCHEMA",
    "VARIANT_IDS",
    "collect_source_episodes",
    "load_registered_source_context",
    "materialize_source_dataset",
    "plan_source_collection",
]
