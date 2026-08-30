from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.stride_hierarchical_ch_source_v2 import (
    _json_bytes,
    _jsonl_bytes,
    _pinned_path,
    _read_json,
    _read_jsonl,
    _relative_materialized_path,
    _resolve_project_path,
    _sha256,
    _write_exact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "lns2.stride.hierarchical_ch_compact_flow_source_config.v1"
MATERIALIZATION_SCHEMA = "lns2.stride.hierarchical_ch_compact_flow_source_materialization.v1"
TASK_MATERIALIZER_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_task_materialization.v1"
)
PLAN_SCHEMA = "lns2.stride.hierarchical_ch_compact_flow_source_plan.v1"
QUALIFICATION_AUDIT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_source_qualification_audit.v1"
)
COLLECTION_TRUST_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_source_collection_trust.v1"
)
EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_source_v1"
BALANCED_CONFIG_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_balanced_source_config.v1"
)
BALANCED_MATERIALIZATION_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_balanced_source_materialization.v1"
)
BALANCED_PLAN_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_balanced_source_plan.v1"
)
BALANCED_QUALIFICATION_AUDIT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_balanced_source_qualification_audit.v1"
)
BALANCED_COLLECTION_TRUST_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_balanced_source_collection_trust.v1"
)
BALANCED_EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_balanced_source_v1"
BALANCED_GATE_PROFILE = {
    "schema": "lns2.stride.hierarchical_ch_compact_flow_gate_profile.v1",
    "profile_id": "split_level_balanced_od_coverage_v1",
    "coverage_unit": "split",
    "minimum_nonzero_states": 96,
    "minimum_nonzero_resets_per_map": 6,
    "minimum_nonzero_high_load_resets_per_map": 2,
    "minimum_nonzero_resets_per_od_variant": 40,
    "minimum_active_maps_per_od_variant": 7,
    "minimum_resets_with_at_least_4_conflicts": 48,
    "minimum_maps_with_a_reset_at_least_4_conflicts": 8,
    "minimum_resets_with_at_least_16_conflicts": 24,
    "minimum_maps_with_a_reset_at_least_16_conflicts": 5,
    "require_all_capacity_strata_active": True,
}
ALLOWED_SPLITS = ("train", "development")
EXPECTED_MAPS_PER_SPLIT = 8
EXPECTED_TASKS_PER_SPLIT = 96
EXPECTED_TASK_COUNT = 192
EXPECTED_EPISODES_PER_SPLIT = 192
EXPECTED_EPISODE_COUNT = 384
SOLVER_SEEDS = (41, 42)
DEFAULT_WORKERS = 16
MAXIMUM_WORKERS = 20
PARENT_V2_IMPORTED_QUALIFICATION_ROWS = 0


Materializer = Callable[..., dict[str, Any]]


def _gate_profile_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _source_profile(config: dict[str, Any]) -> dict[str, Any]:
    identity = (str(config.get("schema", "")), str(config.get("experiment_id", "")))
    if identity == (CONFIG_SCHEMA, EXPERIMENT_ID):
        if config.get("qualification_gate_profile") is not None:
            raise ValueError("legacy compact-flow source cannot change its qualification gates")
        return {
            "balanced": False,
            "experiment_id": EXPERIMENT_ID,
            "config_schema": CONFIG_SCHEMA,
            "materialization_schema": MATERIALIZATION_SCHEMA,
            "plan_schema": PLAN_SCHEMA,
            "qualification_audit_schema": QUALIFICATION_AUDIT_SCHEMA,
            "collection_trust_schema": COLLECTION_TRUST_SCHEMA,
            "gate_profile": None,
            "gate_profile_sha256": None,
        }
    if identity == (BALANCED_CONFIG_SCHEMA, BALANCED_EXPERIMENT_ID):
        gate_profile = dict(config.get("qualification_gate_profile") or {})
        if gate_profile != BALANCED_GATE_PROFILE:
            raise ValueError("balanced compact-flow qualification gate profile changed")
        return {
            "balanced": True,
            "experiment_id": BALANCED_EXPERIMENT_ID,
            "config_schema": BALANCED_CONFIG_SCHEMA,
            "materialization_schema": BALANCED_MATERIALIZATION_SCHEMA,
            "plan_schema": BALANCED_PLAN_SCHEMA,
            "qualification_audit_schema": BALANCED_QUALIFICATION_AUDIT_SCHEMA,
            "collection_trust_schema": BALANCED_COLLECTION_TRUST_SCHEMA,
            "gate_profile": gate_profile,
            "gate_profile_sha256": _gate_profile_sha256(gate_profile),
        }
    raise ValueError("unsupported compact-flow source config or experiment identity")


def _ctx_source_profile(ctx: dict[str, Any]) -> dict[str, Any]:
    """Return the sealed profile, with a legacy default for existing test adapters."""

    value = ctx.get("source_profile")
    if value is not None:
        return dict(value)
    return _source_profile({"schema": CONFIG_SCHEMA, "experiment_id": EXPERIMENT_ID})


def _pin_pair(value: dict[str, Any]) -> tuple[str, str]:
    return str(value.get("path", "")), str(value.get("sha256", "")).lower()


def _split_value(value: Any, split: str) -> Any:
    return dict(value)[split] if isinstance(value, dict) else value


def _load_registration_product(registration: dict[str, Any]) -> dict[str, Any]:
    """Normalize the compact-flow registration's exact static task product."""

    task_product = dict(registration.get("task_product") or {})
    od_variants = tuple(map(str, task_product.get("od_variants") or ()))
    capacity_spec = dict(registration.get("capacity_strata") or {})
    capacity_strata = tuple(
        name for name in ("ultra", "small", "medium", "large") if name in capacity_spec
    )
    map_registration = {
        str(map_id): dict(value)
        for map_id, value in dict(registration.get("map_registration") or {}).items()
    }
    per_map_loads = {
        map_id: tuple(map(int, row.get("loads") or ()))
        for map_id, row in map_registration.items()
    }
    per_map_task_seeds = {
        map_id: tuple(map(int, row.get("task_seeds") or ()))
        for map_id, row in map_registration.items()
    }
    map_capacity_strata = {
        map_id: str(row.get("capacity_stratum", ""))
        for map_id, row in map_registration.items()
    }
    if od_variants != ("opposite_exchange", "uniform_random"):
        raise ValueError("compact-flow registration OD variants changed")
    if len(capacity_strata) != 4 or len(set(capacity_strata)) != 4:
        raise ValueError("compact-flow registration must contain exactly four capacity strata")
    if any(
        len(loads) != 3 or len(set(loads)) != 3 or min(loads) <= 0
        for loads in per_map_loads.values()
    ):
        raise ValueError("compact-flow registration must contain three static loads per map")
    if any(len(seeds) != 2 or len(set(seeds)) != 2 for seeds in per_map_task_seeds.values()):
        raise ValueError("compact-flow registration must contain two task seeds per map")
    if any(value not in capacity_strata for value in map_capacity_strata.values()):
        raise ValueError("compact-flow map capacity stratum differs from registration")
    for map_id, row in map_registration.items():
        count = int(row.get("capacity_free_cells", -1))
        bounds = dict(capacity_spec[map_capacity_strata[map_id]])
        minimum = int(bounds["minimum_inclusive"])
        maximum_raw = bounds.get("maximum_inclusive")
        maximum = None if maximum_raw is None else int(maximum_raw)
        if count < minimum or (maximum is not None and count > maximum):
            raise ValueError(f"compact-flow capacity stratum bounds mismatch: {map_id}")
    if (
        int(task_product.get("tasks_per_map", -1)) != 12
        or int(task_product.get("tasks_per_split", -1)) != EXPECTED_TASKS_PER_SPLIT
        or int(task_product.get("episodes_per_split", -1)) != EXPECTED_EPISODES_PER_SPLIT
        or list(map(int, task_product.get("solver_seeds") or ())) != list(SOLVER_SEEDS)
        or int(task_product.get("workers", -1)) != DEFAULT_WORKERS
    ):
        raise ValueError("compact-flow task product contract changed")
    return {
        "od_variants": od_variants,
        "capacity_strata": capacity_strata,
        "per_map_loads": per_map_loads,
        "per_map_task_seeds": per_map_task_seeds,
        "map_capacity_strata": map_capacity_strata,
    }


def load_registered_source_context(
    config_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = (root / path).resolve()
    config = _read_json(path)
    source_profile = _source_profile(config)
    claim = dict(config.get("claim_boundary") or {})
    if (
        claim.get("training_authorized") is not False
        or claim.get("outcome_filtering_allowed") is not False
        or claim.get("reserve_or_replacement_backfill_allowed") is not False
        or claim.get("sealed_final_semantic_access_authorized") is not False
        or int(claim.get("source_v2_qualification_rows_imported", -1))
        != PARENT_V2_IMPORTED_QUALIFICATION_ROWS
    ):
        raise ValueError("compact-flow source must be no-train, no-backfill, and final-sealed")

    registration_pin = dict(config.get("registration") or {})
    registration_path = _pinned_path(
        root, dict(registration_pin.get("config") or {}), "compact-flow registration"
    )
    report_path = _pinned_path(
        root, dict(registration_pin.get("report") or {}), "compact-flow registration report"
    )
    registration = _read_json(registration_path)
    report = _read_json(report_path)
    registration_sha = _sha256(registration_path)
    report_sha = _sha256(report_path)
    if (
        registration.get("schema") != registration_pin.get("required_config_schema")
        or registration.get("experiment_id") != EXPERIMENT_ID
        or registration.get("status") != "REGISTERED"
        or registration.get("claim_boundary", {}).get("training_authorized") is not False
        or report.get("schema") != registration_pin.get("required_report_schema")
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("status") != "REGISTERED"
        or report.get("passed") is not True
        or report.get("config_sha256") != registration_sha
        or report.get("training_authorized") is not False
    ):
        raise ValueError("compact-flow registration/report identity is invalid")
    map_splits = {
        split: list(map(str, dict(registration["map_splits"])[split]))
        for split in ALLOWED_SPLITS
    }
    if (
        any(len(map_splits[split]) != EXPECTED_MAPS_PER_SPLIT for split in ALLOWED_SPLITS)
        or set(map_splits["train"]) & set(map_splits["development"])
        or any(name not in ALLOWED_SPLITS for name in dict(registration["map_splits"]))
    ):
        raise ValueError("compact-flow registration must contain exact disjoint train/dev maps only")
    product = _load_registration_product(registration)
    allowed_ids = {map_id for split in ALLOWED_SPLITS for map_id in map_splits[split]}
    if set(product["per_map_loads"]) != allowed_ids:
        raise ValueError("compact-flow per-map loads differ from the registered map product")
    required_capacity_counts = {
        split: {str(name): int(value) for name, value in dict(values).items()}
        for split, values in dict(registration.get("required_capacity_counts") or {}).items()
    }
    for split in ALLOWED_SPLITS:
        observed_capacity_counts = collections.Counter(
            product["map_capacity_strata"][map_id] for map_id in map_splits[split]
        )
        if (
            set(observed_capacity_counts) != set(product["capacity_strata"])
            or dict(observed_capacity_counts) != required_capacity_counts.get(split)
        ):
            raise ValueError(f"compact-flow {split} capacity-stratum replication changed")
    registration_task_product = dict(registration.get("task_product") or {})
    report_task_product = dict(report.get("task_product") or {})
    if (
        int(registration.get("expected_task_count", -1)) != EXPECTED_TASK_COUNT
        or int(registration.get("expected_episode_count", -1)) != EXPECTED_EPISODE_COUNT
        or int(report_task_product.get("registered_task_count", -1))
        != EXPECTED_TASK_COUNT
        or int(report_task_product.get("expected_episode_count", -1))
        != EXPECTED_EPISODE_COUNT
        or report_task_product != registration_task_product
        or list(map(int, registration.get("solver_seeds") or ()))
        != list(SOLVER_SEEDS)
        or int(registration.get("workers", -1)) != DEFAULT_WORKERS
    ):
        raise ValueError("compact-flow registration is not the exact 192-task/384-episode design")
    registered_task_pin = dict(registration.get("registered_task_manifest") or {})
    registered_task_path = _pinned_path(
        root, registered_task_pin, "compact-flow registered task manifest"
    )
    registered_task_rows = _read_jsonl(registered_task_path)
    report_task_pin = dict(report.get("registered_task_manifest") or {})
    if (
        len(registered_task_rows) != EXPECTED_TASK_COUNT
        or int(registered_task_pin.get("row_count", -1)) != EXPECTED_TASK_COUNT
        or report_task_pin != registered_task_pin
        or any(
            row.get("schema") != registered_task_pin.get("schema")
            or str(row.get("split")) not in ALLOWED_SPLITS
            for row in registered_task_rows
        )
        or len({str(row["task_id"]) for row in registered_task_rows})
        != EXPECTED_TASK_COUNT
    ):
        raise ValueError("compact-flow registered task manifest integrity failed")
    registered_tasks_by_split = {
        split: [row for row in registered_task_rows if str(row["split"]) == split]
        for split in ALLOWED_SPLITS
    }
    for split in ALLOWED_SPLITS:
        if len(registered_tasks_by_split[split]) != EXPECTED_TASKS_PER_SPLIT:
            raise ValueError(f"compact-flow registered {split} task count changed")
    recovery = dict(registration.get("parent_source_v2_terminal_failure") or {})
    if int(recovery.get("completed_v2_reset_rows_imported", -1)) != 0:
        raise ValueError("compact-flow registration must import zero source-v2 qualification rows")
    if recovery.get("full_schedule_restart") is not True:
        raise ValueError("compact-flow registration must restart its full schedule")

    materializer = dict(config.get("task_materializer") or {})
    materializer_path = _pinned_path(
        root, dict(materializer.get("implementation") or {}), "compact-flow materializer"
    )
    if (
        materializer.get("module")
        != "experiments.stride_hierarchical_ch_compact_flow_tasks_v1"
        or materializer.get("function") != "materialize_compact_flow_tasks"
        or materializer.get("algorithm_schema") != TASK_MATERIALIZER_SCHEMA
    ):
        raise ValueError("compact-flow materializer identity changed")
    collection = dict(config.get("collection") or {})
    base_config_path = _pinned_path(
        root, dict(collection.get("base_config") or {}), "compact-flow collection base"
    )
    controller_pin = dict(collection.get("controller_bundle") or {})
    controller_root = _resolve_project_path(root, str(controller_pin.get("path", "")))
    controller_manifest = controller_root / "controller_manifest.json"
    if (
        not controller_manifest.is_file()
        or _sha256(controller_manifest)
        != str(controller_pin.get("manifest_sha256", "")).lower()
    ):
        raise ValueError("compact-flow v2-full controller bundle SHA256 mismatch")
    if (
        collection.get("controller") != "v2-full"
        or collection.get("executed_policy") != "realized_dynamic"
        or list(map(int, collection.get("solver_seeds") or ())) != list(SOLVER_SEEDS)
        or int(collection.get("workers", -1)) != DEFAULT_WORKERS
        or int(collection.get("maximum_workers", -1)) != MAXIMUM_WORKERS
        or int(collection.get("expected_episode_count", -1)) != EXPECTED_EPISODE_COUNT
        or collection.get("deterministic_pp_replay") is not True
        or collection.get("feature_backend") != "native"
    ):
        raise ValueError("compact-flow v2-full collection contract changed")
    source = dict(config.get("source") or {})
    if (
        tuple(source.get("allowed_splits") or ()) != ALLOWED_SPLITS
        or int(source.get("expected_task_count", -1)) != EXPECTED_TASK_COUNT
        or int(source.get("expected_tasks_per_split", -1)) != EXPECTED_TASKS_PER_SPLIT
    ):
        raise ValueError("compact-flow source materialization contract changed")
    return {
        "project_root": root,
        "source_config_path": path,
        "source_config": config,
        "registration_path": registration_path,
        "registration": registration,
        "registration_sha256": registration_sha,
        "registration_report_path": report_path,
        "registration_report": report,
        "registration_report_sha256": report_sha,
        "materializer_path": materializer_path,
        "registered_task_path": registered_task_path,
        "registered_task_sha256": _sha256(registered_task_path),
        "registered_tasks_by_split": registered_tasks_by_split,
        "base_config_path": base_config_path,
        "base_config": _read_json(base_config_path),
        "controller_root": controller_root,
        "map_splits": map_splits,
        "allowed_ids": allowed_ids,
        "source_profile": source_profile,
        **product,
    }


def _default_materializer(**kwargs: Any) -> dict[str, Any]:
    from experiments.stride_hierarchical_ch_compact_flow_tasks_v1 import (
        materialize_compact_flow_tasks,
    )

    return materialize_compact_flow_tasks(**kwargs)


def _manifest_row_product(row: dict[str, Any]) -> tuple[str, str, int, int]:
    od_variant = str(
        row.get("od_variant_id")
        or row.get("od_variant")
        or row.get("compact_flow_variant")
        or ""
    )
    task_seed = int(row.get("task_seed", row.get("od_seed", -1)))
    return str(row["map_id"]), od_variant, task_seed, int(row["agent_count"])


def _validate_manifest_rows(
    ctx: dict[str, Any], dataset_root: Path, split: str, rows: list[dict[str, Any]]
) -> None:
    if len(rows) != EXPECTED_TASKS_PER_SPLIT:
        raise ValueError(f"compact-flow {split} manifest is not the exact 96-task product")
    expected = {
        (map_id, od_variant, int(task_seed), int(load))
        for map_id in ctx["map_splits"][split]
        for od_variant in ctx["od_variants"]
        for task_seed in ctx["per_map_task_seeds"][map_id]
        for load in ctx["per_map_loads"][map_id]
    }
    observed = {_manifest_row_product(row) for row in rows}
    if observed != expected or len(observed) != len(rows):
        raise ValueError(f"compact-flow {split} manifest differs from registration")
    observed_task_ids = {str(row["task_id"]) for row in rows}
    registered_task_ids = {
        str(row["task_id"]) for row in ctx["registered_tasks_by_split"][split]
    }
    if (
        len(observed_task_ids) != len(rows)
        or observed_task_ids != registered_task_ids
    ):
        raise ValueError(f"compact-flow {split} task IDs are not unique")
    split_root = dataset_root / split
    for row in rows:
        map_id = str(row["map_id"])
        if (
            map_id not in ctx["map_splits"][split]
            or str(row.get("split")) != split
            or row.get("training_authorized") is not False
            or str(row.get("capacity_stratum")) != ctx["map_capacity_strata"][map_id]
        ):
            raise ValueError(f"compact-flow {split} manifest violates split/no-train contract")
        for field, hash_field in (
            ("map_file", "map_sha256"),
            ("scenario_file", "scenario_sha256"),
            ("task_file", "task_sha256"),
            ("map_metadata_file", "map_metadata_sha256"),
        ):
            relative, materialized = _relative_materialized_path(split_root, row[field], field)
            if relative != str(row[field]).replace("\\", "/"):
                raise ValueError(f"compact-flow manifest path is not canonical: {row[field]}")
            if _sha256(materialized) != str(row[hash_field]).lower():
                raise ValueError(f"compact-flow materialized hash mismatch: {row['task_id']}/{field}")


def _validate_materialized(
    ctx: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    profile = _ctx_source_profile(ctx)
    trust = _read_json(output_root / "materialization_trust_report.json")
    if (
        trust.get("schema") != profile["materialization_schema"]
        or trust.get("experiment_id") != profile["experiment_id"]
        or trust.get("status") != "MATERIALIZED"
        or trust.get("registration_sha256") != ctx["registration_sha256"]
        or trust.get("registration_report_sha256") != ctx["registration_report_sha256"]
        or trust.get("training_authorized") is not False
    ):
        raise ValueError("compact-flow orchestration materialization trust is invalid")
    dataset_root = Path(str(trust["dataset_root"])).resolve()
    if dataset_root != (output_root / "dataset").resolve():
        raise ValueError("compact-flow dataset root differs from the output fingerprint")
    materializer_report_path = output_root / "compact_flow_materialization_report.json"
    if (
        not materializer_report_path.is_file()
        or _sha256(materializer_report_path) != trust.get("materializer_report_sha256")
        or int(trust.get("task_count", -1)) != EXPECTED_TASK_COUNT
    ):
        raise ValueError("compact-flow materializer report pin is invalid")
    manifests: dict[str, list[dict[str, Any]]] = {}
    for split in ALLOWED_SPLITS:
        manifest_path = dataset_root / split / "manifest.jsonl"
        expected_hash = str(trust["split_manifests"][split]["sha256"])
        if (
            Path(str(trust["split_manifests"][split]["path"])).resolve()
            != manifest_path.resolve()
            or int(trust["split_manifests"][split].get("task_count", -1))
            != EXPECTED_TASKS_PER_SPLIT
            or _sha256(manifest_path) != expected_hash
        ):
            raise ValueError(f"compact-flow {split} manifest SHA256 mismatch")
        rows = _read_jsonl(manifest_path)
        _validate_manifest_rows(ctx, dataset_root, split, rows)
        manifests[split] = rows
    return trust, manifests


def materialize_source_dataset(
    config_path: str | Path,
    output: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    project_root: str | Path = PROJECT_ROOT,
    materializer: Materializer | None = None,
) -> dict[str, Any]:
    if dry_run and resume:
        raise ValueError("dry-run and resume are mutually exclusive")
    ctx = load_registered_source_context(config_path, project_root=project_root)
    output_root = Path(output).resolve()
    profile = _ctx_source_profile(ctx)
    preview = {
        "schema": profile["materialization_schema"],
        "experiment_id": profile["experiment_id"],
        "status": "DRY_RUN" if dry_run else "MATERIALIZED",
        "registration_sha256": ctx["registration_sha256"],
        "registration_report_sha256": ctx["registration_report_sha256"],
        "expected_task_count": EXPECTED_TASK_COUNT,
        "split_task_counts": {split: EXPECTED_TASKS_PER_SPLIT for split in ALLOWED_SPLITS},
        "source_v2_qualification_rows_imported": 0,
        "sealed_final_semantic_access": False,
        "training_authorized": False,
    }
    if profile["balanced"]:
        preview.update(
            {
                "qualification_gate_profile": profile["gate_profile"],
                "qualification_gate_profile_sha256": profile[
                    "gate_profile_sha256"
                ],
            }
        )
    if dry_run:
        return preview
    materializer = materializer or _default_materializer
    report = materializer(
        config_path=ctx["registration_path"],
        output_root=output_root,
        dry_run=False,
        resume=resume,
        project_root=ctx["project_root"],
    )
    materializer_report_path = output_root / "compact_flow_materialization_report.json"
    if not materializer_report_path.is_file() or report != _read_json(materializer_report_path):
        raise ValueError("compact-flow materializer report is missing or differs from its return value")
    if (
        report.get("schema") != TASK_MATERIALIZER_SCHEMA
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("status") != "MATERIALIZED"
        or int(report.get("task_count", -1)) != EXPECTED_TASK_COUNT
        or report.get("config_sha256") != ctx["registration_sha256"]
        or report.get("registration_report_sha256") != ctx["registration_report_sha256"]
        or report.get("registered_task_manifest_sha256")
        != ctx["registered_task_sha256"]
        or Path(str(report.get("dataset_root", ""))).resolve()
        != (output_root / "dataset").resolve()
        or report.get("sealed_final_semantic_access") is not False
        or report.get("training_authorized") is not False
    ):
        raise ValueError("compact-flow materializer did not produce the registered product")
    dataset_root = Path(str(report["dataset_root"])).resolve()
    split_manifests: dict[str, dict[str, Any]] = {}
    for split in ALLOWED_SPLITS:
        manifest_path = dataset_root / split / "manifest.jsonl"
        rows = _read_jsonl(manifest_path)
        _validate_manifest_rows(ctx, dataset_root, split, rows)
        if (
            int(report["split_manifests"][split]["task_count"]) != EXPECTED_TASKS_PER_SPLIT
            or _sha256(manifest_path) != report["split_manifests"][split]["sha256"]
        ):
            raise ValueError(f"compact-flow materializer {split} report differs from manifest")
        split_manifests[split] = {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "task_count": len(rows),
        }
    preview.update(
        {
            "dataset_root": str(dataset_root),
            "task_count": EXPECTED_TASK_COUNT,
            "materializer_report": str(materializer_report_path),
            "materializer_report_sha256": _sha256(materializer_report_path),
            "split_manifests": split_manifests,
        }
    )
    _write_exact(
        output_root / "materialization_trust_report.json",
        _json_bytes(preview),
        resume=resume,
    )
    return preview


def _effective_workers(ctx: dict[str, Any], workers: int | None) -> int:
    value = int(
        ctx["source_config"]["collection"]["workers"] if workers is None else workers
    )
    if value <= 0 or value > MAXIMUM_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAXIMUM_WORKERS}")
    return value


def _scenario_indices(rows: list[dict[str, Any]]) -> list[int]:
    values: set[int] = set()
    for row in rows:
        if "scenario_index" in row:
            values.add(int(row["scenario_index"]))
        else:
            try:
                values.add(int(str(row["scenario_type"]).rsplit("_", 1)[-1]))
            except (KeyError, ValueError) as error:
                raise ValueError("compact-flow scenario_type lacks a numeric registered suffix") from error
    if len(values) != 4:
        raise ValueError("compact-flow manifest must contain four OD/task-seed scenario indices")
    return sorted(values)


def _collection_config(
    ctx: dict[str, Any], split: str, rows: list[dict[str, Any]], *, workers: int
) -> dict[str, Any]:
    config = json.loads(json.dumps(ctx["base_config"]))
    collection = dict(ctx["source_config"]["collection"])
    maps = list(ctx["map_splits"][split])
    layout_by_map: dict[str, str] = {}
    for map_id in maps:
        layouts = {str(row["layout_mode"]) for row in rows if str(row["map_id"]) == map_id}
        if len(layouts) != 1:
            raise ValueError(f"compact-flow {map_id} has a non-unique layout family")
        layout_by_map[map_id] = next(iter(layouts))
    family_counts = collections.Counter(layout_by_map.values())
    required_families = sorted(family_counts)
    qualification = {
        "mode": "movingai_ood",
        "minimum_nonzero_states": 96,
        "minimum_active_maps": 8,
        "required_layout_families": required_families,
    }
    profile = _ctx_source_profile(ctx)
    if profile["balanced"]:
        qualification.update(
            {
                "source_gate_profile": profile["gate_profile"],
                "source_gate_profile_sha256": profile["gate_profile_sha256"],
            }
        )
    config.update(
        {
            "formal": True,
            "split": split,
            "solver_seeds": list(SOLVER_SEEDS),
            "policies": list(collection["policies_in_config"]),
            "dataset_design": {
                "mode": "movingai_ood",
                "map_count": EXPECTED_MAPS_PER_SPLIT,
                "task_count": EXPECTED_TASKS_PER_SPLIT,
                "scenario_indices": _scenario_indices(rows),
                "layout_family_counts": dict(sorted(family_counts.items())),
                "maps": [
                    {
                        "map_id": map_id,
                        "layout_family": layout_by_map[map_id],
                        "agent_counts": list(ctx["per_map_loads"][map_id]),
                    }
                    for map_id in maps
                ],
                "historical_map_ids": list(
                    ctx["registration"].get("legacy_h1_exclusion", {}).get("map_ids")
                    or ()
                ),
            },
            "environment": {
                **dict(config["environment"]),
                "time_limit": float(collection["environment_time_limit_seconds"]),
            },
            "qualification": qualification,
            "max_decisions": int(collection["max_decisions"]),
            "metric_iteration_budget": int(collection["max_decisions"]),
            "wall_time_budget_seconds": float(collection["wall_time_budget_seconds"]),
            "episode_process_timeout_seconds": float(
                collection["episode_process_timeout_seconds"]
            ),
            "workers": workers,
            "deterministic_pp_replay": True,
            "reference_datasets": [],
            "scientific_status": "compact_flow_source_training_unauthorized",
            "training_authorized": False,
        }
    )
    return config


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
    trust, manifests = _validate_materialized(ctx, output_root)
    effective_workers = _effective_workers(ctx, workers)
    profile = _ctx_source_profile(ctx)
    schedule = [
        {
            "split": split,
            "task_id": str(row["task_id"]),
            "map_id": str(row["map_id"]),
            "od_variant": _manifest_row_product(row)[1],
            "capacity_stratum": str(row["capacity_stratum"]),
            "agent_count": int(row["agent_count"]),
            "task_seed": int(row.get("task_seed", row.get("od_seed"))),
            "solver_seed": int(solver_seed),
            "workers": effective_workers,
            "controller": "v2-full",
            "policy": "realized_dynamic",
            "qualification_required": True,
            "source_v2_qualification_imported": False,
            "training_authorized": False,
        }
        for split in ALLOWED_SPLITS
        for row in manifests[split]
        for solver_seed in SOLVER_SEEDS
    ]
    schedule.sort(
        key=lambda row: (str(row["split"]), str(row["task_id"]), int(row["solver_seed"]))
    )
    if profile["balanced"]:
        for row in schedule:
            row.update(
                {
                    "qualification_gate_profile_id": profile["gate_profile"][
                        "profile_id"
                    ],
                    "qualification_gate_profile_sha256": profile[
                        "gate_profile_sha256"
                    ],
                }
            )
    keys = {(row["split"], row["task_id"], row["solver_seed"]) for row in schedule}
    if len(schedule) != EXPECTED_EPISODE_COUNT or len(keys) != EXPECTED_EPISODE_COUNT:
        raise ValueError("compact-flow schedule is not the exact 384-job product")
    report = {
        "schema": profile["plan_schema"],
        "experiment_id": profile["experiment_id"],
        "status": "DRY_RUN" if dry_run else "PLANNED",
        "source_config_sha256": _sha256(ctx["source_config_path"]),
        "registration_sha256": ctx["registration_sha256"],
        "registration_report_sha256": ctx["registration_report_sha256"],
        "materialization_trust_sha256": _sha256(
            output_root / "materialization_trust_report.json"
        ),
        "materializer_report_sha256": trust["materializer_report_sha256"],
        "workers": effective_workers,
        "solver_seeds": list(SOLVER_SEEDS),
        "task_count": EXPECTED_TASK_COUNT,
        "episode_count": EXPECTED_EPISODE_COUNT,
        "split_episode_counts": {
            split: EXPECTED_EPISODES_PER_SPLIT for split in ALLOWED_SPLITS
        },
        "source_v2_qualification_rows_imported": 0,
        "outcome_filtering": False,
        "sealed_final_semantic_access": False,
        "training_authorized": False,
    }
    if profile["balanced"]:
        fingerprint_payload = {
            "experiment_id": profile["experiment_id"],
            "source_config_sha256": report["source_config_sha256"],
            "registration_sha256": report["registration_sha256"],
            "registration_report_sha256": report["registration_report_sha256"],
            "materialization_trust_sha256": report[
                "materialization_trust_sha256"
            ],
            "workers": effective_workers,
            "solver_seeds": list(SOLVER_SEEDS),
            "episode_count": EXPECTED_EPISODE_COUNT,
            "qualification_gate_profile_sha256": profile[
                "gate_profile_sha256"
            ],
        }
        report.update(
            {
                "qualification_gate_profile": profile["gate_profile"],
                "qualification_gate_profile_sha256": profile[
                    "gate_profile_sha256"
                ],
                "plan_fingerprint": hashlib.sha256(
                    _json_bytes(fingerprint_payload)
                ).hexdigest(),
                "plan_fingerprint_payload": fingerprint_payload,
            }
        )
    if dry_run:
        return {**report, "schedule": schedule}
    for split in ALLOWED_SPLITS:
        _write_exact(
            output_root / "collection_configs" / f"{split}.json",
            _json_bytes(
                _collection_config(ctx, split, manifests[split], workers=effective_workers)
            ),
            resume=resume,
        )
    schedule_path = output_root / "source_schedule.jsonl"
    _write_exact(schedule_path, _jsonl_bytes(schedule), resume=resume)
    report.update(
        {"schedule_path": str(schedule_path), "schedule_sha256": _sha256(schedule_path)}
    )
    _write_exact(output_root / "source_plan_report.json", _json_bytes(report), resume=resume)
    return report


def _manifest_job_product(
    path: Path, expected: set[tuple[str, int]], label: str
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    observed = {(str(row["task_id"]), int(row["solver_seed"])) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"{label} manifest differs from the exact task/seed product")
    return rows


def _validate_schedule(
    ctx: dict[str, Any],
    rows: list[dict[str, Any]],
    manifests: dict[str, list[dict[str, Any]]],
    *,
    workers: int,
) -> None:
    profile = _ctx_source_profile(ctx)
    task_by_key = {
        (split, str(row["task_id"])): row
        for split in ALLOWED_SPLITS
        for row in manifests[split]
    }
    expected = {
        (split, task_id, solver_seed)
        for split, task_id in task_by_key
        for solver_seed in SOLVER_SEEDS
    }
    observed = {
        (str(row.get("split")), str(row.get("task_id")), int(row.get("solver_seed", -1)))
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("compact-flow schedule differs from the exact task/seed product")
    for row in rows:
        split = str(row["split"])
        task = task_by_key[(split, str(row["task_id"]))]
        if (
            str(row.get("map_id")) != str(task["map_id"])
            or str(row.get("od_variant")) != _manifest_row_product(task)[1]
            or str(row.get("capacity_stratum")) != str(task["capacity_stratum"])
            or int(row.get("agent_count", -1)) != int(task["agent_count"])
            or int(row.get("task_seed", -1)) != _manifest_row_product(task)[2]
            or int(row.get("workers", -1)) != workers
            or row.get("controller") != "v2-full"
            or row.get("policy") != "realized_dynamic"
            or row.get("qualification_required") is not True
            or row.get("source_v2_qualification_imported") is not False
            or row.get("training_authorized") is not False
            or (
                profile["balanced"]
                and (
                    row.get("qualification_gate_profile_id")
                    != profile["gate_profile"]["profile_id"]
                    or row.get("qualification_gate_profile_sha256")
                    != profile["gate_profile_sha256"]
                )
            )
        ):
            raise ValueError("compact-flow schedule metadata or no-import boundary changed")


def _qualification_audit(
    ctx: dict[str, Any],
    split: str,
    output: Path,
    jobs: set[tuple[str, int]],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    source_profile = _ctx_source_profile(ctx)
    manifest_path = output / "qualification_manifest.jsonl"
    report_path = output / "qualification_report.json"
    run_config_path = output / "run_config.json"
    if not (manifest_path.is_file() and report_path.is_file() and run_config_path.is_file()):
        return None
    rows = _manifest_job_product(manifest_path, jobs, "compact-flow qualification")
    report = _read_json(report_path)
    run_config = _read_json(run_config_path)
    if type(report.get("passed")) is not bool:
        raise ValueError("compact-flow qualification report lacks passed=true/false")
    if not str(run_config.get("run_fingerprint", "")):
        raise ValueError("compact-flow qualification lacks a run fingerprint")
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
    valid = [
        row
        for row in rows
        if str(row.get("status")) == "ok" and row.get("error") in (None, "")
    ]
    complete = [row for row in valid if bool(row.get("initial_complete", False))]
    consistent = [
        row
        for row in valid
        if bool(row.get("initial_feasible", False))
        == (bool(row.get("initial_complete", False)) and int(row.get("initial_conflicts", -1)) == 0)
    ]
    nonzero = [row for row in valid if int(row.get("initial_conflicts", 0)) > 0]
    per_map = collections.Counter(str(row["map_id"]) for row in nonzero)
    per_map_od: collections.Counter[tuple[str, str]] = collections.Counter()
    high_by_map: collections.Counter[str] = collections.Counter()
    strata_active: set[str] = set()
    for row in nonzero:
        task = task_by_id[str(row["task_id"])]
        map_id = str(row["map_id"])
        od_variant = _manifest_row_product(task)[1]
        per_map_od[(map_id, od_variant)] += 1
        if int(row["agent_count"]) == max(ctx["per_map_loads"][map_id]):
            high_by_map[map_id] += 1
        strata_active.add(ctx["map_capacity_strata"][map_id])
    maps_ge4 = {
        str(row["map_id"])
        for row in valid
        if int(row.get("initial_conflicts", 0)) >= 4
    }
    maps_ge16 = {
        str(row["map_id"])
        for row in valid
        if int(row.get("initial_conflicts", 0)) >= 16
    }
    map_ids = set(ctx["map_splits"][split])
    per_od = {
        od_variant: sum(
            per_map_od[(map_id, od_variant)] for map_id in map_ids
        )
        for od_variant in ctx["od_variants"]
    }
    active_maps_per_od = {
        od_variant: {
            map_id
            for map_id in map_ids
            if per_map_od[(map_id, od_variant)] > 0
        }
        for od_variant in ctx["od_variants"]
    }
    gate_profile = (
        source_profile["gate_profile"]
        if source_profile["balanced"]
        else {
            "minimum_nonzero_states": 96,
            "minimum_nonzero_resets_per_map": 6,
            "minimum_nonzero_high_load_resets_per_map": 2,
            "minimum_resets_with_at_least_4_conflicts": 48,
            "minimum_maps_with_a_reset_at_least_4_conflicts": 8,
            "minimum_resets_with_at_least_16_conflicts": 24,
            "minimum_maps_with_a_reset_at_least_16_conflicts": 5,
        }
    )
    gates = {
        "expected_reset_count": len(rows) == EXPECTED_EPISODES_PER_SPLIT,
        "registered_task_metadata": metadata_consistent,
        "all_resets_valid": len(valid) == EXPECTED_EPISODES_PER_SPLIT,
        "all_resets_complete": len(complete) == EXPECTED_EPISODES_PER_SPLIT,
        "all_initial_states_consistent": len(consistent) == EXPECTED_EPISODES_PER_SPLIT,
        "closed_loop_registered_gates": bool(report["passed"]),
        "minimum_nonzero_states": len(nonzero)
        >= int(gate_profile["minimum_nonzero_states"]),
        "minimum_nonzero_resets_per_map": all(
            per_map[map_id]
            >= int(gate_profile["minimum_nonzero_resets_per_map"])
            for map_id in map_ids
        ),
        "minimum_nonzero_high_load_resets_per_map": all(
            high_by_map[map_id]
            >= int(gate_profile["minimum_nonzero_high_load_resets_per_map"])
            for map_id in map_ids
        ),
        "minimum_resets_with_at_least_4_conflicts": sum(
            int(row.get("initial_conflicts", 0)) >= 4 for row in valid
        )
        >= int(gate_profile["minimum_resets_with_at_least_4_conflicts"]),
        "minimum_maps_with_a_reset_at_least_4_conflicts": len(maps_ge4)
        >= int(gate_profile["minimum_maps_with_a_reset_at_least_4_conflicts"]),
        "minimum_resets_with_at_least_16_conflicts": sum(
            int(row.get("initial_conflicts", 0)) >= 16 for row in valid
        )
        >= int(gate_profile["minimum_resets_with_at_least_16_conflicts"]),
        "minimum_maps_with_a_reset_at_least_16_conflicts": len(maps_ge16)
        >= int(gate_profile["minimum_maps_with_a_reset_at_least_16_conflicts"]),
        "all_capacity_strata_active": strata_active == set(ctx["capacity_strata"]),
        "source_v2_qualification_rows_imported_is_zero": True,
        "no_outcome_subset_or_backfill": True,
    }
    if source_profile["balanced"]:
        gates.update(
            {
                "minimum_nonzero_resets_per_od_variant": all(
                    per_od[od_variant]
                    >= int(gate_profile["minimum_nonzero_resets_per_od_variant"])
                    for od_variant in ctx["od_variants"]
                ),
                "minimum_active_maps_per_od_variant": all(
                    len(active_maps_per_od[od_variant])
                    >= int(gate_profile["minimum_active_maps_per_od_variant"])
                    for od_variant in ctx["od_variants"]
                ),
            }
        )
    else:
        gates["minimum_nonzero_resets_per_map_od_variant"] = all(
            per_map_od[(map_id, od_variant)] >= 2
            for map_id in map_ids
            for od_variant in ctx["od_variants"]
        )
    audit = {
        "schema": source_profile["qualification_audit_schema"],
        "experiment_id": source_profile["experiment_id"],
        "split": split,
        "registration_sha256": ctx["registration_sha256"],
        "qualification_manifest_sha256": _sha256(manifest_path),
        "qualification_report_sha256": _sha256(report_path),
        "run_config_sha256": _sha256(run_config_path),
        "run_fingerprint": str(run_config["run_fingerprint"]),
        "observed": {
            "reset_count": len(rows),
            "valid_reset_count": len(valid),
            "complete_reset_count": len(complete),
            "consistent_initial_state_count": len(consistent),
            "nonzero_reset_count": len(nonzero),
            "nonzero_resets_by_map": dict(sorted(per_map.items())),
            "nonzero_resets_by_map_od_variant": {
                f"{map_id}/{od_variant}": per_map_od[(map_id, od_variant)]
                for map_id in sorted(map_ids)
                for od_variant in ctx["od_variants"]
            },
            "nonzero_high_load_resets_by_map": dict(sorted(high_by_map.items())),
            "resets_with_at_least_4_conflicts": sum(
                int(row.get("initial_conflicts", 0)) >= 4 for row in valid
            ),
            "maps_with_a_reset_at_least_4_conflicts": sorted(maps_ge4),
            "resets_with_at_least_16_conflicts": sum(
                int(row.get("initial_conflicts", 0)) >= 16 for row in valid
            ),
            "maps_with_a_reset_at_least_16_conflicts": sorted(maps_ge16),
            "active_capacity_strata": sorted(strata_active),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "failure_action": "STATE_SUPPLY_FAIL_NO_BACKFILL",
        "source_v2_qualification_rows_imported": 0,
        "outcome_filtering": False,
        "training_authorized": False,
    }
    if source_profile["balanced"]:
        audit.update(
            {
                "qualification_gate_profile": gate_profile,
                "qualification_gate_profile_sha256": source_profile[
                    "gate_profile_sha256"
                ],
            }
        )
        audit["observed"].update(
            {
                "nonzero_resets_by_od_variant": dict(sorted(per_od.items())),
                "active_maps_by_od_variant": {
                    od_variant: sorted(active_maps_per_od[od_variant])
                    for od_variant in ctx["od_variants"]
                },
                "active_map_counts_by_od_variant": {
                    od_variant: len(active_maps_per_od[od_variant])
                    for od_variant in ctx["od_variants"]
                },
            }
        )
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
    ctx = load_registered_source_context(config_path, project_root=project_root)
    output_root = Path(output).resolve()
    source_profile = _ctx_source_profile(ctx)
    _trust, manifests = _validate_materialized(ctx, output_root)
    effective_workers = _effective_workers(ctx, workers)
    plan_path = output_root / "source_plan_report.json"
    schedule_path = output_root / "source_schedule.jsonl"
    plan = _read_json(plan_path)
    if (
        plan.get("schema") != source_profile["plan_schema"]
        or plan.get("experiment_id") != source_profile["experiment_id"]
        or plan.get("status") != "PLANNED"
        or plan.get("source_config_sha256") != _sha256(ctx["source_config_path"])
        or int(plan.get("workers", -1)) != effective_workers
        or int(plan.get("episode_count", -1)) != EXPECTED_EPISODE_COUNT
        or plan.get("registration_sha256") != ctx["registration_sha256"]
        or plan.get("registration_report_sha256")
        != ctx["registration_report_sha256"]
        or plan.get("materialization_trust_sha256")
        != _sha256(output_root / "materialization_trust_report.json")
        or _sha256(schedule_path) != plan.get("schedule_sha256")
        or (
            source_profile["balanced"]
            and (
                plan.get("qualification_gate_profile")
                != source_profile["gate_profile"]
                or plan.get("qualification_gate_profile_sha256")
                != source_profile["gate_profile_sha256"]
                or not str(plan.get("plan_fingerprint", ""))
            )
        )
    ):
        raise ValueError("compact-flow collection plan is missing or has a different fingerprint")
    schedule_rows = _read_jsonl(schedule_path)
    _validate_schedule(ctx, schedule_rows, manifests, workers=effective_workers)
    collection = dict(ctx["source_config"]["collection"])
    executions: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        jobs = {
            (str(row["task_id"]), int(seed))
            for row in manifests[split]
            for seed in SOLVER_SEEDS
        }
        if len(jobs) != EXPECTED_EPISODES_PER_SPLIT:
            raise ValueError(f"compact-flow {split} is not the exact 192-job product")
        split_output = output_root / "collection" / split
        executions.append(
            {
                "split": split,
                "jobs": jobs,
                "output": split_output,
                "common": {
                    "dataset": Path(_trust["dataset_root"]),
                    "config_path": output_root / "collection_configs" / f"{split}.json",
                    "output": split_output,
                    "workers": effective_workers,
                    "trace_format": str(collection["trace_format"]),
                    "controller": "v2-full",
                    "feature_backend": "native",
                    "controller_bundle": ctx["controller_root"],
                    "job_keys": jobs,
                    "cohort_job_keys": jobs,
                    "wall_time_budget_seconds": float(
                        collection["wall_time_budget_seconds"]
                    ),
                    "episode_process_timeout_seconds": float(
                        collection["episode_process_timeout_seconds"]
                    ),
                    "environment_time_limit_seconds": float(
                        collection["environment_time_limit_seconds"]
                    ),
                    "qualification_process_timeout_seconds": float(
                        collection["qualification_process_timeout_seconds"]
                    ),
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
        return {
            "schema": source_profile["collection_trust_schema"],
            "experiment_id": source_profile["experiment_id"],
            "status": "DRY_RUN_PREVIEW",
            "registration_sha256": ctx["registration_sha256"],
            "source_schedule_sha256": _sha256(schedule_path),
            "workers": effective_workers,
            "planned_episode_count": EXPECTED_EPISODE_COUNT,
            "realized_episode_count": 0,
            "qualification_job_count": EXPECTED_EPISODE_COUNT,
            "source_v2_qualification_rows_imported": 0,
            "runner_invoked": False,
            "dispatch": dispatch,
            "training_authorized": False,
            **(
                {
                    "qualification_gate_profile": source_profile["gate_profile"],
                    "qualification_gate_profile_sha256": source_profile[
                        "gate_profile_sha256"
                    ],
                    "plan_fingerprint": plan["plan_fingerprint"],
                }
                if source_profile["balanced"]
                else {}
            ),
        }
    runner = runner or run_closed_loop_collection
    calls: list[dict[str, Any]] = []
    qualification_passed: dict[str, bool] = {}
    qualification_reused: dict[str, bool] = {}
    audits: dict[str, dict[str, Any]] = {}
    # Complete and audit both split cohorts before the first policy call.
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
            calls.append(
                {"split": execution["split"], "phase": "qualify", "reused": False}
            )
            audit = _qualification_audit(
                ctx,
                execution["split"],
                execution["output"],
                execution["jobs"],
                manifests[execution["split"]],
            )
            if audit is None:
                raise ValueError(
                    f"compact-flow {execution['split']} qualification is incomplete"
                )
            qualification_reused[execution["split"]] = False
        else:
            calls.append(
                {"split": execution["split"], "phase": "qualify", "reused": True}
            )
            qualification_reused[execution["split"]] = True
        _write_exact(
            execution["output"] / "compact_flow_qualification_audit.json",
            _json_bytes(audit),
            resume=resume,
        )
        audits[execution["split"]] = audit
        qualification_passed[execution["split"]] = bool(audit["passed"])
    common_report = {
        "schema": source_profile["collection_trust_schema"],
        "experiment_id": source_profile["experiment_id"],
        "source_config_sha256": _sha256(ctx["source_config_path"]),
        "registration_sha256": ctx["registration_sha256"],
        "registration_report_sha256": ctx["registration_report_sha256"],
        "source_schedule_sha256": _sha256(schedule_path),
        "workers": effective_workers,
        "splits": list(ALLOWED_SPLITS),
        "controller": "v2-full",
        "policy": "realized_dynamic",
        "planned_episode_count": EXPECTED_EPISODE_COUNT,
        "qualification_job_count": EXPECTED_EPISODE_COUNT,
        "qualification_passed": qualification_passed,
        "qualification_reused": qualification_reused,
        "qualification_run_fingerprints": {
            split: audits[split]["run_fingerprint"] for split in ALLOWED_SPLITS
        },
        "source_v2_qualification_rows_imported": 0,
        "outcome_filtering": False,
        "sealed_final_semantic_access": False,
        "training_authorized": False,
    }
    if source_profile["balanced"]:
        common_report.update(
            {
                "qualification_gate_profile": source_profile["gate_profile"],
                "qualification_gate_profile_sha256": source_profile[
                    "gate_profile_sha256"
                ],
                "plan_fingerprint": plan["plan_fingerprint"],
            }
        )
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
        calls.append(
            {
                "split": execution["split"],
                "phase": "realized_dynamic",
                "reused": False,
            }
        )
        _manifest_job_product(
            execution["output"] / "realized_dynamic_manifest.jsonl",
            execution["jobs"],
            "compact-flow realized_dynamic",
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
    "DEFAULT_WORKERS",
    "EXPECTED_EPISODE_COUNT",
    "EXPECTED_TASK_COUNT",
    "EXPERIMENT_ID",
    "MATERIALIZATION_SCHEMA",
    "MAXIMUM_WORKERS",
    "PLAN_SCHEMA",
    "QUALIFICATION_AUDIT_SCHEMA",
    "TASK_MATERIALIZER_SCHEMA",
    "collect_source_episodes",
    "load_registered_source_context",
    "materialize_source_dataset",
    "plan_source_collection",
]
