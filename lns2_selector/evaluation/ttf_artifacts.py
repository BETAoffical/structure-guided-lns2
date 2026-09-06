"""Read-only validation of completed, checkpoint-restored TTF lane artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from experiments._common import (
    contained_file,
    episode_id,
    json_fingerprint,
    read_json,
    read_jsonl,
    sha256_file,
)
from experiments.closed_loop_trace_storage import storage_fingerprint, trace_file_metadata
from lns2_selector.evaluation.trace_validation import validate_closed_loop_trace
from lns2_selector.evaluation.ttf_metrics import validate_ttf_summary
from lns2_selector.runtime.artifact_validation import finite_number, strict_integer


TTF_ARTIFACT_VALIDATION_SCHEMA = "lns2.ttf_artifact_validation.v1"


def _require_lane(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid TTF lane: {message}")


def _lane_object(value: Any, label: str) -> dict[str, Any]:
    _require_lane(isinstance(value, dict), f"{label} must be an object")
    return value


def _lane_sha(value: Any, label: str) -> str:
    _require_lane(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label} must be a SHA256",
    )
    return value


def load_ttf_lane(
    lane: Path,
    manifest_name: str,
    *,
    item: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    expected_policy: str,
    expected_controller: str,
    expected_augmentation: dict[str, Any] | None,
    wall_time_budget_seconds: float,
    expected_override: dict[str, Any],
    expected_configuration: Mapping[str, Any],
    expected_frozen_models: Mapping[str, Any],
    expected_controller_bundle: Mapping[str, Any],
    expected_runtime: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return validated inputs, or None for missing/failed work.

    Completed but inconsistent artifacts fail closed before a report is written.
    Historical producer hashes are checked internally, not against today's
    source tree: analyzing evidence is not permission to resume an old run.
    """
    lane = Path(lane).resolve()
    manifest_path = lane / manifest_name
    if not manifest_path.is_file():
        return None
    manifest_sha = sha256_file(manifest_path)
    rows = read_jsonl(manifest_path)
    if not rows:
        return None
    _require_lane(len(rows) == 1, "lane manifest must contain exactly one episode")
    row = _lane_object(rows[0], "manifest row")
    for field in ("task_id", "solver_seed"):
        _require_lane(row.get(field) == item[field], f"manifest {field} mismatch")
    _require_lane(strict_integer(row.get("solver_seed"), minimum=0), "invalid seed")
    if row.get("status") not in {"ok", "resumed"}:
        return None

    run_path = contained_file(lane, "run_config.json", field="TTF run config")
    run_sha = sha256_file(run_path)
    run = _lane_object(read_json(run_path), "run config")
    _require_lane(run.get("schema") == "lns2.closed_loop_confirmation.v1", "run schema")
    _require_lane(type(run.get("schema_version")) is int and run["schema_version"] == 1, "run version")
    configuration = _lane_object(run.get("configuration"), "configuration")
    implementation = _lane_object(run.get("controller_implementation"), "producer")
    files = _lane_object(implementation.get("files"), "producer files")
    _require_lane(bool(files), "producer source inventory is empty")
    for name, digest in files.items():
        _lane_sha(digest, f"producer source {name}")
    native = _lane_object(implementation.get("native_module"), "native identity")
    _lane_sha(native.get("sha256"), "native binary")
    for field in ("path", "repair_timing_schema", "native_semantics_schema"):
        _require_lane(isinstance(native.get(field), str) and bool(native[field]), f"native {field}")
    producer_sha = _lane_sha(implementation.get("sha256"), "producer identity")
    _require_lane(
        producer_sha == json_fingerprint({"files": files, "native_module": native}),
        "producer fingerprint mismatch",
    )
    _require_lane(run.get("configuration_fingerprint") == json_fingerprint(configuration), "configuration fingerprint mismatch")
    dataset_sha = _lane_sha(run.get("dataset_fingerprint"), "dataset fingerprint")
    for field in ("frozen_models", "controller_bundle"):
        _lane_object(run.get(field), field)
    _require_lane(json_fingerprint(run["frozen_models"]) == json_fingerprint(expected_frozen_models), "registered frozen_models mismatch")
    _require_lane(json_fingerprint(run["controller_bundle"]) == json_fingerprint(expected_controller_bundle), "registered controller_bundle mismatch")
    _require_lane("v3_s3_bundle" in run, "missing V3 bundle identity")
    run_fingerprint = json_fingerprint({
        "dataset_fingerprint": dataset_sha,
        "configuration_fingerprint": run["configuration_fingerprint"],
        "freeze_manifest": run["frozen_models"],
        "controller_bundle_manifest": run["controller_bundle"],
        "v3_s3_bundle_manifest": run["v3_s3_bundle"],
        "controller_implementation": implementation,
    })
    _require_lane(run.get("run_fingerprint") == run_fingerprint, "run fingerprint mismatch")
    _require_lane(run.get("controller") == configuration.get("controller") == expected_controller, "controller mismatch")
    proposal = _lane_object(configuration.get("proposal"), "proposal")
    _require_lane(proposal.get("hybridstructpool") == expected_augmentation, "augmentation mismatch")
    expected_proposal = dict(expected_configuration["proposal"])
    if expected_augmentation is not None:
        expected_proposal["hybridstructpool"] = expected_augmentation
    _require_lane(json_fingerprint(proposal) == json_fingerprint(expected_proposal), "registered proposal mismatch")
    expected_environment = dict(expected_configuration["environment"])
    expected_environment["max_repair_iterations"] = 0
    if expected_runtime and "environment_time_limit_seconds" in expected_runtime:
        expected_environment["time_limit"] = expected_runtime["environment_time_limit_seconds"]
    _require_lane(json_fingerprint(configuration.get("environment")) == json_fingerprint(expected_environment), "registered environment mismatch")
    _require_lane(type(configuration.get("max_decisions")) is int and configuration["max_decisions"] == 0, "unexpected iteration limit")
    _require_lane(type(configuration.get("workers")) is int and configuration["workers"] == 1, "timed workers must be one")
    _require_lane(configuration.get("task_ids_override") == [item["task_id"]], "task cohort mismatch")
    _require_lane(configuration.get("cohort_job_keys_override") == [[item["task_id"], item["solver_seed"]]], "seed cohort mismatch")
    _require_lane(expected_policy in configuration.get("policies", []), "policy not registered")
    _require_lane(configuration.get("stopping_rule") == "wall-clock", "stopping rule mismatch")
    budget = configuration.get("wall_time_budget_seconds")
    _require_lane(finite_number(budget, minimum=0) and budget == wall_time_budget_seconds, "wall budget mismatch")
    override_key = f"{item['task_id']}::{item['solver_seed']}"
    _require_lane(configuration.get("episode_override_fingerprints") == {override_key: json_fingerprint(expected_override)}, "checkpoint override fingerprint mismatch")
    if expected_runtime is not None:
        for field in ("feature_backend", "controller_runtime", "verification_profile", "repair_seed_policy", "deterministic_pp_replay", "episode_process_timeout_seconds"):
            if field in expected_runtime:
                _require_lane(json_fingerprint(configuration.get(field)) == json_fingerprint(expected_runtime[field]), f"runtime {field} mismatch")
        environment = _lane_object(configuration.get("environment"), "environment")
        if "environment_time_limit_seconds" in expected_runtime:
            _require_lane(environment.get("time_limit") == expected_runtime["environment_time_limit_seconds"], "native deadline mismatch")

    expected_episode = episode_id(dict(item), int(item["solver_seed"]), expected_policy)
    _require_lane(row.get("episode_id") == expected_episode and row.get("policy") == expected_policy, "episode/policy mismatch")
    for field in ("map_id", "agent_count"):
        _require_lane(row.get(field) == checkpoint[field], f"manifest {field} mismatch")
    summary = validate_ttf_summary(row.get("summary"), wall_time_budget_seconds=wall_time_budget_seconds)
    _require_lane(summary.get("ttf_clock_schema") == "lns2.ttf.reset_inclusive_wall.v1", "TTF clock schema mismatch")
    _require_lane(summary.get("initial_fingerprint") == checkpoint["expected_fingerprint"], "checkpoint state fingerprint mismatch")
    _require_lane(type(summary.get("initial_conflicts")) is int and summary["initial_conflicts"] == checkpoint["expected_conflicts"], "checkpoint conflicts mismatch")
    for field in ("invalid_action_count", "fingerprint_mismatch_count"):
        _require_lane(type(summary.get(field)) is int and summary[field] == 0, f"{field} must be zero")
    trace_path = contained_file(lane, row.get("trace_file"), field="TTF trace")
    metadata = trace_file_metadata(trace_path)
    for field, value in metadata.items():
        _require_lane(type(row.get(field)) is type(value) and row[field] == value, f"{field} mismatch")
    validated = validate_closed_loop_trace(
        trace_path, run_fingerprint,
        expected_episode_id=expected_episode,
        expected_policy=expected_policy,
        expected_solver_seed=int(item["solver_seed"]),
        metric_iteration_budget=configuration.get("metric_iteration_budget"),
        collection_root=lane,
    )
    _require_lane(json_fingerprint(validated["summary"]) == json_fingerprint(summary), "manifest and trace summary mismatch")
    _require_lane(type(row.get("trace_event_count")) is int and row["trace_event_count"] == validated["event_count"], "trace event count mismatch")
    _require_lane("initial_state_ref" in row and row["initial_state_ref"] == validated["initial_state_ref"], "initial state reference mismatch")
    trace_format = validated["trace_format"]
    _require_lane(row.get("trace_format") == run.get("trace_format") == trace_format, "trace format mismatch")
    _require_lane(row.get("storage_fingerprint") == run.get("storage_fingerprint") == storage_fingerprint(trace_format), "trace storage identity mismatch")
    initial = validated["events"][0]
    restored = _lane_object(initial.get("episode_override"), "trace checkpoint provenance")
    expected_provenance = {
        "schema": expected_override["schema"],
        "state_id": checkpoint["checkpoint_id"],
        "source_checkpoint_id": checkpoint["checkpoint_id"],
        "source_checkpoint_identity_sha256": checkpoint["checkpoint_identity_sha256"],
        "source_full_fingerprint": checkpoint["expected_fingerprint"],
        "forced_first_action": False,
    }
    if "repair_structure_fingerprint" in checkpoint:
        expected_provenance["source_repair_fingerprint"] = checkpoint["repair_structure_fingerprint"]
    for field, value in expected_provenance.items():
        _require_lane(type(restored.get(field)) is type(value) and restored[field] == value, f"trace {field} mismatch")
    _require_lane(type(validated["events"][-1].get("success")) is bool and validated["events"][-1]["success"] == summary["success"], "finish success type/value mismatch")
    _require_lane(sha256_file(run_path) == run_sha and sha256_file(manifest_path) == manifest_sha and trace_file_metadata(trace_path) == metadata, "artifacts changed during validation")
    return {
        "manifest": row,
        "summary": summary,
        "producer_fingerprint": producer_sha,
        "dataset_fingerprint": dataset_sha,
        "evidence": {
            "run_config_sha256": run_sha,
            "manifest_sha256": manifest_sha,
            "run_fingerprint": run_fingerprint,
            "producer_fingerprint": producer_sha,
            "native_sha256": native["sha256"],
            **metadata,
        },
    }


def validate_ttf_lane_cohort(rows: list[dict[str, Any]]) -> None:
    """Paired lanes must use one saved producer/native and one dataset identity."""
    for field in ("producer_fingerprint", "dataset_fingerprint"):
        _require_lane(len({row[field] for row in rows}) <= 1, f"mixed lane {field}")


def registered_ttf_model_manifest(
    root: Path, configuration: Mapping[str, Any]
) -> dict[str, Any]:
    """Read the SHA-registered deployment manifest without loading any model."""
    registration = _lane_object(configuration.get("model_registration"), "registered models")
    deployment = contained_file(
        root, str(registration.get("deployment_bundle", "")) + "/portable_manifest.json",
        field="registered TTF deployment manifest",
    )
    expected = _lane_sha(registration.get("deployment_manifest_sha256"), "registered deployment")
    _require_lane(sha256_file(deployment) == expected, "registered deployment manifest changed")
    return _lane_object(read_json(deployment), "deployment manifest")
