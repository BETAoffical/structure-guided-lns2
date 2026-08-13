from __future__ import annotations

import collections
import itertools
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    _native_filesystem_path,
    producer_identity,
    registered_input,
    sha256_file,
)
from experiments.closed_loop_trace_storage import (
    TRACE_FORMAT_DELTA_GZIP_V2,
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
    write_state_blob,
)
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_collection import (
    FULL_POOL_PROPOSAL,
    _paired_action,
    _validate_native_repair,
)
from experiments.stride_pretail_forced_continuation import _collection_path
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.trace_replay import restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import (
    generate_online_candidates,
    score_online_candidates,
)
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome
from lns2_selector.training import load_frozen_policy_bundle


CONFIG_SCHEMA = "lns2.stride.successor_repairability_registration.v1"
SCIENTIFIC_STATUS = (
    "preregistered_bounded_successor_base_pool_repairability_probe"
)
EXPERIMENT_ID = "stride-successor-repairability-v1"
EXPECTED_PARENT = "18272f12ed5a4878817baf78b2a0a241147f750d"
PREPARATION_SCHEMA = "lns2.stride.successor_repairability_preparation.v1"
PREPARATION_REPORT_SCHEMA = (
    "lns2.stride.successor_repairability_preparation_report.v1"
)
TRIAL_SCHEMA = "lns2.stride.successor_repairability_trial.v1"
STATUS_SCHEMA = "lns2.stride.successor_repairability_status.v1"
REPORT_SCHEMA = "lns2.stride.successor_repairability_report.v1"
PROFILE = "realized_dynamic"
INITIAL_TRIALS = tuple(range(8))
EXTENSION_TRIALS = tuple(range(8, 16))
INITIAL_FIRST_HALF = tuple(range(4))
INITIAL_SECOND_HALF = tuple(range(4, 8))
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/closed_loop_trace_storage.py",
    "experiments/feature_schema_v2.py",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/stride_collection.py",
    "experiments/stride_pretail_forced_continuation.py",
    "experiments/stride_repairability_collection.py",
    "experiments/stride_successor_repairability.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/fingerprints.py",
    "lns2_selector/runtime/online_selection.py",
    "lns2_selector/runtime/repair_outcomes.py",
    "lns2_selector/training/policy_bundle.py",
    "scripts/run_stride_successor_repairability.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _mean(values: Iterable[float | int | bool]) -> float:
    rows = [float(value) for value in values]
    return statistics.fmean(rows) if rows else 0.0


def _sign(value: float, *, epsilon: float = 1e-12) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        rank = 0.5 * (position + end - 1)
        for offset in range(position, end):
            result[ordered[offset]] = rank
        position = end
    return result


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("successor-repairability rank vectors do not align")
    first = _ranks(left)
    second = _ranks(right)
    first_mean = _mean(first)
    second_mean = _mean(second)
    covariance = sum(
        (a - first_mean) * (b - second_mean) for a, b in zip(first, second)
    )
    first_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    second_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    if first_scale <= 1e-15 and second_scale <= 1e-15:
        return 1.0
    if first_scale <= 1e-15 or second_scale <= 1e-15:
        return 0.0
    return covariance / (first_scale * second_scale)


def _manifest_index_sha256(collection: Path) -> tuple[int, str]:
    rows = [
        {
            "path": path.relative_to(collection).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(collection.rglob("realized_dynamic_manifest.jsonl"))
    ]
    return len(rows), _fingerprint(rows)


def validate_registration(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status") != SCIENTIFIC_STATUS
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit") != EXPECTED_PARENT
    ):
        raise ValueError("successor-repairability registration identity changed")
    cohort = dict(config.get("cohort") or {})
    if (
        int(cohort.get("occurrence_count", -1)) != 270
        or int(cohort.get("case_count", -1)) != 45
        or int(cohort.get("arm_count", -1)) != 3
        or int(cohort.get("distinct_full_successor_count", -1)) != 234
        or int(cohort.get("distinct_repair_successor_count", -1)) != 223
        or cohort.get("preserve_occurrence_identity") is not True
        or cohort.get("no_result_based_exclusion") is not True
        or cohort.get("future_outcome_forbidden_during_preparation_and_collection")
        is not True
    ):
        raise ValueError("successor-repairability cohort changed")
    probe = dict(config.get("candidate_probe") or {})
    if (
        probe.get("role")
        != "standardized_measurement_instrument_not_deployment_pool"
        or list(map(str, probe.get("families") or ()))
        != ["target", "collision", "random"]
        or list(map(int, probe.get("neighborhood_sizes") or ())) != [4, 8, 16]
        or int(probe.get("candidates_per_family", -1)) != 2
        or int(probe.get("maximum_candidates_per_occurrence", -1)) != 18
        or any(
            probe.get(name) is not False
            for name in (
                "structpool_allowed",
                "slotpool_allowed",
                "topology_boundary_allowed",
                "causal_or_closure_augmentation_allowed",
                "fixed_structpool_sizes_8_16_24_32_allowed",
            )
        )
        or int(probe.get("feature_dimension", -1)) != 124
    ):
        raise ValueError("successor-repairability candidate probe changed")
    execution = dict(config.get("execution") or {})
    if (
        int(execution.get("prepare_workers", -1)) != 16
        or int(execution.get("collection_workers", -1)) != 16
        or int(execution.get("prepare_job_timeout_seconds", -1)) != 600
        or int(execution.get("per_candidate_trial_timeout_seconds", -1)) != 300
        or int(execution.get("preflight_repair_state_count", -1)) != 3
        or list(map(int, execution.get("preflight_trial_indices") or ())) != [0, 1]
        or tuple(map(int, execution.get("initial_trial_indices") or ()))
        != INITIAL_TRIALS
        or tuple(map(int, execution.get("uniform_extension_trial_indices") or ()))
        != EXTENSION_TRIALS
        or execution.get("extension_only_if_initial_stability_fails") is not True
        or execution.get("extension_must_include_every_state_and_candidate")
        is not True
        or execution.get("stop_on_first_error_or_timeout") is not True
        or execution.get("resume_from_atomic_candidate_trial") is not True
        or execution.get("replan_algorithm") != "PP"
        or execution.get("use_sipp") is not True
        or execution.get("repair_order_controlled") is not False
        or execution.get("raw_ttf_timing_run") is not False
    ):
        raise ValueError("successor-repairability execution contract changed")
    gates = dict(config.get("initial_stability_gates") or {})
    if gates != {
        "minimum_mean_top3_overlap": 0.8,
        "minimum_median_repair_change_rank_correlation": 0.6,
        "minimum_median_strict_drop_rank_correlation": 0.6,
        "minimum_anchor_direction_agreement": 0.7,
        "maximum_mean_cross_half_regret": 0.02,
        "every_map_group_must_pass": True,
        "required_error_count": 0,
        "required_timeout_count": 0,
    }:
        raise ValueError("successor-repairability stability gates changed")
    association = dict(config.get("offline_association_gates") or {})
    if association != {
        "primary_burden": "one minus pool mean strict conflict drop rate",
        "minimum_overall_spearman_burden_vs_bounded_auc": 0.1,
        "minimum_maps_with_nonnegative_burden_vs_auc": 2,
        "minimum_within_case_auc_direction_concordance": 0.55,
        "minimum_informative_within_case_pairs": 20,
        "bounded_success_and_auc_never_enter_online_features": True,
    }:
        raise ValueError("successor-repairability association gates changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "mechanism_probe_only": True,
        "candidate_pool_improvement_claim_allowed": False,
        "ranker_improvement_claim_allowed": False,
        "model_training_allowed": False,
        "runtime_integration_allowed": False,
        "ttf_experiment_allowed": False,
        "longtail_prevention_claim_allowed": False,
        "future_outcome_used_for_job_selection": False,
        "future_outcome_used_as_online_feature": False,
        "no_result_based_exclusion": True,
    }:
        raise ValueError("successor-repairability claim boundary changed")


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], Path]:
    path = Path(config_path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    validate_registration(config)
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    collection = (root / str(config["source_collection"]["path"])).resolve()
    count, digest = _manifest_index_sha256(collection)
    if (
        count != int(config["source_collection"]["manifest_file_count"])
        or digest != str(config["source_collection"]["manifest_index_sha256"])
    ):
        raise ValueError("successor-repairability source manifest index changed")
    inventory_report = _read_json(inputs["successor_inventory_report"])
    inventory_status = _read_json(inputs["successor_inventory_status"])
    if (
        inventory_report.get("integrity_passed") is not True
        or inventory_status.get("complete") is not True
        or int(inventory_report.get("overall", {}).get("episode_count", -1)) != 270
        or int(inventory_report.get("distinct_successor_fingerprint_count", -1))
        != 234
        or int(
            inventory_report.get("distinct_successor_repair_fingerprint_count", -1)
        )
        != 223
    ):
        raise ValueError("successor-repairability inventory prerequisite changed")
    return path, root, config, inputs, collection


def _single_manifest(
    collection: Path, item: dict[str, Any]
) -> dict[str, Any]:
    rows = _read_jsonl(collection / "realized_dynamic_manifest.jsonl")
    matching = [
        row
        for row in rows
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matching) != 1:
        raise ValueError("successor occurrence does not resolve exactly one manifest")
    return dict(matching[0])


def _source_replay(run_path: Path, *, task_id: str, solver_seed: int) -> dict[str, Any]:
    run = _read_json(run_path)
    dataset_root = Path(str(run["dataset"])).resolve()
    split = str(run["configuration"].get("split", "balanced_wall_clock"))
    matches = [
        row
        for row in _load_dataset_rows(dataset_root, [split])
        if str(row["task_id"]) == str(task_id)
    ]
    if len(matches) != 1:
        # Historical run configurations do not always repeat the split under
        # configuration.  The manifest split is the authoritative fallback.
        matches = [
            row
            for row in _load_dataset_rows(dataset_root, ["balanced_wall_clock"])
            if str(row["task_id"]) == str(task_id)
        ]
    if len(matches) != 1:
        raise ValueError("successor-repairability task does not resolve exactly once")
    environment = dict(run["configuration"]["environment"])
    if environment.get("replan_algorithm") != "PP" or environment.get("use_sipp") is not True:
        raise ValueError("successor-repairability source is not PP/SIPPS")
    return {
        "dataset_root": str(dataset_root),
        "row": matches[0],
        "environment": environment,
        "solver_seed": int(solver_seed),
        "replay_destroy_strategy": "Adaptive",
    }


def _base_proposal(run: dict[str, Any]) -> dict[str, Any]:
    proposal = dict(run["configuration"]["proposal"])
    proposal.update(FULL_POOL_PROPOSAL)
    for name in ("structpool", "topology_boundary", "causalclosurepool"):
        proposal.pop(name, None)
    return proposal


def _load_v2_model(run: dict[str, Any]) -> Any:
    configuration = dict(run["configuration"])
    bundle = load_frozen_policy_bundle(
        configuration["frozen_models"], dict(configuration["model_registration"])
    )
    return bundle.models[PROFILE]


def _feature_digest(features: dict[str, float]) -> str:
    return _fingerprint({str(name): float(value) for name, value in features.items()})


def _occurrence_id(row: dict[str, Any]) -> str:
    return _fingerprint(
        {
            "case_id": str(row["case_id"]),
            "trial_index": int(row["trial_index"]),
            "arm": str(row["arm"]),
        }
    )


def semantic_state_id(
    *, task_id: str, repair_fingerprint: str, repair_semantics: str
) -> str:
    return _fingerprint(
        {
            "task_id": str(task_id),
            "repair_fingerprint": str(repair_fingerprint),
            "repair_semantics": str(repair_semantics),
        }
    )


def semantic_action_id(
    *,
    task_id: str,
    repair_fingerprint: str,
    repair_semantics: str,
    agents: Iterable[int],
) -> str:
    return _fingerprint(
        {
            "semantic_state_id": semantic_state_id(
                task_id=task_id,
                repair_fingerprint=repair_fingerprint,
                repair_semantics=repair_semantics,
            ),
            "agents": sorted(map(int, agents)),
        }
    )


def _preparation_file(output: Path, occurrence_id: str) -> Path:
    return output / "prepared_occurrences" / f"{occurrence_id}.json"


def _preparation_valid(payload: dict[str, Any], *, job: dict[str, Any]) -> bool:
    candidates = list(payload.get("candidates") or ())
    candidate_ids = [str(row.get("candidate_id")) for row in candidates]
    agent_sets = [tuple(map(int, row.get("agents") or ())) for row in candidates]
    anchors = [row for row in candidates if row.get("is_v2_anchor") is True]
    return bool(
        payload.get("schema") == PREPARATION_SCHEMA
        and payload.get("run_fingerprint") == job["run_fingerprint"]
        and payload.get("occurrence_id") == job["occurrence_id"]
        and payload.get("inventory_row_sha256") == job["inventory_row_sha256"]
        and payload.get("complete") is True
        and len(candidates) > 0
        and len(candidates) <= 18
        and len(candidate_ids) == len(set(candidate_ids))
        and len(agent_sets) == len(set(agent_sets))
        and len(anchors) == 1
        and all(int(row.get("feature_count", -1)) == 124 for row in candidates)
        and all(row.get("candidate_kind") == "base" for row in candidates)
        and payload.get("future_outcome_read") is False
    )


def _prepare_occurrence(job: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(str(job["output_path"]))
    if output_path.is_file():
        existing = _read_json(output_path)
        if bool(job["resume"]) and _preparation_valid(existing, job=job):
            return {
                "status": "resumed",
                "job_id": str(job["occurrence_id"]),
                "occurrence_count": 1,
                "candidate_count": len(existing["candidates"]),
            }
        raise ValueError(f"invalid successor preparation checkpoint: {output_path}")

    item = dict(job["schedule_item"])
    inventory = dict(job["inventory_row"])
    collection = Path(str(job["collection"]))
    manifest = _single_manifest(collection, item)
    if manifest.get("status") != "ok" or manifest.get("trace_format") != TRACE_FORMAT_DELTA_GZIP_V2:
        raise ValueError("successor preparation source episode changed")
    trace_path = (collection / str(manifest["trace_file"])).resolve()
    if sha256_file(_native_filesystem_path(trace_path)) != str(manifest["trace_sha256"]):
        raise ValueError("successor preparation trace SHA-256 changed")
    events = read_trace_events(_native_filesystem_path(trace_path))
    if len(events) < 3 or events[0].get("event") != "initial":
        raise ValueError("successor preparation trace is incomplete")
    initial_event = dict(events[0])
    initial_blob = (collection / str(initial_event["state_blob"])).resolve()
    state = read_state_blob(_native_filesystem_path(initial_blob))
    state.update(dict(initial_event["state_extras"]))
    first = dict(events[1])
    if first.get("event") != "transition" or int(first.get("decision_index", -1)) != 0:
        raise ValueError("successor preparation forced transition changed")
    if state_fingerprint(state) != str(first["before_fingerprint"]):
        raise ValueError("successor preparation initial fingerprint changed")
    successor = apply_state_delta(state, dict(first["state_delta"]))
    successor.update(apply_extras_delta(state, dict(first["state_extras_delta"])))
    successor_full = state_fingerprint(successor)
    successor_repair = repair_structure_fingerprint(successor)
    if (
        successor_full != str(first["after_fingerprint"])
        or successor_full != str(inventory["successor_fingerprint"])
        or successor_repair != str(inventory["successor_repair_fingerprint"])
        or int(successor["num_of_colliding_pairs"])
        != int(inventory["successor_conflicts"])
    ):
        raise ValueError("successor preparation inventory identity changed")
    output_root = Path(str(job["output_root"]))
    relative_blob, blob_path = write_state_blob(output_root, successor)
    run_path = collection / "run_config.json"
    run = _read_json(run_path)
    replay = _source_replay(
        run_path, task_id=str(item["task_id"]), solver_seed=int(item["solver_seed"])
    )
    repair_semantics = _fingerprint(
        {
            "dataset_fingerprint": run.get("dataset_fingerprint"),
            "task_id": str(item["task_id"]),
            "environment": replay["environment"],
            "replay_destroy_strategy": replay["replay_destroy_strategy"],
        }
    )
    environment, restored = restore_repair_state(
        replay, successor, seed=repairability_restore_seed(successor_repair)
    )
    if repair_structure_fingerprint(restored) != successor_repair:
        raise RuntimeError("successor preparation native restore changed")
    proposal = _base_proposal(run)
    candidates, generation = generate_online_candidates(
        environment,
        successor,
        task_id=str(item["task_id"]),
        solver_seed=int(item["solver_seed"]),
        decision_index=1,
        proposal_config=proposal,
        state_hash=successor_full,
        verify_full_state=False,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    if not candidates or len(candidates) > 18:
        raise RuntimeError("successor base candidate count is outside the frozen cap")
    if any(candidate.get("structpool_family_groups") for candidate in candidates):
        raise RuntimeError("successor base probe generated a structural candidate")
    engine = OnlineFeatureEngine(
        successor,
        backend="native",
        required_features={PROFILE: PROFILE_FEATURE_NAMES[PROFILE]},
        dense_output=False,
    )
    feature_rows, feature_metrics = engine.realized_rows(
        candidates, state_hash=successor_full
    )
    anchor_index, scores, _margin = score_online_candidates(
        feature_rows, _load_v2_model(run)
    )
    records: list[dict[str, Any]] = []
    for index, (candidate, feature_row, score) in enumerate(
        zip(candidates, feature_rows, scores)
    ):
        agents = sorted(map(int, candidate["agents"]))
        features = {
            str(name): float(value)
            for name, value in dict(feature_row["features"][PROFILE]).items()
        }
        if (
            set(features) != set(PROFILE_FEATURE_NAMES[PROFILE])
            or len(features) != 124
            or any(not math.isfinite(value) for value in features.values())
        ):
            raise RuntimeError("successor candidate feature schema changed")
        records.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_kind": "base",
                "agents": agents,
                "actual_size": len(agents),
                "selection_families": sorted(
                    map(str, candidate.get("selection_families") or ())
                ),
                "semantic_action_id": semantic_action_id(
                    task_id=str(item["task_id"]),
                    repair_fingerprint=successor_repair,
                    repair_semantics=repair_semantics,
                    agents=agents,
                ),
                "v2_score": float(score),
                "is_v2_anchor": index == anchor_index,
                "feature_schema": "lns2.realized_features.v2",
                "feature_count": len(features),
                "feature_sha256": _feature_digest(features),
                "features": features,
            }
        )
    occurrence = {
        "schema": PREPARATION_SCHEMA,
        "run_fingerprint": str(job["run_fingerprint"]),
        "complete": True,
        "occurrence_id": str(job["occurrence_id"]),
        "case_id": str(item["case_id"]),
        "state_id": str(item["state_id"]),
        "trial_index": int(item["trial_index"]),
        "arm": str(item["arm"]),
        "candidate_id_forced": str(item["candidate_id"]),
        "map_id": str(inventory["map_id"]),
        "task_id": str(item["task_id"]),
        "solver_seed": int(item["solver_seed"]),
        "source_collection": str(collection),
        "source_run_config": str(run_path),
        "source_run_config_sha256": sha256_file(run_path),
        "source_trace_sha256": str(manifest["trace_sha256"]),
        "successor_fingerprint": successor_full,
        "successor_repair_fingerprint": successor_repair,
        "successor_conflicts": int(successor["num_of_colliding_pairs"]),
        "successor_state_blob": str(blob_path),
        "successor_state_blob_relative": relative_blob,
        "successor_state_blob_sha256": sha256_file(blob_path),
        "successor_state_context": dict(successor.get("context") or {}),
        "repair_semantics": repair_semantics,
        "semantic_state_id": semantic_state_id(
            task_id=str(item["task_id"]),
            repair_fingerprint=successor_repair,
            repair_semantics=repair_semantics,
        ),
        "candidate_count": len(records),
        "candidate_generation": generation,
        "feature_metrics": feature_metrics,
        "candidates": records,
        "inventory_row_sha256": str(job["inventory_row_sha256"]),
        "future_outcome_read": False,
    }
    if not _preparation_valid(occurrence, job=job):
        raise RuntimeError("successor preparation artifact failed validation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, occurrence)
    return {
        "status": "ok",
        "job_id": str(job["occurrence_id"]),
        "occurrence_count": 1,
        "candidate_count": len(records),
    }


def _failure(job: dict[str, Any], status: str, error: str) -> dict[str, Any]:
    return {
        "status": str(status),
        "job_id": str(job.get("job_id", job.get("occurrence_id", "unknown"))),
        "error": str(error),
        "occurrence_count": 0,
        "candidate_count": 0,
    }


def select_preflight_occurrences(
    occurrences: list[dict[str, Any]], *, state_count: int
) -> list[dict[str, Any]]:
    maps = sorted({str(row["map_id"]) for row in occurrences})
    if state_count != len(maps):
        raise ValueError("preflight requires exactly one successor state per map")
    selected: list[dict[str, Any]] = []
    for map_id in maps:
        representatives: dict[str, dict[str, Any]] = {}
        for row in occurrences:
            if str(row["map_id"]) != map_id:
                continue
            state_key = str(row["semantic_state_id"])
            previous = representatives.get(state_key)
            if previous is None or str(row["occurrence_id"]) < str(
                previous["occurrence_id"]
            ):
                representatives[state_key] = row
        ordered = sorted(
            representatives.values(),
            key=lambda row: (
                _fingerprint(
                    {
                        "namespace": "stride-successor-repairability-preflight-v1",
                        "map_id": map_id,
                        "semantic_state_id": str(row["semantic_state_id"]),
                    }
                ),
                str(row["occurrence_id"]),
            ),
        )
        if not ordered:
            raise ValueError(f"preflight map has no successor state: {map_id}")
        selected.append(ordered[0])
    return selected


def prepare_successor_repairability(
    *,
    config_path: str | Path,
    output: str | Path,
    workers: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    config_path, root, config, inputs, source_collection = load_registration(
        config_path
    )
    worker_count = int(workers or config["execution"]["prepare_workers"])
    if worker_count != 16:
        raise ValueError("successor-repairability preparation requires 16 workers")
    schedule = _read_jsonl(inputs["pretail_execution_schedule"])
    inventory_rows = _read_jsonl(inputs["successor_inventory_rows"])
    inventory_by_key = {
        (str(row["case_id"]), int(row["trial_index"]), str(row["arm"])): row
        for row in inventory_rows
    }
    if len(schedule) != 270 or len(inventory_by_key) != 270:
        raise ValueError("successor-repairability occurrence count changed")
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    identity = {
        "schema": "lns2.stride.successor_repairability_prepare_run.v1",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "source_manifest_index_sha256": str(
            config["source_collection"]["manifest_index_sha256"]
        ),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "preparation_run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("successor preparation output belongs to another run")
        if not resume:
            raise ValueError("successor preparation output exists; pass resume")
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})

    jobs: list[dict[str, Any]] = []
    for item in schedule:
        key = (str(item["case_id"]), int(item["trial_index"]), str(item["arm"]))
        inventory = inventory_by_key.get(key)
        if inventory is None:
            raise ValueError("successor schedule left the frozen inventory")
        occurrence_id = _occurrence_id(inventory)
        collection = _collection_path(source_collection, item)
        jobs.append(
            {
                "job_id": occurrence_id,
                "occurrence_id": occurrence_id,
                "schedule_item": item,
                "inventory_row": inventory,
                "inventory_row_sha256": _fingerprint(inventory),
                "collection": str(collection),
                "output_root": str(output),
                "output_path": str(_preparation_file(output, occurrence_id)),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    if len({job["occurrence_id"] for job in jobs}) != 270:
        raise ValueError("successor preparation occurrence IDs are not unique")
    status_path = output / "preparation_status.json"
    _write_json(
        status_path,
        {
            "schema": PREPARATION_REPORT_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "status": "running",
            "total_jobs": len(jobs),
            "workers": worker_count,
            "error_jobs": 0,
            "timeout_jobs": 0,
        },
    )
    results = _run_jobs(
        _prepare_occurrence,
        jobs,
        worker_count,
        phase="successor-repairability-prepare",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["execution"]["prepare_job_timeout_seconds"]),
        failure_result=_failure,
        stop_on_failure=True,
    )
    failures = [row for row in results if row.get("status") in {"error", "timeout"}]
    occurrences: list[dict[str, Any]] = []
    jobs_by_id = {str(job["occurrence_id"]): job for job in jobs}
    if not failures and len(results) == len(jobs):
        for occurrence_id, job in sorted(jobs_by_id.items()):
            path = _preparation_file(output, occurrence_id)
            payload = _read_json(path)
            if not _preparation_valid(payload, job=job):
                failures.append(
                    {
                        "status": "error",
                        "job_id": occurrence_id,
                        "error": "invalid preparation artifact",
                    }
                )
                break
            occurrences.append(payload)
    full_keys = {str(row["successor_fingerprint"]) for row in occurrences}
    repair_keys = {str(row["successor_repair_fingerprint"]) for row in occurrences}
    semantic_states = {str(row["semantic_state_id"]) for row in occurrences}
    passed = bool(
        not failures
        and len(occurrences) == int(config["cohort"]["occurrence_count"])
        and len(full_keys) == int(config["cohort"]["distinct_full_successor_count"])
        and len(repair_keys)
        == int(config["cohort"]["distinct_repair_successor_count"])
        and len(semantic_states) == len(repair_keys)
    )
    if passed:
        _write_jsonl(output / "successor_occurrences.jsonl", occurrences)
        preflight = select_preflight_occurrences(
            occurrences,
            state_count=int(config["execution"]["preflight_repair_state_count"]),
        )
        _write_jsonl(output / "preflight_occurrences.jsonl", preflight)
    report = {
        "schema": PREPARATION_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_fingerprint": run_fingerprint,
        "passed": passed,
        "occurrence_count": len(occurrences),
        "distinct_full_successor_count": len(full_keys),
        "distinct_repair_successor_count": len(repair_keys),
        "semantic_state_count": len(semantic_states),
        "occurrence_candidate_count": sum(
            len(row["candidates"]) for row in occurrences
        ),
        "distinct_semantic_action_count": len(
            {
                str(candidate["semantic_action_id"])
                for row in occurrences
                for candidate in row["candidates"]
            }
        ),
        "candidate_count_distribution": dict(
            sorted(
                collections.Counter(
                    int(row["candidate_count"]) for row in occurrences
                ).items()
            )
        ),
        "map_occurrence_counts": dict(
            sorted(collections.Counter(str(row["map_id"]) for row in occurrences).items())
        ),
        "error_jobs": sum(row.get("status") == "error" for row in failures),
        "timeout_jobs": sum(row.get("status") == "timeout" for row in failures),
        "failures": failures,
        "future_outcome_read": False,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": (
            sha256_file(output / "successor_occurrences.jsonl") if passed else None
        ),
        "preflight_manifest_sha256": (
            sha256_file(output / "preflight_occurrences.jsonl") if passed else None
        ),
        "producer": producer,
    }
    _write_json(output / "preparation_report.json", report)
    _write_json(
        status_path,
        {
            **report,
            "status": "complete" if passed else "failed",
            "completed_jobs": len(results) - len(failures),
            "workers": worker_count,
        },
    )
    return report


def _trial_path(collection: Path, action_id: str, trial_index: int) -> Path:
    return (
        collection
        / "trials"
        / action_id[:2]
        / action_id
        / f"trial-{trial_index:02d}.json"
    )


def _trial_valid(payload: dict[str, Any], *, job: dict[str, Any]) -> bool:
    before = int(payload.get("before_conflicts", -1))
    after = payload.get("conflicts_after")
    return bool(
        payload.get("schema") == TRIAL_SCHEMA
        and payload.get("run_fingerprint") == job["run_fingerprint"]
        and payload.get("semantic_action_id") == job["semantic_action_id"]
        and int(payload.get("trial_index", -1)) == int(job["trial_index"])
        and int(payload.get("pp_seed", -1)) == int(job["pp_seed"])
        and payload.get("before_repair_fingerprint")
        == job["successor_repair_fingerprint"]
        and before == int(job["successor_conflicts"])
        and type(after) is int
        and int(after) >= 0
        and payload.get("agents") == job["agents"]
        and payload.get("native_action_validated") is True
        and payload.get("repair_order_controlled") is False
        and payload.get("runtime_fields_stored") is False
        and payload.get("future_trajectory_stored") is False
        and all(
            name not in payload
            for name in (
                "native_step_seconds",
                "pp_replan_seconds",
                "ttf",
                "wall_time",
                "future_conflicts",
            )
        )
    )


def _collect_trial(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(job["output_path"]))
    if path.is_file():
        existing = _read_json(path)
        if bool(job["resume"]) and _trial_valid(existing, job=job):
            return {"status": "resumed", "job_id": str(job["job_id"]), "trial_count": 1}
        raise ValueError(f"invalid successor trial checkpoint: {path}")
    blob_path = Path(str(job["state_blob"]))
    if sha256_file(blob_path) != str(job["state_blob_sha256"]):
        raise ValueError("successor trial state blob SHA-256 changed")
    state = read_state_blob(blob_path)
    state["context"] = dict(job["state_context"])
    if (
        state_fingerprint(state) != str(job["successor_fingerprint"])
        or repair_structure_fingerprint(state)
        != str(job["successor_repair_fingerprint"])
    ):
        raise ValueError("successor trial source state changed")
    run_path = Path(str(job["source_run_config"]))
    if sha256_file(run_path) != str(job["source_run_config_sha256"]):
        raise ValueError("successor trial source run configuration changed")
    replay = _source_replay(
        run_path,
        task_id=str(job["task_id"]),
        solver_seed=int(job["solver_seed"]),
    )
    environment, restored = restore_repair_state(
        replay,
        state,
        seed=repairability_restore_seed(str(job["successor_repair_fingerprint"])),
    )
    before_repair = repair_structure_fingerprint(restored)
    before_conflicts = int(restored["num_of_colliding_pairs"])
    if (
        before_repair != str(job["successor_repair_fingerprint"])
        or before_conflicts != int(job["successor_conflicts"])
    ):
        raise RuntimeError("successor trial native restore changed")
    agents = list(map(int, job["agents"]))
    pp_seed = int(job["pp_seed"])
    result = _plain(environment.step(_paired_action(agents, pp_seed)))
    after, metrics = _validate_native_repair(
        result, expected_agents=agents, expected_seed=pp_seed
    )
    after_conflicts = int(after["num_of_colliding_pairs"])
    after_repair = repair_structure_fingerprint(after)
    payload = {
        "schema": TRIAL_SCHEMA,
        "run_fingerprint": str(job["run_fingerprint"]),
        "semantic_action_id": str(job["semantic_action_id"]),
        "agents": agents,
        "trial_index": int(job["trial_index"]),
        "pp_seed": pp_seed,
        "before_conflicts": before_conflicts,
        "before_repair_fingerprint": before_repair,
        "conflicts_after": after_conflicts,
        "after_repair_fingerprint": after_repair,
        "repair_state_changed": after_repair != before_repair,
        "strict_conflict_drop": after_conflicts < before_conflicts,
        "no_progress": after_conflicts >= before_conflicts,
        "normalized_conflict_reduction": (
            before_conflicts - after_conflicts
        ) / max(1, before_conflicts),
        "replan_success": bool(metrics["replan_success"]),
        "feasible": bool(after.get("feasible")),
        "repair_outcome": classify_repair_outcome(
            before_fingerprint=before_repair,
            after_fingerprint=after_repair,
            replan_success=bool(metrics["replan_success"]),
            conflicts_before=before_conflicts,
            conflicts_after=after_conflicts,
            feasible=bool(after.get("feasible")),
        ),
        "native_action_validated": True,
        "repair_order_controlled": False,
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
    }
    if not _trial_valid(payload, job=job):
        raise RuntimeError("successor trial artifact failed validation")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    return {"status": "ok", "job_id": str(job["job_id"]), "trial_count": 1}


def _mode_trials(config: dict[str, Any], mode: str) -> tuple[int, ...]:
    field = {
        "preflight": "preflight_trial_indices",
        "initial": "initial_trial_indices",
        "extension": "uniform_extension_trial_indices",
    }.get(mode)
    if field is None:
        raise ValueError("successor-repairability mode must be preflight, initial or extension")
    return tuple(map(int, config["execution"][field]))


def _collection_occurrences(output: Path, *, mode: str) -> list[dict[str, Any]]:
    filename = (
        "preflight_occurrences.jsonl"
        if mode == "preflight"
        else "successor_occurrences.jsonl"
    )
    return _read_jsonl(output / filename)


def _job_core(occurrence: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_action_id": str(candidate["semantic_action_id"]),
        "agents": list(map(int, candidate["agents"])),
        "task_id": str(occurrence["task_id"]),
        "solver_seed": int(occurrence["solver_seed"]),
        "successor_fingerprint": str(occurrence["successor_fingerprint"]),
        "successor_repair_fingerprint": str(
            occurrence["successor_repair_fingerprint"]
        ),
        "successor_conflicts": int(occurrence["successor_conflicts"]),
        "state_blob": str(occurrence["successor_state_blob"]),
        "state_blob_sha256": str(occurrence["successor_state_blob_sha256"]),
        "state_context": dict(occurrence["successor_state_context"]),
        "source_run_config": str(occurrence["source_run_config"]),
        "source_run_config_sha256": str(occurrence["source_run_config_sha256"]),
        "repair_semantics": str(occurrence["repair_semantics"]),
        "representative_occurrence_id": str(occurrence["occurrence_id"]),
    }


def _semantic_jobs(
    occurrences: list[dict[str, Any]],
    *,
    trials: tuple[int, ...],
    collection: Path,
    run_fingerprint: str,
    resume: bool,
) -> list[dict[str, Any]]:
    actions: dict[str, dict[str, Any]] = {}
    for occurrence in sorted(occurrences, key=lambda row: str(row["occurrence_id"])):
        for candidate in occurrence["candidates"]:
            core = _job_core(occurrence, candidate)
            action_id = str(core["semantic_action_id"])
            previous = actions.get(action_id)
            if previous is None:
                actions[action_id] = core
                continue
            invariant_fields = (
                "semantic_action_id",
                "agents",
                "task_id",
                "successor_repair_fingerprint",
                "successor_conflicts",
                "repair_semantics",
            )
            if any(previous[field] != core[field] for field in invariant_fields):
                raise ValueError("semantic successor action cache collision")
    jobs: list[dict[str, Any]] = []
    for action_id, core in sorted(actions.items()):
        for trial_index in trials:
            pp_seed = repairability_pp_seed(
                str(core["successor_repair_fingerprint"]), trial_index
            )
            jobs.append(
                {
                    **core,
                    "trial_index": trial_index,
                    "pp_seed": pp_seed,
                    "job_id": f"{action_id}:{trial_index:02d}",
                    "output_path": str(_trial_path(collection, action_id, trial_index)),
                    "run_fingerprint": run_fingerprint,
                    "resume": bool(resume),
                }
            )
    return jobs


def collect_successor_repairability(
    *,
    config_path: str | Path,
    output: str | Path,
    mode: str,
    workers: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    config_path, root, config, _inputs, _source = load_registration(config_path)
    output = Path(output).resolve()
    preparation = _read_json(output / "preparation_report.json")
    if (
        preparation.get("passed") is not True
        or preparation.get("config_sha256") != sha256_file(config_path)
        or preparation.get("manifest_sha256")
        != sha256_file(output / "successor_occurrences.jsonl")
    ):
        raise ValueError("successor-repairability preparation is incomplete")
    worker_count = int(workers or config["execution"]["collection_workers"])
    if worker_count != 16:
        raise ValueError("successor-repairability collection requires 16 workers")
    trials = _mode_trials(config, mode)
    if mode == "initial":
        preflight = _read_json(output / "preflight" / "collection_report.json")
        if (
            preflight.get("mode") != "preflight"
            or preflight.get("integrity_passed") is not True
            or preflight.get("config_sha256") != sha256_file(config_path)
        ):
            raise ValueError("initial collection requires a passed preflight")
    if mode == "extension":
        initial = _read_json(output / "initial" / "collection_report.json")
        if (
            initial.get("mode") != "initial"
            or initial.get("integrity_passed") is not True
            or initial.get("initial_stability_passed") is not False
            or initial.get("config_sha256") != sha256_file(config_path)
        ):
            raise ValueError("extension is allowed only after initial instability")
    occurrences = _collection_occurrences(output, mode=mode)
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    if producer != preparation["producer"]:
        raise ValueError("successor-repairability producer changed after preparation")
    identity = {
        "schema": "lns2.stride.successor_repairability_collection_run.v1",
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "config_sha256": sha256_file(config_path),
        "preparation_manifest_sha256": str(preparation["manifest_sha256"]),
        "selected_occurrence_ids": sorted(
            str(row["occurrence_id"]) for row in occurrences
        ),
        "trial_indices": list(trials),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    collection = output / mode
    collection.mkdir(parents=True, exist_ok=True)
    run_path = collection / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("successor trial output belongs to another run")
        if not resume:
            raise ValueError("successor trial output exists; pass resume")
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    jobs = _semantic_jobs(
        occurrences,
        trials=trials,
        collection=collection,
        run_fingerprint=run_fingerprint,
        resume=resume,
    )
    status_path = collection / "collection_status.json"
    _write_json(
        status_path,
        {
            "schema": STATUS_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "mode": mode,
            "status": "running",
            "total_jobs": len(jobs),
            "completed_jobs": 0,
            "workers": worker_count,
            "error_jobs": 0,
            "timeout_jobs": 0,
        },
    )
    results = _run_jobs(
        _collect_trial,
        jobs,
        worker_count,
        phase=f"successor-repairability-{mode}",
        output_root=collection,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(
            config["execution"]["per_candidate_trial_timeout_seconds"]
        ),
        failure_result=_failure,
        stop_on_failure=True,
    )
    failures = [row for row in results if row.get("status") in {"error", "timeout"}]
    status = {
        "schema": STATUS_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "mode": mode,
        "status": "failed" if failures else "complete",
        "total_jobs": len(jobs),
        "completed_jobs": len(results) - len(failures),
        "workers": worker_count,
        "error_jobs": sum(row.get("status") == "error" for row in failures),
        "timeout_jobs": sum(row.get("status") == "timeout" for row in failures),
        "failures": failures,
    }
    _write_json(status_path, status)
    if not failures and len(results) == len(jobs):
        return analyze_successor_repairability(
            config_path=config_path, output=output, mode=mode
        )
    return status


def aggregate_candidate(
    candidate: dict[str, Any],
    trials: list[dict[str, Any]],
    *,
    first_half: tuple[int, ...],
    second_half: tuple[int, ...],
) -> dict[str, Any]:
    ordered = sorted(trials, key=lambda row: int(row["trial_index"]))
    expected = list(first_half + second_half)
    if [int(row["trial_index"]) for row in ordered] != expected:
        raise ValueError("successor candidate trial product is incomplete")
    split = len(first_half)

    def summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
        return {
            f"{prefix}repair_state_change_rate": _mean(
                row["repair_state_changed"] for row in rows
            ),
            f"{prefix}strict_conflict_drop_rate": _mean(
                row["strict_conflict_drop"] for row in rows
            ),
            f"{prefix}no_progress_rate": _mean(row["no_progress"] for row in rows),
            f"{prefix}replan_success_rate": _mean(
                row["replan_success"] for row in rows
            ),
            f"{prefix}mean_normalized_conflict_reduction": _mean(
                row["normalized_conflict_reduction"] for row in rows
            ),
        }

    return {
        "candidate_id": str(candidate["candidate_id"]),
        "semantic_action_id": str(candidate["semantic_action_id"]),
        "agents": list(map(int, candidate["agents"])),
        "actual_size": int(candidate["actual_size"]),
        "selection_families": list(map(str, candidate["selection_families"])),
        "is_v2_anchor": bool(candidate["is_v2_anchor"]),
        "trial_count": len(ordered),
        **summarize(ordered, ""),
        **summarize(ordered[:split], "first_half_"),
        **summarize(ordered[split:], "second_half_"),
        "distinct_after_repair_fingerprint_count": len(
            {str(row["after_repair_fingerprint"]) for row in ordered}
        ),
    }


def _ranking_key(row: dict[str, Any], prefix: str = "") -> tuple[Any, ...]:
    return (
        -float(row[f"{prefix}repair_state_change_rate"]),
        -float(row[f"{prefix}strict_conflict_drop_rate"]),
        -float(row[f"{prefix}mean_normalized_conflict_reduction"]),
        float(row[f"{prefix}no_progress_rate"]),
        str(row["candidate_id"]),
    )


def summarize_occurrence(
    occurrence: dict[str, Any], aggregates: list[dict[str, Any]]
) -> dict[str, Any]:
    if not aggregates:
        raise ValueError("successor occurrence has no candidate aggregates")
    anchors = [row for row in aggregates if row["is_v2_anchor"]]
    if len(anchors) != 1:
        raise ValueError("successor occurrence V2 anchor changed")
    anchor = anchors[0]
    ordered = sorted(aggregates, key=_ranking_key)
    first_ordered = sorted(
        aggregates, key=lambda row: _ranking_key(row, "first_half_")
    )
    second_ordered = sorted(
        aggregates, key=lambda row: _ranking_key(row, "second_half_")
    )
    top_count = min(3, len(aggregates))
    first_top = {row["candidate_id"] for row in first_ordered[:top_count]}
    second_top = {row["candidate_id"] for row in second_ordered[:top_count]}
    repair_first = [float(row["first_half_repair_state_change_rate"]) for row in aggregates]
    repair_second = [float(row["second_half_repair_state_change_rate"]) for row in aggregates]
    drop_first = [float(row["first_half_strict_conflict_drop_rate"]) for row in aggregates]
    drop_second = [float(row["second_half_strict_conflict_drop_rate"]) for row in aggregates]
    direction_agreements: list[float] = []
    for row in aggregates:
        if row["candidate_id"] == anchor["candidate_id"]:
            continue
        for metric in ("repair_state_change_rate", "strict_conflict_drop_rate"):
            first_delta = float(row[f"first_half_{metric}"]) - float(
                anchor[f"first_half_{metric}"]
            )
            second_delta = float(row[f"second_half_{metric}"]) - float(
                anchor[f"second_half_{metric}"]
            )
            direction_agreements.append(float(_sign(first_delta) == _sign(second_delta)))
    regrets: list[float] = []
    for metric in ("repair_state_change_rate", "strict_conflict_drop_rate"):
        first_best = min(
            aggregates,
            key=lambda row: (-float(row[f"first_half_{metric}"]), str(row["candidate_id"])),
        )
        second_best = min(
            aggregates,
            key=lambda row: (-float(row[f"second_half_{metric}"]), str(row["candidate_id"])),
        )
        regrets.append(
            max(float(row[f"second_half_{metric}"]) for row in aggregates)
            - float(first_best[f"second_half_{metric}"])
        )
        regrets.append(
            max(float(row[f"first_half_{metric}"]) for row in aggregates)
            - float(second_best[f"first_half_{metric}"])
        )
    return {
        "occurrence_id": str(occurrence["occurrence_id"]),
        "case_id": str(occurrence["case_id"]),
        "trial_index": int(occurrence["trial_index"]),
        "arm": str(occurrence["arm"]),
        "map_id": str(occurrence["map_id"]),
        "task_id": str(occurrence["task_id"]),
        "successor_fingerprint": str(occurrence["successor_fingerprint"]),
        "successor_repair_fingerprint": str(
            occurrence["successor_repair_fingerprint"]
        ),
        "semantic_state_id": str(occurrence["semantic_state_id"]),
        "successor_conflicts": int(occurrence["successor_conflicts"]),
        "candidate_count": len(aggregates),
        "v2_anchor": anchor,
        "best_observed_candidate": ordered[0],
        "top3_candidate_ids": [str(row["candidate_id"]) for row in ordered[:top_count]],
        "first_half_top3_candidate_ids": sorted(first_top),
        "second_half_top3_candidate_ids": sorted(second_top),
        "top3_overlap": len(first_top & second_top) / max(1, top_count),
        "repair_change_rank_correlation": _spearman(repair_first, repair_second),
        "strict_drop_rank_correlation": _spearman(drop_first, drop_second),
        "anchor_direction_agreement": _mean(direction_agreements),
        "cross_half_regret": _mean(regrets),
        "zero_repair_candidate_fraction": _mean(
            float(row["repair_state_change_rate"] <= 1e-12) for row in aggregates
        ),
        "zero_strict_drop_candidate_fraction": _mean(
            float(row["strict_conflict_drop_rate"] <= 1e-12) for row in aggregates
        ),
        "pool_mean_repair_state_change_rate": _mean(
            row["repair_state_change_rate"] for row in aggregates
        ),
        "pool_mean_strict_conflict_drop_rate": _mean(
            row["strict_conflict_drop_rate"] for row in aggregates
        ),
        "pool_mean_no_progress_rate": _mean(
            row["no_progress_rate"] for row in aggregates
        ),
        "best_repair_state_change_rate": max(
            float(row["repair_state_change_rate"]) for row in aggregates
        ),
        "best_strict_conflict_drop_rate": max(
            float(row["strict_conflict_drop_rate"]) for row in aggregates
        ),
        "candidate_aggregates": aggregates,
    }


def _stability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "occurrence_count": len(rows),
        "mean_top3_overlap": _mean(row["top3_overlap"] for row in rows),
        "median_repair_change_rank_correlation": (
            statistics.median(
                float(row["repair_change_rank_correlation"]) for row in rows
            )
            if rows
            else 0.0
        ),
        "median_strict_drop_rank_correlation": (
            statistics.median(
                float(row["strict_drop_rank_correlation"]) for row in rows
            )
            if rows
            else 0.0
        ),
        "mean_anchor_direction_agreement": _mean(
            row["anchor_direction_agreement"] for row in rows
        ),
        "mean_cross_half_regret": _mean(row["cross_half_regret"] for row in rows),
        "mean_zero_repair_candidate_fraction": _mean(
            row["zero_repair_candidate_fraction"] for row in rows
        ),
        "mean_zero_strict_drop_candidate_fraction": _mean(
            row["zero_strict_drop_candidate_fraction"] for row in rows
        ),
        "mean_pool_repair_state_change_rate": _mean(
            row["pool_mean_repair_state_change_rate"] for row in rows
        ),
        "mean_pool_strict_conflict_drop_rate": _mean(
            row["pool_mean_strict_conflict_drop_rate"] for row in rows
        ),
    }


def _stability_gates(
    summary: dict[str, Any], gates: dict[str, Any]
) -> dict[str, bool]:
    return {
        "top3_overlap": float(summary["mean_top3_overlap"])
        >= float(gates["minimum_mean_top3_overlap"]),
        "repair_change_rank_correlation": float(
            summary["median_repair_change_rank_correlation"]
        )
        >= float(gates["minimum_median_repair_change_rank_correlation"]),
        "strict_drop_rank_correlation": float(
            summary["median_strict_drop_rank_correlation"]
        )
        >= float(gates["minimum_median_strict_drop_rank_correlation"]),
        "anchor_direction_agreement": float(
            summary["mean_anchor_direction_agreement"]
        )
        >= float(gates["minimum_anchor_direction_agreement"]),
        "cross_half_regret": float(summary["mean_cross_half_regret"])
        <= float(gates["maximum_mean_cross_half_regret"]),
    }


def paired_auc_direction(
    rows: list[dict[str, Any]],
) -> dict[str, float | int]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), int(row["trial_index"]))].append(row)
    informative = 0
    concordant = 0
    for group in grouped.values():
        for left, right in itertools.combinations(group, 2):
            burden_delta = float(left["repairability_burden"]) - float(
                right["repairability_burden"]
            )
            auc_delta = float(left["bounded_normalized_fixed_auc"]) - float(
                right["bounded_normalized_fixed_auc"]
            )
            if _sign(burden_delta) == 0 or _sign(auc_delta) == 0:
                continue
            informative += 1
            concordant += _sign(burden_delta) == _sign(auc_delta)
    return {
        "informative_pair_count": informative,
        "concordant_pair_count": concordant,
        "direction_concordance": concordant / max(1, informative),
    }


def _association_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if bool(row["bounded_success"])]
    censored = [row for row in rows if bool(row["bounded_right_censored"])]
    return {
        "occurrence_count": len(rows),
        "bounded_success_count": len(success),
        "bounded_right_censored_count": len(censored),
        "spearman_burden_vs_bounded_auc": _spearman(
            [float(row["repairability_burden"]) for row in rows],
            [float(row["bounded_normalized_fixed_auc"]) for row in rows],
        ),
        "mean_burden_success": _mean(row["repairability_burden"] for row in success),
        "mean_burden_right_censored": _mean(
            row["repairability_burden"] for row in censored
        ),
        **paired_auc_direction(rows),
    }


def _load_mode_trials(
    *,
    config: dict[str, Any],
    output: Path,
    source_mode: str,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    occurrences = _collection_occurrences(output, mode=source_mode)
    trials = _mode_trials(config, source_mode)
    collection = output / source_mode
    run = _read_json(collection / "run_config.json")
    status = _read_json(collection / "collection_status.json")
    jobs = _semantic_jobs(
        occurrences,
        trials=trials,
        collection=collection,
        run_fingerprint=str(run["run_fingerprint"]),
        resume=True,
    )
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for job in jobs:
        path = Path(str(job["output_path"]))
        if not path.is_file():
            raise ValueError(f"successor-repairability trial is missing: {path}")
        payload = _read_json(path)
        if not _trial_valid(payload, job=job):
            raise ValueError(f"successor-repairability trial is invalid: {path}")
        key = (str(payload["semantic_action_id"]), int(payload["trial_index"]))
        if key in rows:
            raise ValueError("successor-repairability trial key is duplicated")
        rows[key] = payload
    return rows, jobs, status


def _integrity_report(
    *,
    config: dict[str, Any],
    preparation: dict[str, Any],
    occurrences: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    trial_rows: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, bool], bool]:
    expected_occurrences = (
        int(config["execution"]["preflight_repair_state_count"])
        if len(status_rows) == 1 and status_rows[0].get("mode") == "preflight"
        else int(config["cohort"]["occurrence_count"])
    )
    gates = {
        "preparation_passed": preparation.get("passed") is True,
        "occurrence_count": len(occurrences) == expected_occurrences,
        "job_count": len(trial_rows) == len(jobs),
        "status_complete": all(row.get("status") == "complete" for row in status_rows),
        "zero_errors": sum(int(row.get("error_jobs", -1)) for row in status_rows) == 0,
        "zero_timeouts": sum(int(row.get("timeout_jobs", -1)) for row in status_rows)
        == 0,
        "native_actions_valid": all(
            row.get("native_action_validated") is True for row in trial_rows.values()
        ),
        "repair_order_uncontrolled": all(
            row.get("repair_order_controlled") is False for row in trial_rows.values()
        ),
        "no_runtime_or_future_fields": all(
            row.get("runtime_fields_stored") is False
            and row.get("future_trajectory_stored") is False
            for row in trial_rows.values()
        ),
        "occurrence_identity_preserved": len(
            {str(row["occurrence_id"]) for row in occurrences}
        )
        == len(occurrences),
    }
    return gates, all(gates.values())


def analyze_successor_repairability(
    *, config_path: str | Path, output: str | Path, mode: str
) -> dict[str, Any]:
    config_path, _root, config, inputs, _source = load_registration(config_path)
    output = Path(output).resolve()
    preparation = _read_json(output / "preparation_report.json")
    if mode not in {"preflight", "initial", "extended"}:
        raise ValueError("analysis mode must be preflight, initial or extended")
    source_modes = [mode] if mode != "extended" else ["initial", "extension"]
    occurrence_mode = "preflight" if mode == "preflight" else "initial"
    occurrences = _collection_occurrences(output, mode=occurrence_mode)
    all_trials: dict[tuple[str, int], dict[str, Any]] = {}
    all_jobs: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for source_mode in source_modes:
        rows, jobs, status = _load_mode_trials(
            config=config, output=output, source_mode=source_mode
        )
        overlap = set(all_trials) & set(rows)
        if overlap:
            raise ValueError("successor-repairability initial/extension trials overlap")
        all_trials.update(rows)
        all_jobs.extend(jobs)
        statuses.append(status)
    integrity_gates, integrity_passed = _integrity_report(
        config=config,
        preparation=preparation,
        occurrences=occurrences,
        jobs=all_jobs,
        status_rows=statuses,
        trial_rows=all_trials,
    )
    collection_output = output / mode
    collection_output.mkdir(parents=True, exist_ok=True)
    if mode == "preflight":
        report = {
            "schema": REPORT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "mode": mode,
            "integrity_passed": integrity_passed,
            "integrity_gates": integrity_gates,
            "occurrence_count": len(occurrences),
            "semantic_action_count": len(
                {str(job["semantic_action_id"]) for job in all_jobs}
            ),
            "trial_count": len(all_trials),
            "error_count": sum(int(row.get("error_jobs", 0)) for row in statuses),
            "timeout_count": sum(int(row.get("timeout_jobs", 0)) for row in statuses),
            "config_sha256": sha256_file(config_path),
            "preparation_manifest_sha256": str(preparation["manifest_sha256"]),
            "next_action": (
                config["decision_rule"]["preflight_integrity_pass"]
                if integrity_passed
                else "stop_and_preserve_artifacts"
            ),
            "claim_boundary": dict(config["claim_boundary"]),
        }
        _write_json(collection_output / "collection_report.json", report)
        return report

    if mode == "initial":
        all_indices = INITIAL_TRIALS
        first_half = INITIAL_FIRST_HALF
        second_half = INITIAL_SECOND_HALF
    else:
        all_indices = tuple(range(16))
        first_half = tuple(range(8))
        second_half = tuple(range(8, 16))
    inventory_by_id = {
        _occurrence_id(row): row for row in _read_jsonl(inputs["successor_inventory_rows"])
    }
    occurrence_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for occurrence in occurrences:
        aggregates: list[dict[str, Any]] = []
        for candidate in occurrence["candidates"]:
            trials = [
                all_trials[(str(candidate["semantic_action_id"]), trial_index)]
                for trial_index in all_indices
            ]
            aggregate = aggregate_candidate(
                candidate,
                trials,
                first_half=first_half,
                second_half=second_half,
            )
            aggregates.append(aggregate)
            candidate_rows.append(
                {
                    "occurrence_id": str(occurrence["occurrence_id"]),
                    "case_id": str(occurrence["case_id"]),
                    "trial_index": int(occurrence["trial_index"]),
                    "arm": str(occurrence["arm"]),
                    "map_id": str(occurrence["map_id"]),
                    "successor_repair_fingerprint": str(
                        occurrence["successor_repair_fingerprint"]
                    ),
                    **aggregate,
                }
            )
        summary = summarize_occurrence(occurrence, aggregates)
        summary.pop("candidate_aggregates")
        outcome = inventory_by_id.get(str(occurrence["occurrence_id"]))
        if outcome is None:
            raise ValueError("successor occurrence lost its frozen outcome row")
        summary.update(
            {
                "bounded_success": bool(outcome["bounded_success"]),
                "bounded_right_censored": bool(outcome["bounded_right_censored"]),
                "bounded_stop_reason": str(outcome["bounded_stop_reason"]),
                "bounded_normalized_fixed_auc": float(
                    outcome["bounded_normalized_fixed_auc"]
                ),
                "bounded_final_conflicts": int(outcome["bounded_final_conflicts"]),
                "repairability_burden": 1.0
                - float(summary["pool_mean_strict_conflict_drop_rate"]),
            }
        )
        occurrence_rows.append(summary)
    overall_stability = _stability_summary(occurrence_rows)
    stability_gate_config = dict(config["initial_stability_gates"])
    overall_stability_gates = _stability_gates(
        overall_stability, stability_gate_config
    )
    by_map_stability: list[dict[str, Any]] = []
    every_map_passed = True
    for map_id in sorted({str(row["map_id"]) for row in occurrence_rows}):
        group = [row for row in occurrence_rows if str(row["map_id"]) == map_id]
        summary = _stability_summary(group)
        gates = _stability_gates(summary, stability_gate_config)
        passed = all(gates.values())
        every_map_passed &= passed
        by_map_stability.append(
            {"map_id": map_id, **summary, "gates": gates, "passed": passed}
        )
    initial_stability_passed = bool(
        integrity_passed
        and all(overall_stability_gates.values())
        and every_map_passed
    )
    overall_association = _association_summary(occurrence_rows)
    by_map_association: list[dict[str, Any]] = []
    for map_id in sorted({str(row["map_id"]) for row in occurrence_rows}):
        group = [row for row in occurrence_rows if str(row["map_id"]) == map_id]
        by_map_association.append({"map_id": map_id, **_association_summary(group)})
    association_config = dict(config["offline_association_gates"])
    nonnegative_maps = sum(
        float(row["spearman_burden_vs_bounded_auc"]) >= 0.0
        for row in by_map_association
    )
    association_gates = {
        "overall_burden_vs_auc": float(
            overall_association["spearman_burden_vs_bounded_auc"]
        )
        >= float(association_config["minimum_overall_spearman_burden_vs_bounded_auc"]),
        "map_direction_count": nonnegative_maps
        >= int(association_config["minimum_maps_with_nonnegative_burden_vs_auc"]),
        "within_case_auc_direction": float(
            overall_association["direction_concordance"]
        )
        >= float(association_config["minimum_within_case_auc_direction_concordance"]),
        "within_case_pair_count": int(overall_association["informative_pair_count"])
        >= int(association_config["minimum_informative_within_case_pairs"]),
    }
    offline_association_passed = all(association_gates.values())
    if not integrity_passed:
        next_action = "stop_and_preserve_artifacts"
    elif not initial_stability_passed and mode == "initial":
        next_action = config["decision_rule"]["initial_stability_fails"]
    elif not initial_stability_passed:
        next_action = "stop_successor_repairability_target_before_training"
    elif not offline_association_passed:
        next_action = config["decision_rule"]["stability_passes_but_association_fails"]
    else:
        next_action = config["decision_rule"]["all_initial_gates_pass"]
    _write_jsonl(collection_output / "candidate_aggregates.jsonl", candidate_rows)
    _write_jsonl(collection_output / "occurrence_summaries.jsonl", occurrence_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "integrity_passed": integrity_passed,
        "integrity_gates": integrity_gates,
        "occurrence_count": len(occurrence_rows),
        "candidate_aggregate_count": len(candidate_rows),
        "semantic_action_count": len(
            {str(row["semantic_action_id"]) for row in candidate_rows}
        ),
        "unique_native_trial_count": len(all_trials),
        "trial_indices": list(all_indices),
        "overall_stability": overall_stability,
        "overall_stability_gates": overall_stability_gates,
        "by_map_stability": by_map_stability,
        "every_map_stability_passed": every_map_passed,
        "initial_stability_passed": initial_stability_passed,
        "overall_offline_association": overall_association,
        "by_map_offline_association": by_map_association,
        "offline_association_gates": association_gates,
        "offline_association_passed": offline_association_passed,
        "next_action": next_action,
        "config_sha256": sha256_file(config_path),
        "candidate_aggregates_sha256": sha256_file(
            collection_output / "candidate_aggregates.jsonl"
        ),
        "occurrence_summaries_sha256": sha256_file(
            collection_output / "occurrence_summaries.jsonl"
        ),
        "claim_boundary": dict(config["claim_boundary"]),
        "future_outcomes_used_for_analysis_only": True,
        "model_training_performed": False,
        "runtime_integration_performed": False,
        "ttf_experiment_performed": False,
    }
    _write_json(collection_output / "collection_report.json", report)
    return report


__all__ = [
    "aggregate_candidate",
    "analyze_successor_repairability",
    "collect_successor_repairability",
    "load_registration",
    "paired_auc_direction",
    "prepare_successor_repairability",
    "select_preflight_occurrences",
    "semantic_action_id",
    "semantic_state_id",
    "summarize_occurrence",
    "validate_registration",
]
