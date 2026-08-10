from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments._common import contained_file, producer_identity, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.feature_schema_v2 import FEATURE_SCHEMA_ID, PROFILE_FEATURE_NAMES
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
from experiments.state_analysis import analyze_state, analyze_static_grid
from experiments.stride_collection import _paired_action, _validate_native_repair
from experiments.stride_marginalpool_root_diagnostic import (
    CHECKPOINT_SCHEMA,
    _feature_payload,
    _structural,
    load_registration as load_root_registration,
)
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.stride_robustaction_label_collection import (
    _forbidden_hits,
)
from experiments.stride_tailswitch_sequence_forensics import (
    load_registration as load_sequence_registration,
)
from experiments.trace_replay import TARGET_STATE_RESTORE_CONTRACT, restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


CONFIG_SCHEMA = "lns2.stride.marginalpool_action_replay_registration.v1"
RUN_SCHEMA = "lns2.stride.marginalpool_action_replay_run.v1"
STATE_SCHEMA = "lns2.stride.marginalpool_action_replay_state.v1"
TRIAL_SCHEMA = "lns2.stride.marginalpool_action_replay_trial.v1"
AGGREGATE_SCHEMA = "lns2.stride.marginalpool_action_replay_aggregate.v1"
COLLECTION_REPORT_SCHEMA = "lns2.stride.marginalpool_action_replay_collection_report.v1"
ANALYSIS_REPORT_SCHEMA = "lns2.stride.marginalpool_action_replay_analysis_report.v1"
ANALYSIS_STATUS_SCHEMA = "lns2.stride.marginalpool_action_replay_analysis_status.v1"
EXPERIMENT_ID = "stride-marginalpool-action-replay-v1"
SCIENTIFIC_STATUS = "preregistered_all_existing_candidates_paired_current_step_replay"
PARENT_COMMIT = "a9da04e065911ff4e1d8b94b5c9769d2723f3fb7"
TRIAL_INDICES = tuple(range(16))
FIRST_HALF = tuple(range(8))
SECOND_HALF = tuple(range(8, 16))
FEATURE_PROFILE = "realized_dynamic"
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/closed_loop_trace_storage.py",
    "experiments/feature_schema_v2.py",
    "experiments/neighborhood_features.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/stride_collection.py",
    "experiments/stride_marginalpool_action_replay.py",
    "experiments/stride_marginalpool_root_diagnostic.py",
    "experiments/stride_repairability_collection.py",
    "experiments/stride_robustaction_label_collection.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/fingerprints.py",
    "lns2_selector/runtime/repair_outcomes.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _mean(values: Iterable[float | int]) -> float:
    rows = [float(value) for value in values]
    return float(statistics.fmean(rows)) if rows else 0.0


def _feature_digest(features: dict[str, float]) -> str:
    return hashlib.sha256(
        json.dumps(
            features,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_registration(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status") != SCIENTIFIC_STATUS
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit") != PARENT_COMMIT
    ):
        raise ValueError("MarginalPool action-replay registration identity changed")
    cohort = dict(config.get("cohort") or {})
    if (
        int(cohort.get("logical_checkpoint_count", -1)) != 90
        or int(cohort.get("unique_repair_state_count", -1)) != 78
        or int(cohort.get("unique_state_candidate_action_count", -1)) != 2502
        or list(cohort.get("deduplication_key") or ())
        != ["state_fingerprint", "candidate_id"]
        or cohort.get("logical_checkpoint_mapping_is_retained") is not True
        or cohort.get("classification_must_not_affect_collection") is not True
        or cohort.get("no_result_based_exclusion") is not True
    ):
        raise ValueError("MarginalPool action-replay cohort contract changed")
    execution = dict(config.get("execution") or {})
    if (
        int(execution.get("workers", -1)) != 2
        or int(execution.get("per_state_timeout_seconds", -1)) != 7200
        or execution.get("replan_algorithm") != "PP"
        or execution.get("use_sipp") is not True
        or execution.get("native_repair_semantics_unchanged") is not True
        or tuple(map(int, execution.get("trial_indices") or ())) != TRIAL_INDICES
    ):
        raise ValueError("MarginalPool action-replay execution contract changed")
    preflight = dict(config.get("preflight") or {})
    if (
        preflight.get("selection")
        != "first two unique state_fingerprint values in ascending lexical order"
        or tuple(map(int, preflight.get("trial_indices") or ())) != (0, 1)
        or preflight.get("all_candidates_in_selected_states") is not True
        or preflight.get("outcomes_must_not_change_cohort_thresholds_or_protocol")
        is not True
        or preflight.get("full_collection_requires_integrity_pass") is not True
    ):
        raise ValueError("MarginalPool action-replay preflight contract changed")
    labels = dict(config.get("current_step_labels") or {})
    expected_aggregates = {
        "seed_mean",
        "seed_standard_deviation",
        "lower_half_mean",
        "first_fixed_half_mean",
        "second_fixed_half_mean",
        "no_progress_rate",
        "replan_success_rate",
        "feasible_rate",
        "minimum_seed_score",
        "maximum_seed_score",
    }
    if (
        set(map(str, labels.get("aggregates") or ())) != expected_aggregates
        or list(labels.get("first_fixed_half") or ()) != [0, 7]
        or list(labels.get("second_fixed_half") or ()) != [8, 15]
        or labels.get("future_trajectory_forbidden") is not True
        or labels.get("runtime_and_ttf_forbidden") is not True
    ):
        raise ValueError("MarginalPool current-step label contract changed")
    dominance = dict(config.get("stable_dominance") or {})
    if (
        not math.isclose(
            float(dominance.get("minimum_seed_mean_advantage", math.nan)),
            0.02,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or dominance.get("no_progress_rate_must_not_increase") is not True
        or dominance.get("both_fixed_half_mean_differences_must_be_strictly_positive")
        is not True
        or dominance.get("ties_are_not_dominance") is not True
    ):
        raise ValueError("MarginalPool stable-dominance rule changed")
    gates = dict(config.get("integrity_gates") or {})
    if (
        int(gates.get("required_logical_checkpoint_count", -1)) != 90
        or int(gates.get("required_unique_state_count", -1)) != 78
        or int(gates.get("required_unique_state_candidate_action_count", -1))
        != 2502
        or int(gates.get("required_trial_count", -1)) != 40032
        or int(gates.get("required_trial_indices_per_action", -1)) != 16
        or int(gates.get("required_feature_count_per_action", -1)) != 124
        or int(gates.get("required_error_count", -1)) != 0
        or int(gates.get("required_timeout_count", -1)) != 0
    ):
        raise ValueError("MarginalPool action-replay integrity gates changed")
    boundary = dict(config.get("claim_boundary") or {})
    expected_boundary = {
        "paired_current_step_repair_diagnostic_only": True,
        "model_training_allowed": False,
        "solver_modification_allowed": False,
        "candidate_quality_claim_is_checkpoint_local": True,
        "long_term_quality_claim": False,
        "ttf_improvement_claim": False,
        "generalization_claim": False,
        "default_controller_replacement_allowed": False,
        "no_result_based_exclusion": True,
    }
    if boundary != expected_boundary:
        raise ValueError("MarginalPool action-replay claim boundary changed")


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(config_path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    validate_registration(config)
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    return path, root, config, inputs


def _logical_checkpoint_id(row: dict[str, Any]) -> str:
    return _fingerprint(
        {
            "case_id": str(row["case_id"]),
            "checkpoint_kind": str(row["checkpoint_kind"]),
            "state_fingerprint": str(row["state_fingerprint"]),
            "selected_candidate_id": str(row["selected_candidate_id"]),
        }
    )


def _candidate_core(candidate: dict[str, Any]) -> dict[str, Any]:
    agents = sorted(map(int, candidate.get("agents") or ()))
    if not agents:
        raise ValueError("MarginalPool candidate has no agents")
    if len(agents) != len(set(agents)):
        raise ValueError("MarginalPool candidate has duplicate agents")
    candidate_id = str(candidate.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("MarginalPool candidate has no ID")
    # Score/retention and SlotPool support annotations belong to the historical
    # controller that presented the action.  They can differ when the same
    # physical state/action appears in StructPool and SlotPool traces, but they
    # neither alter the native explicit neighborhood nor enter the frozen 124D
    # feature schema.  Keep only action semantics and feature provenance in the
    # deduplicated replay product; the complete source rows remain checksum-
    # pinned in the Stage 1 checkpoint manifest.
    return {
        "candidate_id": candidate_id,
        "agents": agents,
        "actual_size": len(agents),
        "seed_agents": sorted(map(int, candidate.get("seed_agents") or ())),
        "proposal_seeds": sorted(map(int, candidate.get("proposal_seeds") or ())),
        "proposal_count_by_family": {
            str(name): int(value)
            for name, value in sorted(
                dict(candidate.get("proposal_count_by_family") or {}).items()
            )
        },
        "selection_families": list(map(str, candidate.get("selection_families") or ())),
        "selection_rank_by_family": {
            str(name): int(value)
            for name, value in sorted(
                dict(candidate.get("selection_rank_by_family") or {}).items()
            )
        },
        "structpool_family_groups": list(
            map(str, candidate.get("structpool_family_groups") or ())
        ),
        "candidate_kind": "structural" if _structural(candidate) else "base",
    }


def build_frozen_cohort(
    config_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config_path, root, config, inputs = load_registration(config_path)
    root_status = _read_json(inputs["root_status"])
    root_report = _read_json(inputs["root_report"])
    if (
        root_status.get("complete") is not True
        or root_status.get("integrity_passed") is not True
        or root_report.get("integrity_passed") is not True
        or int(root_report.get("checkpoint_count", -1)) != 90
        or root_report.get("current_action_outcome_used") is not False
        or root_report.get("future_trajectory_used") is not False
    ):
        raise ValueError("MarginalPool root diagnostic is not a passed frozen input")
    _, _, _, root_inputs = load_root_registration(inputs["root_registration"])
    _, _, _, sequence_inputs = load_sequence_registration(
        root_inputs["sequence_registration"]
    )
    tailswitch_root = sequence_inputs["tailswitch_report"].parent
    checkpoints = _read_jsonl(inputs["root_checkpoints"])
    if len(checkpoints) != 90:
        raise ValueError("MarginalPool checkpoint count changed")

    states: dict[str, dict[str, Any]] = {}
    logical_rows: list[dict[str, Any]] = []
    checkpoint_root = inputs["root_checkpoints"].parent
    for checkpoint in checkpoints:
        if (
            checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or checkpoint.get("current_action_outcome_used") is not False
        ):
            raise ValueError("MarginalPool checkpoint schema or timing changed")
        state_key = str(checkpoint["state_fingerprint"])
        blob_path = contained_file(
            checkpoint_root,
            str(checkpoint["state_blob"]),
            field="MarginalPool state blob",
        )
        if sha256_file(blob_path) != str(checkpoint["state_blob_sha256"]):
            raise ValueError("MarginalPool state blob hash changed")
        state = read_state_blob(blob_path)
        if state_fingerprint(state) != state_key:
            raise ValueError("MarginalPool state blob fingerprint changed")
        candidates = [_candidate_core(dict(row)) for row in checkpoint["candidate_pool"]]
        candidate_ids = [str(row["candidate_id"]) for row in candidates]
        if (
            len(candidate_ids) != len(set(candidate_ids))
            or str(checkpoint["selected_candidate_id"]) not in candidate_ids
            or int(dict(checkpoint["selected_feature"])["feature_count"]) != 124
        ):
            raise ValueError("MarginalPool checkpoint candidate contract changed")
        state_dir = tailswitch_root / "states" / _fingerprint(
            {"state_id": str(checkpoint["state_id"])}
        )[:20]
        source_run_config = state_dir / str(checkpoint["treatment_policy"]) / "run_config.json"
        if not source_run_config.is_file():
            raise ValueError(f"MarginalPool source run config is missing: {source_run_config}")
        source_run_sha256 = sha256_file(source_run_config)
        logical_id = _logical_checkpoint_id(checkpoint)
        logical = {
            "logical_checkpoint_id": logical_id,
            "case_id": str(checkpoint["case_id"]),
            "checkpoint_kind": str(checkpoint["checkpoint_kind"]),
            "classification": str(checkpoint["classification"]),
            "challenger": str(checkpoint["challenger"]),
            "contrast": str(checkpoint["contrast"]),
            "map_id": str(checkpoint["map_id"]),
            "task_id": str(checkpoint["task_id"]),
            "solver_seed": int(checkpoint["solver_seed"]),
            "decision_index": int(checkpoint["decision_index"]),
            "treatment_policy": str(checkpoint["treatment_policy"]),
            "state_fingerprint": state_key,
            "before_conflicts": int(checkpoint["conflict_pair_count"]),
            "selected_candidate_id": str(checkpoint["selected_candidate_id"]),
            "candidate_ids": candidate_ids,
            "selected_feature": dict(checkpoint["selected_feature"]),
        }
        logical_rows.append(logical)
        record = states.get(state_key)
        if record is None:
            record = {
                "state_fingerprint": state_key,
                "state_blob": str(blob_path),
                "state_blob_sha256": str(checkpoint["state_blob_sha256"]),
                "state_context": dict(checkpoint.get("state_context") or {}),
                "map_id": str(checkpoint["map_id"]),
                "task_id": str(checkpoint["task_id"]),
                "solver_seed": int(checkpoint["solver_seed"]),
                "split": str(dict(checkpoint.get("state_context") or {})["split"]),
                "source_run_config": str(source_run_config),
                "source_run_config_sha256": source_run_sha256,
                "candidates": {},
                "logical_checkpoint_ids": [],
            }
            states[state_key] = record
        elif (
            str(record["state_blob_sha256"]) != str(checkpoint["state_blob_sha256"])
            or dict(record["state_context"])
            != dict(checkpoint.get("state_context") or {})
            or str(record["task_id"]) != str(checkpoint["task_id"])
        ):
            raise ValueError("duplicate MarginalPool repair state has inconsistent metadata")
        record["logical_checkpoint_ids"].append(logical_id)
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            prior = record["candidates"].get(candidate_id)
            if prior is None:
                record["candidates"][candidate_id] = candidate
            elif prior != candidate:
                raise ValueError(
                    "duplicate MarginalPool state/action has inconsistent provenance"
                )

    state_rows = []
    for state_key, record in sorted(states.items()):
        state_rows.append(
            {
                **record,
                "candidates": [
                    record["candidates"][candidate_id]
                    for candidate_id in sorted(record["candidates"])
                ],
                "logical_checkpoint_ids": sorted(record["logical_checkpoint_ids"]),
            }
        )
    logical_rows.sort(key=lambda row: str(row["logical_checkpoint_id"]))
    action_count = sum(len(row["candidates"]) for row in state_rows)
    if len(state_rows) != 78 or len(logical_rows) != 90 or action_count != 2502:
        raise ValueError(
            "MarginalPool frozen cohort changed: "
            f"states={len(state_rows)}, logical={len(logical_rows)}, actions={action_count}"
        )
    metadata = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "tailswitch_report": str(sequence_inputs["tailswitch_report"]),
        "tailswitch_report_sha256": sha256_file(sequence_inputs["tailswitch_report"]),
        "cohort_fingerprint": _fingerprint(
            {
                "states": [
                    {
                        "state_fingerprint": row["state_fingerprint"],
                        "state_blob_sha256": row["state_blob_sha256"],
                        "candidate_ids": [
                            candidate["candidate_id"] for candidate in row["candidates"]
                        ],
                    }
                    for row in state_rows
                ],
                "logical": logical_rows,
            }
        ),
        "claim_boundary": dict(config["claim_boundary"]),
    }
    return metadata, state_rows, logical_rows


def _replay_job(state_record: dict[str, Any]) -> dict[str, Any]:
    run_path = Path(str(state_record["source_run_config"]))
    if sha256_file(run_path) != str(state_record["source_run_config_sha256"]):
        raise ValueError("MarginalPool source run configuration changed")
    run = _read_json(run_path)
    dataset_root = Path(str(run["dataset"])).resolve()
    matches = [
        row
        for row in _load_dataset_rows(dataset_root, [str(state_record["split"])])
        if str(row["task_id"]) == str(state_record["task_id"])
    ]
    if len(matches) != 1:
        raise ValueError("MarginalPool replay task must resolve exactly once")
    environment = dict(run["configuration"]["environment"])
    if environment.get("replan_algorithm") != "PP" or environment.get("use_sipp") is not True:
        raise ValueError("MarginalPool replay PP/SIPPS semantics changed")
    return {
        "dataset_root": str(dataset_root),
        "row": matches[0],
        "environment": environment,
        "solver_seed": int(state_record["solver_seed"]),
        "replay_destroy_strategy": "Adaptive",
    }


def aggregate_candidate(
    candidate: dict[str, Any], trials: list[dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(trials, key=lambda row: int(row["trial_index"]))
    indices = [int(row["trial_index"]) for row in ordered]
    if indices != list(TRIAL_INDICES):
        raise ValueError("MarginalPool aggregate requires all 16 trial indices")
    values = [float(row["normalized_conflict_reduction"]) for row in ordered]
    lower_half = sorted(values)[: len(values) // 2]
    return {
        "schema": AGGREGATE_SCHEMA,
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_kind": str(candidate["candidate_kind"]),
        "agents": list(map(int, candidate["agents"])),
        "actual_size": int(candidate["actual_size"]),
        "selection_families": list(map(str, candidate["selection_families"])),
        "structpool_family_groups": list(
            map(str, candidate["structpool_family_groups"])
        ),
        "trial_count": len(values),
        "seed_mean": statistics.fmean(values),
        "seed_standard_deviation": statistics.pstdev(values),
        "lower_half_mean": statistics.fmean(lower_half),
        "first_fixed_half_mean": statistics.fmean(values[:8]),
        "second_fixed_half_mean": statistics.fmean(values[8:]),
        "no_progress_rate": statistics.fmean(
            float(row["conflicts_after"] >= row["before_conflicts"])
            for row in ordered
        ),
        "replan_success_rate": statistics.fmean(
            float(row["replan_success"]) for row in ordered
        ),
        "feasible_rate": statistics.fmean(float(row["feasible"]) for row in ordered),
        "minimum_seed_score": min(values),
        "maximum_seed_score": max(values),
        "feature_schema": str(candidate["feature_schema"]),
        "feature_profile": str(candidate["feature_profile"]),
        "feature_count": int(candidate["feature_count"]),
        "feature_sha256": str(candidate["feature_sha256"]),
        "features": dict(candidate["features"]),
    }


def stable_dominates(
    candidate: dict[str, Any], reference: dict[str, Any], *, minimum: float = 0.02
) -> bool:
    return (
        float(candidate["seed_mean"]) - float(reference["seed_mean"])
        >= minimum - 1e-15
        and float(candidate["no_progress_rate"])
        <= float(reference["no_progress_rate"]) + 1e-15
        and float(candidate["first_fixed_half_mean"])
        > float(reference["first_fixed_half_mean"]) + 1e-15
        and float(candidate["second_fixed_half_mean"])
        > float(reference["second_fixed_half_mean"]) + 1e-15
    )


def _state_artifact_valid(
    payload: dict[str, Any],
    *,
    state_record: dict[str, Any],
    run_fingerprint: str,
    trial_indices: tuple[int, ...],
) -> bool:
    candidates = payload.get("candidates")
    trials = payload.get("trials")
    if (
        payload.get("schema") != STATE_SCHEMA
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("complete") is not True
        or payload.get("state_fingerprint") != state_record["state_fingerprint"]
        or payload.get("state_record_fingerprint") != _fingerprint(state_record)
        or not isinstance(candidates, list)
        or not isinstance(trials, list)
        or payload.get("native_action_semantics_validated") is not True
        or payload.get("selected_feature_reproduction_validated") is not True
        or payload.get("runtime_fields_stored") is not False
        or payload.get("future_trajectory_stored") is not False
    ):
        return False
    expected_by_id = {
        str(row["candidate_id"]): row for row in state_record["candidates"]
    }
    candidate_by_id = {
        str(row.get("candidate_id", "")): row
        for row in candidates
        if isinstance(row, dict)
    }
    if set(candidate_by_id) != set(expected_by_id):
        return False
    names = set(PROFILE_FEATURE_NAMES[FEATURE_PROFILE])
    for candidate_id, candidate in candidate_by_id.items():
        features = candidate.get("features")
        if (
            list(map(int, candidate.get("agents") or ()))
            != list(map(int, expected_by_id[candidate_id]["agents"]))
            or candidate.get("feature_schema") != FEATURE_SCHEMA_ID
            or candidate.get("feature_profile") != FEATURE_PROFILE
            or int(candidate.get("feature_count", -1)) != 124
            or not isinstance(features, dict)
            or set(features) != names
            or candidate.get("feature_sha256") != _feature_digest(features)
        ):
            return False
    expected_pairs = {
        (candidate_id, trial_index)
        for candidate_id in candidate_by_id
        for trial_index in trial_indices
    }
    observed_pairs = Counter(
        (str(row.get("candidate_id", "")), int(row.get("trial_index", -1)))
        for row in trials
        if isinstance(row, dict)
    )
    if set(observed_pairs) != expected_pairs or any(
        count != 1 for count in observed_pairs.values()
    ):
        return False
    before_repair = str(payload.get("before_repair_fingerprint", ""))
    before_conflicts = int(payload.get("before_conflicts", -1))
    if before_conflicts <= 0:
        return False
    for trial in trials:
        trial_index = int(trial.get("trial_index", -1))
        after = trial.get("conflicts_after")
        if (
            trial.get("schema") != TRIAL_SCHEMA
            or trial.get("state_fingerprint") != state_record["state_fingerprint"]
            or type(after) is not int
            or after < 0
            or int(trial.get("pp_seed", -1))
            != repairability_pp_seed(before_repair, trial_index)
            or not math.isfinite(
                float(trial.get("normalized_conflict_reduction", math.nan))
            )
            or not math.isclose(
                float(trial["normalized_conflict_reduction"]),
                (before_conflicts - after) / max(1, before_conflicts),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not isinstance(trial.get("replan_success"), bool)
            or not isinstance(trial.get("feasible"), bool)
        ):
            return False
    return not _forbidden_hits(payload)


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    state_record = dict(job["state_record"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    trial_indices = tuple(map(int, job["trial_indices"]))
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if _state_artifact_valid(
            existing,
            state_record=state_record,
            run_fingerprint=run_fingerprint,
            trial_indices=trial_indices,
        ):
            return {
                "state_fingerprint": str(state_record["state_fingerprint"]),
                "state_file": str(output_path),
                "status": "resumed",
                "state_count": 1,
                "outcome_count": len(existing["trials"]),
                "candidate_count": len(existing["candidates"]),
                "trial_count": len(existing["trials"]),
                "error_count": 0,
            }
        raise ValueError(f"invalid completed MarginalPool replay state: {output_path}")

    blob_path = Path(str(state_record["state_blob"]))
    if sha256_file(blob_path) != str(state_record["state_blob_sha256"]):
        raise RuntimeError("MarginalPool replay state blob changed")
    state = read_state_blob(blob_path)
    state["context"] = dict(state_record["state_context"])
    state_key = str(state_record["state_fingerprint"])
    if state_fingerprint(state) != state_key:
        raise RuntimeError("MarginalPool replay state fingerprint changed")
    replay = _replay_job(state_record)
    before_repair = repair_structure_fingerprint(state)
    before_conflicts = int(state["num_of_colliding_pairs"])
    restore_seed = repairability_restore_seed(before_repair)
    _environment, restored = restore_repair_state(replay, state, seed=restore_seed)
    if repair_structure_fingerprint(restored) != before_repair:
        raise RuntimeError("MarginalPool replay native restore changed")

    static_grid = analyze_static_grid(state)
    analysis = analyze_state(state, static_grid=static_grid)
    candidates: list[dict[str, Any]] = []
    for source_candidate in state_record["candidates"]:
        candidate = dict(source_candidate)
        feature = _feature_payload(state, candidate, analysis)
        candidates.append({**candidate, **feature})
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
    logical_by_id = dict(job["logical_by_id"])
    for logical_id in state_record["logical_checkpoint_ids"]:
        logical = dict(logical_by_id[logical_id])
        selected = candidate_by_id[str(logical["selected_candidate_id"])]
        expected = dict(logical["selected_feature"])
        if (
            dict(selected["features"]) != dict(expected["features"])
            or str(selected["feature_sha256"]) != str(expected["feature_sha256"])
        ):
            raise RuntimeError("MarginalPool selected 124D feature reproduction changed")

    known_agents = {int(agent["id"]) for agent in state["agents"]}
    if any(
        not set(map(int, candidate["agents"])) <= known_agents
        for candidate in candidates
    ):
        raise RuntimeError("MarginalPool replay candidate contains an unknown agent")
    trials: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        agents = list(map(int, candidate["agents"]))
        for trial_index in trial_indices:
            branch_environment, branch_state = restore_repair_state(
                replay, state, seed=restore_seed
            )
            if repair_structure_fingerprint(branch_state) != before_repair:
                raise RuntimeError("MarginalPool paired branch restore changed")
            pp_seed = repairability_pp_seed(before_repair, trial_index)
            result = _plain(branch_environment.step(_paired_action(agents, pp_seed)))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_repair = repair_structure_fingerprint(after)
            trials.append(
                {
                    "schema": TRIAL_SCHEMA,
                    "state_fingerprint": state_key,
                    "candidate_id": candidate_id,
                    "trial_index": trial_index,
                    "pp_seed": pp_seed,
                    "before_conflicts": before_conflicts,
                    "before_repair_fingerprint": before_repair,
                    "conflicts_after": conflicts_after,
                    "normalized_conflict_reduction": (
                        before_conflicts - conflicts_after
                    )
                    / max(1, before_conflicts),
                    "no_progress": conflicts_after >= before_conflicts,
                    "replan_success": bool(metrics["replan_success"]),
                    "feasible": bool(after.get("feasible")),
                    "repair_outcome": classify_repair_outcome(
                        before_fingerprint=before_repair,
                        after_fingerprint=after_repair,
                        replan_success=bool(metrics["replan_success"]),
                        conflicts_before=before_conflicts,
                        conflicts_after=conflicts_after,
                        feasible=bool(after.get("feasible")),
                    ),
                    "after_repair_fingerprint": after_repair,
                }
            )
    payload = {
        "schema": STATE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_fingerprint": state_key,
        "state_record_fingerprint": _fingerprint(state_record),
        "state_blob_sha256": str(state_record["state_blob_sha256"]),
        "before_repair_fingerprint": before_repair,
        "before_conflicts": before_conflicts,
        "state_restore": {
            "contract": TARGET_STATE_RESTORE_CONTRACT,
            "restore_seed": restore_seed,
            "source_run_config_sha256": str(
                state_record["source_run_config_sha256"]
            ),
        },
        "logical_checkpoint_ids": list(state_record["logical_checkpoint_ids"]),
        "candidates": candidates,
        "trials": trials,
        "native_action_semantics_validated": True,
        "selected_feature_reproduction_validated": True,
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
    }
    if not _state_artifact_valid(
        payload,
        state_record=state_record,
        run_fingerprint=run_fingerprint,
        trial_indices=trial_indices,
    ):
        raise RuntimeError("MarginalPool replay state artifact is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_fingerprint": state_key,
        "state_file": str(output_path),
        "status": "ok",
        "state_count": 1,
        "outcome_count": len(trials),
        "candidate_count": len(candidates),
        "trial_count": len(trials),
        "error_count": 0,
    }


def _validate_preflight(
    preflight_root: Path,
    *,
    config_sha256: str,
    selected_state_fingerprints: list[str],
) -> dict[str, Any]:
    report_path = preflight_root / "collection_report.json"
    if not report_path.is_file():
        raise ValueError("MarginalPool full replay requires completed preflight")
    report = _read_json(report_path)
    if (
        report.get("schema") != COLLECTION_REPORT_SCHEMA
        or report.get("mode") != "preflight"
        or report.get("complete") is not True
        or report.get("integrity_passed") is not True
        or report.get("config_sha256") != config_sha256
        or list(report.get("selected_state_fingerprints") or ())
        != selected_state_fingerprints
        or report.get("outcomes_used_for_protocol_changes") is not False
    ):
        raise ValueError("MarginalPool action-replay preflight contract changed")
    return report


def collect_marginalpool_action_replay(
    *,
    config_path: str | Path,
    output: str | Path,
    mode: str,
    workers: int | None = None,
    resume: bool = False,
    preflight_output: str | Path | None = None,
) -> dict[str, Any]:
    if mode not in {"preflight", "full"}:
        raise ValueError("MarginalPool replay mode must be preflight or full")
    metadata, all_states, logical_rows = build_frozen_cohort(config_path)
    config_path, root, config, _inputs = load_registration(config_path)
    worker_count = int(workers or config["execution"]["workers"])
    if worker_count <= 0:
        raise ValueError("MarginalPool replay workers must be positive")
    first_two = [str(row["state_fingerprint"]) for row in all_states[:2]]
    if mode == "preflight":
        selected_states = all_states[:2]
        trial_indices = tuple(map(int, config["preflight"]["trial_indices"]))
    else:
        if preflight_output is None:
            raise ValueError("MarginalPool full replay requires --preflight-output")
        _validate_preflight(
            Path(preflight_output).resolve(),
            config_sha256=str(metadata["config_sha256"]),
            selected_state_fingerprints=first_two,
        )
        selected_states = all_states
        trial_indices = TRIAL_INDICES

    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    run_identity = {
        "schema": RUN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "config_path": str(config_path),
        "config_sha256": str(metadata["config_sha256"]),
        "input_sha256": dict(metadata["input_sha256"]),
        "tailswitch_report_sha256": str(metadata["tailswitch_report_sha256"]),
        "cohort_fingerprint": str(metadata["cohort_fingerprint"]),
        "selected_state_fingerprints": [
            str(row["state_fingerprint"]) for row in selected_states
        ],
        "trial_indices": list(trial_indices),
        "target_state_restore_contract": TARGET_STATE_RESTORE_CONTRACT,
        "source_run_config_sha256": {
            str(row["state_fingerprint"]): str(row["source_run_config_sha256"])
            for row in selected_states
        },
        "producer": producer,
    }
    run_fingerprint = _fingerprint(run_identity)
    output = Path(output).resolve()
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("MarginalPool replay output belongs to another run")
        if not resume:
            raise ValueError("MarginalPool replay output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**run_identity, "run_fingerprint": run_fingerprint})
    logical_by_id = {
        str(row["logical_checkpoint_id"]): row for row in logical_rows
    }
    jobs = [
        {
            "job_id": str(row["state_fingerprint"]),
            "state_record": row,
            "logical_by_id": {
                logical_id: logical_by_id[logical_id]
                for logical_id in row["logical_checkpoint_ids"]
            },
            "trial_indices": list(trial_indices),
            "output_path": str(
                output / "states" / f"{str(row['state_fingerprint'])}.json"
            ),
            "run_fingerprint": run_fingerprint,
            "resume": bool(resume),
        }
        for row in selected_states
    ]
    status_path = output / "collection_status.json"
    observed: list[dict[str, Any]] = []

    def update_status(result: dict[str, Any]) -> None:
        observed.append(result)
        failures = [
            row for row in observed if row.get("status") in {"error", "timeout"}
        ]
        _write_json(
            status_path,
            {
                "schema": COLLECTION_REPORT_SCHEMA,
                "mode": mode,
                "run_fingerprint": run_fingerprint,
                "requested_state_count": len(jobs),
                "completed_state_count": len(observed) - len(failures),
                "completed_candidate_count": sum(
                    int(row.get("candidate_count", 0)) for row in observed
                ),
                "completed_trial_count": sum(
                    int(row.get("trial_count", 0)) for row in observed
                ),
                "error_state_count": sum(
                    row.get("status") == "error" for row in failures
                ),
                "timeout_state_count": sum(
                    row.get("status") == "timeout" for row in failures
                ),
                "active_jobs": [],
                "status": "running",
                "errors": failures,
            },
        )

    _write_json(
        status_path,
        {
            "schema": COLLECTION_REPORT_SCHEMA,
            "mode": mode,
            "run_fingerprint": run_fingerprint,
            "requested_state_count": len(jobs),
            "completed_state_count": 0,
            "completed_candidate_count": 0,
            "completed_trial_count": 0,
            "error_state_count": 0,
            "timeout_state_count": 0,
            "active_jobs": [],
            "status": "running",
            "errors": [],
        },
    )
    try:
        observed = _run_jobs(
            _collect_state,
            jobs,
            worker_count,
            phase=f"stride-marginalpool-action-replay-{mode}",
            output_root=output,
            run_fingerprint=run_fingerprint,
            timeout_seconds=float(config["execution"]["per_state_timeout_seconds"]),
            on_result=update_status,
        )
    except BaseException as error:
        current = _read_json(status_path)
        _write_json(
            status_path,
            {
                **current,
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                "runner_error": f"{type(error).__name__}: {error}",
            },
        )
        raise

    successes = [
        row for row in observed if row.get("status") in {"ok", "resumed"}
    ]
    errors = [
        {
            "state_fingerprint": str(row.get("state_fingerprint", row.get("job_id"))),
            "status": str(row.get("status")),
            "error": str(row.get("error")),
        }
        for row in observed
        if row.get("status") in {"error", "timeout"}
    ]
    state_by_key = {str(row["state_fingerprint"]): row for row in selected_states}
    all_trials: list[dict[str, Any]] = []
    all_aggregates: list[dict[str, Any]] = []
    state_manifest: list[dict[str, Any]] = []
    valid_state_count = 0
    for result in sorted(successes, key=lambda row: str(row["state_fingerprint"])):
        state_key = str(result["state_fingerprint"])
        state_record = state_by_key[state_key]
        state_file = Path(str(result["state_file"]))
        payload = _read_json(state_file)
        if not _state_artifact_valid(
            payload,
            state_record=state_record,
            run_fingerprint=run_fingerprint,
            trial_indices=trial_indices,
        ):
            errors.append(
                {
                    "state_fingerprint": state_key,
                    "status": "error",
                    "error": "invalid state artifact",
                }
            )
            continue
        valid_state_count += 1
        all_trials.extend(payload["trials"])
        trials_by_candidate: dict[str, list[dict[str, Any]]] = {}
        for trial in payload["trials"]:
            trials_by_candidate.setdefault(str(trial["candidate_id"]), []).append(trial)
        if mode == "full":
            for candidate in payload["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                all_aggregates.append(
                    {
                        **aggregate_candidate(
                            candidate, trials_by_candidate[candidate_id]
                        ),
                        "state_fingerprint": state_key,
                        "map_id": str(state_record["map_id"]),
                        "task_id": str(state_record["task_id"]),
                        "solver_seed": int(state_record["solver_seed"]),
                        "before_conflicts": int(payload["before_conflicts"]),
                        "logical_checkpoint_ids": list(
                            state_record["logical_checkpoint_ids"]
                        ),
                    }
                )
        state_manifest.append(
            {
                "state_fingerprint": state_key,
                "map_id": str(state_record["map_id"]),
                "task_id": str(state_record["task_id"]),
                "solver_seed": int(state_record["solver_seed"]),
                "before_conflicts": int(payload["before_conflicts"]),
                "candidate_count": len(payload["candidates"]),
                "trial_count": len(payload["trials"]),
                "logical_checkpoint_ids": list(state_record["logical_checkpoint_ids"]),
                "state_file": str(state_file),
                "state_file_sha256": sha256_file(state_file),
            }
        )

    candidate_count = sum(int(row["candidate_count"]) for row in state_manifest)
    trial_count = len(all_trials)
    expected_candidate_count = sum(len(row["candidates"]) for row in selected_states)
    expected_trial_count = expected_candidate_count * len(trial_indices)
    seeds_by_state_index: dict[tuple[str, int], set[int]] = {}
    seeds_by_state: dict[str, set[int]] = {}
    for trial in all_trials:
        key = (str(trial["state_fingerprint"]), int(trial["trial_index"]))
        seeds_by_state_index.setdefault(key, set()).add(int(trial["pp_seed"]))
        seeds_by_state.setdefault(key[0], set()).add(int(trial["pp_seed"]))
    gates = {
        "exact_selected_states": valid_state_count == len(selected_states),
        "exact_candidate_product": candidate_count == expected_candidate_count,
        "exact_trial_product": trial_count == expected_trial_count,
        "exact_full_cohort": mode != "full"
        or (
            valid_state_count == 78
            and candidate_count == 2502
            and trial_count == 40032
            and len(logical_rows) == 90
        ),
        "paired_and_distinct_pp_seeds": all(
            len(seeds_by_state_index.get((state_key, trial_index), set())) == 1
            for state_key in state_by_key
            for trial_index in trial_indices
        )
        and all(
            len(seeds_by_state.get(state_key, set())) == len(trial_indices)
            for state_key in state_by_key
        ),
        "all_124d_features": valid_state_count == len(successes),
        "state_blob_and_fingerprint_identity": valid_state_count == len(successes),
        "selected_feature_reproduction": valid_state_count == len(successes),
        "native_action_semantics": valid_state_count == len(successes),
        "zero_errors": not any(row["status"] == "error" for row in errors),
        "zero_timeouts": not any(row["status"] == "timeout" for row in errors),
        "no_runtime_ttf_or_future_fields": not _forbidden_hits(
            [*all_trials, *all_aggregates]
        ),
    }
    integrity_passed = all(gates.values())
    artifacts: dict[str, Any] = {}
    if integrity_passed:
        state_manifest_path = output / "state_manifest.jsonl"
        logical_manifest_path = output / "logical_checkpoint_manifest.jsonl"
        _write_jsonl(state_manifest_path, state_manifest)
        _write_jsonl(logical_manifest_path, logical_rows)
        artifacts.update(
            {
                "state_manifest": str(state_manifest_path),
                "state_manifest_sha256": sha256_file(state_manifest_path),
                "logical_checkpoint_manifest": str(logical_manifest_path),
                "logical_checkpoint_manifest_sha256": sha256_file(
                    logical_manifest_path
                ),
            }
        )
        if mode == "full":
            trials_path = output / "repair_trials.jsonl"
            aggregates_path = output / "candidate_aggregates.jsonl"
            _write_jsonl(trials_path, all_trials)
            _write_jsonl(aggregates_path, all_aggregates)
            artifacts.update(
                {
                    "repair_trials": str(trials_path),
                    "repair_trials_sha256": sha256_file(trials_path),
                    "candidate_aggregates": str(aggregates_path),
                    "candidate_aggregates_sha256": sha256_file(aggregates_path),
                }
            )
    report = {
        "schema": COLLECTION_REPORT_SCHEMA,
        "scientific_status": (
            "execution_integrity_preflight_outcomes_not_analyzed"
            if mode == "preflight"
            else "paired_current_step_all_candidate_replay"
        ),
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "run_fingerprint": run_fingerprint,
        "config_sha256": str(metadata["config_sha256"]),
        "complete": True,
        "integrity_passed": integrity_passed,
        "requested_state_count": len(selected_states),
        "completed_state_count": valid_state_count,
        "candidate_count": candidate_count,
        "trial_count": trial_count,
        "logical_checkpoint_count": len(logical_rows) if mode == "full" else sum(
            len(row["logical_checkpoint_ids"]) for row in selected_states
        ),
        "new_state_count": sum(row.get("status") == "ok" for row in successes),
        "resumed_state_count": sum(
            row.get("status") == "resumed" for row in successes
        ),
        "error_state_count": sum(row["status"] == "error" for row in errors),
        "timeout_state_count": sum(row["status"] == "timeout" for row in errors),
        "selected_state_fingerprints": [
            str(row["state_fingerprint"]) for row in selected_states
        ],
        "trial_indices": list(trial_indices),
        "gates": gates,
        "errors": errors,
        "artifacts": artifacts,
        "outcomes_used_for_protocol_changes": False,
        "runtime_or_ttf_read": False,
        "future_trajectory_read": False,
        "claim_boundary": dict(config["claim_boundary"]),
    }
    report_path = output / "collection_report.json"
    _write_json(report_path, report)
    _write_json(
        status_path,
        {
            **report,
            "status": "complete" if integrity_passed else "failed",
            "collection_report_sha256": sha256_file(report_path),
        },
    )
    return report


def _winner_ids(
    rows: list[dict[str, Any]], field: str, *, tolerance: float = 1e-15
) -> list[str]:
    best = max(float(row[field]) for row in rows)
    return sorted(
        str(row["candidate_id"])
        for row in rows
        if float(row[field]) >= best - tolerance
    )


def _logical_result(
    logical: dict[str, Any], aggregates: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    state_key = str(logical["state_fingerprint"])
    rows = [aggregates[(state_key, candidate_id)] for candidate_id in logical["candidate_ids"]]
    selected = aggregates[(state_key, str(logical["selected_candidate_id"]))]
    alternatives = [
        row for row in rows if str(row["candidate_id"]) != str(selected["candidate_id"])
    ]
    dominators = [row for row in alternatives if stable_dominates(row, selected)]
    full_winners = _winner_ids(rows, "seed_mean")
    first_winners = _winner_ids(rows, "first_fixed_half_mean")
    second_winners = _winner_ids(rows, "second_fixed_half_mean")
    direction_reversal = any(
        (
            float(row["first_fixed_half_mean"])
            > float(selected["first_fixed_half_mean"]) + 1e-15
        )
        != (
            float(row["second_fixed_half_mean"])
            > float(selected["second_fixed_half_mean"]) + 1e-15
        )
        for row in alternatives
    )
    no_positive_candidate = max(float(row["seed_mean"]) for row in rows) <= 0.0
    selected_is_best = str(selected["candidate_id"]) in full_winners and not dominators
    if dominators:
        root_class = "ranker_error"
    elif no_positive_candidate:
        root_class = "pool_or_operator_error"
    elif selected_is_best:
        root_class = "selected_is_current_step_best"
    else:
        root_class = "small_margin_or_seed_uncertain"
    best = max(
        rows,
        key=lambda row: (float(row["seed_mean"]), -float(row["no_progress_rate"]), str(row["candidate_id"])),
    )
    return {
        **logical,
        "root_class": root_class,
        "selected_seed_mean": float(selected["seed_mean"]),
        "selected_no_progress_rate": float(selected["no_progress_rate"]),
        "best_candidate_id": str(best["candidate_id"]),
        "best_candidate_kind": str(best["candidate_kind"]),
        "best_seed_mean": float(best["seed_mean"]),
        "best_no_progress_rate": float(best["no_progress_rate"]),
        "selected_normalized_regret": float(best["seed_mean"])
        - float(selected["seed_mean"]),
        "stable_dominator_count": len(dominators),
        "stable_dominator_ids": sorted(str(row["candidate_id"]) for row in dominators),
        "full_winner_ids": full_winners,
        "first_half_winner_ids": first_winners,
        "second_half_winner_ids": second_winners,
        "half_winner_overlap": sorted(set(first_winners) & set(second_winners)),
        "half_winner_disjoint": not bool(set(first_winners) & set(second_winners)),
        "selected_relative_direction_reversal": direction_reversal,
        "pp_uncertainty": not bool(set(first_winners) & set(second_winners))
        or direction_reversal,
        "current_step_only": True,
    }


def _summarize_logical(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = Counter(str(row["root_class"]) for row in rows)
    return {
        "checkpoint_count": len(rows),
        "root_class_counts": dict(sorted(class_counts.items())),
        "ranker_error_fraction": _mean(
            row["root_class"] == "ranker_error" for row in rows
        ),
        "selected_current_step_best_fraction": _mean(
            row["root_class"] == "selected_is_current_step_best" for row in rows
        ),
        "pool_or_operator_error_fraction": _mean(
            row["root_class"] == "pool_or_operator_error" for row in rows
        ),
        "pp_uncertainty_fraction": _mean(row["pp_uncertainty"] for row in rows),
        "mean_selected_normalized_regret": _mean(
            row["selected_normalized_regret"] for row in rows
        ),
        "mean_stable_dominator_count": _mean(
            row["stable_dominator_count"] for row in rows
        ),
    }


def analyze_marginalpool_action_replay(
    *, config_path: str | Path, collection: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, _root, config, _inputs = load_registration(config_path)
    collection = Path(collection).resolve()
    report_path = collection / "collection_report.json"
    report = _read_json(report_path)
    if (
        report.get("schema") != COLLECTION_REPORT_SCHEMA
        or report.get("mode") != "full"
        or report.get("complete") is not True
        or report.get("integrity_passed") is not True
        or report.get("config_sha256") != sha256_file(config_path)
        or int(report.get("completed_state_count", -1)) != 78
        or int(report.get("candidate_count", -1)) != 2502
        or int(report.get("trial_count", -1)) != 40032
    ):
        raise ValueError("MarginalPool action replay is not a passed full collection")
    artifacts = dict(report["artifacts"])
    aggregate_path = contained_file(
        collection,
        Path(str(artifacts["candidate_aggregates"])).name,
        field="MarginalPool candidate aggregates",
    )
    logical_path = contained_file(
        collection,
        Path(str(artifacts["logical_checkpoint_manifest"])).name,
        field="MarginalPool logical checkpoint manifest",
    )
    if (
        sha256_file(aggregate_path) != artifacts["candidate_aggregates_sha256"]
        or sha256_file(logical_path)
        != artifacts["logical_checkpoint_manifest_sha256"]
    ):
        raise ValueError("MarginalPool action-replay analysis input changed")
    aggregate_rows = _read_jsonl(aggregate_path)
    logical_rows = _read_jsonl(logical_path)
    aggregates = {
        (str(row["state_fingerprint"]), str(row["candidate_id"])): row
        for row in aggregate_rows
    }
    if len(aggregates) != 2502 or len(logical_rows) != 90:
        raise ValueError("MarginalPool action-replay analysis product changed")
    results = [_logical_result(row, aggregates) for row in logical_rows]
    by_classification = {
        label: _summarize_logical(
            [row for row in results if str(row["classification"]) == label]
        )
        for label in sorted({str(row["classification"]) for row in results})
    }
    by_checkpoint = {
        label: _summarize_logical(
            [row for row in results if str(row["checkpoint_kind"]) == label]
        )
        for label in sorted({str(row["checkpoint_kind"]) for row in results})
    }
    by_challenger = {
        label: _summarize_logical(
            [row for row in results if str(row["challenger"]) == label]
        )
        for label in sorted({str(row["challenger"]) for row in results})
    }
    summary = _summarize_logical(results)
    class_counts = Counter(str(row["root_class"]) for row in results)
    if class_counts.get("ranker_error", 0) > 0:
        next_step = "separate ranker-error checkpoints from closure/pool failures before training"
    elif class_counts.get("selected_is_current_step_best", 0) > 0:
        next_step = "preregister two-step repair-closure sequence diagnostic"
    else:
        next_step = "design repair-closure candidates before ranker training"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "logical_checkpoint_results.jsonl"
    _write_jsonl(result_path, results)
    analysis = {
        "schema": ANALYSIS_REPORT_SCHEMA,
        "scientific_status": "checkpoint_local_current_step_root_classification_no_ttf",
        "experiment_id": EXPERIMENT_ID,
        "integrity_passed": True,
        "collection_report_sha256": sha256_file(report_path),
        "candidate_aggregates_sha256": sha256_file(aggregate_path),
        "logical_checkpoint_manifest_sha256": sha256_file(logical_path),
        "logical_checkpoint_results_sha256": sha256_file(result_path),
        "summary": summary,
        "by_classification": by_classification,
        "by_checkpoint_kind": by_checkpoint,
        "by_challenger": by_challenger,
        "next_step": next_step,
        "current_step_only": True,
        "runtime_or_ttf_read": False,
        "future_trajectory_read": False,
        "claim_boundary": dict(config["claim_boundary"]),
    }
    analysis_path = output / "action_replay_analysis.json"
    _write_json(analysis_path, analysis)
    _write_json(
        output / "action_replay_analysis_status.json",
        {
            "schema": ANALYSIS_STATUS_SCHEMA,
            "complete": True,
            "integrity_passed": True,
            "analysis_report_sha256": sha256_file(analysis_path),
            "logical_checkpoint_results_sha256": sha256_file(result_path),
        },
    )
    return analysis
