from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_collection import (
    _paired_action,
    _replay_job,
    _validate_native_repair,
    load_stride_selection,
)
from experiments.stride_repairability_collection import (
    _source_target_state,
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.stride_robustaction_preflight import (
    _state_artifact_valid as preflight_state_artifact_valid,
)
from experiments.trace_replay import (
    TARGET_STATE_RESTORE_CONTRACT,
    restore_repair_state,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


CONFIG_SCHEMA = "lns2.stride.robustaction_structpool_label_collection_config.v1"
RUN_SCHEMA = "lns2.stride.robustaction_structpool_label_collection_run.v1"
STATE_SCHEMA = "lns2.stride.robustaction_structpool_label_collection_state.v1"
TRIAL_SCHEMA = "lns2.stride.robustaction_structpool_label_trial.v1"
AGGREGATE_SCHEMA = "lns2.stride.robustaction_current_step_label.v1"
REPORT_SCHEMA = "lns2.stride.robustaction_structpool_label_collection_report.v1"
SELECTION_POLICIES = ("official_adaptive", "v2-full")
TRIAL_INDICES = tuple(range(16))
FIRST_HALF = tuple(range(8))
SECOND_HALF = tuple(range(8, 16))
FORBIDDEN_PRODUCT_FIELDS = {
    "cost_to_go",
    "future_repair_rounds",
    "future_trajectory",
    "native_step_seconds",
    "pp_replan_seconds",
    "receding_q",
    "repair_runtime",
    "time_to_feasible",
    "ttf",
}
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/repair_collection.py",
    "experiments/stride_collection.py",
    "experiments/stride_repairability_collection.py",
    "experiments/stride_robustaction_label_collection.py",
    "experiments/stride_robustaction_preflight.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/fingerprints.py",
    "lns2_selector/runtime/repair_outcomes.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _registered(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"registered RobustAction label input is missing: {path}")
    observed = sha256_file(path)
    if observed != str(spec["sha256"]):
        raise ValueError(
            f"registered RobustAction label input changed: {path}: "
            f"expected {spec['sha256']}, got {observed}"
        )
    return path


def validate_robustaction_label_collection_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("RobustAction label-collection schema changed")
    if (
        config.get("scientific_status")
        != "preregistered_16_paired_pp_seed_current_step_labels_before_candidate_repairs"
        or config.get("collection_id")
        != "stride-robustaction-structpool-label-collection-v1"
        or config.get("data_line_id")
        != "stride-robustaction-structpool-recovery-data-v2"
        or config.get("planned_model_id") != "stride-robustaction-v1"
        or config.get("candidate_pool_id") != "v2-plus-stride-structpool-v1"
        or config.get("pre_registration_git_commit")
        != "702b66d2e0f993aebd9123aec1afab9889bbf3d4"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
    ):
        raise ValueError("RobustAction label-collection identity changed")

    if set(config.get("inputs") or {}) != {
        "selection",
        "preflight_config",
        "preflight_run_config",
        "preflight_report",
        "preflight_rows",
        "frozen_v2_manifest",
    }:
        raise ValueError("RobustAction label input registry changed")
    if (
        config.get("preflight_state_artifact_root")
        != "build/stride-robustaction-label-preflight-v1/states"
        or config.get("preflight_state_artifact_tree_sha256")
        != "6bc5f8a4d0edcb2881eee4361fc5c5c4362b32f3cbc2953ef8a8a75998c07bc2"
    ):
        raise ValueError("RobustAction preflight state registry changed")

    cohort = dict(config.get("cohort_contract") or {})
    if cohort != {
        "research_split": "train",
        "state_count": 320,
        "episode_count": 216,
        "states_per_source_policy": {
            "official_adaptive": 160,
            "v2-full": 160,
        },
        "map_count": 28,
        "task_count": 56,
        "maximum_states_per_episode": 2,
        "active_structpool_state_count": 98,
        "active_structpool_states_per_source_policy": {
            "official_adaptive": 51,
            "v2-full": 47,
        },
    }:
        raise ValueError("RobustAction label cohort changed")

    candidates = dict(config.get("candidate_contract") or {})
    if candidates != {
        "candidate_count": 6285,
        "candidate_count_distribution": {
            "16": 15,
            "17": 24,
            "18": 183,
            "22": 1,
            "23": 7,
            "24": 90,
        },
        "added_candidate_count_distribution": {"0": 222, "5": 4, "6": 94},
        "feature_schema": "lns2.realized_features.v2",
        "feature_profile": "realized_dynamic",
        "feature_dimension": 124,
        "exact_preflight_candidate_and_feature_reuse": True,
    }:
        raise ValueError("RobustAction label candidate contract changed")

    paired = dict(config.get("paired_repair_contract") or {})
    if (
        tuple(map(int, paired.get("trial_indices") or ())) != TRIAL_INDICES
        or tuple(map(int, paired.get("first_half_indices") or ())) != FIRST_HALF
        or tuple(map(int, paired.get("second_half_indices") or ())) != SECOND_HALF
        or int(paired.get("trials_per_candidate", -1)) != 16
        or int(paired.get("expected_trial_count", -1)) != 100560
        or paired.get("seed_namespace") != "stride-repairability-paired-pp-v1"
        or paired.get("seed_pairing_unit")
        != "same_state_repair_fingerprint_and_trial_index_across_all_candidates"
        or paired.get("restore_contract") != TARGET_STATE_RESTORE_CONTRACT
        or paired.get("replan_algorithm") != "PP"
        or paired.get("runtime_or_pp_time_stored") is not False
        or paired.get("future_trajectory_stored") is not False
    ):
        raise ValueError("RobustAction paired repair contract changed")

    label = dict(config.get("label_contract") or {})
    if (
        label.get("schema") != AGGREGATE_SCHEMA
        or label.get("per_seed_target")
        != "normalized_current_step_conflict_reduction"
        or label.get("formula")
        != "(conflicts_before-conflicts_after)/max(1,conflicts_before)"
        or list(label.get("candidate_aggregates") or ())
        != [
            "seed_mean",
            "seed_standard_deviation",
            "lower_half_mean",
            "first_fixed_half_mean",
            "second_fixed_half_mean",
            "progress_rate",
            "replan_success_rate",
            "feasible_rate",
        ]
        or any(
            label.get(name) is not False
            for name in (
                "single_seed_winner_used",
                "post_structure_penalty_used",
                "repair_runtime_used",
                "future_repair_rounds_used",
                "cost_to_go_used",
                "receding_q_used",
                "ttf_used",
            )
        )
    ):
        raise ValueError("RobustAction current-step label contract changed")

    opportunity = dict(config.get("post_collection_opportunity_audit") or {})
    if (
        opportunity.get("anchor")
        != "frozen_v2_ranking_over_the_exact_same_candidate_pool"
        or float(opportunity.get("minimum_paired_win_fraction", -1.0)) != 0.75
        or float(opportunity.get("minimum_absolute_mean_effect", -1.0)) != 0.02
        or opportunity.get("require_both_fixed_half_mean_directions") is not True
        or float(opportunity.get("tie_epsilon", -1.0)) != 1e-12
        or float(
            opportunity.get("minimum_states_with_nonanchor_robust_win_fraction", -1.0)
        )
        != 0.25
        or float(opportunity.get("minimum_robust_positive_action_fraction", -1.0))
        != 0.04
        or dict(
            opportunity.get("minimum_positive_opportunity_maps_by_topology_group")
            or {}
        )
        != {
            "dao_high_topology": 3,
            "dao_mid_topology": 3,
            "dao_low_topology_control": 2,
        }
        or list(opportunity.get("action_uncertainty_outputs") or ())
        != [
            "fixed_half_exact_winner_agreement",
            "fixed_half_top3_overlap",
            "fixed_half_pairwise_direction_agreement",
            "candidate_seed_standard_deviation",
        ]
    ):
        raise ValueError("RobustAction opportunity audit contract changed")

    gates = dict(config.get("integrity_gates") or {})
    if set(gates) != {
        "require_all_320_states",
        "require_exact_6285_candidates",
        "require_exact_100560_trial_product",
        "require_paired_and_distinct_pp_seeds",
        "require_exact_preflight_candidates_and_features",
        "require_state_and_repair_fingerprint_identity",
        "require_native_action_semantics",
        "require_zero_errors",
        "require_zero_timeouts",
        "require_no_runtime_or_future_fields",
    } or not all(value is True for value in gates.values()):
        raise ValueError("RobustAction label integrity gates changed")

    execution = dict(config.get("execution") or {})
    if execution != {
        "recommended_workers": 2,
        "per_state_timeout_seconds": 7200,
        "resume_required_for_existing_output": True,
        "preserve_complete_failed_product": True,
        "monitor_interval_minutes": 30,
    }:
        raise ValueError("RobustAction label execution contract changed")

    if project_root is not None:
        for spec in dict(config["inputs"]).values():
            _registered(project_root.resolve(), dict(spec))


def state_artifact_tree_sha256(root: str | Path) -> str:
    root = Path(root).resolve()
    rows = [
        {"file": path.name, "sha256": sha256_file(path)}
        for path in sorted(root.glob("*.json"), key=lambda value: value.name)
    ]
    serialized = json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _candidate_signature(candidates: list[dict[str, Any]]) -> str:
    return _fingerprint(
        [
            {
                "candidate_id": str(row["candidate_id"]),
                "agents": list(map(int, row["agents"])),
                "selection_families": sorted(
                    map(str, row.get("selection_families") or ())
                ),
                "structpool_family_groups": sorted(
                    map(str, row.get("structpool_family_groups") or ())
                ),
            }
            for row in candidates
        ]
    )


def _feature_signature(candidates: list[dict[str, Any]]) -> str:
    return _fingerprint(
        [
            {
                "candidate_id": str(row["candidate_id"]),
                "features": dict(row["features"]),
            }
            for row in candidates
        ]
    )


def _aggregate_candidate(
    candidate: dict[str, Any], trials: list[dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(trials, key=lambda row: int(row["trial_index"]))
    values = [float(row["normalized_conflict_reduction"]) for row in ordered]
    lower_half = sorted(values)[: len(values) // 2]
    return {
        "schema": AGGREGATE_SCHEMA,
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_kind": str(candidate["candidate_kind"]),
        "agents": list(map(int, candidate["agents"])),
        "actual_size": int(candidate["actual_size"]),
        "selection_families": list(candidate["selection_families"]),
        "structpool_family_groups": list(candidate["structpool_family_groups"]),
        "trial_count": len(values),
        "seed_mean": statistics.fmean(values),
        "seed_standard_deviation": statistics.pstdev(values),
        "lower_half_mean": statistics.fmean(lower_half),
        "first_fixed_half_mean": statistics.fmean(values[:8]),
        "second_fixed_half_mean": statistics.fmean(values[8:]),
        "progress_rate": statistics.fmean(
            float(row["conflicts_after"] < row["before_conflicts"])
            for row in ordered
        ),
        "replan_success_rate": statistics.fmean(
            float(row["replan_success"]) for row in ordered
        ),
        "feasible_rate": statistics.fmean(float(row["feasible"]) for row in ordered),
        "minimum_seed_score": min(values),
        "maximum_seed_score": max(values),
        "features": dict(candidate["features"]),
    }


def _forbidden_hits(value: Any) -> set[str]:
    hits: set[str] = set()
    if isinstance(value, dict):
        hits.update(FORBIDDEN_PRODUCT_FIELDS & set(map(str, value)))
        for nested in value.values():
            hits.update(_forbidden_hits(nested))
    elif isinstance(value, list):
        for nested in value:
            hits.update(_forbidden_hits(nested))
    return hits


def _state_artifact_valid(
    payload: dict[str, Any],
    *,
    run_fingerprint: str,
    decision: dict[str, Any],
    preflight_payload: dict[str, Any],
) -> bool:
    state_id = str(decision["state_id"])
    if (
        payload.get("schema") != STATE_SCHEMA
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("state_id") != state_id
        or payload.get("decision") != decision
        or payload.get("complete") is not True
        or payload.get("before_fingerprint") != decision.get("before_fingerprint")
        or payload.get("before_conflicts") != decision.get("before_conflicts")
        or payload.get("before_repair_fingerprint")
        != preflight_payload.get("before_repair_fingerprint")
        or payload.get("candidate_signature")
        != preflight_payload.get("candidate_signature")
        or payload.get("feature_signature")
        != preflight_payload.get("feature_signature")
        or payload.get("candidate_repair_trials_executed") is not True
        or payload.get("controller_actions_executed") is not False
        or payload.get("runtime_fields_stored") is not False
        or payload.get("future_trajectory_stored") is not False
        or payload.get("native_action_semantics_validated") is not True
        or _forbidden_hits(payload)
    ):
        return False
    before_repair = str(payload["before_repair_fingerprint"])
    state_restore = payload.get("state_restore")
    if (
        not before_repair
        or not isinstance(state_restore, dict)
        or state_restore.get("contract") != TARGET_STATE_RESTORE_CONTRACT
        or state_restore.get("restore_seed")
        != repairability_restore_seed(before_repair)
        or state_restore.get("repair_structure_fingerprint") != before_repair
    ):
        return False

    candidates = payload.get("candidates")
    trials = payload.get("trials")
    aggregates = payload.get("candidate_aggregates")
    if (
        not isinstance(candidates, list)
        or candidates != preflight_payload.get("candidates")
        or not isinstance(trials, list)
        or not isinstance(aggregates, list)
        or _candidate_signature(candidates) != payload["candidate_signature"]
        or _feature_signature(candidates) != payload["feature_signature"]
    ):
        return False
    candidate_by_id = {str(row.get("candidate_id", "")): row for row in candidates}
    if (
        not all(candidate_by_id)
        or len(candidate_by_id) != len(candidates)
        or len(aggregates) != len(candidates)
    ):
        return False
    required_features = set(PROFILE_FEATURE_NAMES["realized_dynamic"])
    for candidate in candidates:
        features = candidate.get("features")
        if (
            not isinstance(features, dict)
            or set(features) != required_features
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in features.values()
            )
        ):
            return False

    observed_product: set[tuple[str, int]] = set()
    trials_by_candidate: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in candidate_by_id
    }
    before_conflicts = int(payload["before_conflicts"])
    for trial in trials:
        if not isinstance(trial, dict) or trial.get("schema") != TRIAL_SCHEMA:
            return False
        candidate_id = str(trial.get("candidate_id", ""))
        trial_index = trial.get("trial_index")
        if (
            candidate_id not in candidate_by_id
            or type(trial_index) is not int
            or int(trial_index) not in TRIAL_INDICES
            or (candidate_id, int(trial_index)) in observed_product
            or trial.get("state_id") != state_id
            or trial.get("candidate_kind")
            != candidate_by_id[candidate_id].get("candidate_kind")
            or trial.get("before_conflicts") != before_conflicts
            or trial.get("before_fingerprint") != payload["before_fingerprint"]
            or trial.get("before_repair_fingerprint") != before_repair
            or trial.get("pp_seed")
            != repairability_pp_seed(before_repair, int(trial_index))
            or type(trial.get("conflicts_after")) is not int
            or int(trial["conflicts_after"]) < 0
            or type(trial.get("replan_success")) is not bool
            or type(trial.get("feasible")) is not bool
            or not isinstance(trial.get("after_repair_fingerprint"), str)
            or not trial["after_repair_fingerprint"]
        ):
            return False
        expected_reduction = (
            before_conflicts - int(trial["conflicts_after"])
        ) / max(1, before_conflicts)
        if (
            not math.isfinite(float(trial.get("normalized_conflict_reduction", math.nan)))
            or abs(float(trial["normalized_conflict_reduction"]) - expected_reduction)
            > 1e-15
        ):
            return False
        try:
            expected_outcome = classify_repair_outcome(
                before_fingerprint=before_repair,
                after_fingerprint=str(trial["after_repair_fingerprint"]),
                replan_success=bool(trial["replan_success"]),
                conflicts_before=before_conflicts,
                conflicts_after=int(trial["conflicts_after"]),
                feasible=bool(trial["feasible"]),
            )
        except ValueError:
            return False
        if trial.get("repair_outcome") != expected_outcome:
            return False
        observed_product.add((candidate_id, int(trial_index)))
        trials_by_candidate[candidate_id].append(trial)
    expected_product = {
        (candidate_id, trial_index)
        for candidate_id in candidate_by_id
        for trial_index in TRIAL_INDICES
    }
    if observed_product != expected_product:
        return False

    expected_aggregates = {
        candidate_id: _aggregate_candidate(
            candidate, trials_by_candidate[candidate_id]
        )
        for candidate_id, candidate in candidate_by_id.items()
    }
    aggregate_by_id = {
        str(row.get("candidate_id", "")): row
        for row in aggregates
        if isinstance(row, dict)
    }
    return aggregate_by_id == expected_aggregates


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    decision = dict(job["decision"])
    preflight_path = Path(str(job["preflight_path"]))
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    preflight = _read_json(preflight_path)
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if _state_artifact_valid(
            existing,
            run_fingerprint=run_fingerprint,
            decision=decision,
            preflight_payload=preflight,
        ):
            return {
                "state_id": str(decision["state_id"]),
                "state_file": str(output_path),
                "status": "resumed",
                "state_count": 1,
                "outcome_count": len(existing["trials"]),
                "error_count": 0,
                "candidate_count": len(existing["candidates"]),
                "trial_count": len(existing["trials"]),
            }
        raise ValueError(f"invalid completed RobustAction label state: {output_path}")

    replay = _replay_job(decision)
    state, source_manifest, source_trace_path = _source_target_state(decision)
    before_fingerprint = state_fingerprint(state)
    before_repair = repair_structure_fingerprint(state)
    before_conflicts = int(state["num_of_colliding_pairs"])
    if (
        before_fingerprint != str(decision["before_fingerprint"])
        or before_repair != str(preflight["before_repair_fingerprint"])
        or before_conflicts != int(decision["before_conflicts"])
        or before_conflicts <= 0
    ):
        raise RuntimeError("RobustAction label source state differs from preflight")
    restore_seed = repairability_restore_seed(before_repair)
    _environment, restored = restore_repair_state(
        replay, state, seed=restore_seed
    )
    if repair_structure_fingerprint(restored) != before_repair:
        raise RuntimeError("RobustAction label restore changed repair structure")

    candidates = list(preflight["candidates"])
    trials: list[dict[str, Any]] = []
    trials_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        agents = list(map(int, candidate["agents"]))
        candidate_trials: list[dict[str, Any]] = []
        for trial_index in TRIAL_INDICES:
            branch_environment, branch_state = restore_repair_state(
                replay, state, seed=restore_seed
            )
            if repair_structure_fingerprint(branch_state) != before_repair:
                raise RuntimeError("RobustAction paired branch restore changed")
            pp_seed = repairability_pp_seed(before_repair, trial_index)
            result = _plain(
                branch_environment.step(_paired_action(agents, pp_seed))
            )
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_repair = repair_structure_fingerprint(after)
            repair_outcome = classify_repair_outcome(
                before_fingerprint=before_repair,
                after_fingerprint=after_repair,
                replan_success=bool(metrics["replan_success"]),
                conflicts_before=before_conflicts,
                conflicts_after=conflicts_after,
                feasible=bool(after.get("feasible")),
            )
            trial = {
                "schema": TRIAL_SCHEMA,
                "state_id": str(decision["state_id"]),
                "candidate_id": candidate_id,
                "candidate_kind": str(candidate["candidate_kind"]),
                "trial_index": trial_index,
                "pp_seed": pp_seed,
                "before_conflicts": before_conflicts,
                "before_fingerprint": before_fingerprint,
                "before_repair_fingerprint": before_repair,
                "conflicts_after": conflicts_after,
                "normalized_conflict_reduction": (
                    before_conflicts - conflicts_after
                )
                / max(1, before_conflicts),
                "replan_success": bool(metrics["replan_success"]),
                "feasible": bool(after.get("feasible")),
                "repair_outcome": repair_outcome,
                "after_repair_fingerprint": after_repair,
            }
            candidate_trials.append(trial)
            trials.append(trial)
        trials_by_candidate[candidate_id] = candidate_trials

    payload = {
        "schema": STATE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_id": str(decision["state_id"]),
        "decision": decision,
        "before_fingerprint": before_fingerprint,
        "before_repair_fingerprint": before_repair,
        "before_conflicts": before_conflicts,
        "state_restore": {
            "contract": TARGET_STATE_RESTORE_CONTRACT,
            "restore_seed": restore_seed,
            "repair_structure_fingerprint": before_repair,
            "source_trace_file": str(source_manifest["trace_file"]),
            "source_trace_path": str(source_trace_path),
        },
        "preflight_state_file": str(preflight_path),
        "preflight_state_sha256": sha256_file(preflight_path),
        "candidate_signature": str(preflight["candidate_signature"]),
        "feature_signature": str(preflight["feature_signature"]),
        "candidates": candidates,
        "trials": trials,
        "candidate_aggregates": [
            _aggregate_candidate(
                candidate,
                trials_by_candidate[str(candidate["candidate_id"])],
            )
            for candidate in candidates
        ],
        "candidate_repair_trials_executed": True,
        "controller_actions_executed": False,
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
        "native_action_semantics_validated": True,
    }
    if not _state_artifact_valid(
        payload,
        run_fingerprint=run_fingerprint,
        decision=decision,
        preflight_payload=preflight,
    ):
        raise RuntimeError("RobustAction label state artifact is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": str(decision["state_id"]),
        "state_file": str(output_path),
        "status": "ok",
        "state_count": 1,
        "outcome_count": len(trials),
        "error_count": 0,
        "candidate_count": len(candidates),
        "trial_count": len(trials),
    }


def _load_preflight_index(
    *,
    root: Path,
    expected_tree_sha256: str,
    preflight_run_fingerprint: str,
    decisions: list[dict[str, Any]],
) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(f"RobustAction preflight state root is missing: {root}")
    observed_tree = state_artifact_tree_sha256(root)
    if observed_tree != expected_tree_sha256:
        raise ValueError(
            "RobustAction preflight state tree changed: "
            f"expected {expected_tree_sha256}, got {observed_tree}"
        )
    decisions_by_id = {str(row["state_id"]): row for row in decisions}
    index: dict[str, Path] = {}
    candidate_counts: Counter[int] = Counter()
    added_counts: Counter[int] = Counter()
    candidate_total = 0
    for path in sorted(root.glob("*.json"), key=lambda value: value.name):
        payload = _read_json(path)
        state_id = str(payload.get("state_id", ""))
        decision = decisions_by_id.get(state_id)
        if (
            decision is None
            or state_id in index
            or not preflight_state_artifact_valid(
                payload,
                run_fingerprint=preflight_run_fingerprint,
                decision=decision,
            )
        ):
            raise ValueError(f"invalid RobustAction preflight state artifact: {path}")
        index[state_id] = path
        candidate_count = len(payload["candidates"])
        candidate_counts[candidate_count] += 1
        added_counts[int(payload["added_candidate_count"])] += 1
        candidate_total += candidate_count
    if set(index) != set(decisions_by_id):
        raise ValueError("RobustAction preflight state coverage differs from selection")
    if (
        candidate_total != 6285
        or dict(candidate_counts)
        != {16: 15, 17: 24, 18: 183, 22: 1, 23: 7, 24: 90}
        or dict(added_counts) != {0: 222, 5: 4, 6: 94}
    ):
        raise ValueError("RobustAction preflight candidate product changed")
    return index


def collect_robustaction_labels(
    *,
    config_path: str | Path,
    output: str | Path,
    workers: int = 2,
    resume: bool = False,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("RobustAction label workers must be positive")
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robustaction_label_collection_config(
        config, project_root=project_root
    )
    inputs = {
        name: _registered(project_root, dict(spec))
        for name, spec in dict(config["inputs"]).items()
    }
    preflight_run = _read_json(inputs["preflight_run_config"])
    preflight_report = _read_json(inputs["preflight_report"])
    if (
        preflight_report.get("passed") is not True
        or int(preflight_report.get("completed_state_count", -1)) != 320
        or int(preflight_report.get("error_state_count", -1)) != 0
        or preflight_report.get("run_fingerprint")
        != preflight_run.get("run_fingerprint")
        or preflight_report.get("candidate_repair_trials_executed") is not False
        or preflight_report.get("ttf_read") is not False
    ):
        raise ValueError("RobustAction label collection requires passed preflight")
    selected = load_stride_selection(inputs["selection"])
    cohort = dict(config["cohort_contract"])
    policy_counts = Counter(str(row["source_policy"]) for row in selected)
    episode_counts = Counter(str(row["episode_id"]) for row in selected)
    if (
        len(selected) != int(cohort["state_count"])
        or len(episode_counts) != int(cohort["episode_count"])
        or max(episode_counts.values(), default=0)
        > int(cohort["maximum_states_per_episode"])
        or policy_counts != Counter(cohort["states_per_source_policy"])
        or len({str(row["map_id"]) for row in selected}) != int(cohort["map_count"])
        or len({str(row["task_id"]) for row in selected})
        != int(cohort["task_count"])
    ):
        raise ValueError("RobustAction label selection dimensions changed")

    preflight_root = (
        project_root / str(config["preflight_state_artifact_root"])
    ).resolve()
    preflight_index = _load_preflight_index(
        root=preflight_root,
        expected_tree_sha256=str(config["preflight_state_artifact_tree_sha256"]),
        preflight_run_fingerprint=str(preflight_run["run_fingerprint"]),
        decisions=selected,
    )
    producer = producer_identity(
        project_root=project_root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    source_run_hashes = {
        str(Path(str(row["source_root"])).resolve()): sha256_file(
            Path(str(row["source_root"])).resolve() / "run_config.json"
        )
        for row in selected
    }
    identity = {
        "schema": RUN_SCHEMA,
        "config_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "preflight_state_artifact_tree_sha256": state_artifact_tree_sha256(
            preflight_root
        ),
        "selected_state_ids": [str(row["state_id"]) for row in selected],
        "trial_indices": list(TRIAL_INDICES),
        "target_state_restore_contract": TARGET_STATE_RESTORE_CONTRACT,
        "source_run_config_sha256": dict(sorted(source_run_hashes.items())),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    output = Path(output).resolve()
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("RobustAction label output belongs to another run")
        if not resume:
            raise ValueError("RobustAction label output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    jobs: list[dict[str, Any]] = []
    for decision in selected:
        key = _fingerprint(
            {
                "state_id": decision["state_id"],
                "before": decision["before_fingerprint"],
            }
        )[:20]
        jobs.append(
            {
                "job_id": str(decision["state_id"]),
                "state_id": str(decision["state_id"]),
                "solver_seed": int(decision["solver_seed"]),
                "row": decision,
                "decision": decision,
                "preflight_path": str(preflight_index[str(decision["state_id"])]),
                "output_path": str(output / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
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
                "schema": REPORT_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "requested_state_count": len(jobs),
                "completed_state_count": len(observed) - len(failures),
                "error_state_count": sum(
                    row.get("status") == "error" for row in failures
                ),
                "timeout_state_count": sum(
                    row.get("status") == "timeout" for row in failures
                ),
                "completed_candidate_count": sum(
                    int(row.get("candidate_count", 0)) for row in observed
                ),
                "completed_trial_count": sum(
                    int(row.get("trial_count", 0)) for row in observed
                ),
                "status": "running",
                "errors": [
                    {
                        "state_id": str(row.get("state_id", row.get("job_id"))),
                        "status": str(row.get("status")),
                        "error": str(row.get("error")),
                    }
                    for row in failures
                ],
            },
        )

    _write_json(
        status_path,
        {
            "schema": REPORT_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "requested_state_count": len(jobs),
            "completed_state_count": 0,
            "error_state_count": 0,
            "timeout_state_count": 0,
            "completed_candidate_count": 0,
            "completed_trial_count": 0,
            "status": "running",
            "errors": [],
        },
    )
    try:
        observed = _run_jobs(
            _collect_state,
            jobs,
            workers,
            phase="stride-robustaction-label-collection",
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
            "state_id": str(row.get("state_id", row.get("job_id"))),
            "status": str(row.get("status")),
            "error": str(row.get("error")),
        }
        for row in observed
        if row.get("status") in {"error", "timeout"}
    ]
    selected_by_id = {str(row["state_id"]): row for row in selected}
    all_trials: list[dict[str, Any]] = []
    all_aggregates: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    candidate_counts: Counter[int] = Counter()
    for result in sorted(successes, key=lambda row: str(row["state_id"])):
        state_id = str(result["state_id"])
        payload = _read_json(Path(str(result["state_file"])))
        preflight = _read_json(preflight_index[state_id])
        if not _state_artifact_valid(
            payload,
            run_fingerprint=run_fingerprint,
            decision=selected_by_id[state_id],
            preflight_payload=preflight,
        ):
            errors.append(
                {
                    "state_id": state_id,
                    "status": "error",
                    "error": "invalid state artifact",
                }
            )
            continue
        decision = selected_by_id[state_id]
        candidate_counts[len(payload["candidates"])] += 1
        all_trials.extend(payload["trials"])
        for aggregate in payload["candidate_aggregates"]:
            all_aggregates.append(
                {
                    **aggregate,
                    "state_id": state_id,
                    "map_id": str(decision["map_id"]),
                    "task_id": str(decision["task_id"]),
                    "episode_id": str(decision["episode_id"]),
                    "source_policy": str(decision["source_policy"]),
                    "solver_seed": int(decision["solver_seed"]),
                    "decision_index": int(decision["decision_index"]),
                    "decision_stage": str(decision["decision_stage"]),
                    "layout_mode": str(decision["layout_mode"]),
                    "agent_count": int(decision["agent_count"]),
                    "before_conflicts": int(decision["before_conflicts"]),
                    "research_split": str(cohort["research_split"]),
                }
            )
        state_rows.append(
            {
                "state_id": state_id,
                "map_id": str(decision["map_id"]),
                "task_id": str(decision["task_id"]),
                "episode_id": str(decision["episode_id"]),
                "source_policy": str(decision["source_policy"]),
                "layout_mode": str(decision["layout_mode"]),
                "candidate_count": len(payload["candidates"]),
                "trial_count": len(payload["trials"]),
                "candidate_signature": str(payload["candidate_signature"]),
                "feature_signature": str(payload["feature_signature"]),
                "state_file": str(result["state_file"]),
                "state_file_sha256": sha256_file(Path(str(result["state_file"]))),
            }
        )

    trial_count = len(all_trials)
    candidate_count = len(all_aggregates)
    timeout_count = sum(row.get("status") == "timeout" for row in observed)
    error_count = sum(row.get("status") == "error" for row in errors)
    state_file_count = len(list((output / "states").glob("*.json")))
    seeds_by_state_index: dict[tuple[str, int], set[int]] = {}
    seeds_by_state: dict[str, set[int]] = {}
    for trial in all_trials:
        state_id = str(trial["state_id"])
        trial_index = int(trial["trial_index"])
        pp_seed = int(trial["pp_seed"])
        seeds_by_state_index.setdefault((state_id, trial_index), set()).add(pp_seed)
        seeds_by_state.setdefault(state_id, set()).add(pp_seed)
    gates = {
        "all_320_states": len(state_rows) == 320 and state_file_count == 320,
        "exact_6285_candidates": candidate_count == 6285,
        "exact_100560_trial_product": trial_count == 100560,
        "candidate_distribution": dict(candidate_counts)
        == {16: 15, 17: 24, 18: 183, 22: 1, 23: 7, 24: 90},
        "paired_and_distinct_pp_seeds": all(
            len(seeds_by_state_index.get((state_id, trial_index), set())) == 1
            for state_id in selected_by_id
            for trial_index in TRIAL_INDICES
        )
        and all(
            len(seeds_by_state.get(state_id, set())) == 16
            for state_id in selected_by_id
        ),
        "exact_preflight_candidates_and_features": len(state_rows) == len(successes),
        "state_and_repair_fingerprint_identity": len(state_rows) == len(successes),
        "native_action_semantics": len(state_rows) == len(successes),
        "zero_errors": error_count == 0,
        "zero_timeouts": timeout_count == 0,
        "no_runtime_or_future_fields": not _forbidden_hits(
            [*all_trials, *all_aggregates]
        ),
    }
    integrity_passed = all(gates.values())
    artifacts: dict[str, Any] = {}
    if integrity_passed:
        trials_path = output / "repair_trials.jsonl"
        aggregates_path = output / "candidate_aggregates.jsonl"
        states_path = output / "state_manifest.jsonl"
        _write_jsonl(trials_path, all_trials)
        _write_jsonl(aggregates_path, all_aggregates)
        _write_jsonl(states_path, state_rows)
        artifacts = {
            "repair_trials_sha256": sha256_file(trials_path),
            "candidate_aggregates_sha256": sha256_file(aggregates_path),
            "state_manifest_sha256": sha256_file(states_path),
            "state_artifact_tree_sha256": state_artifact_tree_sha256(
                output / "states"
            ),
        }
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "paired_current_step_label_collection_no_timing",
        "collection_id": str(config["collection_id"]),
        "run_fingerprint": run_fingerprint,
        "requested_state_count": len(jobs),
        "completed_state_count": len(state_rows),
        "new_state_count": sum(row.get("status") == "ok" for row in successes),
        "resumed_state_count": sum(
            row.get("status") == "resumed" for row in successes
        ),
        "candidate_count": candidate_count,
        "trial_count": trial_count,
        "error_state_count": error_count,
        "timeout_state_count": timeout_count,
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
        "integrity_gates": gates,
        "integrity_passed": integrity_passed,
        "opportunity_audit_executed": False,
        "runtime_or_pp_time_stored": False,
        "future_trajectory_stored": False,
        "formal_speed_claim": False,
        "errors": errors,
        "artifacts": artifacts,
        "config_sha256": sha256_file(config_path),
        "next_decision": config[
            "next_decision_on_pass"
            if integrity_passed
            else "next_decision_on_failure"
        ],
    }
    _write_json(output / "collection_report.json", report)
    _write_json(
        status_path,
        {
            **report,
            "status": "complete" if integrity_passed else "failed",
        },
    )
    return report


__all__ = [
    "AGGREGATE_SCHEMA",
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "STATE_SCHEMA",
    "TRIAL_SCHEMA",
    "_aggregate_candidate",
    "_state_artifact_valid",
    "collect_robustaction_labels",
    "state_artifact_tree_sha256",
    "validate_robustaction_label_collection_config",
]
