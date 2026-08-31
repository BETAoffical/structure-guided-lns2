"""Independent H1 label-support collection for Dual16 hierarchical admission.

The experiment deliberately stops before model fitting or runtime registration.
It authenticates the existing station-centric warehouse checkpoints, freezes a
V2 winner over the base pool, creates the current Component16 and Hotspot16
arms, deduplicates exact agent sets, and executes only the current repair step
under sixteen paired PP seeds.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import os
from pathlib import Path
from typing import Any, Mapping

from experiments._common import (
    producer_identity,
    read_json,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.neighborhood_candidates import candidate_id
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _plain,
    _run_jobs,
    state_fingerprint,
)
from experiments.state_analysis import analyze_state
from experiments.stride_collection import _paired_action, _validate_native_repair
from experiments.stride_warehouse_disruption_recovery_load_extension_v1 import (
    _checkpoint_inputs,
)
from experiments.trace_replay import (
    CHECKPOINT_BLOB_RESTORE_CONTRACT,
    restore_repair_state,
    target_state_from_checkpoint_blob,
)
from lns2_selector.evaluation.hierarchical_admission_h1 import (
    COMPONENT_ARM,
    FORBIDDEN_H1_FIELDS,
    HOTSPOT_ARM,
    H1_TRIAL_INDICES,
    V2_ARM,
    CandidateArm,
    StabilityPolicy,
    build_hierarchical_h1_labels,
    canonicalize_hierarchy,
    h1_trial_from_mapping,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import (
    generate_online_candidates,
    score_online_candidates,
)
from lns2_selector.runtime.structshell_dual16 import (
    STRUCTSHELL_DUAL16_POOL_ID,
    STRUCTSHELL_DUAL16_RUNTIME_ID,
    structshell_dual16_augmentation,
)
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidate_subset,
)


CONFIG_SCHEMA = "lns2.stride.dual16_hierarchical_admission_h1_config.v1"
EXPERIMENT_ID = "stride-dual16-hierarchical-admission-h1-v1"
DATA_LINE_ID = "stride-dual16-hierarchical-admission-h1-data-v1"
PLANNED_MODEL_ID = "stride-dual16-hierarchical-admission-v1"
PARENT_COMMIT = "e0912b3bf76be1ae3d45bfc248c3b67ec72874ac"

PREFLIGHT_RUN_SCHEMA = "lns2.stride.dual16_hierarchical_admission_h1_preflight_run.v1"
PREFLIGHT_STATE_SCHEMA = "lns2.stride.dual16_hierarchical_admission_h1_preflight_state.v1"
PREFLIGHT_REPORT_SCHEMA = "lns2.stride.dual16_hierarchical_admission_h1_preflight_report.v1"
COLLECTION_RUN_SCHEMA = "lns2.stride.dual16_hierarchical_admission_h1_collection_run.v1"
COLLECTION_STATE_SCHEMA = "lns2.stride.dual16_hierarchical_admission_h1_collection_state.v1"
COLLECTION_REPORT_SCHEMA = "lns2.stride.dual16_hierarchical_admission_h1_collection_report.v1"
LABEL_ROW_SCHEMA = "lns2.stride.dual16_hierarchical_admission_h1_label.v1"

COMPONENT_FAMILY = "structpool-conflict-component:16"
HOTSPOT_FAMILY = "structpool-spatiotemporal-hotspot:16"
EXPECTED_INPUTS = {
    "load_extension_config",
    "checkpoint_manifest",
    "checkpoint_report",
    "checkpoint_worker_results",
    "runtime_template",
    "frozen_v2_manifest",
}
EXPECTED_INPUT_SPECS = {
    "load_extension_config": {
        "path": "configs/stride_warehouse_disruption_recovery_load_extension_v1.json",
        "sha256": "6af88121e232fdd925d5e640a65a174d45fc1a82f837a414c4ea4068f6e44e26",
    },
    "checkpoint_manifest": {
        "path": "build/stride-warehouse-disruption-recovery-load-extension-v1/checkpoints/load_extension_checkpoint_manifest.jsonl",
        "sha256": "a9acfae87a43ed1a703495f085d7215274beb166e9771a4c5f75c3331b7a113e",
    },
    "checkpoint_report": {
        "path": "build/stride-warehouse-disruption-recovery-load-extension-v1/checkpoints/load_extension_checkpoint_report.json",
        "sha256": "5e100bd4340b307ef6aa5785b78c43017626c816dd2bca35b7809f93509a13a9",
    },
    "checkpoint_worker_results": {
        "path": "build/stride-warehouse-disruption-recovery-load-extension-v1/checkpoints/load_extension_worker_results.json",
        "sha256": "f4ea48e8519e104a05bc24d781925e7b8c08dd4fb646228c355a3119fa864209",
    },
    "runtime_template": {
        "path": "configs/stride_warehouse_fixed16_development_runtime_v2.json",
        "sha256": "4a7376f428338abedf8a7defb645588b59fdda081f575918ea2a7abb278f673a",
    },
    "frozen_v2_manifest": {
        "path": "artifacts/initlns-closed-loop-controller-v2/controller_manifest.json",
        "sha256": "1b699182f9890148d0030e691b457c82ac7054760534665ad65484afb0ac82a8",
    },
}
IMPLEMENTATION_FILES = (
    "experiments/_common.py",
    "experiments/compact_controller_model.py",
    "experiments/stride_dual16_hierarchical_admission_h1_collection.py",
    "experiments/feature_schema_v2.py",
    "experiments/neighborhood_candidates.py",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/stride_collection.py",
    "experiments/stride_warehouse_disruption_recovery_load_extension_v1.py",
    "experiments/warehouse_disruption_checkpoints.py",
    "lns2_selector/evaluation/hierarchical_admission_h1.py",
    "lns2_selector/runtime/fingerprints.py",
    "lns2_selector/runtime/online_selection.py",
    "lns2_selector/runtime/structshell_dual16.py",
    "lns2_selector/runtime/topology_candidates.py",
    "lns2_selector/training/policy_bundle.py",
    "experiments/trace_replay.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _registered(root: Path, spec: Mapping[str, Any], *, label: str) -> Path:
    path = (root / str(spec.get("path", ""))).resolve()
    _require(path.is_file(), f"registered H1 input is missing: {label}: {path}")
    expected = str(spec.get("sha256", ""))
    observed = sha256_file(path)
    _require(
        observed == expected,
        f"registered H1 input changed: {label}: expected {expected}, got {observed}",
    )
    return path


def validate_config(
    config: Mapping[str, Any], *, project_root: Path | None = None
) -> dict[str, Path]:
    """Validate the preregistered identity and optionally authenticate inputs."""

    _require(config.get("schema") == CONFIG_SCHEMA, "H1 config schema changed")
    _require(config.get("experiment_id") == EXPERIMENT_ID, "H1 experiment id changed")
    _require(config.get("data_line_id") == DATA_LINE_ID, "H1 data line changed")
    _require(
        config.get("planned_model_id") == PLANNED_MODEL_ID,
        "H1 planned model id changed",
    )
    _require(
        config.get("scientific_status")
        == "preregistered_sequential_development_current_step_label_support_only",
        "H1 scientific status changed",
    )
    _require(
        config.get("pre_registration_parent_commit") == PARENT_COMMIT,
        "H1 preregistration parent changed",
    )

    inputs = dict(config.get("inputs") or {})
    _require(set(inputs) == EXPECTED_INPUTS, "H1 registered input set changed")
    _require(inputs == EXPECTED_INPUT_SPECS, "H1 registered input identity changed")
    for name, spec in inputs.items():
        _require(isinstance(spec, Mapping), f"H1 input spec is invalid: {name}")
        digest = str(spec.get("sha256", ""))
        _require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"H1 input digest is invalid: {name}",
        )

    source = dict(config.get("source_cohort") or {})
    _require(
        source
        == {
            "cohort_id": "stride-warehouse-disruption-recovery-load-extension-v1",
            "role": "sequential_development_only",
            "expected_checkpoint_count": 36,
            "expected_map_count": 6,
            "expected_load_band_counts": {
                "medium_high": 12,
                "high": 12,
                "very_high": 12,
            },
            "controller_outcome_filtering": False,
            "reserve_or_replacement_backfill": False,
            "future_map_disjoint_confirmation_required": True,
        },
        "H1 source cohort changed",
    )

    controller = dict(config.get("controller_contract") or {})
    _require(
        controller
        == {
            "controller_id": "v2-full",
            "bundle_path": "artifacts/initlns-closed-loop-controller-v2",
            "feature_schema": "lns2.realized_features.v2",
            "feature_profile": "realized_dynamic",
            "feature_dimension": 124,
            "proposal_backend": "optimized",
            "anchor_ranking": "base_v2_pool_only_before_structural_generation",
        },
        "H1 frozen V2 controller contract changed",
    )

    dual16 = dict(config.get("dual16_contract") or {})
    _require(
        dual16.get("pool_id") == STRUCTSHELL_DUAL16_POOL_ID
        and dual16.get("runtime_id") == STRUCTSHELL_DUAL16_RUNTIME_ID
        and dual16.get("nominal_size") == 16
        and dual16.get("structural_families")
        == {
            "component": "conflict_component",
            "hotspot": "spatiotemporal_hotspot",
        }
        and dual16.get("maximum_structural_actions") == 2
        and dual16.get("exact_agent_tuple_deduplication") is True
        and dual16.get("frozen_v2_anchor_may_not_change_after_structural_generation")
        is True,
        "H1 Dual16 contract changed",
    )

    paired = dict(config.get("paired_repair_contract") or {})
    _require(
        tuple(paired.get("trial_indices") or ()) == H1_TRIAL_INDICES
        and tuple(paired.get("first_fixed_half") or ()) == tuple(range(8))
        and tuple(paired.get("second_fixed_half") or ()) == tuple(range(8, 16))
        and paired.get("trials_per_unique_action") == 16
        and paired.get("seed_namespace")
        == "stride-dual16-hierarchical-admission-h1-paired-pp-v1"
        and paired.get("same_state_trial_seed_across_all_unique_actions") is True
        and paired.get("fresh_restore_per_action_trial") is True
        and paired.get("execute_each_unique_set_once") is True
        and paired.get("failure_backfill_allowed") is False
        and paired.get("replan_algorithm") == "PP"
        and float(paired.get("per_action_time_limit_seconds", -1)) == 5.0
        and paired.get("runtime_fields_stored") is False
        and paired.get("future_fields_stored") is False,
        "H1 paired-repair contract changed",
    )

    label = dict(config.get("label_contract") or {})
    _require(
        label.get("per_seed_target")
        == "normalized_current_step_conflict_reduction"
        and label.get("formula")
        == "(conflicts_before-conflicts_after)/max(1,conflicts_before)"
        and float(label.get("minimum_mean_advantage", -1)) == 0.02
        and float(label.get("minimum_paired_win_fraction", -1)) == 0.75
        and label.get("require_positive_first_fixed_half") is True
        and label.get("require_positive_second_fixed_half") is True
        and label.get("require_rollback_rate_noninferiority") is True
        and label.get("require_no_progress_rate_noninferiority") is True
        and label.get("stage2")
        == "component_vs_hotspot_only_when_exact_agent_sets_differ"
        and label.get("stage1")
        == "admit_structural_only_when_component_and_hotspot_each_stably_support_structural_over_frozen_v2"
        and label.get("all_shared") == "audit_only"
        and all(
            label.get(name) is False
            for name in (
                "posthoc_best_structural_used",
                "single_seed_winner_used",
                "repair_runtime_used",
                "future_repair_rounds_used",
                "cost_to_go_used",
                "receding_q_used",
                "ttf_used",
            )
        ),
        "H1 label contract changed",
    )

    execution = dict(config.get("execution") or {})
    _require(
        int(execution.get("recommended_workers", 0)) == 16
        and int(execution.get("maximum_workers", 0)) == 20
        and float(execution.get("per_state_process_fuse_seconds", 0.0)) == 420.0
        and execution.get("resume_required_for_existing_output") is True
        and execution.get("preserve_complete_failed_product") is True,
        "H1 execution contract changed",
    )

    gates = dict(config.get("pilot_support_gates") or {})
    _require(
        gates
        == {
            "minimum_stage1_structural_labels": 4,
            "minimum_stage1_v2_labels": 4,
            "minimum_maps_with_stage1_structural_label": 2,
            "minimum_maps_with_stage1_v2_label": 2,
            "minimum_stage2_component_labels": 2,
            "minimum_stage2_hotspot_labels": 2,
            "minimum_stage2_eligible_states": 8,
            "gate_authorizes": "expanded_fresh_label_collection_only",
        },
        "H1 pilot support gates changed",
    )
    _require(
        config.get("next_decision_on_support_pass")
        == "register_expanded_fresh_map_grouped_h1_collection_before_training"
        and config.get("next_decision_on_support_failure")
        == "stop_before_training_and_report_no_go_label_support",
        "H1 next-decision contract changed",
    )

    claims = dict(config.get("claim_boundary") or {})
    _require(
        claims
        == {
            "sequential_development_only": True,
            "training_authorized": False,
            "runtime_integration_authorized": False,
            "default_replacement_authorized": False,
            "ttf_or_speed_claim_authorized": False,
            "map_disjoint_confirmation_required": True,
        },
        "H1 claim boundary changed",
    )

    if project_root is None:
        return {}
    return {
        name: _registered(project_root.resolve(), spec, label=name)
        for name, spec in sorted(inputs.items())
    }


def load_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    config_path = Path(path).resolve()
    root = _root()
    config = dict(read_json(config_path))
    inputs = validate_config(config, project_root=root)
    return config_path, root, config, inputs


def paired_pp_seed(state_repair_fingerprint: str, trial_index: int) -> int:
    """Return one arm-independent seed for a state/trial pair."""

    if (
        not isinstance(state_repair_fingerprint, str)
        or len(state_repair_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in state_repair_fingerprint
        )
    ):
        raise ValueError("H1 paired seed requires a SHA-256 repair fingerprint")
    if type(trial_index) is not int or trial_index not in H1_TRIAL_INDICES:
        raise ValueError("H1 paired seed trial index must be in 0..15")
    return int(
        _fingerprint(
            {
                "namespace": "stride-dual16-hierarchical-admission-h1-paired-pp-v1",
                "repair_state": state_repair_fingerprint,
                "trial_index": trial_index,
            }
        )[:16],
        16,
    ) % (2**31)


def _forbidden_hits(value: Any) -> set[str]:
    hits: set[str] = set()
    if isinstance(value, Mapping):
        hits.update(FORBIDDEN_H1_FIELDS & set(map(str, value)))
        for nested in value.values():
            hits.update(_forbidden_hits(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            hits.update(_forbidden_hits(nested))
    return hits


def validate_candidate_arm(
    row: Mapping[str, Any], *, role: str, agent_count: int
) -> CandidateArm:
    """Validate an exact action identity; C and H must be current fixed16 arms."""

    candidate = CandidateArm(
        candidate_id=str(row.get("candidate_id", "")),
        agents=tuple(row.get("agents") or ()),
    )
    if any(agent >= int(agent_count) for agent in candidate.agents):
        raise ValueError(f"{role} candidate contains an out-of-range agent")
    if candidate.candidate_id != candidate_id(candidate.agents):
        raise ValueError(f"{role} candidate id differs from its exact agent set")
    if role in {COMPONENT_ARM, HOTSPOT_ARM} and len(candidate.agents) != 16:
        raise ValueError(f"{role} candidate must contain exactly 16 agents")
    return candidate


def _role_rows(rows: list[dict[str, Any]], *, agent_count: int) -> dict[str, CandidateArm]:
    found: dict[str, CandidateArm] = {}
    for row in rows:
        families = set(map(str, row.get("selection_families") or ()))
        for role, family in (
            (COMPONENT_ARM, COMPONENT_FAMILY),
            (HOTSPOT_ARM, HOTSPOT_FAMILY),
        ):
            if family not in families:
                continue
            if role in found:
                raise ValueError(f"Dual16 generated multiple {role} fixed16 cells")
            found[role] = validate_candidate_arm(row, role=role, agent_count=agent_count)
    if set(found) != {COMPONENT_ARM, HOTSPOT_ARM}:
        raise ValueError("Dual16 did not materialize both Component16 and Hotspot16")
    return found


def _source_product(
    config_path: Path,
    inputs: Mapping[str, Path],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    source_output = inputs["checkpoint_manifest"].parents[1]
    _source_path, _source_root, source_config, checkpoint_root, checkpoints = (
        _checkpoint_inputs(inputs["load_extension_config"], source_output)
    )
    dataset_root = Path(str(source_config["_dataset_root"])).resolve()
    dataset = {
        str(row["task_id"]): dict(row)
        for row in _load_dataset_rows(dataset_root, ["load_extension"])
    }
    expected = dict(read_json(config_path))["source_cohort"]
    bands = collections.Counter(str(row["load_band"]) for row in checkpoints)
    _require(
        len(checkpoints) == int(expected["expected_checkpoint_count"])
        and len({str(row["map_id"]) for row in checkpoints})
        == int(expected["expected_map_count"])
        and bands == collections.Counter(expected["expected_load_band_counts"])
        and len(dataset) == 18
        and all(str(row["task_id"]) in dataset for row in checkpoints),
        "authenticated H1 checkpoint cohort differs from preregistration",
    )
    return checkpoint_root, source_config, checkpoints, dataset


def _producer(root: Path) -> dict[str, Any]:
    return producer_identity(
        project_root=root,
        source_files=IMPLEMENTATION_FILES,
        native_required=True,
        package_names=("numpy",),
    )


def _runtime_environment(runtime_template: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    environment = dict(runtime_template["environment"])
    environment["time_limit"] = float(
        config["paired_repair_contract"]["per_action_time_limit_seconds"]
    )
    environment["replan_algorithm"] = "PP"
    return environment


def _checkpoint_job(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_root: Path,
    dataset_root: Path,
    dataset_row: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_root": str(dataset_root),
        "row": dict(dataset_row),
        "environment": dict(environment),
        "replay_destroy_strategy": "Adaptive",
        "checkpoint_root": str(checkpoint_root),
        "checkpoint": dict(checkpoint),
    }


def _restore_checkpoint(job: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    checkpoint = dict(job["checkpoint"])
    row = dict(job["row"])
    source, _blob = target_state_from_checkpoint_blob(
        Path(str(job["checkpoint_root"])),
        checkpoint,
        expected_map_id=str(row["map_id"]),
        expected_task_id=str(row["task_id"]),
        expected_agent_count=int(row["agent_count"]),
    )
    environment, restored = restore_repair_state(
        dict(job), source, seed=int(checkpoint["restore_seed"])
    )
    if (
        state_fingerprint(restored) != str(checkpoint["expected_fingerprint"])
        or repair_structure_fingerprint(restored)
        != str(checkpoint["repair_structure_fingerprint"])
        or int(restored["num_of_colliding_pairs"])
        != int(checkpoint["expected_conflicts"])
    ):
        raise RuntimeError("H1 checkpoint restore changed the frozen state")
    return environment, restored, source


def _serialize_plan(plan: Any) -> dict[str, Any]:
    return _plain({
        "partition": plan.partition,
        "arms": {
            V2_ARM: dataclasses.asdict(plan.frozen_v2),
            COMPONENT_ARM: dataclasses.asdict(plan.component),
            HOTSPOT_ARM: dataclasses.asdict(plan.hotspot),
        },
        "executions": [dataclasses.asdict(row) for row in plan.executions],
        "arm_execution_keys": dict(plan.arm_execution_keys),
        "label_execution_keys": list(plan.label_execution_keys),
        "stage2_applicable": bool(plan.stage2_applicable),
    })


def _plan_from_payload(payload: Mapping[str, Any]) -> Any:
    arms = dict(payload["hierarchy"]["arms"])
    plan = canonicalize_hierarchy(
        frozen_v2=CandidateArm(**arms[V2_ARM]),
        component=CandidateArm(**arms[COMPONENT_ARM]),
        hotspot=CandidateArm(**arms[HOTSPOT_ARM]),
    )
    if _serialize_plan(plan) != dict(payload["hierarchy"]):
        raise ValueError("serialized H1 hierarchy changed")
    return plan


def _preflight_state_valid(
    payload: Mapping[str, Any], *, run_fingerprint: str, checkpoint: Mapping[str, Any]
) -> bool:
    try:
        if (
            payload.get("schema") != PREFLIGHT_STATE_SCHEMA
            or payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("data_line_id") != DATA_LINE_ID
            or payload.get("run_fingerprint") != run_fingerprint
            or payload.get("complete") is not True
            or payload.get("checkpoint_id") != checkpoint.get("checkpoint_id")
            or payload.get("checkpoint_identity_sha256")
            != checkpoint.get("checkpoint_identity_sha256")
            or payload.get("before_fingerprint") != checkpoint.get("expected_fingerprint")
            or payload.get("before_repair_fingerprint")
            != checkpoint.get("repair_structure_fingerprint")
            or payload.get("before_conflicts") != checkpoint.get("expected_conflicts")
            or payload.get("map_id") != checkpoint.get("map_id")
            or payload.get("task_id") != checkpoint.get("task_id")
            or payload.get("load_band") != checkpoint.get("load_band")
            or payload.get("disturbance_replica")
            != checkpoint.get("disturbance_replica")
            or payload.get("agent_count") != checkpoint.get("agent_count")
            or payload.get("restore_contract") != CHECKPOINT_BLOB_RESTORE_CONTRACT
            or type(payload.get("base_candidate_count")) is not int
            or int(payload.get("base_candidate_count", 0)) <= 0
            or payload.get("anchor_ranked_on_base_pool_only") is not True
            or payload.get("feature_schema") != "lns2.realized_features.v2"
            or payload.get("feature_profile") != "realized_dynamic"
            or payload.get("feature_dimension") != 124
            or payload.get("candidate_repairs_executed") is not False
            or payload.get("runtime_or_future_fields_stored") is not False
            or _forbidden_hits(payload)
        ):
            return False
        plan = _plan_from_payload(payload)
        agent_count = int(checkpoint["agent_count"])
        for role in (V2_ARM, COMPONENT_ARM, HOTSPOT_ARM):
            validate_candidate_arm(
                dict(payload["hierarchy"]["arms"])[role],
                role=role,
                agent_count=agent_count,
            )
        base_candidates = list(payload.get("base_candidates") or ())
        base_count = int(payload["base_candidate_count"])
        if len(base_candidates) != base_count:
            return False
        validated_base = [
            validate_candidate_arm(row, role=V2_ARM, agent_count=agent_count)
            for row in base_candidates
        ]
        base_ids = [row.candidate_id for row in validated_base]
        base_sets = [row.agents for row in validated_base]
        if (
            len(set(base_ids)) != base_count
            or len(set(base_sets)) != base_count
            or payload.get("base_candidate_signature")
            != _fingerprint(base_candidates)
        ):
            return False
        scores = list(payload.get("base_scores") or ())
        if (
            len(scores) != base_count
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in scores
            )
            or payload.get("base_score_signature")
            != _fingerprint(list(map(float, scores)))
        ):
            return False
        position = payload.get("frozen_v2_base_position")
        if type(position) is not int or not 0 <= position < base_count:
            return False
        stable_scores = [round(float(value), 12) for value in scores]
        order = sorted(
            range(base_count),
            key=lambda index: (-stable_scores[index], base_ids[index]),
        )
        expected_margin = (
            stable_scores[order[0]] - stable_scores[order[1]]
            if base_count > 1
            else stable_scores[order[0]]
        )
        if (
            position != order[0]
            or payload.get("frozen_v2_candidate_id") != base_ids[position]
            or plan.frozen_v2.candidate_id != base_ids[position]
            or plan.frozen_v2.agents != base_sets[position]
            or not math.isclose(
                float(payload.get("frozen_v2_score", float("nan"))),
                float(scores[position]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(payload.get("frozen_v2_margin", float("nan"))),
                expected_margin,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return False
        features = dict(payload.get("realized_features_by_execution") or {})
        if set(features) != {row.execution_key for row in plan.executions}:
            return False
        if any(
            set(map(str, dict(values)))
            != set(PROFILE_FEATURE_NAMES["realized_dynamic"])
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in dict(values).values()
            )
            for values in features.values()
        ):
            return False
        if (
            int(payload.get("planned_unique_action_count", -1))
            != len(plan.label_execution_keys)
            or int(payload.get("planned_trial_count", -1))
            != len(plan.label_execution_keys) * 16
        ):
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def validate_preflight_state_artifact(
    payload: Mapping[str, Any], *, run_fingerprint: str, checkpoint: Mapping[str, Any]
) -> None:
    if not _preflight_state_valid(
        payload, run_fingerprint=run_fingerprint, checkpoint=checkpoint
    ):
        details: dict[str, Any] = {
            "schema": payload.get("schema") == PREFLIGHT_STATE_SCHEMA,
            "run": payload.get("run_fingerprint") == run_fingerprint,
            "checkpoint": payload.get("checkpoint_id") == checkpoint.get("checkpoint_id"),
            "before": payload.get("before_fingerprint") == checkpoint.get("expected_fingerprint"),
            "forbidden": sorted(_forbidden_hits(payload)),
        }
        try:
            plan = _plan_from_payload(payload)
            features = dict(payload.get("realized_features_by_execution") or {})
            details["plan"] = True
            details["feature_keys"] = set(features) == {
                row.execution_key for row in plan.executions
            }
            details["feature_dimensions"] = sorted(
                {len(dict(values)) for values in features.values()}
            )
            details["feature_value_types"] = sorted(
                {
                    type(value).__name__
                    for values in features.values()
                    for value in dict(values).values()
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            details["plan"] = f"{type(error).__name__}: {error}"
        raise ValueError(f"invalid H1 preflight state artifact: {details}")


def _preflight_worker(job: dict[str, Any]) -> dict[str, Any]:
    checkpoint = dict(job["checkpoint"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        existing = read_json(output_path)
        validate_preflight_state_artifact(
            existing, run_fingerprint=run_fingerprint, checkpoint=checkpoint
        )
        return {
            "status": "resumed",
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "state_file": str(output_path),
        }
    if output_path.exists():
        raise ValueError(f"H1 preflight output exists without reusable identity: {output_path}")

    environment, state, _source = _restore_checkpoint(job)
    before_fingerprint = state_fingerprint(state)
    proposal = dict(job["proposal"])
    proposal.pop("topology_boundary", None)
    proposal.pop("structpool", None)
    proposal.pop("hybridstructpool", None)
    base, _generation = generate_online_candidates(
        environment,
        state,
        task_id=str(checkpoint["task_id"]),
        solver_seed=int(checkpoint["screen_solver_seed"]),
        decision_index=0,
        proposal_config=proposal,
        state_hash=before_fingerprint,
        verify_full_state=False,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    if not base:
        raise RuntimeError("H1 frozen V2 base pool is empty")
    for row in base:
        validate_candidate_arm(row, role=V2_ARM, agent_count=int(checkpoint["agent_count"]))

    feature_engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features={
            "realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]
        },
        dense_output=False,
    )
    base_rows, _base_feature_metrics = feature_engine.realized_rows(
        base, state_hash=before_fingerprint
    )
    bundle = load_controller_bundle(Path(str(job["controller_bundle"])))
    model = bundle.main_models["realized_dynamic"]
    anchor_index, scores, margin = score_online_candidates(base_rows, model)
    frozen_v2 = validate_candidate_arm(
        base[anchor_index], role=V2_ARM, agent_count=int(checkpoint["agent_count"])
    )
    base_audit = [
        {
            "candidate_id": str(row["candidate_id"]),
            "agents": sorted(map(int, row["agents"])),
        }
        for row in base
    ]
    base_scores = list(map(float, scores))

    analysis = analyze_state(state)
    dual16 = structshell_dual16_augmentation()
    structural = generate_structpool_candidate_subset(
        state,
        analysis,
        family_sizes={
            name: tuple(sizes)
            for name, sizes in dual16["runtime_structural_family_sizes"].items()
        },
    )
    roles = _role_rows(structural, agent_count=int(checkpoint["agent_count"]))
    plan = canonicalize_hierarchy(
        frozen_v2=frozen_v2,
        component=roles[COMPONENT_ARM],
        hotspot=roles[HOTSPOT_ARM],
    )

    execution_candidates = [
        {
            "candidate_id": execution.candidate_ids[0],
            "agents": list(execution.agents),
        }
        for execution in plan.executions
    ]
    feature_rows, _execution_feature_metrics = feature_engine.realized_rows(
        execution_candidates, state_hash=before_fingerprint
    )
    features_by_candidate = {
        str(row["candidate_id"]): dict(row["features"]["realized_dynamic"])
        for row in feature_rows
    }
    features_by_execution = {
        execution.execution_key: features_by_candidate[execution.candidate_ids[0]]
        for execution in plan.executions
    }
    current = _plain(environment.get_state())
    if state_fingerprint(current) != before_fingerprint:
        raise RuntimeError("H1 proposal or feature generation mutated the checkpoint")

    payload = {
        "schema": PREFLIGHT_STATE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "data_line_id": DATA_LINE_ID,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "checkpoint_identity_sha256": str(checkpoint["checkpoint_identity_sha256"]),
        "map_id": str(checkpoint["map_id"]),
        "task_id": str(checkpoint["task_id"]),
        "load_band": str(checkpoint["load_band"]),
        "disturbance_replica": int(checkpoint["disturbance_replica"]),
        "agent_count": int(checkpoint["agent_count"]),
        "before_fingerprint": before_fingerprint,
        "before_repair_fingerprint": repair_structure_fingerprint(state),
        "before_conflicts": int(state["num_of_colliding_pairs"]),
        "restore_contract": CHECKPOINT_BLOB_RESTORE_CONTRACT,
        "base_candidate_count": len(base),
        "base_candidates": base_audit,
        "base_candidate_signature": _fingerprint(base_audit),
        "base_scores": base_scores,
        "base_score_signature": _fingerprint(base_scores),
        "frozen_v2_base_position": int(anchor_index),
        "frozen_v2_candidate_id": frozen_v2.candidate_id,
        "frozen_v2_score": float(scores[anchor_index]),
        "frozen_v2_margin": float(margin),
        "anchor_ranked_on_base_pool_only": True,
        "hierarchy": _serialize_plan(plan),
        "feature_schema": "lns2.realized_features.v2",
        "feature_profile": "realized_dynamic",
        "feature_dimension": 124,
        "realized_features_by_execution": features_by_execution,
        "planned_unique_action_count": len(plan.label_execution_keys),
        "planned_trial_count": len(plan.label_execution_keys) * 16,
        "candidate_repairs_executed": False,
        "runtime_or_future_fields_stored": False,
    }
    validate_preflight_state_artifact(
        payload, run_fingerprint=run_fingerprint, checkpoint=checkpoint
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "status": "ok",
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "state_file": str(output_path),
    }


def _failure(job: Mapping[str, Any], status: str, error: str) -> dict[str, Any]:
    checkpoint = dict(job.get("checkpoint") or {})
    return {
        "status": status,
        "checkpoint_id": str(checkpoint.get("checkpoint_id", "unknown")),
        "error": error,
    }


def _preflight_identity(
    config_path: Path,
    root: Path,
    inputs: Mapping[str, Path],
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": PREFLIGHT_RUN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "input_sha256": {name: sha256_file(path) for name, path in sorted(inputs.items())},
        "checkpoint_ids": [str(row["checkpoint_id"]) for row in checkpoints],
        "checkpoint_identity_sha256": [
            str(row["checkpoint_identity_sha256"]) for row in checkpoints
        ],
        "producer": _producer(root),
        "anchor_pool": "base_v2_only",
        "candidate_repairs_executed": False,
    }


def run_preflight(
    config_path: str | Path, output: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    path, root, config, inputs = load_config(config_path)
    checkpoint_root, source_config, checkpoints, dataset = _source_product(path, inputs)
    runtime = dict(read_json(inputs["runtime_template"]))
    environment = _runtime_environment(runtime, config)
    identity = _preflight_identity(path, root, inputs, checkpoints)
    run_fingerprint = _fingerprint(identity)
    preflight_root = Path(output).resolve() / "preflight"
    run_path = preflight_root / "run_config.json"
    if run_path.is_file():
        existing = read_json(run_path)
        if existing.get("run_fingerprint") != run_fingerprint:
            raise ValueError("H1 preflight output belongs to another run")
        if not resume:
            raise ValueError("H1 preflight output exists; pass --resume")
    elif preflight_root.exists() and any(preflight_root.iterdir()):
        raise ValueError("H1 preflight directory is nonempty without run identity")
    preflight_root.mkdir(parents=True, exist_ok=True)
    write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})

    dataset_root = Path(str(source_config["_dataset_root"])).resolve()
    jobs = []
    for checkpoint in checkpoints:
        state_key = _fingerprint(
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "identity": checkpoint["checkpoint_identity_sha256"],
            }
        )[:20]
        row = dataset[str(checkpoint["task_id"])]
        jobs.append(
            {
                **_checkpoint_job(
                    checkpoint,
                    checkpoint_root=checkpoint_root,
                    dataset_root=dataset_root,
                    dataset_row=row,
                    environment=environment,
                ),
                "job_id": str(checkpoint["checkpoint_id"]),
                "run_fingerprint": run_fingerprint,
                "proposal": dict(runtime["proposal"]),
                "controller_bundle": str(
                    (root / config["controller_contract"]["bundle_path"]).resolve()
                ),
                "output_path": str(preflight_root / "states" / f"{state_key}.json"),
                "resume": bool(resume),
            }
        )
    results = _run_jobs(
        _preflight_worker,
        jobs,
        int(config["execution"]["recommended_workers"]),
        phase="dual16-hierarchical-admission-h1-preflight",
        output_root=preflight_root,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["execution"]["per_state_process_fuse_seconds"]),
        failure_result=_failure,
        stop_on_failure=False,
    )
    errors = [row for row in results if row.get("status") not in {"ok", "resumed"}]
    payloads = []
    by_checkpoint = {str(row["checkpoint_id"]): row for row in checkpoints}
    for result in results:
        if result.get("status") not in {"ok", "resumed"}:
            continue
        payload = dict(read_json(Path(str(result["state_file"]))))
        validate_preflight_state_artifact(
            payload,
            run_fingerprint=run_fingerprint,
            checkpoint=by_checkpoint[str(result["checkpoint_id"])],
        )
        payloads.append(payload)
    partitions = collections.Counter(str(row["hierarchy"]["partition"]) for row in payloads)
    report = {
        "schema": PREFLIGHT_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_fingerprint": run_fingerprint,
        "requested_state_count": len(checkpoints),
        "completed_state_count": len(payloads),
        "error_state_count": len(errors),
        "map_count": len({str(row["map_id"]) for row in payloads}),
        "load_band_counts": dict(
            sorted(collections.Counter(str(row["load_band"]) for row in payloads).items())
        ),
        "partition_counts": dict(sorted(partitions.items())),
        "planned_unique_action_count": sum(
            int(row["planned_unique_action_count"]) for row in payloads
        ),
        "planned_trial_count": sum(int(row["planned_trial_count"]) for row in payloads),
        "anchor_ranked_on_base_pool_only": all(
            row["anchor_ranked_on_base_pool_only"] is True for row in payloads
        ),
        "candidate_repairs_executed": False,
        "errors": errors,
        "integrity_passed": len(payloads) == 36 and not errors,
    }
    write_json(preflight_root / "report.json", report)
    return report


def _collection_state_valid(
    payload: Mapping[str, Any],
    *,
    run_fingerprint: str,
    checkpoint: Mapping[str, Any],
    preflight: Mapping[str, Any],
    expected_preflight_sha256: str,
) -> bool:
    try:
        if (
            payload.get("schema") != COLLECTION_STATE_SCHEMA
            or payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("data_line_id") != DATA_LINE_ID
            or payload.get("run_fingerprint") != run_fingerprint
            or payload.get("complete") is not True
            or payload.get("checkpoint_id") != checkpoint.get("checkpoint_id")
            or payload.get("checkpoint_identity_sha256")
            != checkpoint.get("checkpoint_identity_sha256")
            or payload.get("preflight_state_sha256")
            != expected_preflight_sha256
            or payload.get("before_fingerprint") != checkpoint.get("expected_fingerprint")
            or payload.get("before_repair_fingerprint")
            != checkpoint.get("repair_structure_fingerprint")
            or payload.get("before_conflicts") != checkpoint.get("expected_conflicts")
            or payload.get("map_id") != checkpoint.get("map_id")
            or payload.get("task_id") != checkpoint.get("task_id")
            or payload.get("load_band") != checkpoint.get("load_band")
            or payload.get("disturbance_replica")
            != checkpoint.get("disturbance_replica")
            or payload.get("execute_each_unique_set_once") is not True
            or payload.get("fresh_restore_per_action_trial") is not True
            or payload.get("runtime_or_future_fields_stored") is not False
            or _forbidden_hits(payload)
        ):
            return False
        plan = _plan_from_payload(preflight)
        if payload.get("partition") != plan.partition:
            return False
        product = dict(payload.get("trials_by_execution") or {})
        if set(product) != set(plan.label_execution_keys):
            return False
        for execution_key, rows in product.items():
            if len(rows) != 16:
                return False
            for row in rows:
                h1_trial_from_mapping(row)
                if (
                    int(row["before_conflicts"])
                    != int(checkpoint["expected_conflicts"])
                    or int(row["pp_seed"])
                    != paired_pp_seed(
                        str(checkpoint["repair_structure_fingerprint"]),
                        int(row["trial_index"]),
                    )
                ):
                    return False
            if tuple(sorted(int(row["trial_index"]) for row in rows)) != H1_TRIAL_INDICES:
                return False
        policy = StabilityPolicy()
        if (
            int(payload.get("executed_unique_action_count", -1)) != len(product)
            or int(payload.get("executed_trial_count", -1))
            != sum(len(rows) for rows in product.values())
        ):
            return False
        expected_labels = _plain(dataclasses.asdict(
            build_hierarchical_h1_labels(plan, product, policy=policy)
        ))
        return expected_labels == payload.get("labels")
    except (KeyError, RuntimeError, TypeError, ValueError):
        return False


def validate_collection_state_artifact(
    payload: Mapping[str, Any],
    *,
    run_fingerprint: str,
    checkpoint: Mapping[str, Any],
    preflight: Mapping[str, Any],
    expected_preflight_sha256: str,
) -> None:
    if not _collection_state_valid(
        payload,
        run_fingerprint=run_fingerprint,
        checkpoint=checkpoint,
        preflight=preflight,
        expected_preflight_sha256=expected_preflight_sha256,
    ):
        raise ValueError("invalid H1 collection state artifact")


def _collect_worker(job: dict[str, Any]) -> dict[str, Any]:
    checkpoint = dict(job["checkpoint"])
    output_path = Path(str(job["output_path"]))
    preflight_path = Path(str(job["preflight_path"]))
    preflight = dict(read_json(preflight_path))
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        existing = read_json(output_path)
        validate_collection_state_artifact(
            existing,
            run_fingerprint=run_fingerprint,
            checkpoint=checkpoint,
            preflight=preflight,
            expected_preflight_sha256=sha256_file(preflight_path),
        )
        return {
            "status": "resumed",
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "state_file": str(output_path),
            "trial_count": sum(len(rows) for rows in existing["trials_by_execution"].values()),
        }
    if output_path.exists():
        raise ValueError(f"H1 collection output exists without reusable identity: {output_path}")

    plan = _plan_from_payload(preflight)
    before_repair = str(checkpoint["repair_structure_fingerprint"])
    product: dict[str, list[dict[str, Any]]] = {}
    execution_by_key = {row.execution_key: row for row in plan.executions}
    for execution_key in plan.label_execution_keys:
        execution = execution_by_key[execution_key]
        rows: list[dict[str, Any]] = []
        for trial_index in H1_TRIAL_INDICES:
            environment, before, _source = _restore_checkpoint(job)
            pp_seed = paired_pp_seed(before_repair, trial_index)
            result = _plain(
                environment.step(_paired_action(list(execution.agents), pp_seed))
            )
            after, metrics = _validate_native_repair(
                result,
                expected_agents=list(execution.agents),
                expected_seed=pp_seed,
            )
            before_conflicts = int(before["num_of_colliding_pairs"])
            after_conflicts = int(after["num_of_colliding_pairs"])
            rolled_back = not bool(metrics["replan_success"])
            if (
                rolled_back
                and repair_structure_fingerprint(after) != before_repair
            ):
                raise RuntimeError(
                    "failed H1 PP repair changed state instead of rolling back"
                )
            row = {
                "trial_index": trial_index,
                "pp_seed": pp_seed,
                "before_conflicts": before_conflicts,
                "after_conflicts": after_conflicts,
                "strict_decrease": after_conflicts < before_conflicts,
                "rolled_back": rolled_back,
                "normalized_conflict_reduction": (
                    before_conflicts - after_conflicts
                )
                / max(1, before_conflicts),
            }
            h1_trial_from_mapping(row)
            rows.append(row)
        product[execution_key] = rows

    policy = StabilityPolicy(
        minimum_mean_advantage=float(job["minimum_mean_advantage"]),
        minimum_paired_win_fraction=float(job["minimum_paired_win_fraction"]),
    )
    labels = build_hierarchical_h1_labels(plan, product, policy=policy)
    payload = {
        "schema": COLLECTION_STATE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "data_line_id": DATA_LINE_ID,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "checkpoint_identity_sha256": str(checkpoint["checkpoint_identity_sha256"]),
        "map_id": str(checkpoint["map_id"]),
        "task_id": str(checkpoint["task_id"]),
        "load_band": str(checkpoint["load_band"]),
        "disturbance_replica": int(checkpoint["disturbance_replica"]),
        "before_fingerprint": str(checkpoint["expected_fingerprint"]),
        "before_repair_fingerprint": before_repair,
        "before_conflicts": int(checkpoint["expected_conflicts"]),
        "preflight_state_sha256": sha256_file(preflight_path),
        "partition": plan.partition,
        "trials_by_execution": product,
        "labels": _plain(dataclasses.asdict(labels)),
        "executed_unique_action_count": len(product),
        "executed_trial_count": sum(len(rows) for rows in product.values()),
        "execute_each_unique_set_once": True,
        "fresh_restore_per_action_trial": True,
        "runtime_or_future_fields_stored": False,
    }
    validate_collection_state_artifact(
        payload,
        run_fingerprint=run_fingerprint,
        checkpoint=checkpoint,
        preflight=preflight,
        expected_preflight_sha256=sha256_file(preflight_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "status": "ok",
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "state_file": str(output_path),
        "trial_count": int(payload["executed_trial_count"]),
    }


def _load_preflight_index(
    output: Path,
    *,
    checkpoints: list[dict[str, Any]],
    expected_run_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    preflight_root = output / "preflight"
    run = dict(read_json(preflight_root / "run_config.json"))
    report = dict(read_json(preflight_root / "report.json"))
    if (
        run.get("schema") != PREFLIGHT_RUN_SCHEMA
        or run.get("run_fingerprint") != expected_run_fingerprint
        or report.get("schema") != PREFLIGHT_REPORT_SCHEMA
        or report.get("run_fingerprint") != run.get("run_fingerprint")
        or report.get("integrity_passed") is not True
        or int(report.get("completed_state_count", -1)) != 36
        or int(report.get("error_state_count", -1)) != 0
        or report.get("candidate_repairs_executed") is not False
    ):
        raise ValueError("H1 collection requires a passed preflight")
    checkpoint_by_id = {str(row["checkpoint_id"]): row for row in checkpoints}
    index: dict[str, Path] = {}
    for path in sorted((preflight_root / "states").glob("*.json")):
        payload = dict(read_json(path))
        checkpoint_id = str(payload.get("checkpoint_id", ""))
        checkpoint = checkpoint_by_id.get(checkpoint_id)
        if checkpoint is None or checkpoint_id in index:
            raise ValueError("H1 preflight state identity is duplicated or unknown")
        validate_preflight_state_artifact(
            payload,
            run_fingerprint=str(run["run_fingerprint"]),
            checkpoint=checkpoint,
        )
        index[checkpoint_id] = path
    if set(index) != set(checkpoint_by_id):
        raise ValueError("H1 preflight state coverage changed")
    return run, index


def _collection_identity(
    config_path: Path,
    root: Path,
    inputs: Mapping[str, Path],
    preflight_run: Mapping[str, Any],
    preflight_index: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "schema": COLLECTION_RUN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "input_sha256": {name: sha256_file(path) for name, path in sorted(inputs.items())},
        "preflight_run_fingerprint": str(preflight_run["run_fingerprint"]),
        "preflight_state_sha256": {
            checkpoint_id: sha256_file(path)
            for checkpoint_id, path in sorted(preflight_index.items())
        },
        "producer": _producer(root),
        "trial_indices": list(H1_TRIAL_INDICES),
        "seed_namespace": "stride-dual16-hierarchical-admission-h1-paired-pp-v1",
        "current_step_only": True,
        "training_authorized": False,
        "runtime_integration_authorized": False,
    }


def run_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    workers: int | None = None,
    preflight_output: str | Path | None = None,
) -> dict[str, Any]:
    path, root, config, inputs = load_config(config_path)
    output_root = Path(output).resolve()
    if preflight_output is not None and Path(preflight_output).resolve() != output_root:
        raise ValueError("H1 preflight and collection must share one output root")
    checkpoint_root, source_config, checkpoints, dataset = _source_product(path, inputs)
    expected_preflight_fingerprint = _fingerprint(
        _preflight_identity(path, root, inputs, checkpoints)
    )
    preflight_run, preflight_index = _load_preflight_index(
        output_root,
        checkpoints=checkpoints,
        expected_run_fingerprint=expected_preflight_fingerprint,
    )
    identity = _collection_identity(path, root, inputs, preflight_run, preflight_index)
    run_fingerprint = _fingerprint(identity)
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = dict(read_json(run_path))
        if existing.get("run_fingerprint") != run_fingerprint:
            raise ValueError("H1 collection output belongs to another run")
        if not resume:
            raise ValueError("H1 collection output exists; pass --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})

    requested_workers = int(workers or config["execution"]["recommended_workers"])
    if not 1 <= requested_workers <= int(config["execution"]["maximum_workers"]):
        raise ValueError("H1 collection workers exceed the registered range")
    runtime = dict(read_json(inputs["runtime_template"]))
    environment = _runtime_environment(runtime, config)
    dataset_root = Path(str(source_config["_dataset_root"])).resolve()
    jobs = []
    for checkpoint in checkpoints:
        state_key = _fingerprint(
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "identity": checkpoint["checkpoint_identity_sha256"],
            }
        )[:20]
        jobs.append(
            {
                **_checkpoint_job(
                    checkpoint,
                    checkpoint_root=checkpoint_root,
                    dataset_root=dataset_root,
                    dataset_row=dataset[str(checkpoint["task_id"])],
                    environment=environment,
                ),
                "job_id": str(checkpoint["checkpoint_id"]),
                "run_fingerprint": run_fingerprint,
                "preflight_path": str(preflight_index[str(checkpoint["checkpoint_id"])]),
                "output_path": str(output_root / "states" / f"{state_key}.json"),
                "resume": bool(resume),
                "minimum_mean_advantage": float(
                    config["label_contract"]["minimum_mean_advantage"]
                ),
                "minimum_paired_win_fraction": float(
                    config["label_contract"]["minimum_paired_win_fraction"]
                ),
            }
        )
    status_path = output_root / "collection_status.json"
    observed: list[dict[str, Any]] = []

    def update_status(result: dict[str, Any]) -> None:
        observed.append(result)
        failures = [row for row in observed if row.get("status") not in {"ok", "resumed"}]
        write_json(
            status_path,
            {
                "schema": COLLECTION_REPORT_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "status": "running",
                "requested_state_count": len(jobs),
                "completed_state_count": len(observed) - len(failures),
                "completed_trial_count": sum(int(row.get("trial_count", 0)) for row in observed),
                "error_state_count": len(failures),
                "errors": failures,
            },
        )

    write_json(
        status_path,
        {
            "schema": COLLECTION_REPORT_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "status": "running",
            "requested_state_count": len(jobs),
            "completed_state_count": 0,
            "completed_trial_count": 0,
            "error_state_count": 0,
            "errors": [],
        },
    )
    results = _run_jobs(
        _collect_worker,
        jobs,
        requested_workers,
        phase="dual16-hierarchical-admission-h1-collection",
        output_root=output_root,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["execution"]["per_state_process_fuse_seconds"]),
        on_result=update_status,
        failure_result=_failure,
        stop_on_failure=False,
    )
    errors = [row for row in results if row.get("status") not in {"ok", "resumed"}]
    write_json(
        status_path,
        {
            "schema": COLLECTION_REPORT_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "status": "complete" if not errors else "complete_with_errors",
            "requested_state_count": len(jobs),
            "completed_state_count": len(jobs) - len(errors),
            "completed_trial_count": sum(int(row.get("trial_count", 0)) for row in results),
            "error_state_count": len(errors),
            "errors": errors,
        },
    )
    return analyze_collection(path, output_root)


def _support_gate(
    config: Mapping[str, Any], payloads: list[dict[str, Any]]
) -> tuple[dict[str, int], dict[str, bool]]:
    stage1 = collections.Counter(str(row["labels"]["stage1"]["label"]) for row in payloads)
    stage2 = collections.Counter(str(row["labels"]["stage2"]["label"]) for row in payloads)
    structural_maps = {
        str(row["map_id"])
        for row in payloads
        if row["labels"]["stage1"]["label"] == "admit_structural_consensus"
    }
    v2_maps = {
        str(row["map_id"])
        for row in payloads
        if row["labels"]["stage1"]["label"] == "keep_v2"
    }
    observed = {
        "stage1_structural_labels": stage1["admit_structural_consensus"],
        "stage1_v2_labels": stage1["keep_v2"],
        "maps_with_stage1_structural_label": len(structural_maps),
        "maps_with_stage1_v2_label": len(v2_maps),
        "stage2_component_labels": stage2["component"],
        "stage2_hotspot_labels": stage2["hotspot"],
        "stage2_eligible_states": sum(bool(row["labels"]["stage2"]["applicable"]) for row in payloads),
    }
    required = {
        "stage1_structural_labels": int(config["pilot_support_gates"]["minimum_stage1_structural_labels"]),
        "stage1_v2_labels": int(config["pilot_support_gates"]["minimum_stage1_v2_labels"]),
        "maps_with_stage1_structural_label": int(config["pilot_support_gates"]["minimum_maps_with_stage1_structural_label"]),
        "maps_with_stage1_v2_label": int(config["pilot_support_gates"]["minimum_maps_with_stage1_v2_label"]),
        "stage2_component_labels": int(config["pilot_support_gates"]["minimum_stage2_component_labels"]),
        "stage2_hotspot_labels": int(config["pilot_support_gates"]["minimum_stage2_hotspot_labels"]),
        "stage2_eligible_states": int(config["pilot_support_gates"]["minimum_stage2_eligible_states"]),
    }
    return observed, {name: observed[name] >= value for name, value in required.items()}


def analyze_collection(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root_path, config, inputs = load_config(config_path)
    output_root = Path(output).resolve()
    _checkpoint_root, _source_config, checkpoints, _dataset = _source_product(path, inputs)
    expected_preflight_fingerprint = _fingerprint(
        _preflight_identity(path, _root(), inputs, checkpoints)
    )
    preflight_run, preflight_index = _load_preflight_index(
        output_root,
        checkpoints=checkpoints,
        expected_run_fingerprint=expected_preflight_fingerprint,
    )
    run = dict(read_json(output_root / "run_config.json"))
    expected_identity = _collection_identity(
        path, _root(), inputs, preflight_run, preflight_index
    )
    expected_fingerprint = _fingerprint(expected_identity)
    if run.get("schema") != COLLECTION_RUN_SCHEMA or run.get("run_fingerprint") != expected_fingerprint:
        raise ValueError("H1 collection run identity changed")
    checkpoint_by_id = {str(row["checkpoint_id"]): row for row in checkpoints}
    payloads: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_checkpoint_ids: set[str] = set()
    for state_path in sorted((output_root / "states").glob("*.json")):
        payload = dict(read_json(state_path))
        checkpoint_id = str(payload.get("checkpoint_id", ""))
        checkpoint = checkpoint_by_id.get(checkpoint_id)
        preflight_path = preflight_index.get(checkpoint_id)
        if checkpoint_id in seen_checkpoint_ids:
            errors.append({"checkpoint_id": checkpoint_id, "error": "duplicate state artifact"})
            continue
        seen_checkpoint_ids.add(checkpoint_id)
        if checkpoint is None or preflight_path is None:
            errors.append({"checkpoint_id": checkpoint_id, "error": "unknown state artifact"})
            continue
        try:
            validate_collection_state_artifact(
                payload,
                run_fingerprint=expected_fingerprint,
                checkpoint=checkpoint,
                preflight=read_json(preflight_path),
                expected_preflight_sha256=sha256_file(preflight_path),
            )
        except ValueError as error:
            errors.append({"checkpoint_id": checkpoint_id, "error": str(error)})
            continue
        payloads.append(payload)
    observed_ids = {str(row["checkpoint_id"]) for row in payloads}
    missing = sorted(set(checkpoint_by_id) - observed_ids)
    errors.extend({"checkpoint_id": value, "error": "missing state artifact"} for value in missing)

    observed_support, gate_checks = _support_gate(config, payloads)
    integrity_passed = len(payloads) == 36 and not errors and not any(
        _forbidden_hits(row) for row in payloads
    )
    support_passed = integrity_passed and all(gate_checks.values())
    partitions = collections.Counter(str(row["partition"]) for row in payloads)
    stage1 = collections.Counter(str(row["labels"]["stage1"]["label"]) for row in payloads)
    stage2 = collections.Counter(str(row["labels"]["stage2"]["label"]) for row in payloads)
    per_band: dict[str, dict[str, Any]] = {}
    for band in ("medium_high", "high", "very_high"):
        rows = [row for row in payloads if row["load_band"] == band]
        per_band[band] = {
            "state_count": len(rows),
            "stage1_labels": dict(sorted(collections.Counter(row["labels"]["stage1"]["label"] for row in rows).items())),
            "stage2_labels": dict(sorted(collections.Counter(row["labels"]["stage2"]["label"] for row in rows).items())),
        }
    labels_rows = [
        {
            "schema": LABEL_ROW_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "data_line_id": DATA_LINE_ID,
            "run_fingerprint": expected_fingerprint,
            "checkpoint_id": row["checkpoint_id"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "load_band": row["load_band"],
            "disturbance_replica": row["disturbance_replica"],
            "partition": row["partition"],
            "stage1": row["labels"]["stage1"],
            "stage2": row["labels"]["stage2"],
        }
        for row in sorted(payloads, key=lambda value: str(value["checkpoint_id"]))
    ]
    labels_path = output_root / "labels.jsonl"
    write_jsonl(labels_path, labels_rows)
    report = {
        "schema": COLLECTION_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "data_line_id": DATA_LINE_ID,
        "run_fingerprint": expected_fingerprint,
        "scientific_status": "sequential_development_current_step_label_support_only",
        "requested_state_count": 36,
        "completed_state_count": len(payloads),
        "map_count": len({str(row["map_id"]) for row in payloads}),
        "load_band_counts": dict(sorted(collections.Counter(str(row["load_band"]) for row in payloads).items())),
        "partition_counts": dict(sorted(partitions.items())),
        "executed_unique_action_count": sum(int(row["executed_unique_action_count"]) for row in payloads),
        "executed_trial_count": sum(int(row["executed_trial_count"]) for row in payloads),
        "stage1_label_counts": dict(sorted(stage1.items())),
        "stage2_label_counts": dict(sorted(stage2.items())),
        "labels_file": labels_path.name,
        "labels_sha256": sha256_file(labels_path),
        "per_load_band": per_band,
        "pilot_support_observed": observed_support,
        "pilot_support_gate_checks": gate_checks,
        "pilot_support_passed": support_passed,
        "integrity_passed": integrity_passed,
        "errors": errors,
        "training_authorized": False,
        "runtime_integration_authorized": False,
        "ttf_or_speed_claim_authorized": False,
        "map_disjoint_confirmation_required": True,
        "next_decision": (
            config["next_decision_on_support_pass"]
            if support_passed
            else config["next_decision_on_support_failure"]
        ),
    }
    write_json(output_root / "report.json", report)
    return report


__all__ = [
    "analyze_collection",
    "load_config",
    "paired_pp_seed",
    "run_collection",
    "run_preflight",
    "validate_candidate_arm",
    "validate_collection_state_artifact",
    "validate_config",
    "validate_preflight_state_artifact",
]
