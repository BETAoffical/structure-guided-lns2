from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from experiments import stride_dual16_hierarchical_admission_h1_collection as subject
from experiments.neighborhood_candidates import candidate_id
from lns2_selector.evaluation.hierarchical_admission_h1 import (
    COMPONENT_ARM,
    HOTSPOT_ARM,
    V2_ARM,
    CandidateArm,
    build_hierarchical_h1_labels,
    canonicalize_hierarchy,
    h1_trial_from_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "stride_dual16_hierarchical_admission_h1_v1.json"
)
REPAIR_FINGERPRINT = "a" * 64


def _plain(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True))


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _candidate(role: str, start: int) -> dict:
    agents = list(range(start, start + 16))
    return {
        "role": role,
        "candidate_id": candidate_id(agents),
        "agents": agents,
        "actual_size": 16,
    }


def _trial_row(
    trial_index: int,
    *,
    after_conflicts: int,
    pp_seed: int | None = None,
) -> dict:
    before_conflicts = 100
    return {
        "trial_index": trial_index,
        "pp_seed": (
            subject.paired_pp_seed(REPAIR_FINGERPRINT, trial_index)
            if pp_seed is None
            else pp_seed
        ),
        "before_conflicts": before_conflicts,
        "after_conflicts": after_conflicts,
        "strict_decrease": after_conflicts < before_conflicts,
        "rolled_back": False,
        "normalized_conflict_reduction": (
            before_conflicts - after_conflicts
        )
        / before_conflicts,
    }


def _checkpoint() -> dict:
    return {
        "checkpoint_id": "checkpoint-a",
        "checkpoint_identity_sha256": "b" * 64,
        "map_id": "warehouse-map-a",
        "task_id": "warehouse-task-a",
        "load_band": "very_high",
        "disturbance_replica": 0,
        "expected_fingerprint": "c" * 64,
        "repair_structure_fingerprint": REPAIR_FINGERPRINT,
        "expected_conflicts": 100,
        "agent_count": 64,
    }


def _preflight_fixture() -> tuple[dict, object]:
    v2 = _candidate(V2_ARM, 0)
    structural = _candidate(COMPONENT_ARM, 16)
    base_candidates = [
        {"candidate_id": v2["candidate_id"], "agents": v2["agents"]},
        {
            "candidate_id": candidate_id(range(32, 48)),
            "agents": list(range(32, 48)),
        },
        {
            "candidate_id": candidate_id(range(48, 64)),
            "agents": list(range(48, 64)),
        },
    ]
    base_scores = [0.9, 0.5, 0.2]
    plan = canonicalize_hierarchy(
        frozen_v2=CandidateArm(v2["candidate_id"], tuple(v2["agents"])),
        component=CandidateArm(
            structural["candidate_id"], tuple(structural["agents"])
        ),
        hotspot=CandidateArm(
            structural["candidate_id"], tuple(reversed(structural["agents"]))
        ),
    )
    hierarchy = subject._serialize_plan(plan)
    features = {
        execution.execution_key: {
            name: float(index)
            for index, name in enumerate(
                subject.PROFILE_FEATURE_NAMES["realized_dynamic"]
            )
        }
        for execution in plan.executions
    }
    checkpoint = _checkpoint()
    return (
        {
            "schema": subject.PREFLIGHT_STATE_SCHEMA,
            "experiment_id": subject.EXPERIMENT_ID,
            "data_line_id": subject.DATA_LINE_ID,
            "run_fingerprint": "preflight-run",
            "complete": True,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_identity_sha256": checkpoint[
                "checkpoint_identity_sha256"
            ],
            "before_fingerprint": checkpoint["expected_fingerprint"],
            "before_repair_fingerprint": checkpoint[
                "repair_structure_fingerprint"
            ],
            "before_conflicts": checkpoint["expected_conflicts"],
            "map_id": checkpoint["map_id"],
            "task_id": checkpoint["task_id"],
            "load_band": checkpoint["load_band"],
            "disturbance_replica": checkpoint["disturbance_replica"],
            "agent_count": checkpoint["agent_count"],
            "restore_contract": subject.CHECKPOINT_BLOB_RESTORE_CONTRACT,
            "base_candidate_count": len(base_candidates),
            "base_candidates": base_candidates,
            "base_candidate_signature": subject._fingerprint(base_candidates),
            "base_scores": base_scores,
            "base_score_signature": subject._fingerprint(base_scores),
            "frozen_v2_base_position": 0,
            "frozen_v2_candidate_id": v2["candidate_id"],
            "frozen_v2_score": base_scores[0],
            "frozen_v2_margin": base_scores[0] - base_scores[1],
            "anchor_ranked_on_base_pool_only": True,
            "feature_schema": "lns2.realized_features.v2",
            "feature_profile": "realized_dynamic",
            "feature_dimension": 124,
            "candidate_repairs_executed": False,
            "runtime_or_future_fields_stored": False,
            "hierarchy": hierarchy,
            "realized_features_by_execution": features,
            "planned_unique_action_count": len(plan.label_execution_keys),
            "planned_trial_count": len(plan.label_execution_keys) * 16,
        },
        plan,
    )


def _collection_fixture(plan: object) -> dict:
    trials_by_execution = {}
    for execution_index, execution_key in enumerate(plan.label_execution_keys):
        trials_by_execution[execution_key] = [
            _trial_row(
                trial_index,
                after_conflicts=90 - 10 * execution_index,
            )
            for trial_index in range(16)
        ]
    labels = build_hierarchical_h1_labels(plan, trials_by_execution)
    checkpoint = _checkpoint()
    return {
        "schema": subject.COLLECTION_STATE_SCHEMA,
        "experiment_id": subject.EXPERIMENT_ID,
        "data_line_id": subject.DATA_LINE_ID,
        "run_fingerprint": "collection-run",
        "complete": True,
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_identity_sha256": checkpoint["checkpoint_identity_sha256"],
        "preflight_state_sha256": "d" * 64,
        "before_fingerprint": checkpoint["expected_fingerprint"],
        "before_repair_fingerprint": checkpoint["repair_structure_fingerprint"],
        "before_conflicts": checkpoint["expected_conflicts"],
        "map_id": checkpoint["map_id"],
        "task_id": checkpoint["task_id"],
        "load_band": checkpoint["load_band"],
        "disturbance_replica": checkpoint["disturbance_replica"],
        "partition": plan.partition,
        "execute_each_unique_set_once": True,
        "fresh_restore_per_action_trial": True,
        "runtime_or_future_fields_stored": False,
        "executed_unique_action_count": len(trials_by_execution),
        "executed_trial_count": sum(
            len(rows) for rows in trials_by_execution.values()
        ),
        "trials_by_execution": trials_by_execution,
        "labels": _plain(dataclasses.asdict(labels)),
    }


def test_config_identity_and_registered_hashes_fail_closed() -> None:
    config = _config()
    subject.validate_config(config, project_root=PROJECT_ROOT)

    wrong_identity = copy.deepcopy(config)
    wrong_identity["experiment_id"] = "different-experiment"
    with pytest.raises(ValueError, match="experiment id changed"):
        subject.validate_config(wrong_identity, project_root=PROJECT_ROOT)

    wrong_hash = copy.deepcopy(config)
    wrong_hash["inputs"]["frozen_v2_manifest"]["sha256"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="registered input identity changed|registered H1 input changed",
    ):
        subject.validate_config(wrong_hash, project_root=PROJECT_ROOT)

    changed_gates = copy.deepcopy(config)
    changed_gates["pilot_support_gates"]["minimum_stage1_structural_labels"] = 3
    with pytest.raises(ValueError, match="pilot support gates changed"):
        subject.validate_config(changed_gates, project_root=PROJECT_ROOT)


def test_paired_seed_is_arm_independent_and_trial_specific() -> None:
    # Arm identity is deliberately absent, so every action in one state/trial
    # receives the same PP seed while different trials receive different seeds.
    first = subject.paired_pp_seed(REPAIR_FINGERPRINT, 0)
    assert first == subject.paired_pp_seed(REPAIR_FINGERPRINT, 0)
    assert first != subject.paired_pp_seed(REPAIR_FINGERPRINT, 1)
    assert first != subject.paired_pp_seed("b" * 64, 0)
    with pytest.raises(ValueError, match="SHA-256 repair fingerprint"):
        subject.paired_pp_seed("not-a-repair-fingerprint", 0)


@pytest.mark.parametrize("role", (COMPONENT_ARM, HOTSPOT_ARM))
def test_structural_candidate_is_exact_size16_with_exact_set_identity(role: str) -> None:
    record = _candidate(role, 0)
    validated = subject.validate_candidate_arm(record, role=role, agent_count=32)
    assert validated.agents == tuple(range(16))
    assert validated.candidate_id == candidate_id(range(16))

    wrong_size = _candidate(role, 0)
    wrong_size["agents"] = wrong_size["agents"][:-1]
    wrong_size["actual_size"] = 15
    wrong_size["candidate_id"] = candidate_id(wrong_size["agents"])
    with pytest.raises(ValueError, match="exactly 16"):
        subject.validate_candidate_arm(wrong_size, role=role, agent_count=32)

    wrong_identity = _candidate(role, 0)
    wrong_identity["candidate_id"] = "neighborhood-not-the-exact-set"
    with pytest.raises(ValueError, match="exact agent set"):
        subject.validate_candidate_arm(wrong_identity, role=role, agent_count=32)

    out_of_range = _candidate(role, 17)
    with pytest.raises(ValueError, match="out-of-range"):
        subject.validate_candidate_arm(out_of_range, role=role, agent_count=32)


@pytest.mark.parametrize(
    "forbidden_field",
    (
        "time_to_feasible",
        "repair_runtime",
        "future_repair_rounds",
        "remaining_repair_rounds",
        "cost_to_go",
        "receding_q",
    ),
)
def test_trial_artifact_rejects_runtime_and_future_fields(
    forbidden_field: str,
) -> None:
    row = _trial_row(0, after_conflicts=90)
    row[forbidden_field] = 1
    with pytest.raises(ValueError, match="forbidden H1 fields|fields changed"):
        h1_trial_from_mapping(row)


def test_preflight_resume_validator_fails_closed() -> None:
    payload, _plan = _preflight_fixture()
    checkpoint = _checkpoint()
    subject.validate_preflight_state_artifact(
        payload,
        run_fingerprint="preflight-run",
        checkpoint=checkpoint,
    )

    for field, replacement in (
        ("schema", "wrong-schema"),
        ("run_fingerprint", "wrong-run"),
        ("complete", False),
    ):
        corrupted = copy.deepcopy(payload)
        corrupted[field] = replacement
        with pytest.raises(ValueError, match="invalid H1 preflight"):
            subject.validate_preflight_state_artifact(
                corrupted,
                run_fingerprint="preflight-run",
                checkpoint=checkpoint,
            )

    leaked = copy.deepcopy(payload)
    leaked["future_repair_rounds"] = 2
    with pytest.raises(ValueError, match="invalid H1 preflight"):
        subject.validate_preflight_state_artifact(
            leaked,
            run_fingerprint="preflight-run",
            checkpoint=checkpoint,
        )

    wrong_position = copy.deepcopy(payload)
    wrong_position["frozen_v2_base_position"] = 1
    with pytest.raises(ValueError, match="invalid H1 preflight"):
        subject.validate_preflight_state_artifact(
            wrong_position,
            run_fingerprint="preflight-run",
            checkpoint=checkpoint,
        )

    wrong_selected_v = copy.deepcopy(payload)
    wrong_selected_v["frozen_v2_candidate_id"] = wrong_selected_v[
        "base_candidates"
    ][1]["candidate_id"]
    with pytest.raises(ValueError, match="invalid H1 preflight"):
        subject.validate_preflight_state_artifact(
            wrong_selected_v,
            run_fingerprint="preflight-run",
            checkpoint=checkpoint,
        )

    execution_key = next(iter(payload["realized_features_by_execution"]))
    wrong_feature_name = copy.deepcopy(payload)
    feature_row = wrong_feature_name["realized_features_by_execution"][
        execution_key
    ]
    feature_row["not_a_registered_feature"] = feature_row.pop(next(iter(feature_row)))
    with pytest.raises(ValueError, match="invalid H1 preflight"):
        subject.validate_preflight_state_artifact(
            wrong_feature_name,
            run_fingerprint="preflight-run",
            checkpoint=checkpoint,
        )

    nonfinite_feature = copy.deepcopy(payload)
    feature_row = nonfinite_feature["realized_features_by_execution"][
        execution_key
    ]
    feature_row[next(iter(feature_row))] = float("nan")
    with pytest.raises(ValueError, match="invalid H1 preflight"):
        subject.validate_preflight_state_artifact(
            nonfinite_feature,
            run_fingerprint="preflight-run",
            checkpoint=checkpoint,
        )


def test_collection_resume_validator_fails_closed() -> None:
    preflight, plan = _preflight_fixture()
    payload = _collection_fixture(plan)
    checkpoint = _checkpoint()
    arguments = {
        "run_fingerprint": "collection-run",
        "checkpoint": checkpoint,
        "preflight": preflight,
        "expected_preflight_sha256": "d" * 64,
    }
    subject.validate_collection_state_artifact(payload, **arguments)

    wrong_identity = copy.deepcopy(payload)
    wrong_identity["run_fingerprint"] = "wrong-run"
    with pytest.raises(ValueError, match="invalid H1 collection"):
        subject.validate_collection_state_artifact(wrong_identity, **arguments)

    execution_key = next(iter(payload["trials_by_execution"]))
    missing_trial = copy.deepcopy(payload)
    missing_trial["trials_by_execution"][execution_key].pop()
    with pytest.raises(ValueError, match="invalid H1 collection"):
        subject.validate_collection_state_artifact(missing_trial, **arguments)

    wrong_seed = copy.deepcopy(payload)
    wrong_seed["trials_by_execution"][execution_key][0]["pp_seed"] += 1
    with pytest.raises(ValueError, match="invalid H1 collection"):
        subject.validate_collection_state_artifact(wrong_seed, **arguments)

    leaked = copy.deepcopy(payload)
    leaked["labels"]["future_repair_rounds"] = 2
    with pytest.raises(ValueError, match="invalid H1 collection"):
        subject.validate_collection_state_artifact(leaked, **arguments)
