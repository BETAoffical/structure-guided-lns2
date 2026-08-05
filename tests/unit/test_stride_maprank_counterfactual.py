from __future__ import annotations

import copy

from experiments.stride_maprank_counterfactual import (
    COLLECTION_SCHEMA,
    REPORT_SCHEMA,
    STATE_SCHEMA,
    STATUS_SCHEMA,
    TRIAL_SCHEMA,
    _state_artifact_valid,
)
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)


def test_counterfactual_artifact_schemas_are_distinct() -> None:
    assert len({COLLECTION_SCHEMA, REPORT_SCHEMA, STATE_SCHEMA, STATUS_SCHEMA}) == 4
    assert COLLECTION_SCHEMA.endswith("collection.v1")
    assert REPORT_SCHEMA.endswith("report.v1")


def _payload() -> dict:
    selection = {
        "state_id": "state",
        "before_fingerprint": "before-state",
        "before_conflicts": 5,
        "baseline_candidate_id": "v2",
        "challenger_candidate_id": "maprank",
    }
    candidates = [
        {
            "candidate_id": "v2",
            "role": "v2_augmented_selected",
            "agents": [0],
        },
        {
            "candidate_id": "maprank",
            "role": "maprank_selected",
            "agents": [1],
        },
    ]
    before_repair = "before-repair"
    trials = [
        {
            "schema": TRIAL_SCHEMA,
            "state_id": "state",
            "candidate_role": candidate["role"],
            "candidate_id": candidate["candidate_id"],
            "trial_index": trial_index,
            "pp_seed": repairability_pp_seed(before_repair, trial_index),
            "before_conflicts": 5,
            "conflicts_after": 4,
            "normalized_conflict_reduction": 0.2,
            "progress": True,
            "feasible": False,
            "replan_success": True,
            "low_level_generated": 10,
            "low_level_expanded": 5,
            "pp_replan_seconds": 0.01,
        }
        for candidate in candidates
        for trial_index in range(16)
    ]
    return {
        "schema": STATE_SCHEMA,
        "run_fingerprint": "run",
        "state_id": "state",
        "complete": True,
        "selection": selection,
        "before_fingerprint": "before-state",
        "before_repair_fingerprint": before_repair,
        "before_conflicts": 5,
        "restore_seed": repairability_restore_seed(before_repair),
        "candidates": candidates,
        "trials": trials,
    }


def test_counterfactual_state_requires_exact_two_by_sixteen_product() -> None:
    payload = _payload()
    assert _state_artifact_valid(
        payload,
        run_fingerprint="run",
        state_id="state",
        selection=payload["selection"],
    )
    payload["trials"].pop()
    assert not _state_artifact_valid(
        payload,
        run_fingerprint="run",
        state_id="state",
        selection=payload["selection"],
    )


def test_counterfactual_state_resume_identity_is_strict() -> None:
    payload = _payload()
    assert not _state_artifact_valid(
        payload,
        run_fingerprint="other",
        state_id="state",
        selection=payload["selection"],
    )
    assert not _state_artifact_valid(
        payload,
        run_fingerprint="run",
        state_id="other",
        selection=payload["selection"],
    )


def test_counterfactual_state_rejects_semantic_tampering() -> None:
    mutations = (
        lambda payload: payload["trials"][0].__setitem__(
            "candidate_role", "maprank_selected"
        ),
        lambda payload: payload["trials"][0].__setitem__(
            "normalized_conflict_reduction", float("nan")
        ),
        lambda payload: payload["trials"][0].__setitem__(
            "pp_seed", payload["trials"][0]["pp_seed"] + 1
        ),
    )
    for mutate in mutations:
        payload = copy.deepcopy(_payload())
        mutate(payload)
        assert not _state_artifact_valid(
            payload,
            run_fingerprint="run",
            state_id="state",
            selection=payload["selection"],
        )
