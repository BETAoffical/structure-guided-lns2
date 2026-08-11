from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.stride_marginalpool_action_replay import (
    CONFIG_SCHEMA,
    EXPERIMENT_ID,
    PARTIAL_STATE_SCHEMA,
    TRIAL_SCHEMA,
    _import_recovery_states,
    _partial_state_artifact_valid,
    _partial_state_payload,
    _validate_recovery_registration,
    aggregate_candidate,
    build_frozen_cohort,
    stable_dominates,
    validate_registration,
)
from experiments.stride_repairability_collection import repairability_pp_seed
from experiments.repair_collection import _fingerprint


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_marginalpool_action_replay_v1_registration.json"


def _registration() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _aggregate(*, mean: float, no_progress: float, first: float, second: float) -> dict:
    return {
        "seed_mean": mean,
        "no_progress_rate": no_progress,
        "first_fixed_half_mean": first,
        "second_fixed_half_mean": second,
    }


def test_registration_and_frozen_cohort() -> None:
    config = _registration()
    validate_registration(config)
    assert config["schema"] == CONFIG_SCHEMA
    assert config["experiment_id"] == EXPERIMENT_ID
    metadata, states, logical = build_frozen_cohort(CONFIG)
    assert len(states) == 78
    assert len(logical) == 90
    assert sum(len(row["candidates"]) for row in states) == 2502
    assert len(metadata["cohort_fingerprint"]) == 64


def test_registration_rejects_trial_or_claim_drift() -> None:
    config = _registration()
    changed = copy.deepcopy(config)
    changed["execution"]["trial_indices"] = list(range(15))
    with pytest.raises(ValueError, match="execution contract"):
        validate_registration(changed)
    changed = copy.deepcopy(config)
    changed["claim_boundary"]["ttf_improvement_claim"] = True
    with pytest.raises(ValueError, match="claim boundary"):
        validate_registration(changed)


def test_stable_dominance_requires_all_registered_conditions() -> None:
    reference = _aggregate(mean=0.10, no_progress=0.50, first=0.09, second=0.11)
    candidate = _aggregate(mean=0.12, no_progress=0.50, first=0.10, second=0.14)
    assert stable_dominates(candidate, reference)
    worse_no_progress = {**candidate, "no_progress_rate": 0.51}
    assert not stable_dominates(worse_no_progress, reference)
    reversed_half = {**candidate, "first_fixed_half_mean": 0.08}
    assert not stable_dominates(reversed_half, reference)
    insufficient_mean = {**candidate, "seed_mean": 0.119}
    assert not stable_dominates(insufficient_mean, reference)


def test_candidate_aggregate_uses_fixed_halves_and_no_progress() -> None:
    candidate = {
        "candidate_id": "candidate-a",
        "candidate_kind": "structural",
        "agents": [1, 2],
        "actual_size": 2,
        "selection_families": ["structpool-test:2"],
        "structpool_family_groups": ["test"],
        "feature_schema": "lns2.realized_features.v2",
        "feature_profile": "realized_dynamic",
        "feature_count": 124,
        "feature_sha256": "0" * 64,
        "features": {"feature": 1.0},
    }
    trials = [
        {
            "trial_index": index,
            "normalized_conflict_reduction": index / 100.0,
            "conflicts_after": 10 if index == 0 else 9,
            "before_conflicts": 10,
            "replan_success": index % 2 == 0,
            "feasible": False,
        }
        for index in range(16)
    ]
    row = aggregate_candidate(candidate, trials)
    assert row["trial_count"] == 16
    assert row["first_fixed_half_mean"] == pytest.approx(0.035)
    assert row["second_fixed_half_mean"] == pytest.approx(0.115)
    assert row["no_progress_rate"] == pytest.approx(1 / 16)
    assert row["replan_success_rate"] == pytest.approx(0.5)


def test_partial_state_checkpoint_requires_complete_paired_candidate_trials() -> None:
    state_record = {
        "state_fingerprint": "state-a",
        "state_blob_sha256": "1" * 64,
        "source_run_config_sha256": "2" * 64,
        "logical_checkpoint_ids": ["logical-a"],
    }
    candidates = [{"candidate_id": "candidate-a", "agents": [1], "actual_size": 1}]
    before_repair = "3" * 64
    trials = [
        {
            "schema": TRIAL_SCHEMA,
            "state_fingerprint": "state-a",
            "candidate_id": "candidate-a",
            "trial_index": index,
            "pp_seed": repairability_pp_seed(before_repair, index),
            "before_conflicts": 10,
            "before_repair_fingerprint": before_repair,
            "conflicts_after": 9,
            "normalized_conflict_reduction": 0.1,
            "replan_success": True,
            "feasible": False,
        }
        for index in range(16)
    ]
    payload = _partial_state_payload(
        state_record=state_record,
        run_fingerprint="run",
        before_repair=before_repair,
        before_conflicts=10,
        restore_seed=7,
        candidates=candidates,
        completed_candidate_ids={"candidate-a"},
        trials=trials,
    )
    assert payload["schema"] == PARTIAL_STATE_SCHEMA
    assert _partial_state_artifact_valid(
        payload,
        state_record=state_record,
        run_fingerprint="run",
        trial_indices=tuple(range(16)),
        candidates=candidates,
        before_repair=before_repair,
        before_conflicts=10,
    )
    incomplete = copy.deepcopy(payload)
    incomplete["trials"].pop()
    assert not _partial_state_artifact_valid(
        incomplete,
        state_record=state_record,
        run_fingerprint="run",
        trial_indices=tuple(range(16)),
        candidates=candidates,
        before_repair=before_repair,
        before_conflicts=10,
    )


def test_partial_recovery_import_rebinds_run_identity(tmp_path: Path) -> None:
    state_record = {
        "state_fingerprint": "state-a",
        "state_blob_sha256": "1" * 64,
        "source_run_config_sha256": "2" * 64,
        "logical_checkpoint_ids": ["logical-a"],
    }
    candidates = [{"candidate_id": "candidate-a", "agents": [1], "actual_size": 1}]
    before_repair = "3" * 64
    trials = [
        {
            "schema": TRIAL_SCHEMA,
            "state_fingerprint": "state-a",
            "candidate_id": "candidate-a",
            "trial_index": index,
            "pp_seed": repairability_pp_seed(before_repair, index),
            "before_conflicts": 10,
            "before_repair_fingerprint": before_repair,
            "conflicts_after": 9,
            "normalized_conflict_reduction": 0.1,
            "replan_success": True,
            "feasible": False,
        }
        for index in range(16)
    ]
    source = _partial_state_payload(
        state_record=state_record,
        run_fingerprint="source-run",
        before_repair=before_repair,
        before_conflicts=10,
        restore_seed=7,
        candidates=candidates,
        completed_candidate_ids={"candidate-a"},
        trials=trials,
    )
    descriptor = {
        "source_run_fingerprint": "source-run",
        "import_policy": (
            "all valid completed and partial states without outcome filtering"
        ),
        "imported_state_sha256": {},
        "imported_partial_state_sha256": {"state-a": "4" * 64},
    }
    _import_recovery_states(
        {},
        imported_partials={"state-a": source},
        descriptor=descriptor,
        selected_states=[state_record],
        trial_indices=tuple(range(16)),
        output=tmp_path,
        run_fingerprint="continuation-run",
    )
    imported = json.loads(
        (tmp_path / "states" / "state-a.json.partial").read_text(encoding="utf-8")
    )
    assert imported["run_fingerprint"] == "continuation-run"
    assert imported["recovery_provenance"]["source_run_fingerprint"] == "source-run"
    assert _partial_state_artifact_valid(
        imported,
        state_record=state_record,
        run_fingerprint="continuation-run",
        trial_indices=tuple(range(16)),
        candidates=candidates,
        before_repair=before_repair,
        before_conflicts=10,
    )


def test_partial_continuation_registration_pins_source_and_progress(
    tmp_path: Path,
) -> None:
    descriptor = {
        "source_run_config_sha256": "1" * 64,
        "source_status_sha256": "2" * 64,
        "source_collection_report_sha256": "3" * 64,
        "import_policy": (
            "all valid completed and partial states without outcome filtering"
        ),
        "imported_state_count": 70,
        "imported_state_sha256": {"complete": "4" * 64},
        "imported_partial_state_count": 8,
        "imported_partial_state_sha256": {"partial": "5" * 64},
        "imported_partial_state_progress": {
            "partial": {
                "completed_candidate_count": 1,
                "candidate_count": 2,
                "trial_count": 16,
            }
        },
    }
    registration = {
        "schema": (
            "lns2.stride.marginalpool_action_replay_recovery_registration.v2"
        ),
        "scientific_status": (
            "operational_partial_checkpoint_continuation_without_label_change"
        ),
        "experiment_id": "stride-marginalpool-action-replay-continuation-v1",
        "base_registration": {"sha256": "6" * 64},
        "recovery_source": {
            "source_run_config_sha256": descriptor["source_run_config_sha256"],
            "source_status_sha256": descriptor["source_status_sha256"],
            "source_collection_report_sha256": descriptor[
                "source_collection_report_sha256"
            ],
            "imported_state_count": 70,
            "imported_state_sha256_digest": _fingerprint(
                descriptor["imported_state_sha256"]
            ),
            "imported_partial_state_count": 8,
            "imported_partial_state_sha256": descriptor[
                "imported_partial_state_sha256"
            ],
            "imported_partial_state_progress": descriptor[
                "imported_partial_state_progress"
            ],
            "import_policy": descriptor["import_policy"],
        },
        "effective_execution": {
            "workers": 4,
            "per_state_attempt_timeout_seconds": 1800.0,
            "maximum_state_attempts": 4,
            "maximum_state_wall_seconds": 7200.0,
            "candidate_atomic_checkpoints": True,
            "stop_after_consecutive_no_progress_attempts": 2,
        },
        "partial_selection_policy": "all valid partial states from the registered source",
        "outcomes_used_for_recovery_policy": False,
        "labels_or_candidate_set_changed": False,
        "no_result_based_exclusion": True,
    }
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")
    resolved, digest = _validate_recovery_registration(
        path,
        metadata={"config_sha256": "6" * 64},
        descriptor=descriptor,
        worker_count=4,
        per_attempt_timeout_seconds=1800.0,
        maximum_state_attempts=4,
    )
    assert resolved == str(path.resolve())
    assert len(digest) == 64
