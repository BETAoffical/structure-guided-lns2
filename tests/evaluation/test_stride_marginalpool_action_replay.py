from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.stride_marginalpool_action_replay import (
    CONFIG_SCHEMA,
    EXPERIMENT_ID,
    aggregate_candidate,
    build_frozen_cohort,
    stable_dominates,
    validate_registration,
)


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
