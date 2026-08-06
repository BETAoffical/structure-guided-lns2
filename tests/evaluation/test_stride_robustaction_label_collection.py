from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.stride_robustaction_label_collection import (
    STATE_SCHEMA,
    TRIAL_SCHEMA,
    _aggregate_candidate,
    _candidate_signature,
    _feature_signature,
    _state_artifact_valid,
    state_artifact_tree_sha256,
    validate_robustaction_label_collection_config,
)
from experiments.trace_replay import TARGET_STATE_RESTORE_CONTRACT


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_label_collection.json"
)


def _candidate() -> dict[str, object]:
    return {
        "candidate_id": "candidate-0",
        "candidate_kind": "base",
        "agents": [0, 1, 2, 3],
        "actual_size": 4,
        "selection_families": ["target:4"],
        "structpool_family_groups": [],
        "features": {
            name: 0.0 for name in PROFILE_FEATURE_NAMES["realized_dynamic"]
        },
    }


def _state_product() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    decision = {
        "state_id": "state-0",
        "before_fingerprint": "full-before",
        "before_conflicts": 16,
    }
    candidate = _candidate()
    candidates = [candidate]
    before_repair = "repair-before"
    trials = []
    for trial_index in range(16):
        trials.append(
            {
                "schema": TRIAL_SCHEMA,
                "state_id": "state-0",
                "candidate_id": "candidate-0",
                "candidate_kind": "base",
                "trial_index": trial_index,
                "pp_seed": repairability_pp_seed(before_repair, trial_index),
                "before_conflicts": 16,
                "before_fingerprint": "full-before",
                "before_repair_fingerprint": before_repair,
                "conflicts_after": 15,
                "normalized_conflict_reduction": 1.0 / 16.0,
                "replan_success": True,
                "feasible": False,
                "repair_outcome": "conflict_reduced",
                "after_repair_fingerprint": f"repair-after-{trial_index}",
            }
        )
    preflight = {
        "before_repair_fingerprint": before_repair,
        "candidate_signature": _candidate_signature(candidates),
        "feature_signature": _feature_signature(candidates),
        "candidates": candidates,
    }
    payload = {
        "schema": STATE_SCHEMA,
        "run_fingerprint": "run-0",
        "complete": True,
        "state_id": "state-0",
        "decision": decision,
        "before_fingerprint": "full-before",
        "before_repair_fingerprint": before_repair,
        "before_conflicts": 16,
        "state_restore": {
            "contract": TARGET_STATE_RESTORE_CONTRACT,
            "restore_seed": repairability_restore_seed(before_repair),
            "repair_structure_fingerprint": before_repair,
        },
        "preflight_state_file": "preflight.json",
        "preflight_state_sha256": "0" * 64,
        "candidate_signature": preflight["candidate_signature"],
        "feature_signature": preflight["feature_signature"],
        "candidates": candidates,
        "trials": trials,
        "candidate_aggregates": [_aggregate_candidate(candidate, trials)],
        "candidate_repair_trials_executed": True,
        "controller_actions_executed": False,
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
        "native_action_semantics_validated": True,
    }
    return decision, preflight, payload


class RobustActionLabelCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_config_inputs_and_tree_are_valid(self) -> None:
        validate_robustaction_label_collection_config(
            self.config, project_root=ROOT
        )
        observed = state_artifact_tree_sha256(
            ROOT / self.config["preflight_state_artifact_root"]
        )
        self.assertEqual(
            observed, self.config["preflight_state_artifact_tree_sha256"]
        )
        self.assertEqual(
            self.config["paired_repair_contract"]["expected_trial_count"],
            100560,
        )

    def test_candidate_aggregate_keeps_mean_and_risk_views_separate(self) -> None:
        candidate = _candidate()
        trials = []
        for index in range(16):
            value = index / 15.0
            trials.append(
                {
                    "trial_index": index,
                    "normalized_conflict_reduction": value,
                    "before_conflicts": 20,
                    "conflicts_after": 20 - index,
                    "replan_success": index % 2 == 0,
                    "feasible": index == 15,
                }
            )
        aggregate = _aggregate_candidate(candidate, trials)
        self.assertAlmostEqual(aggregate["seed_mean"], 0.5)
        self.assertLess(aggregate["lower_half_mean"], aggregate["seed_mean"])
        self.assertLess(
            aggregate["first_fixed_half_mean"],
            aggregate["second_fixed_half_mean"],
        )
        self.assertEqual(aggregate["trial_count"], 16)

    def test_state_artifact_requires_exact_paired_product_without_timing(self) -> None:
        decision, preflight, payload = _state_product()
        self.assertTrue(
            _state_artifact_valid(
                payload,
                run_fingerprint="run-0",
                decision=decision,
                preflight_payload=preflight,
            )
        )
        changed = copy.deepcopy(payload)
        changed["trials"][0]["native_step_seconds"] = 1.0
        self.assertFalse(
            _state_artifact_valid(
                changed,
                run_fingerprint="run-0",
                decision=decision,
                preflight_payload=preflight,
            )
        )

    def test_runtime_and_single_seed_boundaries_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["label_contract"]["single_seed_winner_used"] = True
        with self.assertRaisesRegex(ValueError, "label contract changed"):
            validate_robustaction_label_collection_config(changed)

        changed = copy.deepcopy(self.config)
        changed["paired_repair_contract"]["runtime_or_pp_time_stored"] = True
        with self.assertRaisesRegex(ValueError, "paired repair contract changed"):
            validate_robustaction_label_collection_config(changed)


if __name__ == "__main__":
    unittest.main()
