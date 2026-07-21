from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.rescue_policy_audit import (
    _balanced_map_folds,
    _index_collection,
    _simulate_sequence,
    _validate_state_fingerprints,
    enumerate_rescue_policies,
)


def _outcome(
    *,
    reduction: int = 0,
    success: bool = False,
    state_changed: bool | None = None,
    seconds: float = 1.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "conflicts_before": 10,
        "conflicts_after": 10 - reduction,
        "conflict_reduction": reduction,
        "replan_success": success,
        "hard_failure": not success,
        "feasible": False,
        "repair_seconds": seconds,
        "pp_replan_seconds": seconds * 0.8,
        "generated": 10,
        "expanded": 20,
        "reopened": 2,
    }
    if state_changed is not None:
        row["state_changed"] = state_changed
    return row


def _trial(
    state: str,
    candidate: str,
    trial_index: int,
    *,
    seed: int = 7,
    split: str = "policy_train",
) -> dict[str, object]:
    return {
        "state_id": state,
        "candidate_id": candidate,
        "trial_index": trial_index,
        "random_seed": seed,
        "split": split,
        "complete": True,
        "status": "ok",
        "outcome": _outcome(),
    }


def _feature(
    state: str,
    candidate: str,
    *,
    split: str,
    map_id: str,
    layout: str = "layout",
) -> dict[str, object]:
    return {
        "state_id": state,
        "candidate_id": candidate,
        "split": split,
        "map_id": map_id,
        "layout_mode": layout,
        "agent_count": 400,
    }


class RescuePolicyGridTests(unittest.TestCase):
    def test_grid_contains_adaptive_and_all_nonempty_permutations(self) -> None:
        policies = enumerate_rescue_policies()
        self.assertEqual(len(policies), 16)
        self.assertEqual(policies[0].policy_id, "adaptive")
        self.assertEqual(len({row.policy_id for row in policies}), 16)
        self.assertIn("4>8>16>adaptive", {row.policy_id for row in policies})
        self.assertIn("16>8>4>adaptive", {row.policy_id for row in policies})


class RescueSequenceSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "state_id": "state",
            "split": "policy_train",
            "map_id": "map",
            "layout_mode": "layout",
            "agent_count": 400,
        }

    def test_hard_failure_advances_and_explicit_state_change_stops(self) -> None:
        first = _trial("state", "size4", 0)
        first["outcome"] = _outcome(success=False, state_changed=False, seconds=1.25)
        second = _trial("state", "size8", 0)
        second["outcome"] = _outcome(
            reduction=3, success=True, state_changed=True, seconds=2.5
        )
        adaptive = _trial("state", "adaptive", 0)
        adaptive["outcome"] = _outcome(
            reduction=5, success=True, state_changed=True, seconds=4.0
        )
        trials = {
            ("state", "size4", 0): first,
            ("state", "size8", 0): second,
            ("state", "adaptive", 0): adaptive,
        }
        row = _simulate_sequence(
            metadata=self.metadata,
            trial_index=0,
            policy_id="4>8>adaptive",
            candidate_sequence=["size4", "size8"],
            adaptive_id="adaptive",
            trials=trials,
            assumption="conflict-reduction-stop",
            reference_only=False,
        )
        self.assertEqual(row["attempt_count"], 2)
        self.assertEqual(row["conflict_reduction"], 3)
        self.assertAlmostEqual(row["repair_seconds"], 3.75)
        self.assertNotIn("adaptive", row["attempted_candidate_ids"])

    def test_legacy_noop_is_evaluated_under_both_bounds(self) -> None:
        candidate = _trial("state", "size4", 0)
        candidate["outcome"] = _outcome(success=True, seconds=1.0)
        adaptive = _trial("state", "adaptive", 0)
        adaptive["outcome"] = _outcome(
            reduction=2, success=True, state_changed=True, seconds=3.0
        )
        trials = {
            ("state", "size4", 0): candidate,
            ("state", "adaptive", 0): adaptive,
        }
        optimistic = _simulate_sequence(
            metadata=self.metadata,
            trial_index=0,
            policy_id="4>adaptive",
            candidate_sequence=["size4"],
            adaptive_id="adaptive",
            trials=trials,
            assumption="replan-success-stop",
            reference_only=False,
        )
        conservative = _simulate_sequence(
            metadata=self.metadata,
            trial_index=0,
            policy_id="4>adaptive",
            candidate_sequence=["size4"],
            adaptive_id="adaptive",
            trials=trials,
            assumption="conflict-reduction-stop",
            reference_only=False,
        )
        self.assertEqual(optimistic["attempt_count"], 1)
        self.assertEqual(conservative["attempt_count"], 2)
        self.assertEqual(conservative["conflict_reduction"], 2)

    def test_missing_policy_branch_is_rejected(self) -> None:
        adaptive = _trial("state", "adaptive", 0)
        with self.assertRaisesRegex(ValueError, "missing policy branch"):
            _simulate_sequence(
                metadata=self.metadata,
                trial_index=0,
                policy_id="4>adaptive",
                candidate_sequence=["missing"],
                adaptive_id="adaptive",
                trials={("state", "adaptive", 0): adaptive},
                assumption="replan-success-stop",
                reference_only=False,
            )


class RescueAuditValidationTests(unittest.TestCase):
    def test_paired_seed_mismatch_is_rejected(self) -> None:
        features = [
            _feature("train", "a", split="policy_train", map_id="train-map"),
            _feature("train", "b", split="policy_train", map_id="train-map"),
            _feature(
                "validation",
                "c",
                split="policy_validation",
                map_id="validation-map",
            ),
        ]
        trials = [
            _trial("train", "a", 0, seed=1),
            _trial("train", "b", 0, seed=2),
            _trial(
                "validation",
                "c",
                0,
                seed=1,
                split="policy_validation",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "paired seed mismatch"):
            _index_collection(features, trials)

    def test_training_and_diagnostic_maps_must_be_disjoint(self) -> None:
        features = [
            _feature("train", "a", split="policy_train", map_id="shared"),
            _feature(
                "validation",
                "b",
                split="policy_validation",
                map_id="shared",
            ),
        ]
        trials = [
            _trial("train", "a", 0),
            _trial(
                "validation",
                "b",
                0,
                split="policy_validation",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "maps overlap"):
            _index_collection(features, trials)

    def test_map_folds_never_mix_diagnostic_states_into_training(self) -> None:
        metadata = {
            **{
                f"train-{index}": {
                    "state_id": f"train-{index}",
                    "split": "policy_train",
                    "map_id": f"map-{index}",
                    "layout_mode": "layout",
                }
                for index in range(4)
            },
            "diagnostic": {
                "state_id": "diagnostic",
                "split": "policy_validation",
                "map_id": "diagnostic-map",
                "layout_mode": "layout",
            },
        }
        folds = _balanced_map_folds(metadata)
        self.assertEqual(len(folds), 4)
        self.assertTrue(
            all(
                "diagnostic-map" not in fold["train_maps"]
                and "diagnostic-map" not in fold["validation_maps"]
                for fold in folds
            )
        )

    def test_missing_state_fingerprint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            state_dir = source / "collection" / "states" / "policy_train"
            state_dir.mkdir(parents=True)
            (state_dir / "state.json").write_text(
                json.dumps(
                    {
                        "state": {
                            "state_id": "state",
                            "before_fingerprint": "a" * 64,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "before_repair_fingerprint"):
                _validate_state_fingerprints(source, {"state"})


if __name__ == "__main__":
    unittest.main()
