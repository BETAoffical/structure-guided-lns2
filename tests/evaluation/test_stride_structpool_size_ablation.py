from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_structpool_size_ablation import (
    TRIAL_SCHEMA,
    _best_size_counts_by_context,
    _copy_reused_trial,
    _family_variant,
    _fixed_half_consistency,
    _grouped_family_size_quality,
    _validate_label_matrix,
    validate_size_ablation_config,
)
from experiments.stride_repairability_collection import repairability_pp_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_structpool_size_ablation_v1.json"


class StructPoolSizeAblationTests(unittest.TestCase):
    def test_registered_config_is_valid(self) -> None:
        validate_size_ablation_config(
            json.loads(CONFIG.read_text(encoding="utf-8"))
        )

    def test_config_rejects_size_or_ttf_drift(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["candidate_grid"]["allowed_sizes"] = [8, 16, 32]
        with self.assertRaisesRegex(ValueError, "four-size grid"):
            validate_size_ablation_config(changed)
        changed = copy.deepcopy(config)
        changed["analysis"]["formal_ttf_claim"] = True
        with self.assertRaisesRegex(ValueError, "TTF claim"):
            validate_size_ablation_config(changed)

    def test_family_variants_keep_boundary_subtypes_separate(self) -> None:
        self.assertEqual(
            _family_variant("structpool-boundary-articulation:16"),
            ("topology_boundary_articulation", 16, "topology_boundary"),
        )
        self.assertEqual(
            _family_variant("structpool-boundary-low_degree:32"),
            ("topology_boundary_low_degree", 32, "topology_boundary"),
        )

    def test_reused_trial_changes_only_collection_provenance(self) -> None:
        source = {
            "state_id": "state",
            "candidate_id": "candidate",
            "candidate_kind": "structpool",
            "trial_index": 3,
            "pp_seed": 17,
            "before_conflicts": 5,
            "before_fingerprint": "before",
            "before_repair_fingerprint": "repair-before",
            "conflicts_after": 2,
            "normalized_conflict_reduction": 0.6,
            "replan_success": True,
            "feasible": False,
            "repair_outcome": "conflict_reduced",
            "after_repair_fingerprint": "repair-after",
        }
        copied = _copy_reused_trial(source, candidate_kind="structpool-grid")
        self.assertEqual(copied["schema"], TRIAL_SCHEMA)
        self.assertEqual(copied["candidate_kind"], "structpool-grid")
        self.assertEqual(copied["trial_source"], "reused_exact_robustaction_v1")
        self.assertEqual(copied["normalized_conflict_reduction"], 0.6)

    def test_grouped_analysis_preserves_context_and_half_consistency(self) -> None:
        def row(candidate_id: str, size: int, first: float, second: float) -> dict:
            return {
                "candidate_id": candidate_id,
                "family": "conflict_component",
                "nominal_size": size,
                "map_id": "map-a",
                "agent_band": "high",
                "seed_mean": (first + second) / 2.0,
                "first_fixed_half_mean": first,
                "second_fixed_half_mean": second,
                "lower_half_mean": min(first, second),
                "no_progress_rate": 0.0,
                "seed_standard_deviation": 0.1,
                "repair_success_rate": 1.0,
                "feasible_rate": 0.0,
            }

        rows = [
            row("a", 8, 0.7, 0.6),
            row("b", 16, 0.6, 0.8),
            row("c", 24, 0.5, 0.5),
            row("d", 32, 0.4, 0.9),
        ]
        half = _fixed_half_consistency(rows)
        self.assertFalse(half["exact_winner_agreement"])
        self.assertEqual(half["top3_overlap"], 0.5)
        grouped = _grouped_family_size_quality(rows, ["map_id", "agent_band"])
        self.assertEqual(len(grouped), 8)
        self.assertEqual({item["dimension"] for item in grouped}, {"map_id", "agent_band"})
        context_rows = [
            {
                "map_id": "map-a",
                "family": "conflict_component",
                "best_size": 8,
            },
            {
                "map_id": "map-a",
                "family": "conflict_component",
                "best_size": 16,
            },
        ]
        counts = _best_size_counts_by_context(context_rows, ["map_id"])
        self.assertEqual(
            counts["map_id"]["map-a"]["conflict_component"],
            {8: 1, 16: 1},
        )

    def test_label_matrix_requires_exact_trials_pairing_and_features(self) -> None:
        features = {f"f{i}": float(i) for i in range(124)}
        expected = {
            ("state", "candidate-a"): {
                "agents": [1, 2],
                "features": features,
                "before_conflicts": 4,
            },
            ("state", "candidate-b"): {
                "agents": [2, 3],
                "features": features,
                "before_conflicts": 4,
            },
        }
        aggregates = [
            {
                "state_id": "state",
                "candidate_id": candidate_id,
                "agent_count": 8,
                "actual_size": 2,
                "agents": agents,
                "features": features,
            }
            for candidate_id, agents in (
                ("candidate-a", [1, 2]),
                ("candidate-b", [2, 3]),
            )
        ]
        trials = []
        before_repair = "repair-before"
        for candidate_id in ("candidate-a", "candidate-b"):
            for trial_index in range(16):
                trials.append(
                    {
                        "schema": TRIAL_SCHEMA,
                        "state_id": "state",
                        "candidate_id": candidate_id,
                        "trial_index": trial_index,
                        "pp_seed": repairability_pp_seed(
                            before_repair, trial_index
                        ),
                        "before_conflicts": 4,
                        "before_fingerprint": "before",
                        "before_repair_fingerprint": before_repair,
                    }
                )
        validation = _validate_label_matrix(
            trials=trials,
            aggregates=aggregates,
            expected_candidates=expected,
        )
        self.assertTrue(validation["passed"])
        broken = copy.deepcopy(trials)
        broken[-1]["trial_index"] = 14
        validation = _validate_label_matrix(
            trials=broken,
            aggregates=aggregates,
            expected_candidates=expected,
        )
        self.assertFalse(validation["passed"])
        self.assertTrue(
            any("exactly 0-15" in message for message in validation["errors"])
        )

    def test_label_matrix_rejects_changed_before_fingerprint(self) -> None:
        before_repair = "repair-before"
        candidate = {
            "state_id": "state",
            "candidate_id": "candidate",
            "agents": [0, 1],
            "actual_size": 2,
            "agent_count": 2,
            "features": {f"feature_{index}": 0.0 for index in range(124)},
        }
        trials = [
            {
                "schema": TRIAL_SCHEMA,
                "state_id": "state",
                "candidate_id": "candidate",
                "trial_index": trial_index,
                "pp_seed": repairability_pp_seed(before_repair, trial_index),
                "before_conflicts": 4,
                "before_fingerprint": "wrong-before",
                "before_repair_fingerprint": before_repair,
            }
            for trial_index in range(16)
        ]
        validation = _validate_label_matrix(
            trials=trials,
            aggregates=[candidate],
            expected_candidates={
                ("state", "candidate"): {
                    "agents": [0, 1],
                    "features": dict(candidate["features"]),
                    "before_conflicts": 4,
                    "before_fingerprint": "expected-before",
                    "before_repair_fingerprint": before_repair,
                }
            },
        )
        self.assertFalse(validation["passed"])
        self.assertIn(
            "candidate before fingerprint changed: state candidate",
            validation["errors"],
        )


if __name__ == "__main__":
    unittest.main()
