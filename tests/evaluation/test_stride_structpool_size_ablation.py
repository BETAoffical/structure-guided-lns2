from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_structpool_size_ablation import (
    TRIAL_SCHEMA,
    _copy_reused_trial,
    _family_variant,
    validate_size_ablation_config,
)


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


if __name__ == "__main__":
    unittest.main()
