from __future__ import annotations

import unittest

from lns2_selector.runtime.fingerprints import semantic_fingerprint
from lns2_selector.runtime.portable_scalar import (
    PORTABLE_SCALAR_MODEL_SCHEMA,
    load_portable_scalar_model,
)
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome
from lns2_selector.training.tree_utils import balanced_map_folds


class PortableRuntimeTests(unittest.TestCase):
    def test_repair_outcome_preserves_native_rollback_contract(self) -> None:
        self.assertEqual(
            classify_repair_outcome(
                before_fingerprint="same",
                after_fingerprint="same",
                replan_success=False,
                conflicts_before=4,
                conflicts_after=4,
            ),
            "hard_failure",
        )
        with self.assertRaisesRegex(ValueError, "failed PP changed"):
            classify_repair_outcome(
                before_fingerprint="before",
                after_fingerprint="after",
                replan_success=False,
                conflicts_before=4,
                conflicts_after=4,
            )

    def test_portable_scalar_model_validates_and_predicts(self) -> None:
        tree = [
            {
                "value": 0.0,
                "feature_idx": 0,
                "num_threshold": 0.5,
                "missing_go_to_left": True,
                "left": 1,
                "right": 2,
                "is_leaf": False,
            },
            {
                "value": -1.0,
                "feature_idx": 0,
                "num_threshold": 0.0,
                "missing_go_to_left": False,
                "left": 0,
                "right": 0,
                "is_leaf": True,
            },
            {
                "value": 2.0,
                "feature_idx": 0,
                "num_threshold": 0.0,
                "missing_go_to_left": False,
                "left": 0,
                "right": 0,
                "is_leaf": True,
            },
        ]
        semantic = {
            "name": "progress",
            "profile": "realized_dynamic",
            "feature_names": ["x"],
            "baseline": 1.0,
            "trees": [tree],
            "transform": "identity",
        }
        payload = {
            "schema": PORTABLE_SCALAR_MODEL_SCHEMA,
            **semantic,
            "semantic_fingerprint": semantic_fingerprint(semantic),
        }
        model = load_portable_scalar_model(payload)
        rows = [
            {
                "feature_profile": "realized_dynamic",
                "feature_names": ["x"],
                "feature_values": [0.0],
            },
            {
                "feature_profile": "realized_dynamic",
                "feature_names": ["x"],
                "feature_values": [1.0],
            },
        ]
        self.assertEqual(model.predict(rows), [0.0, 3.0])

    def test_balanced_map_folds_cover_each_map_once(self) -> None:
        rows = [
            {"map_id": f"map-{index}", "layout_mode": "regular"}
            for index in range(4)
        ]
        folds = balanced_map_folds(rows)
        validation_maps = [
            map_id for fold in folds for map_id in fold["validation_maps"]
        ]
        self.assertEqual(sorted(validation_maps), sorted(row["map_id"] for row in rows))
