from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.feature_schema_v2 import canonicalize_features
from experiments.stride_feature_cache_audit import (
    REGISTERED_PATHS,
    _feature_dict,
    validate_feature_cache_audit_config,
)


class StrideFeatureCacheAuditTest(unittest.TestCase):
    def _config(self) -> dict:
        root = Path(__file__).resolve().parents[2]
        return json.loads(
            (root / "configs" / "stride_feature_cache_audit.json").read_text(
                encoding="utf-8"
            )
        )

    def test_contract_is_action_preserving_and_non_formal(self) -> None:
        config = self._config()
        validate_feature_cache_audit_config(config)
        self.assertEqual(tuple(config["registered_paths"]), REGISTERED_PATHS)
        self.assertFalse(config["candidate_actions_may_change"])
        self.assertFalse(config["formal_speed_claim"])
        self.assertEqual(config["floating_tolerance"], 1e-12)

    def test_contract_rejects_path_or_action_drift(self) -> None:
        config = self._config()
        config["registered_paths"] = config["registered_paths"][:-1]
        with self.assertRaisesRegex(ValueError, "contract changed"):
            validate_feature_cache_audit_config(config)
        config = self._config()
        config["candidate_actions_may_change"] = True
        with self.assertRaisesRegex(ValueError, "contract changed"):
            validate_feature_cache_audit_config(config)

    def test_sparse_rows_are_canonicalized_before_equivalence_check(self) -> None:
        raw = {
            "state.conflicts": 3.0,
            "proposal.actual_size": 8.0,
            "proposal.selection_family": "target",
        }
        row = {"features": {"realized_dynamic": raw}}
        self.assertEqual(
            _feature_dict(row), canonicalize_features(raw, "realized_dynamic")
        )

    def test_dense_rows_preserve_the_requested_projection(self) -> None:
        row = {"feature_names": ["a", "b"], "feature_values": [1, 2.5]}
        self.assertEqual(_feature_dict(row), {"a": 1.0, "b": 2.5})


if __name__ == "__main__":
    unittest.main()
