from __future__ import annotations

import unittest

import numpy as np

from experiments.mixed_full_v2 import _feature_ranges, _portable_payload
from experiments.context_audit import PairwiseModel


class _Estimator:
    classes_ = [0, 1]
    _baseline_prediction = np.asarray([[0.25]])
    _predictors = []


class MixedFullV2Tests(unittest.TestCase):
    def test_feature_ranges_include_missing_values_as_zero(self) -> None:
        rows = [
            {"features": {"a": 2.0}},
            {"features": {"a": 4.0, "b": -1.0}},
        ]
        self.assertEqual(
            _feature_ranges(rows, ["a", "b"]),
            {"a": [2.0, 4.0], "b": [-1.0, 0.0]},
        )

    def test_portable_payload_registers_full_base_feature_schema(self) -> None:
        model = PairwiseModel("realized_dynamic", ["state.a", "x"], _Estimator())
        payload = _portable_payload(model, "abc")
        self.assertEqual(payload["feature_names"], ["state.a", "x"])
        self.assertEqual(payload["source_model_sha256"], "abc")
        self.assertEqual(payload["baseline"], 0.25)


if __name__ == "__main__":
    unittest.main()
