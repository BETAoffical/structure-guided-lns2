from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from experiments.stride_topology_anchor_failure import validate_topology_anchor_failure_config
from experiments.stride_topology_boundary_coverage import (
    collect_topology_boundary_coverage,
    validate_topology_boundary_coverage_config,
)


class StrideTopologyBoundaryCoverageTest(unittest.TestCase):
    def _read(self, name: str) -> dict:
        root = Path(__file__).resolve().parents[2]
        return json.loads((root / "configs" / name).read_text(encoding="utf-8"))

    def test_failure_decomposition_is_posthoc_and_read_only(self) -> None:
        config = self._read("stride_topology_anchor_failure.json")
        validate_topology_anchor_failure_config(config)
        self.assertFalse(config["candidate_repair_trials_allowed"])
        self.assertFalse(config["training_allowed"])

    def test_boundary_revision_is_size_16_capped_and_proposal_only(self) -> None:
        config = self._read("stride_topology_boundary_coverage.json")
        validate_topology_boundary_coverage_config(config)
        self.assertEqual(config["candidate_generator_id"], "stride-topoboundary-v1")
        self.assertEqual(config["augmentation"]["neighborhood_size"], 16)
        self.assertEqual(config["augmentation"]["maximum_added_candidates_per_state"], 2)
        self.assertNotIn(".step(", inspect.getsource(collect_topology_boundary_coverage))
        config["augmentation"]["topology_core_budget"] = 5
        with self.assertRaisesRegex(ValueError, "augmentation protocol"):
            validate_topology_boundary_coverage_config(config)


if __name__ == "__main__":
    unittest.main()
