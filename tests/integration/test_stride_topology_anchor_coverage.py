from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from experiments.stride_topology_anchor_coverage import (
    analyze_topology_anchor_coverage_rows,
    collect_topology_anchor_coverage,
    validate_topology_anchor_coverage_config,
)


class StrideTopologyAnchorCoverageTest(unittest.TestCase):
    def _config(self) -> dict:
        root = Path(__file__).resolve().parents[2]
        return json.loads(
            (root / "configs" / "stride_topology_anchor_coverage.json").read_text(
                encoding="utf-8"
            )
        )

    def test_registered_config_is_capped_and_proposal_only(self) -> None:
        config = self._config()
        validate_topology_anchor_coverage_config(config)
        self.assertEqual(config["candidate_generator_id"], "stride-topoanchor-v1")
        self.assertEqual(
            config["augmentation"]["maximum_total_candidates_per_state"], 24
        )
        self.assertNotIn(".step(", inspect.getsource(collect_topology_anchor_coverage))
        config["augmentation"]["maximum_added_candidates_per_state"] = 7
        with self.assertRaisesRegex(ValueError, "augmentation protocol"):
            validate_topology_anchor_coverage_config(config)

    def test_augmented_analyzer_enforces_candidate_cap(self) -> None:
        config = self._config()
        groups = config["topology_group_gates"]["required_groups"]
        rows = []
        for task_index in range(12):
            for solver_seed in (1, 2):
                rows.append(
                    {
                        "state_id": f"state-{task_index}-{solver_seed}",
                        "task_id": f"task-{task_index}",
                        "solver_seed": solver_seed,
                        "layout_family": groups[task_index % len(groups)],
                        "state_fingerprint_preserved": True,
                        "proposal_repetitions_deterministic": True,
                        "candidate_count": 24,
                        "base_candidate_count": 18,
                        "added_candidate_count": 6,
                        "requested_sizes_represented": [4, 8, 16],
                        "articulation_relevant": task_index < 8,
                        "low_degree_relevant": True,
                        "max_incident_articulation_coverage": 0.8,
                        "max_internal_articulation_coverage": 0.5,
                        "max_incident_low_degree_coverage": 0.9,
                        "max_internal_low_degree_coverage": 0.6,
                    }
                )
        report = analyze_topology_anchor_coverage_rows(config, rows)
        self.assertTrue(report["passed"])
        rows[0]["candidate_count"] = 25
        self.assertFalse(analyze_topology_anchor_coverage_rows(config, rows)["passed"])


if __name__ == "__main__":
    unittest.main()
