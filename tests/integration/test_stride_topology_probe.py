from __future__ import annotations

import collections
import json
import unittest
from pathlib import Path

from experiments.state_analysis import ConflictEvent, StateAnalysis
from experiments.stride_topology_probe import (
    FEATURE_NAMES,
    topology_interaction_features,
    validate_topology_probe_config,
)


class StrideTopologyProbeTest(unittest.TestCase):
    def test_registered_config_is_diagnostic_and_forbids_candidate_trials(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "configs" / "stride_topology_probe.json").read_text(
                encoding="utf-8"
            )
        )
        validate_topology_probe_config(config)
        self.assertEqual(config["expected_state_count"], 48)
        self.assertEqual(config["expected_candidate_count"], 854)
        self.assertEqual(tuple(config["feature_names"]), FEATURE_NAMES)
        config["candidate_repair_trials_allowed_during_extraction"] = True
        with self.assertRaisesRegex(ValueError, "no-PP"):
            validate_topology_probe_config(config)

    def test_candidate_relative_topology_features_have_registered_semantics(self) -> None:
        state = {
            "agents": [
                {"id": 0, "path": [0, 1]},
                {"id": 1, "path": [2, 1]},
                {"id": 2, "path": [5, 0]},
            ]
        }
        analysis = StateAnalysis(
            rows=2,
            cols=3,
            free_cells=set(range(6)),
            degrees={0: 1, 1: 2, 2: 3, 3: 2, 4: 3, 5: 1},
            articulation={1},
            obstacle_rate_2={},
            obstacle_rate_4={},
            visit_heat=collections.Counter({0: 1, 1: 3, 2: 1, 5: 2}),
            agent_heat=collections.Counter(),
            events=[
                ConflictEvent(1, "vertex", 0, 1, (1,)),
                ConflictEvent(2, "vertex", 1, 2, (0,)),
            ],
            pair_set={(0, 1), (1, 2)},
            component_id={0: 0, 1: 0, 2: 0},
            component_members={0: {0, 1, 2}},
        )
        values = topology_interaction_features(state, analysis, [0, 1])
        self.assertEqual(tuple(values), FEATURE_NAMES)
        self.assertAlmostEqual(values[FEATURE_NAMES[0]], 0.5)
        self.assertAlmostEqual(values[FEATURE_NAMES[1]], 1.0)
        self.assertAlmostEqual(
            values["topology.realized.internal_articulation_event_coverage"], 1.0
        )
        self.assertAlmostEqual(
            values["topology.realized.boundary_low_degree_event_ratio"], 0.5
        )
        self.assertAlmostEqual(
            values["topology.realized.low_degree_agent_coverage"], 2.0 / 3.0
        )
        self.assertAlmostEqual(
            values["topology.realized.path_low_degree_cell_coverage"], 2.0 / 3.0
        )
        self.assertAlmostEqual(
            values["topology.realized.low_degree_visit_heat_coverage"], 4.0 / 6.0
        )


if __name__ == "__main__":
    unittest.main()
