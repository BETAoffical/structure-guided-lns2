from __future__ import annotations

import unittest

from experiments.critical_conflicts import (
    CRITICAL_CONFIG_SCHEMA,
    critical_agent_features,
    load_critical_seed_config,
    select_critical_seed_agents,
    update_edge_ages,
)


def _state() -> dict:
    return {
        "rows": 2,
        "cols": 3,
        "obstacles": [0, 0, 0, 0, 0, 0],
        "conflict_edges": [[0, 1], [1, 2], [2, 3]],
        "agents": [
            {
                "id": index,
                "conflict_degree": 2 if index in {1, 2} else 1,
                "delay": index,
                "path": [index, index + 1],
            }
            for index in range(4)
        ],
    }


class CriticalConflictTests(unittest.TestCase):
    def test_graph_features_identify_bridges_and_articulation(self) -> None:
        rows = critical_agent_features(_state())
        self.assertEqual(rows[1]["articulation"], 1.0)
        self.assertEqual(rows[2]["articulation"], 1.0)
        self.assertEqual(rows[1]["bridge_incidence"], 2.0)
        self.assertEqual(rows[0]["bridge_incidence"], 1.0)

    def test_edge_ages_reset_removed_edges(self) -> None:
        first = update_edge_ages({}, _state())
        self.assertEqual(first[(0, 1)], 1)
        changed = _state()
        changed["conflict_edges"] = [[0, 1], [1, 3]]
        second = update_edge_ages(first, changed)
        self.assertEqual(second, {(0, 1): 2, (1, 3): 1})

    def test_two_seed_audit_ranks_original_four_seed_universe(self) -> None:
        selected, diagnostic = select_critical_seed_agents(
            _state(),
            profile="graph",
            margin_threshold=0.0,
            minimum_seeds=2,
            maximum_seeds=2,
            state_hash="0" * 64,
        )
        self.assertEqual(len(diagnostic["legacy_seed_agents"]), 4)
        self.assertEqual(len(diagnostic["ranked_seed_agents"]), 4)
        self.assertEqual(selected, diagnostic["ranked_seed_agents"][:2])

    def test_unpromoted_config_is_not_loadable(self) -> None:
        with self.assertRaisesRegex(ValueError, "did not pass"):
            load_critical_seed_config(
                {
                    "schema": CRITICAL_CONFIG_SCHEMA,
                    "deployment_promoted": False,
                    "profile": "graph",
                    "margin_threshold": 0.1,
                    "minimum_seeds": 2,
                    "maximum_seeds": 4,
                }
            )

    def test_unpromoted_diagnostic_requires_explicit_runtime_opt_in(self) -> None:
        payload = {
            "schema": CRITICAL_CONFIG_SCHEMA,
            "deployment_promoted": False,
            "diagnostic_only": True,
            "profile": "temporal",
            "margin_threshold": 0.2,
            "minimum_seeds": 2,
            "maximum_seeds": 4,
        }
        config = load_critical_seed_config(
            payload, allow_unpromoted_diagnostic=True
        )
        self.assertTrue(config.diagnostic_only)
        self.assertFalse(config.raw["deployment_promoted"])

    def test_promoted_config_is_loaded_exactly(self) -> None:
        config = load_critical_seed_config(
            {
                "schema": CRITICAL_CONFIG_SCHEMA,
                "deployment_promoted": True,
                "profile": "temporal",
                "margin_threshold": 0.1,
                "minimum_seeds": 2,
                "maximum_seeds": 4,
                "source": {"audit": "unit"},
            }
        )
        self.assertEqual(config.profile, "temporal")
        self.assertEqual(config.margin_threshold, 0.1)
        self.assertEqual(config.source, {"audit": "unit"})


if __name__ == "__main__":
    unittest.main()
