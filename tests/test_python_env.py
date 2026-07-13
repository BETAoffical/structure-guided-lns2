from __future__ import annotations

import os
import unittest

try:
    import lns2_env
except ModuleNotFoundError:
    lns2_env = None


@unittest.skipUnless(
    lns2_env is not None and "LNS2_TEST_MAP" in os.environ,
    "the native LNS2 module is tested by Linux CTest",
)
class RepairEnvironmentTests(unittest.TestCase):
    def make_env(self) -> lns2_env.LNS2RepairEnv:
        return lns2_env.LNS2RepairEnv(
            os.environ["LNS2_TEST_MAP"],
            os.environ["LNS2_TEST_SCEN"],
            agent_count=80,
            time_limit=30.0,
            neighborhood_size=8,
            max_repair_iterations=3,
            context={"layout_mode": "random", "task_flow": "benchmark"},
        )

    def test_reset_is_deterministic_and_exposes_context(self) -> None:
        first = self.make_env().reset(seed=17)
        second = self.make_env().reset(seed=17)
        self.assertEqual(
            first["num_of_colliding_pairs"],
            second["num_of_colliding_pairs"],
        )
        self.assertEqual(
            [agent["path"] for agent in first["agents"]],
            [agent["path"] for agent in second["agents"]],
        )
        self.assertEqual(first["context"]["layout_mode"], "random")
        self.assertEqual(
            len(first["conflict_edges"]),
            first["num_of_colliding_pairs"],
        )

    def test_invalid_and_explicit_actions(self) -> None:
        env = self.make_env()
        state = env.reset(seed=19)
        if state["done"]:
            self.skipTest("initial soft PP was already feasible")

        result = env.step(
            {
                "mode": "seed",
                "heuristic": "collision",
                "seed_agent": 10_000,
                "neighborhood_size": 8,
                "random_seed": 123,
            }
        )
        self.assertFalse(result["metrics"]["action_valid"])
        self.assertTrue(result["metrics"]["generated"])
        self.assertEqual(result["metrics"]["requested_random_seed"], 123)

        state = result["observation"]
        if state["done"] or not state["conflict_edges"]:
            return
        edge = state["conflict_edges"][0]
        result = env.step(
            {"mode": "explicit_neighborhood", "agents": list(edge)}
        )
        self.assertTrue(result["metrics"]["action_valid"])
        self.assertEqual(sorted(result["metrics"]["neighborhood"]), sorted(edge))
        self.assertNotIn("reward", result)


if __name__ == "__main__":
    unittest.main()
