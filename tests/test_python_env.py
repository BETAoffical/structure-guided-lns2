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

    def test_proposal_is_deterministic_and_does_not_change_state(self) -> None:
        env = self.make_env()
        state = env.reset(seed=29)
        if state["done"] or not state["conflict_edges"]:
            self.skipTest("initial soft PP was already feasible")
        seed_agent = state["conflict_edges"][0][0]
        action = {
            "mode": "seed",
            "heuristic": "collision",
            "seed_agent": seed_agent,
            "neighborhood_size": 8,
            "random_seed": 31002,
        }
        first = env.propose(action)
        second = env.propose(action)
        batch = env.propose_batch([action, action])
        self.assertTrue(first["action_valid"])
        self.assertTrue(first["generated"])
        self.assertEqual(first["neighborhood"], second["neighborhood"])
        self.assertEqual(batch, [first, first])
        self.assertEqual(state, env.get_state())

        result = env.step(action)
        self.assertEqual(first["neighborhood"], result["metrics"]["neighborhood"])

        invalid = env.propose({**action, "heuristic": "adaptive"})
        self.assertFalse(invalid["action_valid"])
        self.assertFalse(invalid["generated"])

    def test_proposal_requires_seeded_followup_repair(self) -> None:
        env = self.make_env()
        state = env.reset(seed=29)
        if state["done"] or not state["conflict_edges"]:
            self.skipTest("initial soft PP was already feasible")
        edge = list(state["conflict_edges"][0])
        action = {
            "mode": "seed",
            "heuristic": "collision",
            "seed_agent": edge[0],
            "neighborhood_size": 8,
            "random_seed": 31002,
        }
        env.propose(action)
        with self.assertRaises(ValueError):
            env.step({"mode": "explicit_neighborhood", "agents": edge})
        result = env.step(
            {"mode": "explicit_neighborhood", "agents": edge, "random_seed": 31003}
        )
        self.assertTrue(result["metrics"]["action_valid"])

    def test_partial_proposal_batch_still_protects_global_rng(self) -> None:
        env = self.make_env()
        state = env.reset(seed=29)
        if state["done"] or not state["conflict_edges"]:
            self.skipTest("initial soft PP was already feasible")
        edge = list(state["conflict_edges"][0])
        action = {
            "mode": "seed",
            "heuristic": "collision",
            "seed_agent": edge[0],
            "neighborhood_size": 8,
            "random_seed": 32002,
        }
        with self.assertRaises((TypeError, ValueError)):
            env.propose_batch([action, "not-an-action"])
        with self.assertRaises(ValueError):
            env.step({"mode": "explicit_neighborhood", "agents": edge})


if __name__ == "__main__":
    unittest.main()
