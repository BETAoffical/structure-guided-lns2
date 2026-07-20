from __future__ import annotations

import math
import os
import unittest

from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine

try:
    import lns2_env
except ModuleNotFoundError:
    lns2_env = None


@unittest.skipUnless(
    lns2_env is not None and "LNS2_TEST_MAP" in os.environ,
    "the native LNS2 module is tested by Linux CTest",
)
class RepairEnvironmentTests(unittest.TestCase):
    def test_native_timing_schema_is_current(self) -> None:
        self.assertEqual(
            lns2_env.repair_timing_schema,
            "lns2.repair_timing.v1",
        )

    def test_portable_tree_supports_raw_and_sigmoid_outputs(self) -> None:
        trees = [
            [
                {
                    "value": 0.0,
                    "feature_idx": 0,
                    "num_threshold": 0.0,
                    "missing_go_to_left": True,
                    "left": 1,
                    "right": 2,
                    "is_leaf": False,
                },
                {"value": -0.5, "is_leaf": True},
                {"value": 1.5, "is_leaf": True},
            ]
        ]
        predictor = lns2_env.PortableTreeEnsemble(2.0, trees)
        raw = predictor.predict_raw([[-1.0], [1.0]])
        positive = predictor.predict_positive([[-1.0], [1.0]])
        self.assertEqual(raw, [1.5, 3.5])
        self.assertAlmostEqual(positive[0], 1.0 / (1.0 + math.exp(-1.5)))
        self.assertAlmostEqual(positive[1], 1.0 / (1.0 + math.exp(-3.5)))
        pair_scores = predictor.score_pairwise_dense(
            [[-1.0], [1.0]], [0], [0]
        )
        expected_pair = (
            1.0 / (1.0 + math.exp(-1.5))
            + 1.0
            - 1.0 / (1.0 + math.exp(-3.5))
        ) / 2.0
        self.assertAlmostEqual(pair_scores[0], expected_pair)
        self.assertAlmostEqual(pair_scores[1], 1.0 - expected_pair)

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
        first_env = self.make_env()
        first = first_env.reset(seed=17)
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
        timings = first_env.get_last_reset_timings()
        for name in (
            "agent_and_solver_setup_seconds",
            "initial_solution_seconds",
            "state_snapshot_seconds",
            "state_to_python_seconds",
            "reset_total_seconds",
        ):
            self.assertGreaterEqual(timings[name], 0.0)
        self.assertLessEqual(
            timings["agent_and_solver_setup_seconds"]
            + timings["initial_solution_seconds"]
            + timings["state_snapshot_seconds"]
            + timings["state_to_python_seconds"],
            timings["reset_total_seconds"] + 1e-5,
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
        for name in (
            "native_step_seconds",
            "native_neighborhood_generation_seconds",
            "native_replan_seconds",
            "pp_replan_seconds",
            "native_state_snapshot_seconds",
            "native_repair_bookkeeping_seconds",
            "native_residual_seconds",
            "binding_solver_call_seconds",
            "binding_state_snapshot_seconds",
            "state_to_python_seconds",
            "metrics_to_python_seconds",
            "binding_residual_seconds",
            "binding_total_seconds",
        ):
            self.assertGreaterEqual(result["metrics"][name], 0.0)
        native_partition = sum(
            result["metrics"][name]
            for name in (
                "native_neighborhood_generation_seconds",
                "native_replan_seconds",
                "native_state_snapshot_seconds",
                "native_repair_bookkeeping_seconds",
                "native_residual_seconds",
            )
        )
        self.assertAlmostEqual(
            result["metrics"]["pp_replan_seconds"],
            result["metrics"]["native_replan_seconds"],
        )
        self.assertTrue(
            math.isclose(
                native_partition,
                result["metrics"]["native_step_seconds"],
                rel_tol=0.01,
                abs_tol=max(
                    1e-6, 0.01 * result["metrics"]["native_step_seconds"]
                ),
            )
        )
        binding_partition = sum(
            result["metrics"][name]
            for name in (
                "binding_solver_call_seconds",
                "binding_state_snapshot_seconds",
                "state_to_python_seconds",
                "metrics_to_python_seconds",
                "binding_residual_seconds",
            )
        )
        self.assertGreaterEqual(
            result["metrics"]["binding_solver_call_seconds"] + 1e-5,
            result["metrics"]["native_step_seconds"],
        )
        self.assertTrue(
            math.isclose(
                binding_partition,
                result["metrics"]["binding_total_seconds"],
                rel_tol=0.01,
                abs_tol=max(
                    1e-6, 0.01 * result["metrics"]["binding_total_seconds"]
                ),
            )
        )

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
        compact = env.propose_batch_compact([action, action])
        self.assertTrue(first["action_valid"])
        self.assertTrue(first["generated"])
        self.assertEqual(first["neighborhood"], second["neighborhood"])
        self.assertEqual(batch, [first, first])
        self.assertEqual(
            compact,
            [
                (
                    first["action_valid"],
                    first["generated"],
                    first["neighborhood"],
                ),
                (
                    first["action_valid"],
                    first["generated"],
                    first["neighborhood"],
                ),
            ],
        )
        self.assertEqual(state, env.get_state())

        result = env.step(action)
        self.assertEqual(first["neighborhood"], result["metrics"]["neighborhood"])

        invalid = env.propose({**action, "heuristic": "adaptive"})
        self.assertFalse(invalid["action_valid"])
        self.assertFalse(invalid["generated"])

    def test_compact_proposals_match_reference_for_all_families(self) -> None:
        env = self.make_env()
        state = env.reset(seed=31)
        if state["done"] or not state["conflict_edges"]:
            self.skipTest("initial soft PP was already feasible")
        seed_agent = state["conflict_edges"][0][0]
        actions = [
            {
                "mode": "seed",
                "heuristic": heuristic,
                "seed_agent": seed_agent,
                "neighborhood_size": size,
                "random_seed": 41000 + family_index * 100 + size,
            }
            for family_index, heuristic in enumerate(
                ("target", "collision", "random")
            )
            for size in (4, 8, 16)
        ]
        revision = env.get_state_revision()
        reference = env.propose_batch(actions)
        compact = env.propose_batch_compact(actions)
        self.assertEqual(len(reference), len(compact))
        for expected, actual in zip(reference, compact):
            self.assertEqual(
                (
                    expected["action_valid"],
                    expected["generated"],
                    expected["neighborhood"],
                ),
                actual,
            )
        self.assertEqual(revision, env.get_state_revision())
        self.assertEqual(state, env.get_state())

    def test_dense_native_features_match_projected_dicts(self) -> None:
        env = self.make_env()
        state = env.reset(seed=37)
        if state["done"] or not state["conflict_edges"]:
            self.skipTest("initial soft PP was already feasible")
        agents = sorted(map(int, state["conflict_edges"][0]))
        candidate = {
            "candidate_id": "native-dense-test",
            "agents": agents,
            "seed_agents": [agents[0]],
            "proposal_seeds": [42001],
            "selection_families": ["collision:4"],
            "proposal_count_by_family": {"collision:4": 1},
        }
        names = PROFILE_FEATURE_NAMES["realized_dynamic"]
        required = {"realized_dynamic": names}
        dictionary = OnlineFeatureEngine(
            state,
            backend="native",
            required_features=required,
        )
        dense = OnlineFeatureEngine(
            state,
            backend="native",
            required_features=required,
            dense_output=True,
        )
        dict_rows, _ = dictionary.realized_rows([candidate], state_hash="state")
        dense_rows, _ = dense.realized_rows([candidate], state_hash="state")
        expected = dict_rows[0]["features"]["realized_dynamic"]
        actual = dict(
            zip(dense_rows[0]["feature_names"], dense_rows[0]["feature_values"])
        )
        self.assertEqual(set(expected), set(actual))
        self.assertLessEqual(
            max(abs(float(expected[name]) - float(actual[name])) for name in expected),
            1e-12,
        )

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

    def test_cross_environment_rng_interference_requires_explicit_seed(self) -> None:
        first = self.make_env()
        first_state = first.reset(seed=29)
        if first_state["done"] or not first_state["conflict_edges"]:
            self.skipTest("first initial soft PP was already feasible")

        second = self.make_env()
        second_state = second.reset(seed=31)
        if second_state["done"] or not second_state["conflict_edges"]:
            self.skipTest("second initial soft PP was already feasible")

        with self.assertRaisesRegex(ValueError, "another LNS2RepairEnv"):
            first.step({"mode": "official"})

        recovered = first.step(
            {
                "mode": "seed",
                "heuristic": "collision",
                "seed_agent": 10_000,
                "neighborhood_size": 8,
                "random_seed": 31003,
            }
        )
        self.assertFalse(recovered["metrics"]["action_valid"])
        self.assertEqual(recovered["metrics"]["requested_random_seed"], 31003)

        with self.assertRaisesRegex(ValueError, "another LNS2RepairEnv"):
            second.step({"mode": "official"})

    def test_cross_environment_proposal_transfers_rng_ownership(self) -> None:
        first = self.make_env()
        first_state = first.reset(seed=29)
        if first_state["done"] or not first_state["conflict_edges"]:
            self.skipTest("first initial soft PP was already feasible")

        second = self.make_env()
        second_state = second.reset(seed=31)
        if second_state["done"] or not second_state["conflict_edges"]:
            self.skipTest("second initial soft PP was already feasible")

        proposal = first.propose(
            {
                "mode": "seed",
                "heuristic": "collision",
                "seed_agent": first_state["conflict_edges"][0][0],
                "neighborhood_size": 8,
                "random_seed": 31004,
            }
        )
        self.assertTrue(proposal["action_valid"])
        with self.assertRaisesRegex(ValueError, "another LNS2RepairEnv"):
            second.step({"mode": "official"})

    def test_invalid_proposal_does_not_steal_rng_ownership(self) -> None:
        first = self.make_env()
        first_state = first.reset(seed=29)
        if first_state["done"] or not first_state["conflict_edges"]:
            self.skipTest("first initial soft PP was already feasible")

        second = self.make_env()
        second_state = second.reset(seed=31)
        if second_state["done"] or not second_state["conflict_edges"]:
            self.skipTest("second initial soft PP was already feasible")

        invalid = first.propose(
            {
                "mode": "seed",
                "heuristic": "adaptive",
                "seed_agent": first_state["conflict_edges"][0][0],
                "neighborhood_size": 8,
                "random_seed": 31005,
            }
        )
        self.assertFalse(invalid["action_valid"])
        continued = second.step({"mode": "official"})
        self.assertEqual(continued["metrics"]["requested_random_seed"], -1)

    def test_explicit_repair_order_is_applied_and_deterministic(self) -> None:
        def run(order_seed: int) -> tuple[dict, dict]:
            env = self.make_env()
            state = env.reset(seed=29)
            if state["done"] or not state["conflict_edges"]:
                self.skipTest("initial soft PP was already feasible")
            agents = list(state["conflict_edges"][0])
            order = list(reversed(agents))
            result = env.step(
                {
                    "mode": "explicit_neighborhood",
                    "agents": agents,
                    "repair_order": order,
                    "random_seed": order_seed,
                }
            )
            self.assertTrue(result["metrics"]["action_valid"])
            self.assertEqual(result["metrics"]["requested_repair_order"], order)
            self.assertEqual(result["metrics"]["repair_order"], order)
            return result["observation"], result["metrics"]

        first, _ = run(33001)
        second, _ = run(33001)
        self.assertEqual(
            [agent["path"] for agent in first["agents"]],
            [agent["path"] for agent in second["agents"]],
        )

        for case in ("incomplete", "duplicate", "unknown", "seed_mode"):
            env = self.make_env()
            state = env.reset(seed=29)
            if state["done"] or not state["conflict_edges"]:
                return
            agents = list(state["conflict_edges"][0])
            order = list(agents)
            mode = "explicit_neighborhood"
            if case == "incomplete":
                order = agents[:1]
            elif case == "duplicate":
                order = [agents[0], agents[0]]
            elif case == "unknown":
                order = [agents[0], 10_000]
            else:
                mode = "seed"
            action = {
                "mode": mode,
                "agents": agents,
                "repair_order": order,
                "random_seed": 33002,
            }
            if mode == "seed":
                action.update(
                    {
                        "heuristic": "collision",
                        "seed_agent": agents[0],
                        "neighborhood_size": 8,
                    }
                )
            invalid = env.step(action)
            self.assertFalse(invalid["metrics"]["action_valid"], case)

        gcbs = lns2_env.LNS2RepairEnv(
            os.environ["LNS2_TEST_MAP"],
            os.environ["LNS2_TEST_SCEN"],
            agent_count=80,
            time_limit=30.0,
            neighborhood_size=8,
            replan_algorithm="GCBS",
            max_repair_iterations=1,
            context={},
        )
        state = gcbs.reset(seed=29)
        if not state["done"] and state["conflict_edges"]:
            agents = list(state["conflict_edges"][0])
            invalid = gcbs.step(
                {
                    "mode": "explicit_neighborhood",
                    "agents": agents,
                    "repair_order": agents,
                    "random_seed": 33003,
                }
            )
            self.assertFalse(invalid["metrics"]["action_valid"])

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
