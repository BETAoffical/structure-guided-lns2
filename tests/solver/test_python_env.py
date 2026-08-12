from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments._common import NATIVE_SEMANTICS_SCHEMA as EXPERIMENT_NATIVE_SCHEMA
from experiments.repair_collection import state_fingerprint
from lns2_selector.solver.native import (
    NATIVE_SEMANTICS_SCHEMA as SELECTOR_NATIVE_SCHEMA,
    load_native_module,
    native_identity,
)

try:
    import lns2_env
except ModuleNotFoundError:
    lns2_env = None

from scripts import collect_closed_loop_confirmation as collector_cli


class NativeModuleDiscoveryTests(unittest.TestCase):
    def test_native_semantics_schema_is_shared_across_boundaries(self) -> None:
        self.assertEqual(EXPERIMENT_NATIVE_SCHEMA, SELECTOR_NATIVE_SCHEMA)

    def test_canonical_loader_rejects_incompatible_native_semantics(self) -> None:
        incompatible = SimpleNamespace(
            LNS2RepairEnv=object,
            native_semantics_schema="lns2.native_semantics.corrected.v1",
            repair_timing_schema="lns2.repair_timing.v2",
        )
        with (
            patch(
                "lns2_selector.solver.native.importlib.import_module",
                return_value=incompatible,
            ),
            self.assertRaisesRegex(RuntimeError, "unsupported native semantics"),
        ):
            load_native_module()

        with self.assertRaisesRegex(RuntimeError, "unsupported native semantics"):
            native_identity(incompatible)

    def test_collector_cli_registers_existing_native_build(self) -> None:
        if collector_cli.NATIVE_BUILD.is_dir():
            self.assertIn(str(collector_cli.NATIVE_BUILD), sys.path)

    def test_collector_cli_builds_a_registered_seed_subset(self) -> None:
        self.assertEqual(
            collector_cli._selected_job_keys(["task-b", "task-a"], [3, 1]),
            {
                ("task-a", 1),
                ("task-a", 3),
                ("task-b", 1),
                ("task-b", 3),
            },
        )
        self.assertIsNone(collector_cli._selected_job_keys(["task-a"], None))
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            collector_cli._selected_job_keys(None, [1])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            collector_cli._selected_job_keys(["task-a"], [-1])


@unittest.skipUnless(
    lns2_env is not None and "LNS2_TEST_MAP" in os.environ,
    "the native LNS2 module is tested by Linux CTest",
)
class RepairEnvironmentTests(unittest.TestCase):
    def test_native_timing_schema_is_current(self) -> None:
        self.assertEqual(
            lns2_env.repair_timing_schema,
            "lns2.repair_timing.v2",
        )
        self.assertEqual(
            lns2_env.native_semantics_schema,
            "lns2.native_semantics.upstream_compatible.v1",
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

    def test_portable_tree_rejects_invalid_structure(self) -> None:
        cyclic_tree = [
            [
                {
                    "value": 0.0,
                    "feature_idx": 0,
                    "num_threshold": 0.0,
                    "missing_go_to_left": True,
                    "left": 0,
                    "right": 1,
                    "is_leaf": False,
                },
                {"value": 1.0, "is_leaf": True},
            ]
        ]
        with self.assertRaisesRegex(ValueError, "cycle"):
            lns2_env.PortableTreeEnsemble(0.0, cyclic_tree)

    def test_portable_tree_validates_a_deep_chain_without_recursion(self) -> None:
        depth = 20_000
        nodes = [
            {
                "value": 0.0,
                "feature_idx": 0,
                "num_threshold": 0.0,
                "missing_go_to_left": True,
                "left": index + 1,
                "right": depth,
                "is_leaf": False,
            }
            for index in range(depth)
        ]
        nodes.append({"value": 2.0, "is_leaf": True})
        predictor = lns2_env.PortableTreeEnsemble(0.0, [nodes])
        self.assertEqual(predictor.predict_raw([[-1.0]]), [2.0])

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

    def test_default_context_is_isolated_and_observations_return_copies(self) -> None:
        supplied_context = {"nested": {"source": True}}
        first_env = lns2_env.LNS2RepairEnv(
            os.environ["LNS2_TEST_MAP"],
            os.environ["LNS2_TEST_SCEN"],
            agent_count=80,
            context=supplied_context,
        )
        second_env = lns2_env.LNS2RepairEnv(
            os.environ["LNS2_TEST_MAP"],
            os.environ["LNS2_TEST_SCEN"],
            agent_count=80,
        )
        first = first_env.reset(seed=17)
        second = second_env.reset(seed=19)
        supplied_context["nested"]["source"] = False
        self.assertIsNot(first["context"], second["context"])
        first["context"]["leaked"] = True
        first["context"]["nested"]["source"] = False
        self.assertNotIn("leaked", second["context"])
        fresh = first_env.get_state()["context"]
        self.assertNotIn("leaked", fresh)
        self.assertTrue(fresh["nested"]["source"])

    def test_constructor_rejects_unsafe_instance_inputs(self) -> None:
        map_path = os.environ["LNS2_TEST_MAP"]
        scenario_path = os.environ["LNS2_TEST_SCEN"]
        with self.assertRaisesRegex(ValueError, "agent_count"):
            lns2_env.LNS2RepairEnv(map_path, scenario_path, agent_count=0)
        with self.assertRaises(TypeError):
            lns2_env.LNS2RepairEnv(map_path, scenario_path)
        with tempfile.TemporaryDirectory() as directory:
            missing_scenario = Path(directory) / "missing.scen"
            with self.assertRaisesRegex(ValueError, "scenario_path"):
                lns2_env.LNS2RepairEnv(
                    map_path,
                    str(missing_scenario),
                    agent_count=2,
                )
            self.assertFalse(missing_scenario.exists())

            truncated_scenario = Path(directory) / "truncated.scen"
            truncated_scenario.write_text("version 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fewer than 2 agents"):
                lns2_env.LNS2RepairEnv(
                    map_path,
                    str(truncated_scenario),
                    agent_count=2,
                )

            whitespace_scenario = Path(directory) / "whitespace.scen"
            whitespace_scenario.write_text(
                "version 1\n"
                "0 map 32 32 0 0 1 1 1.0\n"
                "0 map 32 32 1 1 2 2 1.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "malformed"):
                lns2_env.LNS2RepairEnv(
                    map_path,
                    str(whitespace_scenario),
                    agent_count=2,
                )

            tab_header_map = Path(directory) / "tab-header.map"
            tab_header_map.write_text(
                "type octile\n"
                "height\t3\n"
                "width 3\n"
                "map\n"
                "...\n"
                "...\n"
                "...\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "map header is malformed"):
                lns2_env.LNS2RepairEnv(
                    str(tab_header_map),
                    scenario_path,
                    agent_count=1,
                )

            duplicate_start = Path(directory) / "duplicate-start.scen"
            duplicate_start.write_text(
                "2\n0,0,2,2\n0,0,2,1\n",
                encoding="utf-8",
            )
            custom_map = Path(directory) / "custom.map"
            custom_map.write_text("3,3\n...\n...\n...\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate start"):
                lns2_env.LNS2RepairEnv(
                    str(custom_map),
                    str(duplicate_start),
                    agent_count=2,
                )

            duplicate_goal = Path(directory) / "duplicate-goal.scen"
            duplicate_goal.write_text(
                "2\n0,0,2,2\n0,1,2,2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate goal"):
                lns2_env.LNS2RepairEnv(
                    str(custom_map),
                    str(duplicate_goal),
                    agent_count=2,
                )

    def test_constructor_accepts_crlf_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            map_path = Path(directory) / "custom.map"
            scenario_path = Path(directory) / "custom.scen"
            map_path.write_bytes(b"3,3\r\n...\r\n...\r\n...\r\n")
            scenario_path.write_bytes(
                b"2\r\n0,0,2,2\r\n0,2,2,0\r\n"
            )
            env = lns2_env.LNS2RepairEnv(
                str(map_path),
                str(scenario_path),
                agent_count=2,
                time_limit=1.0,
            )
            state = env.reset(seed=7)
            self.assertEqual((state["rows"], state["cols"]), (3, 3))
            self.assertEqual(len(state["agents"]), 2)

            moving_map_path = Path(directory) / "moving.map"
            moving_scenario_path = Path(directory) / "moving.scen"
            moving_map_path.write_bytes(
                b"type octile\r\n"
                b"height 3\r\n"
                b"width 3\r\n"
                b"map\r\n"
                b"...\r\n"
                b"...\r\n"
                b"...\r\n"
            )
            moving_scenario_path.write_bytes(
                b"version 1\r\n"
                b"0\tmoving.map\t3\t3\t0\t0\t2\t2\t4\r\n"
            )
            moving_env = lns2_env.LNS2RepairEnv(
                str(moving_map_path),
                str(moving_scenario_path),
                agent_count=1,
                time_limit=1.0,
            )
            moving_state = moving_env.reset(seed=11)
            self.assertEqual(
                (moving_state["rows"], moving_state["cols"]),
                (3, 3),
            )
            self.assertEqual(len(moving_state["agents"]), 1)

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
        self.assertTrue(result["metrics"]["step_applied"])
        self.assertFalse(result["metrics"]["action_valid"])
        self.assertTrue(result["metrics"]["generated"])
        self.assertEqual(result["metrics"]["requested_random_seed"], 123)
        for name in (
            "native_step_seconds",
            "episode_runtime_delta_seconds",
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
        self.assertEqual(
            result["metrics"]["step_runtime"],
            result["metrics"]["native_step_seconds"],
        )
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
        self.assertGreater(
            result["metrics"]["native_state_snapshot_seconds"],
            0.0,
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

    def test_step_runtime_excludes_time_between_calls(self) -> None:
        env = self.make_env()
        state = env.reset(seed=29)
        if state["done"]:
            self.skipTest("initial soft PP was already feasible")
        time.sleep(0.05)
        result = env.step(
            {
                "mode": "seed",
                "heuristic": "collision",
                "seed_agent": 10_000,
                "neighborhood_size": 8,
                "random_seed": 123,
            }
        )
        metrics = result["metrics"]
        self.assertEqual(metrics["step_runtime"], metrics["native_step_seconds"])
        self.assertGreaterEqual(metrics["episode_runtime_delta_seconds"], 0.04)
        self.assertGreater(
            metrics["episode_runtime_delta_seconds"],
            metrics["step_runtime"],
        )

    def test_live_deadline_stops_state_and_proposals(self) -> None:
        env = lns2_env.LNS2RepairEnv(
            os.environ["LNS2_TEST_MAP"],
            os.environ["LNS2_TEST_SCEN"],
            agent_count=80,
            time_limit=0.05,
            neighborhood_size=8,
            context={},
        )
        state = env.reset(seed=29)
        if state["done"] or not state["conflict_edges"]:
            self.skipTest("deadline source did not enter repair")
        edge = state["conflict_edges"][0]
        time.sleep(0.08)
        expired = env.get_state()
        self.assertTrue(expired["done"])
        self.assertEqual(expired["runtime"], 0.05)
        time.sleep(0.02)
        self.assertEqual(env.get_state()["runtime"], expired["runtime"])
        proposal = env.propose(
            {
                "mode": "seed",
                "heuristic": "collision",
                "seed_agent": edge[0],
                "neighborhood_size": 8,
                "random_seed": 555,
            }
        )
        self.assertFalse(proposal["action_valid"])
        self.assertFalse(proposal["generated"])
        terminal = env.step({"mode": "official", "random_seed": 556})
        self.assertFalse(terminal["metrics"]["step_applied"])
        self.assertFalse(terminal["terminated"])
        self.assertTrue(terminal["truncated"])
        self.assertTrue(terminal["observation"]["done"])
        self.assertEqual(terminal["metrics"]["native_step_seconds"], 0.0)

    def test_non_time_terminal_runtime_is_frozen(self) -> None:
        env = lns2_env.LNS2RepairEnv(
            os.environ["LNS2_TEST_MAP"],
            os.environ["LNS2_TEST_SCEN"],
            agent_count=80,
            time_limit=30.0,
            neighborhood_size=8,
            max_repair_iterations=1,
            context={},
        )
        initial = env.reset(seed=29)
        self.assertFalse(initial["done"], "runtime-freeze fixture must need repair")
        terminal = env.step(
            {"mode": "official", "random_seed": 31005}
        )["observation"]
        self.assertTrue(terminal["done"])
        time.sleep(0.02)
        self.assertEqual(env.get_state()["runtime"], terminal["runtime"])

        incomplete_env = lns2_env.LNS2RepairEnv(
            os.environ["LNS2_TEST_MAP"],
            os.environ["LNS2_TEST_SCEN"],
            agent_count=80,
            time_limit=0.0,
            neighborhood_size=8,
            context={},
        )
        incomplete = incomplete_env.reset(seed=29)
        self.assertTrue(incomplete["done"])
        self.assertFalse(incomplete["initial_solution_complete"])
        time.sleep(0.02)
        self.assertEqual(
            incomplete_env.get_state()["runtime"],
            incomplete["runtime"],
        )

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
        after_proposals = env.get_state()
        self.assertGreaterEqual(after_proposals["runtime"], state["runtime"])
        self.assertEqual(
            {key: value for key, value in state.items() if key != "runtime"},
            {
                key: value
                for key, value in after_proposals.items()
                if key != "runtime"
            },
        )

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
        after_proposals = env.get_state()
        self.assertGreaterEqual(after_proposals["runtime"], state["runtime"])
        self.assertEqual(
            {key: value for key, value in state.items() if key != "runtime"},
            {
                key: value
                for key, value in after_proposals.items()
                if key != "runtime"
            },
        )

    def test_restore_paths_rebuilds_repair_state_and_resets_budget(self) -> None:
        source = self.make_env()
        initial = source.reset(seed=31)
        paths = [list(agent["path"]) for agent in initial["agents"]]

        env = self.make_env()
        restored = env.reset_paths(paths, seed=41007)

        self.assertEqual(env.get_state_revision(), 1)
        self.assertEqual(
            [agent["path"] for agent in restored["agents"]], paths
        )
        self.assertEqual(
            restored["num_of_colliding_pairs"],
            initial["num_of_colliding_pairs"],
        )
        self.assertEqual(restored["conflict_edges"], initial["conflict_edges"])
        self.assertEqual(restored["sum_of_costs"], initial["sum_of_costs"])
        self.assertEqual(restored["iteration"], 0)
        self.assertEqual(
            restored["low_level"],
            {"expanded": 0, "generated": 0, "reopened": 0, "runs": 0},
        )
        self.assertLess(restored["runtime"], 0.1)
        self.assertEqual(
            env.get_last_reset_timings()["initial_solution_seconds"], 0.0
        )
        self.assertGreaterEqual(
            env.get_last_reset_timings()["path_restore_seconds"], 0.0
        )

        if restored["done"]:
            self.skipTest("warm-start fixture is already terminal")
        stepped = env.step({"mode": "official"})
        self.assertTrue(stepped["metrics"]["step_applied"])
        self.assertGreater(stepped["observation"]["low_level"]["runs"], 0)

        restored_again = env.restore_paths(paths, seed=41009)
        self.assertEqual(env.get_state_revision(), 3)
        self.assertEqual(restored_again["conflict_edges"], initial["conflict_edges"])
        self.assertEqual(
            restored_again["low_level"],
            {"expanded": 0, "generated": 0, "reopened": 0, "runs": 0},
        )
        restore_timings = env.get_last_reset_timings()
        self.assertEqual(restore_timings["initial_solution_seconds"], 0.0)
        self.assertGreaterEqual(restore_timings["path_restore_seconds"], 0.0)

        invalid = [list(path) for path in paths]
        invalid[0][0] = invalid[0][0] + 1
        before_invalid = env.get_state()
        revision_before_invalid = env.get_state_revision()
        with self.assertRaisesRegex(ValueError, "does not start"):
            env.reset_paths(invalid, seed=41008)
        after_invalid = env.get_state()
        self.assertEqual(env.get_state_revision(), revision_before_invalid)
        self.assertEqual(
            {key: value for key, value in after_invalid.items() if key != "runtime"},
            {
                key: value
                for key, value in before_invalid.items()
                if key != "runtime"
            },
        )

        other = self.make_env()
        with self.assertRaisesRegex(ValueError, "does not start"):
            other.reset_paths(invalid, seed=41010)
        # The rejected reset did not touch the process-global random stream,
        # so the still-live owner can continue without an explicit seed.
        continued = env.step({"mode": "official"})
        self.assertTrue(continued["metrics"]["step_applied"])

    def test_reset_rejects_negative_seed_without_replacing_state(self) -> None:
        env = self.make_env()
        state = env.reset(seed=31)
        revision = env.get_state_revision()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            env.reset(seed=-1)
        self.assertEqual(env.get_state_revision(), revision)
        after = env.get_state()
        self.assertEqual(
            {key: value for key, value in after.items() if key != "runtime"},
            {key: value for key, value in state.items() if key != "runtime"},
        )

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

    def test_invalid_proposal_does_not_require_a_seed_in_the_same_environment(
        self,
    ) -> None:
        env = self.make_env()
        state = env.reset(seed=29)
        if state["done"] or not state["conflict_edges"]:
            self.skipTest("initial soft PP was already feasible")
        invalid = env.propose(
            {
                "mode": "seed",
                "heuristic": "adaptive",
                "seed_agent": state["conflict_edges"][0][0],
                "neighborhood_size": 8,
                "random_seed": 31005,
            }
        )
        self.assertFalse(invalid["action_valid"])
        continued = env.step({"mode": "official"})
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
                    "collect_pp_diagnostics": True,
                }
            )
            self.assertTrue(result["metrics"]["action_valid"])
            self.assertEqual(result["metrics"]["requested_repair_order"], order)
            self.assertEqual(result["metrics"]["repair_order"], order)
            metrics = result["metrics"]
            self.assertIn(
                metrics["pp_failure_reason"],
                {"none", "conflict_bound_exceeded", "time_limit"},
            )
            self.assertEqual(
                metrics["pp_attempted_agent_count"],
                len(metrics["pp_agent_diagnostics"]),
            )
            self.assertLessEqual(
                metrics["pp_inserted_agent_count"],
                metrics["pp_attempted_agent_count"],
            )
            for index, diagnostic in enumerate(metrics["pp_agent_diagnostics"]):
                self.assertEqual(diagnostic["order_index"], index)
                self.assertEqual(diagnostic["agent_id"], order[index])
                self.assertTrue(
                    set(diagnostic["external_blocker_agents"]).isdisjoint(agents)
                )
                self.assertTrue(
                    set(diagnostic["internal_blocker_agents"]).issubset(agents)
                )
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

    def test_replay_neighborhood_accepts_recorded_no_conflict_noop(self) -> None:
        env = self.make_env()
        state = env.reset(seed=29)
        if state["done"]:
            self.skipTest("initial soft PP was already feasible")
        agents = [
            int(agent["id"])
            for agent in state["agents"]
            if int(agent["conflict_degree"]) == 0
        ][:2]
        if not agents:
            self.skipTest("test state has no non-conflicting replay agent")
        result = env.step(
            {
                "mode": "replay_neighborhood",
                "agents": agents,
            }
        )
        self.assertEqual(result["metrics"]["requested_mode"], "replay_neighborhood")
        self.assertTrue(result["metrics"]["action_valid"])
        self.assertEqual(result["metrics"]["neighborhood"], sorted(agents))
        self.assertFalse(result["metrics"]["replan_success"])
        self.assertEqual(result["metrics"]["pp_failure_reason"], "not_run")
        self.assertEqual(result["metrics"]["pp_agent_diagnostics"], [])

    def test_replay_neighborhood_rejects_empty_action(self) -> None:
        env = self.make_env()
        state = env.reset(seed=29)
        if state["done"]:
            self.skipTest("initial soft PP was already feasible")
        result = env.step({"mode": "replay_neighborhood", "agents": []})
        self.assertFalse(result["metrics"]["action_valid"])

    def test_pp_random_seed_is_independent_of_action_rng_history(self) -> None:
        def run(action_seed: int) -> tuple[list[list[int]], dict[str, object]]:
            env = self.make_env()
            state = env.reset(seed=29)
            if state["done"] or not state["conflict_edges"]:
                self.skipTest("initial soft PP was already feasible")
            agents = list(state["conflict_edges"][0])
            result = env.step(
                {
                    "mode": "explicit_neighborhood",
                    "agents": agents,
                    "repair_order": list(reversed(agents)),
                    "random_seed": action_seed,
                    "pp_random_seed": 44001,
                }
            )
            return (
                [list(agent["path"]) for agent in result["observation"]["agents"]],
                result["metrics"],
            )

        first_paths, first_metrics = run(12001)
        second_paths, second_metrics = run(12002)
        self.assertEqual(first_paths, second_paths)
        self.assertEqual(first_metrics["requested_pp_random_seed"], 44001)
        self.assertEqual(first_metrics["applied_pp_random_seed"], 44001)
        self.assertEqual(second_metrics["applied_pp_random_seed"], 44001)

    def test_seeded_official_prefix_replays_exactly(self) -> None:
        source = self.make_env()
        state = source.reset(seed=29)
        recorded: list[tuple[dict[str, object], str]] = []
        pp_steps = 0
        for decision_index in range(2):
            if state["done"]:
                break
            result = source.step(
                {
                    "mode": "official",
                    "pp_random_seed": 45000 + decision_index,
                }
            )
            state = result["observation"]
            metrics = result["metrics"]
            action: dict[str, object] = {
                "mode": "replay_neighborhood",
                "agents": list(metrics["neighborhood"]),
                "repair_order": list(metrics["repair_order"]),
            }
            applied_seed = int(metrics["applied_pp_random_seed"])
            if applied_seed >= 0:
                action["pp_random_seed"] = applied_seed
                pp_steps += 1
            recorded.append((action, state_fingerprint(state)))

        if not pp_steps:
            self.skipTest("the short source prefix did not invoke PP")
        replay = self.make_env()
        replay_state = replay.reset(seed=29)
        for action, expected_fingerprint in recorded:
            replay_state = replay.step(action)["observation"]
            self.assertEqual(state_fingerprint(replay_state), expected_fingerprint)

    def test_proposal_batch_parse_failure_is_atomic(self) -> None:
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
        continued = env.step(
            {"mode": "explicit_neighborhood", "agents": edge}
        )
        self.assertTrue(continued["metrics"]["action_valid"])


if __name__ == "__main__":
    unittest.main()
