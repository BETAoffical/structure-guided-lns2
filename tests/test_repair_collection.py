from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.repair_collection import (
    _horizon_outcomes,
    _prepare_run,
    candidate_actions,
    select_seed_agents,
    state_fingerprint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sample_state() -> dict:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": False,
        "done": False,
        "iteration": 3,
        "rows": 2,
        "cols": 3,
        "sum_of_costs": 12,
        "num_of_colliding_pairs": 3,
        "runtime": 0.5,
        "context": {"layout_mode": "ignored"},
        "low_level": {
            "expanded": 10,
            "generated": 20,
            "reopened": 1,
            "runs": 4,
        },
        "obstacles": [0, 0, 1, 0, 0, 0],
        "conflict_edges": [[0, 1], [0, 2], [2, 3]],
        "agents": [
            {
                "id": 0,
                "start": 0,
                "goal": 5,
                "path_cost": 4,
                "shortest_path_cost": 3,
                "delay": 1,
                "conflict_degree": 2,
                "path": [0, 1, 4, 5],
            },
            {
                "id": 1,
                "start": 1,
                "goal": 4,
                "path_cost": 3,
                "shortest_path_cost": 2,
                "delay": 1,
                "conflict_degree": 1,
                "path": [1, 0, 3, 4],
            },
            {
                "id": 2,
                "start": 3,
                "goal": 2,
                "path_cost": 5,
                "shortest_path_cost": 2,
                "delay": 3,
                "conflict_degree": 2,
                "path": [3, 4, 3, 0, 1, 2],
            },
            {
                "id": 3,
                "start": 5,
                "goal": 0,
                "path_cost": 4,
                "shortest_path_cost": 3,
                "delay": 1,
                "conflict_degree": 1,
                "path": [5, 4, 3, 0],
            },
        ],
    }


class RepairCollectionTests(unittest.TestCase):
    def test_transfer_config_has_exact_splits_and_volume(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs" / "repair_transfer_pilot.json").read_text(
                encoding="utf-8"
            )
        )
        split_counts = {
            name: sum(split["layout_counts"].values())
            * int(split.get("tasks_per_map", config["tasks_per_map"]))
            for name, split in config["splits"].items()
        }
        self.assertEqual(
            split_counts,
            {
                "train": 24,
                "validation": 12,
                "test_id": 12,
                "test_ood_layout": 24,
                "test_ood_task": 12,
                "test_ood_density": 6,
                "test_joint_ood": 12,
            },
        )
        self.assertEqual(sum(split_counts.values()), 102)
        seen = set(config["splits"]["train"]["layout_counts"])
        unseen = set(config["splits"]["test_ood_layout"]["layout_counts"])
        self.assertTrue(seen.isdisjoint(unseen))

    def test_state_fingerprint_excludes_only_runtime_and_context(self) -> None:
        first = sample_state()
        second = sample_state()
        second["runtime"] = 999.0
        second["context"] = {"layout_mode": "different"}
        self.assertEqual(state_fingerprint(first), state_fingerprint(second))
        second["agents"][0]["path"][-1] = 4
        self.assertNotEqual(state_fingerprint(first), state_fingerprint(second))

    def test_candidate_selection_is_deterministic_and_stratified(self) -> None:
        state = sample_state()
        first = select_seed_agents(state, 3)
        second = select_seed_agents(state, 3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertIn(0, first)
        self.assertIn(2, first)
        actions = candidate_actions(
            state,
            maximum_seeds=3,
            heuristics=["target", "collision", "random"],
            neighborhood_sizes=[4, 8],
        )
        self.assertEqual(len(actions), 18)
        self.assertEqual(
            {action["heuristic"] for action in actions},
            {"target", "collision", "random"},
        )

    def test_run_config_rejects_a_different_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "output"
            (dataset / "train").mkdir(parents=True)
            (dataset / "dataset_summary.json").write_text(
                json.dumps({"splits": {"train": {}}}), encoding="utf-8"
            )
            for name, value in (
                ("map.map", "type octile\nheight 1\nwidth 1\nmap\n.\n"),
                ("task.scen", "version 1\n"),
                ("map.json", "{}\n"),
                ("task.json", "{}\n"),
            ):
                (dataset / "train" / name).write_text(value, encoding="utf-8")
            (dataset / "train" / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "map_file": "map.map",
                        "scenario_file": "task.scen",
                        "map_metadata_file": "map.json",
                        "task_file": "task.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _prepare_run(
                dataset,
                output,
                {"schema_version": 1, "value": "first"},
                ["train"],
                resume=False,
            )
            with self.assertRaisesRegex(ValueError, "different dataset or"):
                _prepare_run(
                    dataset,
                    output,
                    {"schema_version": 1, "value": "second"},
                    ["train"],
                    resume=True,
                )

            second_output = root / "second-output"
            stable_config = {"schema_version": 1, "value": "stable"}
            _prepare_run(
                dataset,
                second_output,
                stable_config,
                ["train"],
                resume=False,
            )
            (dataset / "train" / "map.map").write_text(
                "type octile\nheight 1\nwidth 1\nmap\n@\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "different dataset or"):
                _prepare_run(
                    dataset,
                    second_output,
                    stable_config,
                    ["train"],
                    resume=True,
                )

    def test_early_solution_padding_does_not_repeat_runtime(self) -> None:
        initial = sample_state()
        solved = sample_state()
        solved["feasible"] = True
        solved["done"] = True
        solved["num_of_colliding_pairs"] = 0
        points = [
            {"step": 0, "state": initial, "step_runtime": 0.0},
            {"step": 1, "state": solved, "step_runtime": 0.25},
        ]
        outcome = _horizon_outcomes(initial, points, [4])[0]
        self.assertTrue(outcome["available"])
        self.assertTrue(outcome["solved"])
        self.assertEqual(outcome["conflict_auc"], 1.5)
        self.assertEqual(outcome["branch_runtime"], 0.25)
        self.assertEqual(outcome["time_to_feasible"], 0.25)


if __name__ == "__main__":
    unittest.main()
