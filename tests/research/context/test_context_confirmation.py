from __future__ import annotations

import collections
import unittest
from pathlib import Path

from research.studies.context.context_confirmation import (
    _categoricalize_sizes,
    _dominates,
    _permuted_context_rows,
)
from experiments.repair_collection import _validate_config
from generators.config import load_json
from generators.dataset import _layout_schedule, _task_schedules


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _row(task: str, size: int, context: float) -> dict:
    features = {
        "action.neighborhood_size": float(size),
        "action.heuristic=target": 1.0,
        "state.collisions": 2.0,
        "context.signal": context,
    }
    return {
        "state_id": f"state-{task}",
        "task_id": task,
        "candidate_key": f"0:target:{size}",
        "candidate_action": {
            "seed_agent": 0,
            "heuristic": "target",
            "neighborhood_size": size,
        },
        "features": {
            "action_seed": {
                key: value
                for key, value in features.items()
                if not key.startswith(("state.", "context."))
            },
            "dynamic": {
                key: value for key, value in features.items() if not key.startswith("context.")
            },
            "full_context": dict(features),
        },
    }


class ContextConfirmationTests(unittest.TestCase):
    def test_neighborhood_size_is_categorical(self) -> None:
        row = _categoricalize_sizes([_row("task-a", 8, 1.0)])[0]
        for profile in ("action_seed", "dynamic", "full_context"):
            self.assertNotIn("action.neighborhood_size", row["features"][profile])
            self.assertEqual(
                row["features"][profile]["action.neighborhood_size=8"], 1.0
            )

    def test_runtime_only_changes_sensitivity_dominance(self) -> None:
        left = {
            "solved": False,
            "solved_rate": 0.0,
            "conflicts_after": 3,
            "conflict_auc": 7,
            "generated": 20,
            "branch_runtime": 2.0,
        }
        right = dict(left, branch_runtime=3.0)
        self.assertFalse(_dominates(left, right, "primary"))
        self.assertTrue(_dominates(left, right, "runtime_sensitivity"))

    def test_context_permutation_keeps_action_and_dynamic_features(self) -> None:
        rows = [_row("task-a", 4, 1.0), _row("task-b", 8, 2.0)]
        permuted = _permuted_context_rows(rows, 0, 0)
        for source, changed in zip(rows, permuted):
            self.assertEqual(source["features"]["action_seed"], changed["features"]["action_seed"])
            self.assertEqual(source["features"]["dynamic"], changed["features"]["dynamic"])
        self.assertEqual(
            sorted(row["features"]["full_context"]["context.signal"] for row in rows),
            sorted(row["features"]["full_context"]["context.signal"] for row in permuted),
        )

    def test_task_pool_is_exactly_balanced(self) -> None:
        config = {
            "tasks_per_map": 2,
            "task_variant_pool": [
                {"name": name, "task": {"scenario_type": name}}
                for name in ("a", "b", "c", "d")
            ],
        }
        schedules = _task_schedules(config, {}, 6, 7)
        counts = collections.Counter(
            item["name"] for schedule in schedules for item in schedule
        )
        self.assertEqual(len(schedules), 6)
        self.assertTrue(all(len(schedule) == 2 for schedule in schedules))
        self.assertEqual(set(counts.values()), {3})

    def test_task_pool_rejects_oversubscription(self) -> None:
        config = {
            "tasks_per_map": 3,
            "task_variant_pool": [
                {"name": "a", "task": {}},
                {"name": "b", "task": {}},
            ],
        }
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            _task_schedules(config, {}, 2, 0)

    def test_confirmation_configs_match_registered_volume(self) -> None:
        dataset = load_json(
            PROJECT_ROOT
            / "research"
            / "configs"
            / "context"
            / "context_confirmation_dataset.json"
        )
        collection = load_json(
            PROJECT_ROOT
            / "research"
            / "configs"
            / "context"
            / "context_confirmation_collection.json"
        )
        _validate_config(collection)
        self.assertEqual(set(dataset["splits"]), {"train", "validation"})
        map_counts = {
            split: len(_layout_schedule(value))
            for split, value in dataset["splits"].items()
        }
        self.assertEqual(map_counts, {"train": 12, "validation": 6})
        task_count = sum(
            value * dataset["tasks_per_map"] for value in map_counts.values()
        )
        instance_seeds = task_count * len(collection["solver_seeds"])
        counterfactual = collection["counterfactual"]
        maximum_outcomes = (
            instance_seeds
            * counterfactual["max_states_per_episode"]
            * counterfactual["max_seed_agents"]
            * len(counterfactual["heuristics"])
            * len(counterfactual["neighborhood_sizes"])
            * counterfactual["trials"]
        )
        self.assertEqual(instance_seeds, 72)
        self.assertEqual(maximum_outcomes, 23328)


if __name__ == "__main__":
    unittest.main()
