from __future__ import annotations

import collections
import json
import tempfile
import unittest
from pathlib import Path

from experiments.rescue_confirmation_qualification import (
    QUALIFICATION_SPLIT,
    summarize_recipe_yield,
    validate_qualification_isolation,
)
from experiments.rescue_lite_locked_confirmation import (
    select_locked_task_ids,
    source_state_capacity,
    validate_frozen_qualification,
)
from experiments.rescue_lite_confirmation import AGENT_COUNTS, LAYOUTS


def _task(layout: str, agents: int, recipe: str, map_index: int) -> dict[str, object]:
    map_id = f"{layout}-map-{map_index}"
    return {
        "split": QUALIFICATION_SPLIT,
        "layout_mode": layout,
        "agent_count": agents,
        "task_variant": recipe,
        "map_id": map_id,
        "task_id": f"{map_id}-{recipe}",
    }


class RecipeQualificationTests(unittest.TestCase):
    def test_recipe_requires_distributed_states_and_is_frozen_per_cell(self) -> None:
        tasks = []
        failures = []
        counts: collections.Counter[tuple[str, str]] = collections.Counter()
        for layout in LAYOUTS:
            for agents in AGENT_COUNTS:
                cell = f"{layout}__agents_{agents}"
                for recipe in (f"good_{agents}", f"single_task_{agents}"):
                    for map_index in range(3):
                        task = _task(layout, agents, recipe, map_index)
                        tasks.append(task)
                        counts[(cell, recipe)] += 40
                        repeat = 2 if recipe.startswith("good") else (8 if map_index == 0 else 0)
                        for decision_index in range(repeat):
                            failures.append(
                                {
                                    "task_id": task["task_id"],
                                    "map_id": task["map_id"],
                                    "decision_index": decision_index,
                                }
                            )
        summary, selected, gate = summarize_recipe_yield(
            task_rows=tasks,
            failure_decisions=failures,
            decision_counts=counts,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(value.startswith("good_") for value in selected.values()))
        single = [row for row in summary if str(row["recipe"]).startswith("single")]
        self.assertTrue(all(row["capped_no_progress_state_count"] == 2 for row in single))
        self.assertTrue(all(not row["qualification_passed"] for row in single))

    def test_missing_cell_prevents_locked_confirmation(self) -> None:
        task = _task("compartmentalized", 400, "good_400", 0)
        summary, selected, gate = summarize_recipe_yield(
            task_rows=[task],
            failure_decisions=[],
            decision_counts=collections.Counter(),
        )
        self.assertEqual(len(summary), 1)
        self.assertFalse(gate["passed"])
        self.assertEqual(selected, {})


class QualificationIsolationTests(unittest.TestCase):
    def _dataset(self, root: Path, content: str) -> Path:
        split = root / QUALIFICATION_SPLIT
        (split / "maps").mkdir(parents=True)
        (split / "maps" / "map.map").write_text(content, encoding="utf-8")
        (split / "manifest.jsonl").write_text(
            json.dumps({"map_id": "map", "map_file": "maps/map.map"}) + "\n",
            encoding="utf-8",
        )
        (root / "dataset_summary.json").write_text(
            json.dumps(
                {
                    "master_seed": 1,
                    "splits": {QUALIFICATION_SPLIT: {}},
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_previous_map_content_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = self._dataset(root / "current", "same")
            previous = self._dataset(root / "previous", "same")
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_qualification_isolation(current, [previous])

    def test_disjoint_map_content_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = self._dataset(root / "current", "new")
            previous = self._dataset(root / "previous", "old")
            report = validate_qualification_isolation(current, [previous])
            self.assertTrue(report["passed"])


class LockedProtocolTests(unittest.TestCase):
    def _recipes(self) -> dict[str, str]:
        return {
            f"{layout}__agents_{agents}": f"recipe_{agents}"
            for layout in LAYOUTS
            for agents in AGENT_COUNTS
        }

    def test_passed_qualification_freezes_exactly_six_cells(self) -> None:
        recipes = self._recipes()
        observed = validate_frozen_qualification(
            {
                "decision": "qualification_passed_freeze_recipes",
                "qualification_gate": {"passed": True},
                "frozen_recipe_by_cell": recipes,
            }
        )
        self.assertEqual(observed, recipes)

    def test_locked_tasks_use_only_each_cells_frozen_recipe(self) -> None:
        recipes = self._recipes()
        rows = []
        for layout in LAYOUTS:
            for agents in AGENT_COUNTS:
                for map_index in range(4):
                    rows.append(
                        _task(layout, agents, f"recipe_{agents}", map_index)
                    )
                    rows.append(
                        _task(layout, agents, f"unused_{agents}", map_index)
                    )
        task_ids, counts = select_locked_task_ids(rows, recipes)
        self.assertEqual(len(task_ids), 24)
        self.assertEqual(set(counts.values()), {4})
        self.assertFalse(any("unused" in task_id for task_id in task_ids))

    def test_incomplete_locked_task_cell_is_rejected(self) -> None:
        recipes = self._recipes()
        rows = []
        for layout in LAYOUTS:
            for agents in AGENT_COUNTS:
                limit = 3 if (layout, agents) == ("regular_beltway", 400) else 4
                for map_index in range(limit):
                    rows.append(
                        _task(layout, agents, f"recipe_{agents}", map_index)
                    )
        with self.assertRaisesRegex(ValueError, "coverage"):
            select_locked_task_ids(rows, recipes)

    def test_preregistered_eight_task_capacity_is_enforced(self) -> None:
        recipes = self._recipes()
        rows = []
        for layout in LAYOUTS:
            for agents in AGENT_COUNTS:
                for map_index in range(8):
                    rows.append(
                        _task(layout, agents, f"recipe_{agents}", map_index)
                    )
        task_ids, counts = select_locked_task_ids(
            rows, recipes, expected_tasks_per_cell=8
        )
        self.assertEqual(len(task_ids), 48)
        self.assertEqual(set(counts.values()), {8})

        rows.pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            select_locked_task_ids(
                rows, recipes, expected_tasks_per_cell=8
            )

    def test_source_capacity_applies_two_state_task_cap_before_replay(self) -> None:
        decisions = [
            {
                "cell": "regular_beltway__agents_400",
                "task_id": "task-a",
                "map_id": "map-a",
            }
            for _ in range(20)
        ]
        decisions.extend(
            {
                "cell": "regular_beltway__agents_400",
                "task_id": "task-b",
                "map_id": "map-b",
            }
            for _ in range(10)
        )
        capacity = source_state_capacity(decisions)
        observed = capacity["regular_beltway__agents_400"]
        self.assertEqual(observed["raw_state_count"], 30)
        self.assertEqual(observed["capped_state_capacity"], 4)
        self.assertEqual(observed["task_count"], 2)


if __name__ == "__main__":
    unittest.main()
