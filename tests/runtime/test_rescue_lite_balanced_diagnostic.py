from __future__ import annotations

import unittest

from experiments.rescue_lite_balanced_diagnostic import (
    diagnostic_decision,
    select_balanced_diagnostic_states,
)
from experiments.rescue_lite_confirmation import AGENT_COUNTS, LAYOUTS


def _prepared(layout: str, agents: int, index: int) -> dict[str, object]:
    task_index = index // 2
    return {
        "valid": True,
        "state": {
            "state_id": f"{layout}-{agents}-{index}",
            "cell": f"{layout}__agents_{agents}",
            "layout_mode": layout,
            "agent_count": agents,
            "map_id": f"{layout}-map-{task_index}",
            "task_id": f"{layout}-{agents}-task-{task_index}",
            "decision_index": index,
        },
    }


class BalancedDiagnosticSelectionTests(unittest.TestCase):
    def test_selects_four_states_per_cell_with_two_state_task_cap(self) -> None:
        rows = [
            _prepared(layout, agents, index)
            for layout in LAYOUTS
            for agents in AGENT_COUNTS
            for index in range(6)
        ]
        selected, counts = select_balanced_diagnostic_states(rows)
        self.assertEqual(len(selected), 24)
        self.assertEqual(set(counts.values()), {4})
        task_counts: dict[str, int] = {}
        for row in selected:
            task = str(row["state"]["task_id"])
            task_counts[task] = task_counts.get(task, 0) + 1
        self.assertLessEqual(max(task_counts.values()), 2)

    def test_missing_cell_is_rejected_instead_of_lowering_quota(self) -> None:
        rows = [
            _prepared(layout, agents, index)
            for layout in LAYOUTS
            for agents in AGENT_COUNTS
            for index in range(
                3 if (layout, agents) == ("regular_beltway", 400) else 4
            )
        ]
        with self.assertRaisesRegex(ValueError, "cannot fill quotas"):
            select_balanced_diagnostic_states(rows)


class DiagnosticDecisionTests(unittest.TestCase):
    def test_raw_confirmation_support_is_never_labeled_as_promotion(self) -> None:
        decision, recommendation = diagnostic_decision("rescue_lite_confirmed")
        self.assertEqual(decision, "diagnostic_supports_fixed_rescue")
        self.assertIn("without_promotion", recommendation)

    def test_failed_fixed_and_learned_gate_points_to_v3(self) -> None:
        decision, recommendation = diagnostic_decision("proceed_to_v3")
        self.assertEqual(decision, "diagnostic_supports_v3")
        self.assertEqual(recommendation, "proceed_to_v3_design")


if __name__ == "__main__":
    unittest.main()
