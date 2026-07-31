from __future__ import annotations

import unittest

from scripts.compare_corrected_native_semantics import compare


def _report(
    *,
    initial: str,
    proposal: list[int],
    repair: str,
    final: str,
) -> dict[str, object]:
    return {
        "native": {"sha256": initial},
        "input": {
            "map_sha256": "a" * 64,
            "scenario_sha256": "b" * 64,
            "agent_count": 2,
            "repair_steps": 1,
            "seeds": [7],
        },
        "episodes": [
            {
                "seed": 7,
                "initial": {
                    "conflicts": 3,
                    "state_fingerprint": initial,
                },
                "proposals": [
                    {
                        "seed_agent": 1,
                        "heuristic": "target",
                        "requested_size": 4,
                        "random_seed": 99,
                        "neighborhood": proposal,
                    }
                ],
                "repair_fingerprint": repair,
                "repair_trajectory": [3, 0],
                "final": {"state_fingerprint": final},
            }
        ],
    }


class CorrectedNativeSemanticsTests(unittest.TestCase):
    def test_comparison_counts_changed_semantics(self) -> None:
        report = compare(
            _report(initial="same", proposal=[1, 2], repair="old", final="old"),
            _report(initial="same", proposal=[1, 3], repair="new", final="new"),
        )

        self.assertEqual(report["initial_state_changed_count"], 0)
        self.assertEqual(report["changed_proposal_count"], 1)
        self.assertEqual(report["changed_proposal_count_by_family"], {"target": 1})
        self.assertEqual(report["repair_sequence_changed_count"], 1)
        self.assertEqual(report["final_state_changed_count"], 1)

    def test_comparison_rejects_different_seed_coverage(self) -> None:
        baseline = _report(
            initial="same", proposal=[1, 2], repair="same", final="same"
        )
        corrected = _report(
            initial="same", proposal=[1, 2], repair="same", final="same"
        )
        corrected["episodes"][0]["seed"] = 8  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "different seeds"):
            compare(baseline, corrected)

    def test_comparison_treats_a_changed_seed_agent_as_a_semantic_change(
        self,
    ) -> None:
        baseline = _report(
            initial="same", proposal=[1, 2], repair="same", final="same"
        )
        corrected = _report(
            initial="same", proposal=[1, 2], repair="same", final="same"
        )
        corrected["episodes"][0]["proposals"][0]["seed_agent"] = 2  # type: ignore[index]

        report = compare(baseline, corrected)

        self.assertEqual(report["changed_proposal_count"], 1)
        changed = report["episodes"][0]["changed_proposals"][0]
        self.assertEqual(changed["baseline_seed_agent"], 1)
        self.assertEqual(changed["corrected_seed_agent"], 2)

    def test_comparison_rejects_different_inputs(self) -> None:
        baseline = _report(
            initial="same", proposal=[1, 2], repair="same", final="same"
        )
        corrected = _report(
            initial="same", proposal=[1, 2], repair="same", final="same"
        )
        corrected["input"]["agent_count"] = 3  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "different agent_count"):
            compare(baseline, corrected)


if __name__ == "__main__":
    unittest.main()
