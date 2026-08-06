from __future__ import annotations

import copy
import json
import random
import unittest
from collections import Counter
from pathlib import Path

from experiments.stride_robustaction_state_selection import (
    select_episode_first_states,
    select_episode_first_structpool_states,
    validate_robustaction_state_selection_design,
    validate_robustaction_state_selection_v2_design,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_combined_state_selection_design.json"
)
CONFIG_V2 = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_combined_state_selection_design_v2.json"
)


def _row(
    policy: str, episode: int, decision: int, *, conflicts: int | None = None
) -> dict[str, object]:
    return {
        "state_id": f"{policy}-{episode}-{decision}",
        "source_namespace": "source-v4" if episode < 2 else "da2-stability-v2",
        "source_policy": policy,
        "episode_id": f"episode-{episode}",
        "decision_index": decision,
        "before_fingerprint": f"fingerprint-{policy}-{episode}-{decision}",
        "before_conflicts": 10 + decision if conflicts is None else conflicts,
        "agent_count": 150,
        "decision_stage": "early" if decision < 4 else "late",
        "conflict_band": "low_1_10" if decision == 0 else "medium_11_100",
    }


class RobustActionStateSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.config_v2 = json.loads(CONFIG_V2.read_text(encoding="utf-8"))

    def test_registered_design_and_inputs_are_valid(self) -> None:
        validate_robustaction_state_selection_design(
            self.config, project_root=ROOT
        )
        self.assertEqual(
            self.config["selection_contract"]["target_state_count"], 320
        )
        self.assertEqual(
            self.config["selection_contract"]["expected_selected_episode_count"],
            216,
        )

    def test_episode_first_selection_is_order_independent(self) -> None:
        pool = [
            _row(policy, episode, decision)
            for policy in ("official_adaptive", "v2-full")
            for episode in range(3)
            for decision in range(3)
        ]
        first, reports = select_episode_first_states(pool, target_per_policy=4)
        shuffled = list(pool)
        random.Random(20260806).shuffle(shuffled)
        second, _ = select_episode_first_states(shuffled, target_per_policy=4)
        self.assertEqual(
            [row["state_id"] for row in first],
            [row["state_id"] for row in second],
        )
        counts = Counter(
            (
                row["source_namespace"],
                row["source_policy"],
                row["episode_id"],
            )
            for row in first
        )
        self.assertEqual(len(first), 8)
        self.assertEqual(len(counts), 6)
        self.assertEqual(sorted(counts.values()), [1, 1, 1, 1, 2, 2])
        self.assertTrue(
            all(report["selected_episode_count"] == 3 for report in reports.values())
        )

    def test_target_cannot_discard_an_eligible_episode(self) -> None:
        pool = [
            _row(policy, episode, 0)
            for policy in ("official_adaptive", "v2-full")
            for episode in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "discard eligible episodes"):
            select_episode_first_states(pool, target_per_policy=2)

    def test_v2_prefers_one_structpool_eligible_primary_per_episode(self) -> None:
        pool: list[dict[str, object]] = []
        for policy in ("official_adaptive", "v2-full"):
            for episode in range(3):
                pool.append(_row(policy, episode, 0, conflicts=2))
                pool.append(_row(policy, episode, 1, conflicts=20))
                pool.append(_row(policy, episode, 2, conflicts=3))
        selected, reports = select_episode_first_structpool_states(
            pool, target_per_policy=4
        )
        selected_by_episode: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in selected:
            selected_by_episode.setdefault(
                (str(row["source_policy"]), str(row["episode_id"])), []
            ).append(row)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(selected_by_episode), 6)
        self.assertTrue(
            all(
                any(int(row["before_conflicts"]) >= 16 for row in rows)
                for rows in selected_by_episode.values()
            )
        )
        self.assertTrue(
            all(
                report["selected_structpool_eligible_state_count"] >= 3
                for report in reports.values()
            )
        )

    def test_registered_v2_amendment_is_valid(self) -> None:
        validate_robustaction_state_selection_v2_design(
            self.config_v2, project_root=ROOT
        )
        self.assertEqual(
            self.config_v2["observed_prelabel_feasibility"][
                "v1_structpool_eligible_state_count"
            ],
            70,
        )
        self.assertEqual(
            self.config_v2["selection_contract"][
                "minimum_structpool_eligible_states"
            ],
            80,
        )

    def test_outcome_boundary_cannot_be_enabled(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["claim_boundary"]["ttf_read"] = True
        with self.assertRaisesRegex(ValueError, "outcome boundary changed"):
            validate_robustaction_state_selection_design(changed)

        changed = copy.deepcopy(self.config)
        changed["selection_contract"]["target_state_count"] = 415
        with self.assertRaisesRegex(ValueError, "sampling contract changed"):
            validate_robustaction_state_selection_design(changed)


if __name__ == "__main__":
    unittest.main()
