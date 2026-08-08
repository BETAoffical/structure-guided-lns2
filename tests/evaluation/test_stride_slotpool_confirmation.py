from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_slotpool_confirmation import (
    validate_slotpool_confirmation_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_slotpool_fresh_confirmation_v1.json"


class SlotPoolConfirmationTests(unittest.TestCase):
    def test_registered_confirmation_is_map_disjoint_and_valid(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_slotpool_confirmation_config(config)
        self.assertFalse(
            set(config["development_map_ids"]) & set(config["confirmation"]["map_ids"])
        )
        self.assertEqual(
            len(config["confirmation"]["task_ids"])
            * len(config["confirmation"]["solver_seeds"]),
            config["confirmation"]["expected_state_count"],
        )

    def test_confirmation_rejects_filtering_or_retraining_drift(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["confirmation"]["result_based_state_filtering"] = True
        with self.assertRaisesRegex(ValueError, "cohort"):
            validate_slotpool_confirmation_config(changed)
        changed = copy.deepcopy(config)
        changed["claim_boundary"]["frozen_model_no_retraining"] = False
        with self.assertRaisesRegex(ValueError, "claim boundary"):
            validate_slotpool_confirmation_config(changed)

    def test_confirmation_rejects_development_map_overlap(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["confirmation"]["map_ids"][0] = changed["development_map_ids"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_slotpool_confirmation_config(changed)

    def test_registered_coverage_has_v2_base_families_for_every_state(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        candidate_path = PROJECT_ROOT / config["inputs"]["coverage_candidates"]["path"]
        base_counts: dict[str, int] = {}
        for line in candidate_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            families = tuple(map(str, row.get("selection_families") or ()))
            if families and all(
                family.startswith(("target:", "collision:", "random:"))
                for family in families
            ):
                state_id = str(row["state_id"])
                base_counts[state_id] = base_counts.get(state_id, 0) + 1
        self.assertEqual(len(base_counts), config["confirmation"]["expected_state_count"])
        self.assertTrue(all(count == 18 for count in base_counts.values()))


if __name__ == "__main__":
    unittest.main()
