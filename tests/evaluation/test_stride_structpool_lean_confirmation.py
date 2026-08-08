from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.repair_collection import _read_jsonl
from experiments.stride_structpool_lean_confirmation import (
    CONTROLLERS,
    load_structpool_lean_confirmation_config,
    run_structpool_lean_confirmation,
    structpool_lean_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_lean_confirmation.json"


class StructPoolLeanConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _path, cls.root, cls.config = load_structpool_lean_confirmation_config(CONFIG)

    def test_schedule_is_complete_paired_and_strictly_rotated(self) -> None:
        schedule = structpool_lean_schedule(self.config)
        self.assertEqual(len(schedule), 120)
        by_key: dict[tuple[str, str, int], list[dict[str, object]]] = {}
        for row in schedule:
            key = (
                str(row["group_id"]),
                str(row["task_id"]),
                int(row["solver_seed"]),
            )
            by_key.setdefault(key, []).append(row)
        self.assertEqual(len(by_key), 30)
        for index, rows in enumerate(by_key.values()):
            ordered = sorted(rows, key=lambda row: int(row["within_key_position"]))
            offset = index % len(CONTROLLERS)
            expected = CONTROLLERS[offset:] + CONTROLLERS[:offset]
            self.assertEqual(tuple(str(row["controller"]) for row in ordered), expected)

    def test_cohort_is_exact_label_map_disjoint_subset(self) -> None:
        label_maps = {
            str(row["map_id"])
            for row in _read_jsonl(
                ROOT / "build/stride-robustaction-label-collection-v1/state_manifest.jsonl"
            )
        }
        groups = list(self.config["cohort"]["groups"])
        self.assertEqual(
            [str(row["id"]) for row in groups],
            ["den300", "maze100", "random500", "room400", "warehouse600"],
        )
        self.assertTrue(all(str(row["map_id"]) not in label_maps for row in groups))
        self.assertIn("orz200d", label_maps)

    def test_dry_run_registers_one_hundred_twenty_entries(self) -> None:
        report = run_structpool_lean_confirmation(
            CONFIG,
            ROOT / "build/unused-lean-confirmation-dry-run",
            dry_run=True,
        )
        self.assertEqual(report["schedule_entry_count"], 120)
        self.assertEqual(len(str(report["schedule_sha256"])), 64)

    def test_performance_and_claim_boundaries_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["performance_gates"][
            "minimum_mean_raw_ttf_improvement_vs_full_structpool"
        ] = 0.0
        temporary = ROOT / "build/invalid-lean-confirmation-config.json"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(changed), encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "performance gates"):
                load_structpool_lean_confirmation_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
