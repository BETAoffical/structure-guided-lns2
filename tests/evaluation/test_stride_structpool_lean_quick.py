from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_structpool_lean_quick import (
    CONTROLLERS,
    _controller_kwargs,
    load_structpool_lean_quick_config,
    run_structpool_lean_quick,
    structpool_lean_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_lean_quick.json"


class StructPoolLeanQuickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _path, cls.root, cls.config = load_structpool_lean_quick_config(CONFIG)

    def test_schedule_is_complete_paired_and_strictly_rotated(self) -> None:
        schedule = structpool_lean_schedule(self.config)
        self.assertEqual(len(schedule), 32)
        by_key: dict[tuple[str, str, int], list[dict[str, object]]] = {}
        for row in schedule:
            key = (
                str(row["group_id"]),
                str(row["task_id"]),
                int(row["solver_seed"]),
            )
            by_key.setdefault(key, []).append(row)
        self.assertEqual(len(by_key), 8)
        for index, rows in enumerate(by_key.values()):
            ordered = sorted(rows, key=lambda row: int(row["within_key_position"]))
            offset = index % len(CONTROLLERS)
            expected = CONTROLLERS[offset:] + CONTROLLERS[:offset]
            self.assertEqual(tuple(str(row["controller"]) for row in ordered), expected)

    def test_only_lean_controller_receives_registered_filter(self) -> None:
        full = _controller_kwargs(self.root, self.config, "v2-plus-structpool")
        lean = _controller_kwargs(
            self.root, self.config, "v2-plus-structpool-lean"
        )
        self.assertNotIn("lean_filter", full["structpool_augmentation"])
        self.assertEqual(
            lean["structpool_augmentation"]["lean_filter"]["filter_id"],
            "stride-structpool-pure-bottleneck-filter-v1",
        )
        for field in (
            "neighborhood_sizes",
            "maximum_added_candidates",
            "maximum_jaccard_similarity",
            "maximum_total_candidates",
            "activation_gate",
        ):
            self.assertEqual(
                full["structpool_augmentation"][field],
                lean["structpool_augmentation"][field],
            )

    def test_dry_run_registers_thirty_two_entries(self) -> None:
        report = run_structpool_lean_quick(
            CONFIG, ROOT / "build" / "unused-lean-quick-dry-run", dry_run=True
        )
        self.assertEqual(report["schedule_entry_count"], 32)
        self.assertEqual(len(str(report["schedule_sha256"])), 64)

    def test_performance_and_claim_boundaries_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["performance_gates"][
            "minimum_paired_faster_fraction_vs_full_structpool"
        ] = 0.25
        temporary = ROOT / "build" / "invalid-lean-quick-config.json"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(changed), encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "performance gates"):
                load_structpool_lean_quick_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
