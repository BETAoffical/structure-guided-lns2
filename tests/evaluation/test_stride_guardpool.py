from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.closed_loop_confirmation import _pool_runtime_modes
from experiments.stride_guardpool import (
    audit_guardpool_threshold,
    validate_guardpool_registration,
)
from lns2_selector.runtime.online_selection import (
    guardpool_runtime_augmentation,
    slotpool_runtime_augmentation,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_guardpool_v1_registration.json"


class GuardPoolTests(unittest.TestCase):
    def test_registration_is_frozen_and_valid(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_guardpool_registration(config, project_root=ROOT)

    def test_registration_rejects_threshold_or_time_guard_drift(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["stall_guard"]["no_progress_limit"] = 5
        with self.assertRaisesRegex(ValueError, "stall guard"):
            validate_guardpool_registration(changed)
        changed = copy.deepcopy(config)
        changed["claim_boundary"]["time_limit_guard"] = True
        with self.assertRaisesRegex(ValueError, "claim boundary"):
            validate_guardpool_registration(changed)

    def test_slotpool_is_not_mislabeled_as_guardpool(self) -> None:
        self.assertEqual(
            _pool_runtime_modes(slotpool_runtime_augmentation()),
            (True, False),
        )
        guardpool = guardpool_runtime_augmentation()
        self.assertEqual(_pool_runtime_modes(guardpool), (True, True))
        self.assertNotIn("tabu_scope", guardpool["stall_guard"])

    def test_registered_historical_threshold_audit_passes(self) -> None:
        output = ROOT / "build" / "stride-guardpool-threshold-audit-v1"
        report = audit_guardpool_threshold(config_path=CONFIG, output=output)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["selected_no_progress_limit"], 8)
        self.assertGreater(report["false_trigger_rates"]["5"], 0.01)
        self.assertLessEqual(report["false_trigger_rates"]["8"], 0.01)


if __name__ == "__main__":
    unittest.main()
