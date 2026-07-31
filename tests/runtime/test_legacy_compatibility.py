from __future__ import annotations

import unittest

from lns2_selector.compatibility.controller_diagnostics import (
    LegacyControllerDiagnosticError,
    STALL_SHADOW_TRANSITION_SCHEMA,
    validate_legacy_controller_diagnostics,
)


class LegacyControllerCompatibilityTests(unittest.TestCase):
    def test_stall_shadow_diagnostics_remain_readable_without_runtime_code(self) -> None:
        controller = {
            "controller_mode": "v2-stall-shadow",
            "selected_candidate_id": "candidate-1",
            "stall_shadow": {
                "schema": STALL_SHADOW_TRANSITION_SCHEMA,
                "route": "model",
                "base_selection_preserved": True,
                "action_preserved": True,
                "effective_selected_candidate_id": "candidate-1",
                "state_unchanged": False,
                "repair_outcome": "conflict_reduced",
            },
        }
        metrics = {
            "replan_success": True,
            "conflicts_before": 4,
            "conflicts_after": 3,
        }

        self.assertTrue(
            validate_legacy_controller_diagnostics(
                controller, metrics, {"feasible": False}, "model"
            )
        )

    def test_legacy_diagnostics_fail_closed_on_schema_tampering(self) -> None:
        controller = {
            "controller_mode": "v2-stall-shadow",
            "selected_candidate_id": "candidate-1",
            "stall_shadow": {
                "schema": "tampered",
                "route": "model",
                "base_selection_preserved": True,
                "action_preserved": True,
                "effective_selected_candidate_id": "candidate-1",
                "state_unchanged": True,
                "repair_outcome": "accepted_noop",
            },
        }
        with self.assertRaisesRegex(
            LegacyControllerDiagnosticError, "schema mismatch"
        ):
            validate_legacy_controller_diagnostics(
                controller,
                {
                    "replan_success": True,
                    "conflicts_before": 4,
                    "conflicts_after": 4,
                },
                {"feasible": False},
                "model",
            )

    def test_active_controller_diagnostics_are_not_legacy(self) -> None:
        self.assertFalse(
            validate_legacy_controller_diagnostics(
                {"controller_mode": "v3-s3"}, {}, {}, "model"
            )
        )


if __name__ == "__main__":
    unittest.main()
