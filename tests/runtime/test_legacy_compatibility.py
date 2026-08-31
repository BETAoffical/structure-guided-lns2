from __future__ import annotations

import hashlib
import json
import unittest

from lns2_selector.compatibility.controller_diagnostics import (
    LegacyControllerDiagnosticError,
    STALL_SHADOW_TRANSITION_SCHEMA,
    validate_legacy_controller_diagnostics,
)
from lns2_selector.compatibility.retired_pool_profiles import (
    guardpool_runtime_augmentation,
    slotpool_runtime_augmentation,
)
from lns2_selector.runtime.online_selection import validate_structpool_augmentation


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

    def test_retired_pool_profiles_are_exact_detached_and_not_executable(
        self,
    ) -> None:
        slotpool = slotpool_runtime_augmentation()
        guardpool = guardpool_runtime_augmentation()

        def canonical_sha256(value: dict) -> str:
            payload = json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        self.assertEqual(
            canonical_sha256(slotpool),
            "2a5759cb02fd8f67b0a1f71b1b9c007c65446980fb326ef6ea7b2f755ef6324b",
        )
        self.assertEqual(
            canonical_sha256(guardpool),
            "b5c9576c719107c01851742f5939a8e2a94599ec09c16504a47af06750869b08",
        )
        slotpool["activation_gate"]["minimum_conflict_pair_count"] = 0
        self.assertEqual(
            slotpool_runtime_augmentation()["activation_gate"][
                "minimum_conflict_pair_count"
            ],
            16,
        )
        for retired in (
            slotpool_runtime_augmentation(),
            guardpool_runtime_augmentation(),
        ):
            with self.assertRaisesRegex(ValueError, "unsupported StructPool"):
                validate_structpool_augmentation(retired)


if __name__ == "__main__":
    unittest.main()
