from __future__ import annotations

import unittest

from experiments import receding_q_variance_audit as module


def _matrix_rows(values: list[list[float]]) -> list[dict]:
    return [
        {
            "candidate_id": f"candidate-{candidate}",
            "trial_index": trial,
            "value": value,
        }
        for candidate, row in enumerate(values)
        for trial, value in enumerate(row)
    ]


def _decompose(values: list[list[float]]) -> dict:
    return module.decompose_balanced_candidate_seed(
        _matrix_rows(values), value=lambda row: float(row["value"])
    )


def _rollout(candidate: str, trial: int) -> dict:
    steps = []
    for step_index in range(1, 4):
        seed = 1_000 + 10 * trial + step_index
        steps.append(
            {
                "step": step_index,
                "requested_pp_seed": seed,
                "applied_pp_seed": seed,
                "before_fingerprint": f"full-{candidate}-{trial}-{step_index}",
                "after_fingerprint": f"full-{candidate}-{trial}-{step_index + 1}",
                "before_repair_fingerprint": (
                    f"repair-{candidate}-{trial}-{step_index}"
                ),
                "after_repair_fingerprint": (
                    f"repair-{candidate}-{trial}-{step_index + 1}"
                ),
            }
        )
    return {
        "state_id": "state-a",
        "candidate_id": candidate,
        "trial_index": trial,
        "steps": steps,
    }


class RecedingQVarianceAuditTests(unittest.TestCase):
    def test_audit_schema_is_v2(self) -> None:
        self.assertEqual(
            module.RECEDING_Q_VARIANCE_AUDIT_SCHEMA,
            "lns2.receding_q_variance_audit.v2",
        )

    def test_decomposition_finds_pure_candidate_effect(self) -> None:
        result = _decompose([[1.0, 1.0], [3.0, 3.0]])
        self.assertAlmostEqual(result["candidate_fraction"], 1.0)
        self.assertAlmostEqual(result["seed_fraction"], 0.0)
        self.assertAlmostEqual(result["interaction_fraction"], 0.0)

    def test_decomposition_finds_pure_seed_effect(self) -> None:
        result = _decompose([[1.0, 3.0], [1.0, 3.0]])
        self.assertAlmostEqual(result["candidate_fraction"], 0.0)
        self.assertAlmostEqual(result["seed_fraction"], 1.0)
        self.assertAlmostEqual(result["interaction_fraction"], 0.0)

    def test_decomposition_finds_candidate_seed_interaction(self) -> None:
        result = _decompose([[0.0, 2.0], [2.0, 0.0]])
        self.assertAlmostEqual(result["candidate_fraction"], 0.0)
        self.assertAlmostEqual(result["seed_fraction"], 0.0)
        self.assertAlmostEqual(result["interaction_fraction"], 1.0)
        self.assertAlmostEqual(result["closure_error"], 0.0)

    def test_decomposition_rejects_incomplete_matrix(self) -> None:
        rows = _matrix_rows([[0.0, 1.0], [2.0, 3.0]])[:-1]
        with self.assertRaisesRegex(ValueError, "balanced"):
            module.decompose_balanced_candidate_seed(
                rows, value=lambda row: float(row["value"])
            )

    def test_root_state_changed_uses_repair_structure_fingerprints(self) -> None:
        row = _rollout("candidate-a", 0)
        root = row["steps"][0]
        root["after_fingerprint"] = "different-full-state"
        root["after_repair_fingerprint"] = root["before_repair_fingerprint"]
        self.assertEqual(module._state_changed(row), 0.0)
        root["after_repair_fingerprint"] = "different-repair-state"
        self.assertEqual(module._state_changed(row), 1.0)

    def test_four_seed_validation_checks_every_continuation_step(self) -> None:
        rows = [
            _rollout(candidate, trial)
            for candidate in ("candidate-a", "candidate-b")
            for trial in range(4)
        ]
        rows[-1]["steps"][1]["requested_pp_seed"] = 999_999
        with self.assertRaisesRegex(ValueError, "paired PP seeds at step 2"):
            module.validate_four_seed_rows(
                rows, target_state_ids=["state-a"]
            )

    def test_four_seed_validation_allows_unapplied_continuation_seed(self) -> None:
        rows = [
            _rollout(candidate, trial)
            for candidate in ("candidate-a", "candidate-b")
            for trial in range(4)
        ]
        rows[-1]["steps"][2]["applied_pp_seed"] = -1
        coverage = module.validate_four_seed_rows(
            rows, target_state_ids=["state-a"]
        )
        self.assertEqual(coverage["paired_step_group_count"], 12)
        self.assertEqual(coverage["continuation_step_group_count"], 8)

    def test_classification_prefers_candidate_conditioned_order_probe(self) -> None:
        rows = []
        for metric, candidate, seed, interaction in (
            ("root_conflict_reduction", 0.25, 0.10, 0.65),
            ("root_state_changed", 0.30, 0.10, 0.60),
            ("h3_normalized_step_auc", 0.20, 0.10, 0.70),
        ):
            rows.append(
                {
                    "metric": metric,
                    "agent_group": "600",
                    "candidate_fraction": candidate,
                    "seed_fraction": seed,
                    "interaction_fraction": interaction,
                    "noncandidate_fraction": seed + interaction,
                }
            )
        decision, diagnostic = module.classify_variance_source(rows)
        self.assertEqual(decision, "probe_candidate_conditioned_pp_order")
        self.assertAlmostEqual(
            diagnostic["six_hundred_root_order_fraction"], 0.75
        )

    def test_classification_distinguishes_continuation_amplification(self) -> None:
        rows = []
        for metric, candidate, seed, interaction in (
            ("root_conflict_reduction", 0.80, 0.05, 0.15),
            ("root_state_changed", 0.85, 0.05, 0.10),
            ("h3_normalized_step_auc", 0.40, 0.10, 0.50),
        ):
            rows.append(
                {
                    "metric": metric,
                    "agent_group": "600",
                    "candidate_fraction": candidate,
                    "seed_fraction": seed,
                    "interaction_fraction": interaction,
                    "noncandidate_fraction": seed + interaction,
                }
            )
        decision, _ = module.classify_variance_source(rows)
        self.assertEqual(
            decision, "fix_h3_continuation_labels_before_pp_order"
        )


if __name__ == "__main__":
    unittest.main()
