from __future__ import annotations

import unittest

from experiments import receding_q_paired_label_audit as module


def _row(candidate: str, trial: int, value: float) -> dict:
    return {
        "candidate_id": candidate,
        "trial_index": trial,
        "feasible": True,
        "final_conflict_ratio": value,
        "normalized_step_auc": value,
        "normalized_wall_auc_seconds": value,
        "observed_total_seconds": value,
    }


def _policy(method: str, risk_lambda: float = 0.0) -> dict:
    return {
        "policy_id": method,
        "method": method,
        "risk_lambda": risk_lambda,
        "complexity": 0,
    }


class RecedingQPairedLabelAuditTests(unittest.TestCase):
    def test_audit_schema_is_v2(self) -> None:
        self.assertEqual(
            module.RECEDING_Q_PAIRED_LABEL_AUDIT_SCHEMA,
            "lns2.receding_q_paired_label_audit.v2",
        )

    def test_average_rank_uses_each_seed_as_a_paired_block(self) -> None:
        rows = [
            _row("a", 0, 0.0),
            _row("a", 1, 0.0),
            _row("a", 2, 100.0),
            _row("b", 0, 10.0),
            _row("b", 1, 10.0),
            _row("b", 2, 11.0),
        ]
        self.assertEqual(
            module.select_paired_label_candidate(
                rows, policy=_policy("seed_average_rank")
            ),
            "a",
        )

    def test_average_rank_assigns_midranks_to_exact_seed_ties(self) -> None:
        rows = [
            _row("a", 0, 1.0),
            _row("a", 1, 1.0),
            _row("a", 2, 1.0),
            _row("b", 0, 1.0),
            _row("b", 1, 1.0),
            _row("b", 2, 0.0),
        ]
        self.assertEqual(
            module.select_paired_label_candidate(
                rows, policy=_policy("seed_average_rank")
            ),
            "b",
        )

    def test_paired_selection_rejects_unbalanced_trials(self) -> None:
        rows = [
            _row("a", 0, 0.0),
            _row("a", 1, 0.0),
            _row("b", 0, 1.0),
        ]
        with self.assertRaisesRegex(ValueError, "unbalanced"):
            module.select_paired_label_candidate(
                rows, policy=_policy("seed_average_rank")
            )

    def test_copeland_prefers_candidate_with_majority_seed_wins(self) -> None:
        rows = [
            _row("a", 0, 0.0),
            _row("a", 1, 0.0),
            _row("a", 2, 10.0),
            _row("b", 0, 1.0),
            _row("b", 1, 1.0),
            _row("b", 2, 0.0),
        ]
        self.assertEqual(
            module.select_paired_label_candidate(
                rows, policy=_policy("seed_pairwise_copeland")
            ),
            "a",
        )

    def test_paired_centering_removes_common_seed_shift(self) -> None:
        rows = [
            _row("a", 0, 0.0),
            _row("a", 1, 100.0),
            _row("a", 2, 200.0),
            _row("b", 0, 1.0),
            _row("b", 1, 101.0),
            _row("b", 2, 201.0),
        ]
        self.assertEqual(
            module.select_paired_label_candidate(
                rows,
                policy=_policy("paired_quality_se", risk_lambda=2.0),
            ),
            "a",
        )

    def test_gate_uses_only_map_group_oof_results(self) -> None:
        loo_rows = [
            {"map_id": f"map-{index}"}
            for index in range(4)
        ]
        oof = {
            "feasible_rate_delta": 0.0,
            "mean_normalized_step_auc_delta": 0.0,
            "mean_total_seconds_delta": 0.0,
            "net_wins": 0,
        }
        high_load = {
            "feasible_rate_delta": 0.0,
            "mean_normalized_step_auc_delta": 0.0,
            "mean_total_seconds_delta": 0.0,
            "net_wins": 0,
        }
        checks = module._paired_label_checks(
            loo_rows=loo_rows,
            target_state_ids=["state-a"],
            policies=[_policy("mean")],
            oof_summary=oof,
            high_load=high_load,
        )
        self.assertNotIn(
            "some_fixed_method_passes_all_six_hundred_gates", checks
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(
            module.paired_label_audit_decision(checks),
            "paired_h3_label_transform_promising",
        )

    def test_gate_rejects_six_hundred_agent_feasibility_regression(self) -> None:
        checks = module._paired_label_checks(
            loo_rows=[{"map_id": f"map-{index}"} for index in range(4)],
            target_state_ids=["state-a"],
            policies=[_policy("mean")],
            oof_summary={
                "feasible_rate_delta": 0.0,
                "mean_normalized_step_auc_delta": 0.0,
                "mean_total_seconds_delta": 0.0,
                "net_wins": 0,
            },
            high_load={
                "feasible_rate_delta": -0.1,
                "mean_normalized_step_auc_delta": 0.0,
                "mean_total_seconds_delta": 0.0,
                "net_wins": 0,
            },
        )
        self.assertFalse(
            checks["six_hundred_oof_feasible_rate_not_below_v2"]
        )
        self.assertEqual(
            module.paired_label_audit_decision(checks),
            "paired_h3_label_transform_not_supported_by_current_audit",
        )

    def test_failed_gate_has_neutral_noncausal_decision(self) -> None:
        decision = module.paired_label_audit_decision(
            {"six_hundred_oof_auc_not_below_v2": False}
        )
        self.assertEqual(
            decision,
            "paired_h3_label_transform_not_supported_by_current_audit",
        )
        self.assertNotIn("continuation", decision)
        self.assertNotIn("recollect", decision)


if __name__ == "__main__":
    unittest.main()
