from __future__ import annotations

import unittest

from experiments.v3_value_uncertainty import (
    aggregate_arm,
    cluster_bootstrap_mean_interval,
    compare_outcomes,
    leave_one_seed_out,
    select_expected_arm,
    subgroup_summary,
    summarize_policies,
)


def _row(
    state: str,
    arm: str,
    trial: int,
    *,
    feasible: bool = True,
    seconds: float = 1.0,
    final_conflicts: int = 0,
    aliases: tuple[str, ...] = (),
) -> dict:
    return {
        "state_id": state,
        "arm_id": arm,
        "arm_aliases": aliases,
        "trial_index": trial,
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 400,
        "initial_conflicts": 10,
        "final_conflicts": final_conflicts,
        "repair_iterations": 2,
        "feasible": feasible,
        "censored": not feasible,
        "observed_total_seconds": seconds,
        "normalized_conflict_auc_seconds": seconds,
    }


class V3ValueUncertaintySelectionTests(unittest.TestCase):
    def test_feasible_outcome_dominates_faster_censored_outcome(self) -> None:
        feasible = _row("state", "a", 0, feasible=True, seconds=10.0)
        censored = _row(
            "state",
            "b",
            0,
            feasible=False,
            seconds=1.0,
            final_conflicts=1,
        )
        self.assertEqual(compare_outcomes(feasible, censored), -1)

    def test_threshold_keeps_v2_for_small_auc_improvement(self) -> None:
        v2 = {
            **aggregate_arm(
                [
                    _row("state", "v2", trial, seconds=1.0)
                    for trial in range(3)
                ]
            ),
            "mean_normalized_auc_seconds": 1.0,
        }
        candidate = {
            **aggregate_arm(
                [
                    _row("state", "candidate", trial, seconds=0.98)
                    for trial in range(3)
                ]
            ),
            "mean_normalized_auc_seconds": 0.98,
        }
        selected, reason = select_expected_arm(
            [v2, candidate],
            v2_arm_id="v2",
            improvement_threshold=0.05,
        )
        self.assertEqual(selected, "v2")
        self.assertEqual(reason, "auc_threshold_guard")

    def test_higher_feasible_rate_can_override_v2(self) -> None:
        v2 = aggregate_arm(
            [
                _row(
                    "state",
                    "v2",
                    trial,
                    feasible=(trial < 2),
                    final_conflicts=0 if trial < 2 else 5,
                )
                for trial in range(3)
            ]
        )
        candidate = aggregate_arm(
            [_row("state", "candidate", trial) for trial in range(3)]
        )
        selected, reason = select_expected_arm(
            [v2, candidate],
            v2_arm_id="v2",
            improvement_threshold=0.10,
        )
        self.assertEqual(selected, "candidate")
        self.assertEqual(reason, "higher_feasible_rate")


class V3ValueUncertaintyAuditTests(unittest.TestCase):
    def test_leave_one_seed_out_uses_three_training_trials(self) -> None:
        rows = []
        for trial in range(4):
            rows.append(
                _row(
                    "state",
                    "v2",
                    trial,
                    seconds=2.0,
                    aliases=("v2_full",),
                )
            )
            rows.append(
                _row(
                    "state",
                    "candidate",
                    trial,
                    seconds=1.0,
                    aliases=("model_s3",),
                )
            )
        predictions = leave_one_seed_out(
            rows,
            thresholds=(0.05,),
            required_trials=4,
        )
        policy = [
            row
            for row in predictions
            if row["policy_id"] == "loo_expected_t0.05"
        ]
        self.assertEqual(len(policy), 4)
        self.assertTrue(all(row["outcome"] == "win" for row in policy))
        self.assertTrue(all(row["selected_arm_id"] == "candidate" for row in policy))

    def test_policy_summary_reports_paired_wins(self) -> None:
        rows = [
            {
                "policy_id": "policy",
                "state_id": f"state-{index // 2}",
                "map_id": f"map-{index // 2}",
                "deviated_from_v2": True,
                "outcome": "win" if index < 3 else "loss",
                "outcome_score": 1 if index < 3 else -1,
                "selected_feasible": True,
                "v2_feasible": True,
                "selected_total_seconds": 1.0,
                "v2_total_seconds": 2.0,
                "selected_final_conflict_ratio": 0.0,
                "v2_final_conflict_ratio": 0.0,
                "selected_normalized_auc_seconds": 1.0,
                "v2_normalized_auc_seconds": 2.0,
            }
            for index in range(4)
        ]
        summary = summarize_policies(rows)[0]
        self.assertEqual(summary["wins"], 3)
        self.assertEqual(summary["losses"], 1)
        self.assertEqual(summary["feasible_rate_delta"], 0.0)

    def test_subgroup_summary_preserves_map_level_direction(self) -> None:
        rows = [
            {
                "map_id": "good" if index < 2 else "bad",
                "state_id": f"state-{index}",
                "outcome": "win" if index < 2 else "loss",
                "selected_feasible": True,
                "v2_feasible": True,
                "selected_total_seconds": 1.0,
                "v2_total_seconds": 2.0,
            }
            for index in range(4)
        ]
        summaries = subgroup_summary(rows, group_key="map_id")
        lookup = {row["map_id"]: row for row in summaries}
        self.assertEqual(lookup["good"]["net_wins"], 2)
        self.assertEqual(lookup["bad"]["net_wins"], -2)

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        rows = [
            {"state_id": "a", "map_id": "map-a", "score": 1.0},
            {"state_id": "b", "map_id": "map-b", "score": -1.0},
        ]
        first = cluster_bootstrap_mean_interval(
            rows,
            value_key="score",
            cluster_key="map_id",
            samples=100,
            seed=7,
        )
        second = cluster_bootstrap_mean_interval(
            rows,
            value_key="score",
            cluster_key="map_id",
            samples=100,
            seed=7,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
