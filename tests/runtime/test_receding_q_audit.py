from __future__ import annotations

import unittest

from experiments.receding_q_audit import (
    build_actual_candidate_rows,
    fixed_horizon_metrics,
    seed_stability,
)


def _trial(
    sequence: str,
    trial: int,
    *,
    candidate: str = "candidate",
    agents: tuple[int, ...] = (1, 2, 3, 4),
    before: int = 10,
    after: int = 5,
) -> dict:
    return {
        "state_id": "state",
        "sequence_id": sequence,
        "trial_index": trial,
        "split": "policy_train",
        "conflict_trajectory": [before, after],
        "steps": [
            {
                "step": 1,
                "executed": True,
                "candidate_id": candidate,
                "agents": list(agents),
                "conflicts_before": before,
                "conflicts_after": after,
                "repair_outcome": "conflict_reduced",
                "total_seconds": 2.0,
            }
        ],
    }


def _feature(sequence: str, second_size: int) -> dict:
    return {
        "state_id": "state",
        "sequence_id": sequence,
        "split": "policy_train",
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 100,
        "source_stratum": "ordinary_progress",
        "feature_names": [
            "proposal.actual_size",
            "state.colliding_pairs",
            "sequence.step1.requested_size",
            "sequence.step2.requested_size",
        ],
        "feature_values": [4.0, 10.0, 4.0, float(second_size)],
        "templates": [
            {"template_key": "collision:size4:rep0"},
            {"template_key": f"target:size{second_size}:rep0"},
            {"template_key": "random:size8:rep0"},
        ],
    }


class RecedingQMetricTests(unittest.TestCase):
    def test_fixed_horizon_pads_feasible_trajectory_with_zero(self) -> None:
        trial = _trial("sequence", 0, before=10, after=0)
        trial["steps"][0]["repair_outcome"] = "feasible"
        metrics = fixed_horizon_metrics(trial)
        self.assertEqual(metrics["final_conflict_ratio"], 0.0)
        self.assertAlmostEqual(metrics["normalized_step_auc"], 1.0 / 6.0)
        self.assertEqual(metrics["feasible"], 1.0)

    def test_hard_failure_stops_and_pads_unchanged_state(self) -> None:
        trial = _trial("sequence", 0, before=10, after=10)
        trial["steps"][0]["repair_outcome"] = "hard_failure"
        metrics = fixed_horizon_metrics(trial)
        self.assertEqual(metrics["final_conflict_ratio"], 1.0)
        self.assertEqual(metrics["normalized_step_auc"], 1.0)
        self.assertEqual(metrics["no_progress"], 1.0)


class RecedingQAggregationTests(unittest.TestCase):
    def test_future_template_features_are_removed_before_grouping(self) -> None:
        features = [_feature("a", 8), _feature("b", 16)]
        trials = [
            _trial(sequence, trial)
            for sequence in ("a", "b")
            for trial in range(2)
        ]
        baselines = [
            {
                "controller": "v2-full",
                "split": "policy_train",
                "state_id": "state",
                "trial_index": trial,
                "steps": [
                    {
                        "step": 1,
                        "candidate_id": "candidate",
                        "executed": True,
                    }
                ],
            }
            for trial in range(2)
        ]
        rows = build_actual_candidate_rows(
            features,
            trials,
            baselines,
            candidate_feature_names=(
                "proposal.actual_size",
                "state.colliding_pairs",
            ),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["feature_names"],
            ("proposal.actual_size", "state.colliding_pairs"),
        )
        self.assertEqual(rows[0]["observation_count"], 4)
        self.assertTrue(rows[0]["v2_selected"])

    def test_actual_feature_change_across_future_tails_is_rejected(self) -> None:
        features = [_feature("a", 8), _feature("b", 16)]
        features[1]["feature_values"][0] = 8.0
        trials = [
            _trial(sequence, trial)
            for sequence in ("a", "b")
            for trial in range(2)
        ]
        with self.assertRaisesRegex(ValueError, "depend on future templates"):
            build_actual_candidate_rows(
                features,
                trials,
                [],
                candidate_feature_names=(
                    "proposal.actual_size",
                    "state.colliding_pairs",
                ),
            )

    def test_seed_stability_reports_exact_winner_agreement(self) -> None:
        rows = [
            {
                "state_id": "state",
                "candidate_id": "a",
                "observations": [
                    {
                        "trial_index": seed,
                        "feasible": 1.0,
                        "final_conflict_ratio": 0.0,
                        "normalized_step_auc": 0.1,
                        "no_progress": 0.0,
                        "log_total_seconds": 0.2,
                    }
                    for seed in (0, 1)
                ],
            },
            {
                "state_id": "state",
                "candidate_id": "b",
                "observations": [
                    {
                        "trial_index": seed,
                        "feasible": 0.0,
                        "final_conflict_ratio": 0.5,
                        "normalized_step_auc": 0.7,
                        "no_progress": 0.0,
                        "log_total_seconds": 0.1,
                    }
                    for seed in (0, 1)
                ],
            },
        ]
        stability = seed_stability(rows)
        self.assertEqual(stability["state_count"], 1)
        self.assertEqual(stability["exact_oracle_winner_agreement_rate"], 1.0)
        self.assertAlmostEqual(stability["mean_candidate_rank_correlation"], 1.0)


if __name__ == "__main__":
    unittest.main()
