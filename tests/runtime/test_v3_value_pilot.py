from __future__ import annotations

import unittest

from experiments.v3_value_pilot import (
    _extend_replay_repair_budget,
    _winner_key,
    analyze_value_rollouts,
    build_state_arms,
)


def _sequence_trial(
    sequence_id: str,
    template: str,
    candidate_id: str,
    agents: list[int],
) -> dict:
    family, size, representative = template.split(":")
    return {
        "sequence_id": sequence_id,
        "templates": [
            {
                "family": family,
                "requested_size": int(size.removeprefix("size")),
                "representative": int(representative.removeprefix("rep")),
                "template_key": template,
            }
        ],
        "steps": [
            {
                "step": 1,
                "executed": True,
                "candidate_id": candidate_id,
                "agents": agents,
                "selection_seconds": 0.1,
            }
        ],
    }


def _rollout(
    state: str,
    arm: str,
    trial: int,
    *,
    feasible: bool,
    seconds: float,
    final_conflicts: int,
) -> dict:
    return {
        "state_id": state,
        "arm_id": arm,
        "trial_index": trial,
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 400,
        "initial_conflicts": 10,
        "final_conflicts": final_conflicts,
        "feasible": feasible,
        "censored": not feasible,
        "observed_total_seconds": seconds,
        "continuation_iterations": 1,
        "normalized_conflict_auc_seconds": seconds,
    }


class V3ValueArmTests(unittest.TestCase):
    def test_rollout_budget_extends_beyond_s3_source_horizon(self) -> None:
        replay = {"environment": {"max_repair_iterations": 12}}
        prepared = _extend_replay_repair_budget(
            replay,
            prefix_length=9,
            max_repairs=8,
        )
        self.assertEqual(
            prepared["environment"]["max_repair_iterations"],
            17,
        )
        self.assertEqual(replay["environment"]["max_repair_iterations"], 12)

    def test_duplicate_agent_sets_are_deduplicated_with_aliases(self) -> None:
        payload = {
            "trials": [
                _sequence_trial(
                    "model",
                    "collision:size4:rep0",
                    "same",
                    [1, 2, 3, 4],
                ),
                _sequence_trial(
                    "oracle",
                    "collision:size4:rep0",
                    "same",
                    [1, 2, 3, 4],
                ),
                _sequence_trial(
                    "quality",
                    "target:size8:rep0",
                    "quality",
                    list(range(8)),
                ),
            ],
            "external_baselines": [
                {
                    "controller": "v2-full",
                    "steps": [
                        {
                            "step": 1,
                            "candidate_id": "same",
                            "action": {"agents": [1, 2, 3, 4]},
                        }
                    ],
                },
                {
                    "controller": "v2-full",
                    "steps": [
                        {
                            "step": 1,
                            "candidate_id": "same",
                            "action": {"agents": [1, 2, 3, 4]},
                        }
                    ],
                },
            ],
        }
        oracle = {
            "model_sequence_id": "model",
            "oracle_s3_efficiency_sequence_id": "oracle",
            "oracle_s3_quality_time_sequence_id": "quality",
            "oracle_h1_efficiency_first_template": "target:size8:rep0",
        }
        arms = build_state_arms(payload, oracle)
        self.assertEqual(len(arms), 2)
        self.assertEqual(
            arms[0]["aliases"],
            ["v2_full", "model_s3", "oracle_s3_efficiency"],
        )
        self.assertEqual(
            arms[1]["aliases"],
            ["oracle_s3_quality_time", "oracle_h1_efficiency"],
        )

    def test_winner_prefers_feasible_before_censored(self) -> None:
        feasible = _rollout(
            "state",
            "feasible",
            0,
            feasible=True,
            seconds=10.0,
            final_conflicts=0,
        )
        censored = _rollout(
            "state",
            "censored",
            0,
            feasible=False,
            seconds=1.0,
            final_conflicts=1,
        )
        self.assertLess(_winner_key(feasible, 0.1), _winner_key(censored, 0.1))


class V3ValueAnalysisTests(unittest.TestCase):
    def test_smoke_analysis_records_signal_without_promotion(self) -> None:
        rows = [
            _rollout(
                "state",
                arm,
                trial,
                feasible=(arm == "good"),
                seconds=1.0 if arm == "good" else 2.0,
                final_conflicts=0 if arm == "good" else 5,
            )
            for arm in ("good", "bad")
            for trial in (0, 1)
        ]
        report, states, sensitivity = analyze_value_rollouts(
            rows,
            expected_jobs=4,
            smoke_only=True,
        )
        self.assertEqual(report["decision"], "smoke_completed_not_scientific")
        self.assertEqual(report["action_sensitive_state_fraction"], 1.0)
        self.assertEqual(report["uncensored_branch_fraction"], 0.5)
        self.assertEqual(states[0]["arm_count"], 2)
        self.assertTrue(sensitivity)

    def test_analysis_rejects_duplicate_keys(self) -> None:
        row = _rollout(
            "state",
            "arm",
            0,
            feasible=True,
            seconds=1.0,
            final_conflicts=0,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_value_rollouts(
                [row, dict(row)],
                expected_jobs=2,
                smoke_only=True,
            )


if __name__ == "__main__":
    unittest.main()
