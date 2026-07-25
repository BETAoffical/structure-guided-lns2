from __future__ import annotations

import unittest

from experiments.v3_value_stability import (
    analyze_stability_followup,
    build_stability_jobs,
    identify_stability_targets,
    merge_stability_rollouts,
)


def _row(
    state: str,
    arm: str,
    trial: int,
    *,
    feasible: bool,
    seconds: float,
    censored: bool = False,
) -> dict:
    return {
        "state_id": state,
        "arm_id": arm,
        "trial_index": trial,
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 400,
        "initial_conflicts": 10,
        "final_conflicts": 0 if feasible else 5,
        "feasible": feasible,
        "censored": censored,
        "observed_total_seconds": seconds,
        "continuation_iterations": 1,
        "normalized_conflict_auc_seconds": seconds,
        "repair_iterations": 2,
        "complete": True,
    }


def _state(state_id: str) -> dict:
    return {
        "state_id": state_id,
        "arms": [
            {"arm_id": "a", "agents": [1, 2, 3, 4]},
            {"arm_id": "b", "agents": [5, 6, 7, 8]},
        ],
    }


class V3ValueStabilityTargetTests(unittest.TestCase):
    def test_targets_are_union_of_unstable_and_censored_states(self) -> None:
        rows = [
            _row("unstable", "a", 0, feasible=True, seconds=1.0),
            _row("unstable", "b", 0, feasible=True, seconds=2.0),
            _row("unstable", "a", 1, feasible=True, seconds=2.0),
            _row("unstable", "b", 1, feasible=True, seconds=1.0),
            _row("censored", "a", 0, feasible=False, seconds=3.0, censored=True),
            _row("censored", "b", 0, feasible=True, seconds=1.0),
            _row("censored", "a", 1, feasible=False, seconds=3.0, censored=True),
            _row("censored", "b", 1, feasible=True, seconds=1.0),
        ]
        targets = identify_stability_targets(rows)
        self.assertEqual(
            targets["target_state_ids"],
            ["censored", "unstable"],
        )
        self.assertEqual(len(targets["censored_rollout_keys"]), 2)

    def test_jobs_extend_censored_and_add_only_target_seeds(self) -> None:
        rows = [
            _row("stable", arm, trial, feasible=True, seconds=1.0)
            for arm in ("a", "b")
            for trial in (0, 1)
        ] + [
            _row(
                "target",
                arm,
                trial,
                feasible=(arm == "b"),
                seconds=1.0 if arm == "b" else 3.0,
                censored=(arm == "a"),
            )
            for arm in ("a", "b")
            for trial in (0, 1)
        ]
        targets, jobs = build_stability_jobs(
            plan={"states": [_state("stable"), _state("target")]},
            source_rows=rows,
            total_trials=4,
            max_repairs=60,
            wall_clock_seconds=120.0,
        )
        self.assertEqual(targets["target_state_ids"], ["target"])
        self.assertEqual(
            sum(job["reason"] == "extend_censored" for job in jobs),
            2,
        )
        self.assertEqual(
            sum(job["reason"] == "new_seed" for job in jobs),
            4,
        )
        self.assertFalse(
            any(job["state"]["state_id"] == "stable" for job in jobs)
        )


class V3ValueStabilityAnalysisTests(unittest.TestCase):
    def test_merge_replaces_only_censored_source_rollout(self) -> None:
        source = [
            _row("state", "a", 0, feasible=False, seconds=5.0, censored=True),
            _row("state", "b", 0, feasible=True, seconds=1.0),
        ]
        replacement = _row("state", "a", 0, feasible=True, seconds=2.0)
        replacement["followup_reason"] = "extend_censored"
        merged = merge_stability_rollouts(source, [replacement])
        self.assertEqual(len(merged), 2)
        self.assertTrue(
            next(row for row in merged if row["arm_id"] == "a")["feasible"]
        )

    def test_merge_refuses_to_replace_uncensored_source_rollout(self) -> None:
        source = [_row("state", "a", 0, feasible=True, seconds=1.0)]
        replacement = _row("state", "a", 0, feasible=True, seconds=2.0)
        replacement["followup_reason"] = "extend_censored"
        with self.assertRaisesRegex(ValueError, "uncensored"):
            merge_stability_rollouts(source, [replacement])

    def test_analysis_reports_complete_four_seed_target(self) -> None:
        source = [
            _row("state", arm, trial, feasible=True, seconds=seconds)
            for arm, seconds in (("a", 1.0), ("b", 2.0))
            for trial in (0, 1)
        ]
        followup = []
        for arm, seconds in (("a", 1.0), ("b", 2.0)):
            for trial in (2, 3):
                row = _row(
                    "state",
                    arm,
                    trial,
                    feasible=True,
                    seconds=seconds,
                )
                row["followup_reason"] = "new_seed"
                followup.append(row)
        report, states, arms = analyze_stability_followup(
            source_rows=source,
            followup_rows=followup,
            target_state_ids=["state"],
            total_trials=4,
        )
        self.assertTrue(report["checks"]["followup_coverage_complete"])
        self.assertEqual(report["merged_rollout_count"], 8)
        self.assertEqual(states[0]["winner_purity"], 1.0)
        self.assertEqual(states[0]["pairwise_winner_agreement"], 1.0)
        self.assertEqual(report["target_pairwise_winner_agreement"], 1.0)
        self.assertEqual(len(arms), 2)


if __name__ == "__main__":
    unittest.main()
