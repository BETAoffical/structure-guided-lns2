from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.repair_quality import (
    analyze_collection,
    pareto_indices,
    render_markdown,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _horizon(
    horizon: int,
    *,
    solved: bool,
    conflicts: int,
    auc: float,
    generated: int,
    runtime: float,
) -> dict:
    return {
        "horizon": horizon,
        "available": True,
        "solved": solved,
        "solved_step": horizon if solved else None,
        "conflicts_after": conflicts,
        "conflict_reduction": 5 - conflicts,
        "conflict_auc": auc,
        "sum_of_costs_after": 100,
        "cost_improvement": 0,
        "low_level_delta": {
            "expanded": generated,
            "generated": generated,
            "reopened": 0,
            "runs": 1,
        },
        "branch_runtime": runtime,
        "time_to_feasible": runtime if solved else None,
    }


class RepairQualityTests(unittest.TestCase):
    def test_calibration_config_has_expected_volume(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "repair_collection_calibration.json").read_text(
                encoding="utf-8"
            )
        )
        counterfactual = config["counterfactual"]

        self.assertEqual(config["solver_seeds"], [0, 1])
        self.assertEqual(
            config["policies"],
            [
                "official_adaptive",
                "fixed_target",
                "fixed_collision",
                "fixed_random",
            ],
        )
        self.assertEqual(
            counterfactual["eligible_splits"], ["train", "validation"]
        )
        self.assertEqual(counterfactual["max_states_per_episode"], 3)
        self.assertEqual(counterfactual["max_seed_agents"], 6)
        self.assertEqual(
            counterfactual["heuristics"], ["target", "collision", "random"]
        )
        self.assertEqual(counterfactual["neighborhood_sizes"], [4, 8, 16])
        self.assertEqual(counterfactual["trials"], 1)
        self.assertEqual(counterfactual["horizons"], [1, 4])
        self.assertEqual(config["workers"], 4)

        source_episodes = 72
        candidates_per_state = (
            counterfactual["max_seed_agents"]
            * len(counterfactual["heuristics"])
            * len(counterfactual["neighborhood_sizes"])
            * counterfactual["trials"]
        )
        self.assertEqual(
            source_episodes
            * counterfactual["max_states_per_episode"]
            * candidates_per_state,
            11664,
        )

    def test_pareto_indices_keep_tradeoffs_and_remove_dominated_values(
        self,
    ) -> None:
        values = [
            _horizon(
                4,
                solved=True,
                conflicts=0,
                auc=4.0,
                generated=100,
                runtime=1.0,
            ),
            _horizon(
                4,
                solved=False,
                conflicts=1,
                auc=5.0,
                generated=80,
                runtime=0.8,
            ),
            _horizon(
                4,
                solved=False,
                conflicts=2,
                auc=7.0,
                generated=120,
                runtime=1.2,
            ),
        ]
        self.assertEqual(pareto_indices(values), [0, 1])

    def _collection(self, root: Path) -> Path:
        run_fingerprint = "run-1"
        _write_json(
            root / "run_config.json",
            {
                "schema_version": 1,
                "run_fingerprint": run_fingerprint,
                "configuration": {
                    "schema_version": 1,
                    "solver_seeds": [0],
                    "policies": ["official_adaptive"],
                    "environment": {},
                    "counterfactual": {
                        "eligible_splits": ["train"],
                        "source_policy": "official_adaptive",
                        "max_states_per_episode": 1,
                        "max_seed_agents": 1,
                        "heuristics": ["target", "collision"],
                        "neighborhood_sizes": [4],
                        "trials": 1,
                        "horizons": [1, 4],
                    },
                },
            },
        )
        _write_jsonl(
            root / "qualification_manifest.jsonl",
            [{"status": "ok", "repairable": True, "split": "train"}],
        )
        episode_id = "task__seed_0000__official_adaptive"
        _write_jsonl(
            root / "collection_manifest.jsonl",
            [
                {
                    "episode_id": episode_id,
                    "split": "train",
                    "policy": "official_adaptive",
                    "status": "ok",
                    "summary": {
                        "success": True,
                        "time_to_feasible": 0.2,
                        "conflict_auc": 7.0,
                    },
                }
            ],
        )
        directory = root / "counterfactual" / "train" / episode_id
        state_id = episode_id + "__decision_0000"
        state = {
            "schema_version": 1,
            "run_fingerprint": run_fingerprint,
            "episode_id": episode_id,
            "state_id": state_id,
            "decision_index": 0,
            "candidate_count": 2,
            "state": {
                "context": {
                    "split": "train",
                    "layout_mode": "regular_beltway",
                    "scenario_type": "balanced_bidirectional",
                    "task_variant": "balanced_80",
                    "agent_count": 80,
                },
                "agents": [{"id": 3, "conflict_degree": 1, "delay": 0}],
            },
        }
        _write_jsonl(directory / "states.jsonl", [state])
        outcomes = []
        for index, heuristic in enumerate(("target", "collision")):
            outcomes.append(
                {
                    "schema_version": 1,
                    "run_fingerprint": run_fingerprint,
                    "episode_id": episode_id,
                    "state_id": state_id,
                    "candidate_index": index,
                    "candidate_action": {
                        "mode": "seed",
                        "seed_agent": 3,
                        "heuristic": heuristic,
                        "neighborhood_size": 4,
                        "random_seed": index,
                    },
                    "trial_index": 0,
                    "action_valid": True,
                    "horizon_outcomes": [
                        _horizon(
                            1,
                            solved=False,
                            conflicts=4 - index,
                            auc=4.5 - index,
                            generated=100 - 10 * index,
                            runtime=0.1,
                        ),
                        _horizon(
                            4,
                            solved=index == 0,
                            conflicts=index,
                            auc=5.0 + index,
                            generated=100 - 20 * index,
                            runtime=0.3 - 0.1 * index,
                        ),
                    ],
                }
            )
        _write_jsonl(directory / "outcomes.jsonl", outcomes)
        _write_jsonl(directory / "errors.jsonl", [])
        _write_jsonl(
            root / "counterfactual_manifest.jsonl",
            [
                {
                    "episode_id": episode_id,
                    "split": "train",
                    "status": "ok",
                    "complete": True,
                    "error_count": 0,
                    "state_count": 1,
                    "outcome_count": 2,
                    "states_file": directory.relative_to(root).as_posix()
                    + "/states.jsonl",
                    "outcomes_file": directory.relative_to(root).as_posix()
                    + "/outcomes.jsonl",
                    "errors_file": directory.relative_to(root).as_posix()
                    + "/errors.jsonl",
                }
            ],
        )
        return directory / "outcomes.jsonl"

    def test_collection_analysis_passes_and_reports_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._collection(root)
            manifest_path = root / "counterfactual_manifest.jsonl"
            manifests = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            manifests[0]["status"] = "resumed"
            _write_jsonl(manifest_path, manifests)
            baseline_path = root / "collection_manifest.jsonl"
            baselines = [
                json.loads(line)
                for line in baseline_path.read_text(encoding="utf-8").splitlines()
            ]
            baselines[0]["status"] = "resumed"
            _write_jsonl(baseline_path, baselines)
            report = analyze_collection(
                root,
                expected_qualification_runs=1,
                expected_baseline_episodes=1,
                expected_source_episodes=1,
                minimum_states=1,
                minimum_outcomes=2,
                minimum_informative_rate=1.0,
                maximum_family_dominance=1.0,
            )
            self.assertTrue(report["acceptance"]["passed"])
            self.assertEqual(report["counts"]["states"], 1)
            self.assertEqual(report["horizons"]["4"]["informative_rate"], 1.0)
            self.assertEqual(report["coverage"]["episode_count"], 1)
            self.assertEqual(report["coverage"]["seed_agent_selection_count"], 1)
            self.assertEqual(report["coverage"]["full_seed_coverage_rate"], 1.0)
            self.assertEqual(
                report["performance_by_horizon"]["4"]["heuristic"]["target"][
                    "outcome_count"
                ],
                1,
            )
            self.assertIn(
                "regular_beltway",
                report["contexts_horizon_4"]["layout_mode"],
            )
            self.assertIn(
                "target:4",
                report["stage_action_preferences_horizon_4"]["only"][
                    "pareto_state_count_by_family"
                ],
            )
            self.assertIn("Overall acceptance: **PASS**", render_markdown(report))

    def test_invalid_action_and_ood_state_fail_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcomes_path = self._collection(root)
            outcomes = [
                json.loads(line)
                for line in outcomes_path.read_text(encoding="utf-8").splitlines()
            ]
            outcomes[0]["action_valid"] = False
            _write_jsonl(outcomes_path, outcomes)
            states_path = outcomes_path.with_name("states.jsonl")
            states = [json.loads(states_path.read_text(encoding="utf-8"))]
            states[0]["state"]["context"]["split"] = "test_ood_task"
            _write_jsonl(states_path, states)
            manifest_path = root / "counterfactual_manifest.jsonl"
            manifests = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            manifests[0]["split"] = "test_ood_task"
            _write_jsonl(manifest_path, manifests)
            report = analyze_collection(
                root,
                expected_qualification_runs=1,
                expected_baseline_episodes=1,
                expected_source_episodes=1,
                minimum_states=1,
                minimum_outcomes=2,
                minimum_informative_rate=0.0,
                maximum_family_dominance=1.0,
            )
            self.assertFalse(report["acceptance"]["passed"])
            self.assertEqual(report["integrity"]["invalid_actions"], 1)
            self.assertEqual(report["integrity"]["test_ood_manifests"], 1)
            self.assertEqual(report["integrity"]["test_ood_states"], 1)


if __name__ == "__main__":
    unittest.main()
