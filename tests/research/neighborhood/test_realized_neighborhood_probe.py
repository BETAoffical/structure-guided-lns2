from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from research.studies.neighborhood.realized_neighborhood_probe import (
    analyze_realized_records,
    build_candidate_rows,
    evaluation_seed,
    select_representative_neighborhoods,
)
from experiments.repair_collection import _job_label


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _proposal(agents: list[int], family: str, seed: int) -> dict:
    return {
        "agents": agents,
        "family": family,
        "seed_agent": agents[0],
        "proposal_seed": seed,
    }


def _source_collection(root: Path) -> None:
    episode_id = "task-a__seed_0000__official_adaptive"
    state_id = f"{episode_id}__decision_0000"
    _write_jsonl(
        root / "collection_manifest.jsonl",
        [
            {
                "status": "ok",
                "policy": "official_adaptive",
                "episode_id": episode_id,
                "split": "probe",
                "map_id": "map-a",
                "task_id": "task-a",
                "agent_count": 8,
                "solver_seed": 0,
            }
        ],
    )
    state = {
        "state_id": state_id,
        "state_fingerprint": "fingerprint-a",
        "episode_id": episode_id,
        "decision_index": 0,
        "prefix_actions": [],
        "state": {
            "num_of_colliding_pairs": 4,
            "context": {
                "split": "probe",
                "map_id": "map-a",
                "task_id": "task-a",
                "layout_mode": "regular_beltway",
                "task_variant": "balanced_80",
                "agent_count": 8,
            },
        },
    }
    states_path = root / "counterfactual" / "states.jsonl"
    outcomes_path = root / "counterfactual" / "outcomes.jsonl"
    _write_jsonl(states_path, [state])
    outcomes = []
    for family, agents in (
        ("target:4", [0, 1, 2, 3]),
        ("target:4", [0, 1, 4, 5]),
        ("collision:4", [0, 2, 4, 6]),
        ("collision:4", [0, 3, 5, 7]),
    ):
        heuristic, size = family.split(":")
        for trial in range(2):
            outcomes.append(
                {
                    "state_id": state_id,
                    "trial_seed": 100 + len(outcomes),
                    "trial_index": trial,
                    "action_valid": True,
                    "candidate_action": {
                        "heuristic": heuristic,
                        "neighborhood_size": int(size),
                        "seed_agent": 0,
                    },
                    "steps": [
                        {"step": 0},
                        {
                            "step": 1,
                            "metrics": {"neighborhood": agents},
                            "conflicts": trial,
                        },
                    ],
                    "horizon_outcomes": [
                        {"horizon": 1, "conflicts_after": trial}
                    ],
                }
            )
    _write_jsonl(outcomes_path, outcomes)
    _write_jsonl(
        root / "counterfactual_manifest.jsonl",
        [
            {
                "status": "ok",
                "states_file": states_path.relative_to(root).as_posix(),
                "outcomes_file": outcomes_path.relative_to(root).as_posix(),
            }
        ],
    )


def _analysis_fixture() -> tuple[list[dict], list[dict], list[dict], dict, dict]:
    candidates = []
    outcomes = []
    manifest = []
    for state_number in range(2):
        state_id = f"state-{state_number}"
        candidate_rows = []
        for candidate_number, agents in enumerate(([0, 1], [0, 2])):
            candidate_id = f"candidate-{candidate_number}"
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "agents": agents,
                    "actual_size": 2,
                    "selection_families": [
                        "target:4" if candidate_number == 0 else "collision:4"
                    ],
                    "proposal_seeds": [10 + candidate_number],
                }
            )
            for trial in range(8):
                conflicts = candidate_number if state_number == 0 else 1 - candidate_number
                outcomes.append(
                    {
                        "state_id": state_id,
                        "candidate_id": candidate_id,
                        "actual_neighborhood": agents,
                        "evaluation_trial_index": trial,
                        "evaluation_seed": 1000 + trial,
                        "evaluation_seed_disjoint": True,
                        "solved": conflicts == 0,
                        "conflicts_after": conflicts,
                        "generated": 10 + conflicts,
                        "runtime": 0.01,
                    }
                )
        candidates.append(
            {
                "state_id": state_id,
                "state_fingerprint": f"fingerprint-{state_number}",
                "episode_id": f"episode-{state_number}",
                "split": "probe",
                "map_id": f"map-{state_number}",
                "task_id": f"task-{state_number}",
                "layout_mode": "regular_beltway",
                "task_variant": "balanced_80",
                "state": {"num_of_colliding_pairs": 4},
                "candidates": candidate_rows,
            }
        )
        manifest.append(
            {
                "state_id": state_id,
                "status": "ok",
                "error_count": 0,
            }
        )
    config = {
        "evaluation_trials": 8,
        "thresholds": {
            "minimum_state_count": 2,
            "minimum_candidates_per_state": 2,
            "minimum_action_eta_squared": 0.0,
            "minimum_eta_improvement": -1.0,
            "minimum_trial_split_spearman": -1.0,
            "minimum_pareto_candidate_jaccard": 0.0,
            "minimum_best_candidate_jaccard": 0.0,
            "minimum_distinct_outcome_state_rate": 0.0,
            "maximum_fixed_proposal_family_share": 1.0,
        },
    }
    nominal = {"action_effect": {"mean_eta_squared": 0.4}}
    return candidates, outcomes, manifest, config, nominal


class RealizedNeighborhoodProbeTests(unittest.TestCase):
    def test_realized_worker_job_exposes_solver_seed_to_process_runner(self) -> None:
        job = {
            "row": {"task_id": "task-a"},
            "solver_seed": 7,
            "state_row": {"solver_seed": 7},
        }
        self.assertEqual(_job_label(job), "task-a__seed_0007")

    def test_selection_is_frequency_first_then_diversity_and_deterministic(self) -> None:
        proposals = [
            _proposal([0, 1, 2, 3], "target:4", seed) for seed in range(3)
        ]
        proposals.extend(
            [
                _proposal([0, 1, 2, 4], "target:4", 10),
                _proposal([0, 4, 5, 6], "target:4", 11),
                _proposal([0, 2, 4, 6], "collision:4", 12),
                _proposal([0, 3, 5, 7], "collision:4", 13),
            ]
        )
        first = select_representative_neighborhoods(proposals, 2)
        second = select_representative_neighborhoods(list(reversed(proposals)), 2)
        self.assertEqual(first, second)
        target = {
            tuple(row["agents"])
            for row in first
            if "target:4" in row["selection_families"]
        }
        self.assertIn((0, 1, 2, 3), target)
        self.assertIn((0, 4, 5, 6), target)

    def test_evaluation_seed_is_deterministic_and_disjoint(self) -> None:
        first = evaluation_seed("state", "candidate", 0, [])
        self.assertEqual(first, evaluation_seed("state", "candidate", 0, []))
        shifted = evaluation_seed("state", "candidate", 0, [first])
        self.assertNotEqual(first, shifted)
        self.assertNotIn(shifted, {first})

    def test_candidate_build_ignores_post_repair_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source_collection(root)
            first = build_candidate_rows(root, 2)
            path = root / "counterfactual" / "outcomes.jsonl"
            outcomes = [json.loads(line) for line in path.read_text().splitlines()]
            for row in outcomes:
                row["steps"][1]["conflicts"] = 999
                row["horizon_outcomes"][0]["conflicts_after"] = 999
            _write_jsonl(path, outcomes)
            second = build_candidate_rows(root, 2)
            self.assertEqual(first, second)
            self.assertEqual(first[0]["candidate_count"], 4)

    def test_analysis_detects_trial_and_explicit_set_integrity(self) -> None:
        candidates, outcomes, manifest, config, nominal = _analysis_fixture()
        report = analyze_realized_records(
            candidates, outcomes, manifest, config, nominal
        )
        self.assertTrue(report["integrity"]["passed"])
        self.assertEqual(report["integrity"]["outcome_count"], 32)

        missing = analyze_realized_records(
            candidates, outcomes[:-1], manifest, config, nominal
        )
        self.assertFalse(missing["integrity"]["passed"])
        self.assertEqual(missing["integrity"]["errors"]["trial_mismatches"], 1)

        changed = copy.deepcopy(outcomes)
        changed[0]["actual_neighborhood"] = [0, 7]
        mismatch = analyze_realized_records(
            candidates, changed, manifest, config, nominal
        )
        self.assertFalse(mismatch["integrity"]["passed"])
        self.assertEqual(
            mismatch["integrity"]["errors"]["neighborhood_mismatches"], 1
        )

    def test_analysis_rejects_test_or_ood_states(self) -> None:
        candidates, outcomes, manifest, config, nominal = _analysis_fixture()
        candidates[0]["split"] = "test_ood_layout"
        report = analyze_realized_records(
            candidates, outcomes, manifest, config, nominal
        )
        self.assertFalse(report["integrity"]["passed"])
        self.assertEqual(report["integrity"]["errors"]["non_probe_states"], 1)


if __name__ == "__main__":
    unittest.main()
