from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from research.studies.neighborhood.independent_layout_probe import (
    analyze_probe_records,
    holm_adjust,
    paired_difference_test,
    qualification_summary,
    validate_probe_dataset,
)


LAYOUTS = ("regular_beltway", "compartmentalized", "dead_end_aisles")
VARIANTS = ("balanced_80", "balanced_100", "bottleneck_80", "bottleneck_100")


def _settings() -> dict:
    return {
        "expected_split": "probe",
        "expected_layouts": {layout: 2 for layout in LAYOUTS},
        "expected_task_variants": list(VARIANTS),
        "expected_solver_seed": 0,
        "minimum_repairable_tasks": 18,
        "minimum_repairable_tasks_per_layout": 6,
        "minimum_labeled_maps_per_layout": 2,
        "minimum_od_modes_per_map": 2,
        "minimum_densities_per_map": 2,
        "minimum_paired_units": 6,
        "minimum_trials_per_candidate": 8,
        "minimum_mean_action_eta_squared": 0.5,
        "minimum_trial_split_spearman": 0.5,
        "minimum_pareto_family_jaccard": 0.5,
        "minimum_realized_neighborhood_jaccard": 0.5,
        "maximum_fixed_unique_pareto_share": 0.8,
        "maximum_holm_p_value": 0.05,
        "winner_bootstrap_samples": 20,
        "map_bootstrap_samples": 20,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _make_dataset(root: Path) -> list[dict]:
    rows = []
    task_seed = 1000
    map_index = 0
    for layout in LAYOUTS:
        for replica in range(2):
            map_id = f"probe_{layout}_{replica:04d}"
            map_seed = 100 + map_index
            map_index += 1
            for task_index, variant in enumerate(VARIANTS):
                od_mode, density_text = variant.rsplit("_", 1)
                density = int(density_text)
                task_id = f"{map_id}__task_{task_index:04d}"
                task_file = f"instances/{task_id}.json"
                starts = [[task_seed + index, index] for index in range(density)]
                goals = [[index, task_seed + index + 1] for index in range(density)]
                payload = {
                    "task_id": task_id,
                    "map_id": map_id,
                    "starts": starts,
                    "goals": goals,
                }
                path = root / "probe" / task_file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")
                rows.append(
                    {
                        "split": "probe",
                        "map_id": map_id,
                        "task_id": task_id,
                        "map_seed": map_seed,
                        "task_seed": task_seed,
                        "task_file": task_file,
                        "layout_mode": layout,
                        "task_variant": variant,
                        "agent_count": density,
                        "scenario_type": (
                            "balanced_bidirectional"
                            if od_mode == "balanced"
                            else "bottleneck_pressure"
                        ),
                    }
                )
                task_seed += 1
    _write_jsonl(root / "probe" / "manifest.jsonl", rows)
    return rows


def _qualification(rows: list[dict]) -> list[dict]:
    return [
        {
            "status": "ok",
            "solver_seed": 0,
            "split": "probe",
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row["layout_mode"],
            "task_variant": row["task_variant"],
            "agent_count": row["agent_count"],
            "initial_conflicts": 10,
            "repairable": True,
        }
        for row in rows
    ]


def _records(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    baseline = []
    states = []
    outcomes = []
    for row_index, row in enumerate(rows):
        episode_id = f"{row['task_id']}__seed_0000__official_adaptive"
        state_id = f"{episode_id}__decision_0000"
        fingerprint = f"fingerprint-{row_index}"
        baseline.append(
            {
                "status": "ok",
                "policy": "official_adaptive",
                "task_id": row["task_id"],
                "episode_id": episode_id,
                "summary": {"repairable": True, "initial_conflicts": 10},
            }
        )
        states.append(
            {
                "state_id": state_id,
                "state_fingerprint": fingerprint,
                "episode_id": episode_id,
                "decision_index": 0,
                "candidate_count": 2,
                "state": {
                    "context": {
                        "split": "probe",
                        "map_id": row["map_id"],
                        "task_id": row["task_id"],
                        "layout_mode": row["layout_mode"],
                        "task_variant": row["task_variant"],
                        "scenario_type": row["scenario_type"],
                        "agent_count": row["agent_count"],
                    }
                },
            }
        )
        od_mode = row["task_variant"].split("_", 1)[0]
        winner = "target" if od_mode == "balanced" else "collision"
        for heuristic in ("target", "collision"):
            conflicts = 0 if heuristic == winner else 4
            for trial in range(8):
                outcomes.append(
                    {
                        "state_id": state_id,
                        "state_fingerprint": fingerprint,
                        "episode_id": episode_id,
                        "trial_index": trial,
                        "trial_seed": 9000 + trial,
                        "action_valid": True,
                        "candidate_action": {
                            "seed_agent": 0,
                            "heuristic": heuristic,
                            "neighborhood_size": 4,
                        },
                        "conflict_trajectory": [10, conflicts],
                        "steps": [
                            {"step": 0, "conflicts": 10},
                            {
                                "step": 1,
                                "metrics": {
                                    "neighborhood": (
                                        [0, 1, 2, 3]
                                        if heuristic == "target"
                                        else [0, 4, 5, 6]
                                    )
                                },
                            },
                        ],
                        "horizon_outcomes": [
                            {
                                "horizon": 1,
                                "available": True,
                                "solved": conflicts == 0,
                                "conflicts_after": conflicts,
                                "conflict_auc": (10 + conflicts) / 2,
                                "branch_runtime": 0.01,
                                "low_level_delta": {"generated": 10 + conflicts},
                            }
                        ],
                    }
                )
    return baseline, states, outcomes


def _collection_config() -> dict:
    return {
        "solver_seeds": [0],
        "policies": ["official_adaptive"],
        "counterfactual": {
            "source_policy": "official_adaptive",
            "heuristics": ["target", "collision"],
            "neighborhood_sizes": [4],
            "trials": 8,
        },
    }


class IndependentLayoutProbeTests(unittest.TestCase):
    def test_dataset_has_complete_independent_factorial_and_seed_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            rows = _make_dataset(dataset)
            report = validate_probe_dataset(dataset, _settings())
            self.assertTrue(report["passed"])
            self.assertEqual(report["map_count"], 6)
            self.assertEqual(report["independent_density_task_pairs"], 12)
            self.assertEqual(report["layout_counts"], {layout: 2 for layout in LAYOUTS})

            reference = root / "reference"
            _write_jsonl(
                reference / "train" / "manifest.jsonl",
                [{"map_seed": rows[0]["map_seed"], "task_seed": 999999}],
            )
            overlap = validate_probe_dataset(dataset, _settings(), [reference])
            self.assertFalse(overlap["passed"])
            self.assertTrue(overlap["reference_seed_checks"][0]["overlap"])

    def test_qualification_requires_all_layouts_and_paired_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            rows = _make_dataset(dataset)
            validation = validate_probe_dataset(dataset, _settings())
            report = qualification_summary(
                rows, _qualification(rows), _settings(), validation
            )
            self.assertTrue(report["passed"])
            damaged = _qualification(rows)
            for row in damaged:
                if row["layout_mode"] == "dead_end_aisles" and row[
                    "task_variant"
                ].startswith("bottleneck"):
                    row["repairable"] = False
                    row["initial_conflicts"] = 0
            failed = qualification_summary(rows, damaged, _settings(), validation)
            self.assertFalse(failed["gates"]["within_map_od_density_coverage"])

    def test_holm_adjustment_is_monotone(self) -> None:
        self.assertEqual(
            holm_adjust({"layout": 0.01, "od": 0.04, "density": 0.2}),
            {"density": 0.2, "layout": 0.03, "od": 0.08},
        )

    def test_paired_permutation_detects_aligned_family_changes(self) -> None:
        differences = [
            {"target:4": 1.0, "collision:4": -1.0} for _ in range(6)
        ]
        report = paired_difference_test(
            differences, ["target:4", "collision:4"]
        )
        self.assertEqual(report["assignment_count"], 64)
        self.assertLessEqual(report["upper_tail_p_value"], 0.05)

    def test_complete_trials_pass_and_non_probe_labels_fail_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            rows = _make_dataset(dataset)
            settings = _settings()
            validation = validate_probe_dataset(dataset, settings)
            qualification = _qualification(rows)
            baseline, states, outcomes = _records(rows)
            report = analyze_probe_records(
                rows,
                qualification,
                baseline,
                states,
                outcomes,
                _collection_config(),
                settings,
                validation,
            )
            self.assertTrue(report["integrity"]["passed"])
            self.assertEqual(report["integrity"]["minimum_trials_per_candidate"], 8)
            self.assertEqual(report["label_coverage"]["od_pair_count"], 12)
            self.assertEqual(report["label_coverage"]["density_pair_count"], 12)

            leaked_states = copy.deepcopy(states)
            leaked_states[0]["state"]["context"]["split"] = "test_ood_layout"
            leaked = analyze_probe_records(
                rows,
                qualification,
                baseline,
                leaked_states,
                outcomes,
                _collection_config(),
                settings,
                validation,
            )
            self.assertFalse(leaked["integrity"]["passed"])
            self.assertEqual(leaked["integrity"]["errors"]["non_probe_labels"], 1)

            incomplete = analyze_probe_records(
                rows,
                qualification,
                baseline,
                states,
                outcomes[:-1],
                _collection_config(),
                settings,
                validation,
            )
            self.assertFalse(incomplete["integrity"]["passed"])
            self.assertEqual(incomplete["integrity"]["errors"]["trial_mismatches"], 1)

            missing_candidate = analyze_probe_records(
                rows,
                qualification,
                baseline,
                states,
                outcomes[:-8],
                _collection_config(),
                settings,
                validation,
            )
            self.assertFalse(missing_candidate["integrity"]["passed"])
            self.assertEqual(
                missing_candidate["integrity"]["errors"][
                    "candidate_count_mismatches"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
