from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.policy_visited_aggregation import (
    policy_visited_dataset_design,
    policy_visited_qualification_report,
)
from experiments.policy_visited_aggregation_analysis import (
    inverse_layout_state_weights,
    train_equal_state_pairwise_model,
)
from experiments.policy_visited_independent_confirmation import _validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ("regular_beltway", "compartmentalized", "dead_end_aisles")
VARIANTS = ("balanced_80", "balanced_100", "bottleneck_80", "bottleneck_100")


def dataset_rows(split_counts: dict[str, int]) -> list[dict]:
    rows = []
    for split, map_count in split_counts.items():
        per_layout = map_count // len(LAYOUTS)
        for layout in LAYOUTS:
            for map_index in range(per_layout):
                map_id = f"{split}_{layout}_{map_index}"
                for task_index, variant in enumerate(VARIANTS):
                    rows.append(
                        {
                            "split": split,
                            "map_id": map_id,
                            "task_id": f"{map_id}_{variant}",
                            "layout_mode": layout,
                            "task_variant": variant,
                            "agent_count": 80 if variant.endswith("80") else 100,
                            "map_seed": 1000 + len(rows) // 4,
                            "task_seed": 2000 + len(rows),
                        }
                    )
    return rows


def natural_config() -> dict:
    return {
        "study_role": "development",
        "splits": ["policy_train", "policy_validation"],
        "solver_seeds": [1, 2, 3],
        "dataset_design": {
            "tasks_per_map": 4,
            "task_variants": list(VARIANTS),
            "layout_counts": {
                "policy_train": {layout: 4 for layout in LAYOUTS},
                "policy_validation": {layout: 2 for layout in LAYOUTS},
            },
        },
        "qualification": {
            "mode": "natural_distribution_development",
            "minimum_nonzero_by_split": {
                "policy_train": 96,
                "policy_validation": 48,
            },
            "minimum_nonzero_per_layout": {
                "policy_train": 24,
                "policy_validation": 12,
            },
            "minimum_active_maps": {
                "policy_train": 12,
                "policy_validation": 6,
            },
        },
    }


def ranking_rows() -> list[dict]:
    rows = []
    for state_number, layout in enumerate(("regular_beltway",) * 2 + ("compartmentalized",) * 4):
        for candidate in range(2):
            winner = candidate == state_number % 2
            rows.append(
                {
                    "state_id": f"state-{state_number}",
                    "candidate_id": f"candidate-{candidate}",
                    "layout_mode": layout,
                    "features": {
                        "realized_dynamic": {
                            "state.value": float(state_number),
                            "candidate.value": float(candidate),
                        }
                    },
                    "outcome": {
                        "solved_rate": float(winner),
                        "conflicts_after": 0.0 if winner else 2.0,
                    },
                }
            )
    return rows


class PolicyVisitedNaturalDistributionTests(unittest.TestCase):
    def test_natural_gate_keeps_zero_conflicts_and_uses_minimum_counts(self) -> None:
        rows = dataset_rows({"policy_train": 12, "policy_validation": 6})
        config = natural_config()
        design = policy_visited_dataset_design(rows, config)
        qualification = []
        nonzero_limits = {
            ("policy_train", "regular_beltway"): 29,
            ("policy_train", "compartmentalized"): 48,
            ("policy_train", "dead_end_aisles"): 38,
            ("policy_validation", "regular_beltway"): 23,
            ("policy_validation", "compartmentalized"): 24,
            ("policy_validation", "dead_end_aisles"): 16,
        }
        extra_seen = {key: 0 for key in nonzero_limits}
        map_counts = {"policy_train": 4, "policy_validation": 2}
        for source in rows:
            for seed in (1, 2, 3):
                key = (source["split"], source["layout_mode"])
                mandatory = source["task_variant"] == "balanced_80" and seed == 1
                extra_limit = nonzero_limits[key] - map_counts[source["split"]]
                nonzero = mandatory or extra_seen[key] < extra_limit
                if not mandatory and nonzero:
                    extra_seen[key] += 1
                qualification.append(
                    {
                        **source,
                        "solver_seed": seed,
                        "initial_conflicts": 2 if nonzero else 0,
                        "state_fingerprint": f"{source['task_id']}-{seed}",
                        "status": "ok",
                        "error": None,
                    }
                )
        report = policy_visited_qualification_report(
            rows, qualification, config, design, {"passed": True}, formal=True
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["decision"], "eligible_for_development_collection")
        self.assertEqual(report["initial_feasible_count"], 38)
        self.assertFalse(report["confirmation_evidence"])
        self.assertAlmostEqual(
            report["repairable_rate_by_split_layout"][
                "policy_train/regular_beltway"
            ],
            29 / 48,
        )

    def test_legacy_mode_still_applies_strict_layout_threshold(self) -> None:
        rows = dataset_rows({"policy_train": 12, "policy_validation": 6})
        config = natural_config()
        config.pop("study_role")
        config["qualification"] = {
            "minimum_nonzero_by_split": {"policy_train": 108, "policy_validation": 54},
            "minimum_nonzero_per_layout": {"policy_train": 36, "policy_validation": 18},
            "minimum_active_maps": {"policy_train": 11, "policy_validation": 5},
        }
        qualification = []
        for index, source in enumerate(rows):
            for seed in (1, 2, 3):
                nonzero = not (
                    source["split"] == "policy_train"
                    and source["layout_mode"] == "regular_beltway"
                    and index % 4 >= 2
                )
                qualification.append(
                    {
                        **source,
                        "solver_seed": seed,
                        "initial_conflicts": int(nonzero),
                        "state_fingerprint": f"{source['task_id']}-{seed}",
                        "status": "ok",
                    }
                )
        report = policy_visited_qualification_report(
            rows,
            qualification,
            config,
            policy_visited_dataset_design(rows, config),
            {"passed": True},
            formal=True,
        )
        self.assertEqual(report["qualification_mode"], "strict_layout_coverage")
        self.assertFalse(report["passed"])

    def test_inverse_layout_weights_equalize_layout_mass(self) -> None:
        rows = ranking_rows()
        weights = inverse_layout_state_weights(rows)
        mass = {layout: 0.0 for layout in ("regular_beltway", "compartmentalized")}
        for row in rows[::2]:
            mass[row["layout_mode"]] += weights[row["state_id"]]
        self.assertAlmostEqual(mass["regular_beltway"], mass["compartmentalized"])
        parameters = {
            "learning_rate": 0.05,
            "max_iter": 10,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 2,
            "l2_regularization": 0.1,
            "random_state": 20260714,
        }
        _, primary = train_equal_state_pairwise_model(
            rows, "realized_dynamic", parameters
        )
        _, sensitivity = train_equal_state_pairwise_model(
            rows,
            "realized_dynamic",
            parameters,
            state_weight_multipliers=weights,
        )
        self.assertEqual(primary["state_weighting"], "equal_state")
        self.assertEqual(sensitivity["state_weighting"], "custom")
        self.assertTrue(primary["equal_state_weight"])
        self.assertFalse(sensitivity["equal_state_weight"])

    def test_confirmation_configs_are_fixed_and_separate(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/policy_visited_independent_confirmation.json").read_text(
                encoding="utf-8"
            )
        )
        _validate_config(config)
        dataset = json.loads(
            (PROJECT_ROOT / "configs/policy_visited_confirmation_dataset.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(dataset["master_seed"], 20270421)
        self.assertEqual(set(dataset["splits"]), {"policy_confirmation"})
        self.assertIn("build/initlns-policy-visited-v1", config["reference_datasets"])


if __name__ == "__main__":
    unittest.main()
