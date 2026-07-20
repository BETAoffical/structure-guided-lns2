from __future__ import annotations

import unittest

from experiments.closed_loop_confirmation import (
    closed_loop_qualification_report,
    configured_policies,
    movingai_ood_dataset_design,
)
from experiments.movingai_ood_confirmation import (
    family_auc_comparison,
    movingai_ood_acceptance,
)


def dataset_rows() -> list[dict]:
    rows = []
    specifications = (
        ("random-a", "random", (100, 200)),
        ("maze-a", "maze", (100, 200)),
        ("room-a", "room", (400, 600)),
        ("warehouse-a", "warehouse", (300, 500)),
        ("game-a", "game", (200, 300)),
    )
    for map_id, family, counts in specifications:
        for scenario in (4, 5):
            for agents in counts:
                rows.append(
                    {
                        "split": "movingai_ood",
                        "map_id": map_id,
                        "task_id": f"{map_id}-{scenario}-{agents}",
                        "layout_mode": family,
                        "scenario_type": f"movingai_random_{scenario}",
                        "task_variant": f"random_{scenario}_agents_{agents}",
                        "agent_count": agents,
                    }
                )
    return rows


def dataset_design() -> dict:
    maps = []
    for row in dataset_rows():
        if any(value["map_id"] == row["map_id"] for value in maps):
            continue
        maps.append(
            {
                "map_id": row["map_id"],
                "layout_family": row["layout_mode"],
                "agent_counts": sorted(
                    {value["agent_count"] for value in dataset_rows() if value["map_id"] == row["map_id"]}
                ),
            }
        )
    return {
        "mode": "movingai_ood",
        "map_count": 5,
        "task_count": 20,
        "scenario_indices": [4, 5],
        "layout_family_counts": {
            "random": 1,
            "maze": 1,
            "room": 1,
            "warehouse": 1,
            "game": 1,
        },
        "maps": maps,
    }


def manifest_row(policy: str, task: str, family: str, auc: float) -> dict:
    return {
        "status": "ok",
        "policy": policy,
        "task_id": task,
        "map_id": f"{family}-map",
        "layout_mode": family,
        "solver_seed": 1,
        "summary": {
            "repairable": True,
            "success": True,
            "fixed_budget_conflict_auc": auc,
        },
    }


class MovingAIOODConfirmationTests(unittest.TestCase):
    def test_dataset_design_checks_scenarios_agents_and_families(self) -> None:
        report = movingai_ood_dataset_design(
            dataset_rows(), "movingai_ood", dataset_design()
        )
        self.assertTrue(report["passed"])
        changed = dataset_rows()[:-1]
        self.assertFalse(
            movingai_ood_dataset_design(changed, "movingai_ood", dataset_design())["passed"]
        )

    def test_five_policy_configuration_is_supported(self) -> None:
        policies = [
            "official_adaptive",
            "fixed_target",
            "fixed_collision",
            "fixed_random",
            "realized_dynamic",
        ]
        self.assertEqual(configured_policies({"policies": policies}), tuple(policies))

    def test_ood_qualification_keeps_zero_conflict_and_requires_all_families(self) -> None:
        rows = dataset_rows()
        results = []
        for row in rows:
            for seed in (1, 2, 3):
                results.append(
                    {
                        **row,
                        "solver_seed": seed,
                        "initial_conflicts": 0 if row["task_id"].endswith("-100") else 3,
                        "state_fingerprint": f"{row['task_id']}-{seed}",
                        "status": "ok",
                    }
                )
        report = closed_loop_qualification_report(
            rows,
            results,
            {
                "solver_seeds": [1, 2, 3],
                "qualification": {
                    "mode": "movingai_ood",
                    "minimum_nonzero_states": 1,
                    "minimum_active_maps": 5,
                    "required_layout_families": ["random", "maze", "room", "warehouse", "game"],
                },
                "severity_thresholds": {"low_max": 0.001, "medium_max": 0.01},
            },
            movingai_ood_dataset_design(rows, "movingai_ood", dataset_design()),
            {"passed": True},
            formal=True,
        )
        self.assertTrue(report["passed"])
        self.assertGreater(report["initial_feasible_count"], 0)

    def test_family_comparison_and_registered_acceptance(self) -> None:
        families = ("random", "maze", "room", "warehouse", "game")
        adaptive = [manifest_row("official_adaptive", name, name, 100.0) for name in families]
        realized = [manifest_row("realized_dynamic", name, name, 80.0) for name in families]
        family = family_auc_comparison(adaptive, realized)
        self.assertEqual(family["families_no_worse"], 5)
        summaries = {
            policy: {
                "error_count": 0,
                "success_count": 5,
                "invalid_action_count": 0,
                "fingerprint_mismatch_count": 0,
            }
            for policy in (
                "official_adaptive",
                "fixed_target",
                "fixed_collision",
                "fixed_random",
                "realized_dynamic",
            )
        }
        base = {
            "qualification": {"passed": True},
            "integrity": {"passed": True},
            "policy_summaries": summaries,
            "comparisons": {
                "realized_dynamic_vs_official_adaptive": {
                    "metrics": {
                        "fixed_budget_conflict_auc": {
                            "relative_improvement": 0.2,
                            "maps_no_worse": 12,
                            "bootstrap": {"improvement_95_ci": [0.1, 0.3]},
                        }
                    }
                }
            },
        }
        manifests = {
            "official_adaptive": adaptive,
            "realized_dynamic": realized,
            "fixed_target": adaptive,
            "fixed_collision": adaptive,
            "fixed_random": adaptive,
        }
        acceptance = movingai_ood_acceptance(
            base,
            manifests,
            {
                "ood_thresholds": {
                    "minimum_auc_improvement": 0.05,
                    "bootstrap_lower_bound": 0.0,
                    "minimum_maps_no_worse": 8,
                    "minimum_layout_families_no_worse": 4,
                }
            },
        )
        self.assertTrue(acceptance["passed"])


if __name__ == "__main__":
    unittest.main()
