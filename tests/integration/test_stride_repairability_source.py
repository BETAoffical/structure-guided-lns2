from __future__ import annotations

import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from experiments.stride_repairability_source import (
    validate_repairability_source_cohort_config,
)
from experiments.stride_repairability_selection import (
    _select_repairability_source_pool,
)


class StrideRepairabilitySourceTest(unittest.TestCase):
    def _root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _config(self) -> dict:
        return json.loads(
            (
                self._root()
                / "configs"
                / "stride_repairability_source_cohort.json"
            ).read_text(encoding="utf-8")
        )

    def test_source_cohort_freezes_effective_split_and_replacements(self) -> None:
        config = self._config()
        validate_repairability_source_cohort_config(config)
        split = config["effective_map_split"]
        self.assertEqual(len(split["train"]), 16)
        self.assertEqual(len(split["validation"]), 6)
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertEqual(
            config["replacements"],
            {
                "arena2": "den206d",
                "den001d": "den011d",
                "den005d": "ht_mansion_n",
            },
        )
        self.assertEqual(config["expected_task_count"], 44)

    def test_source_cohort_rejects_outcome_and_identity_drift(self) -> None:
        config = self._config()
        config["selection_inputs_forbidden"].remove("controller_ttf")
        with self.assertRaisesRegex(ValueError, "outcome boundary changed"):
            validate_repairability_source_cohort_config(config)

        config = self._config()
        config["controller_id"] = "v2-full"
        with self.assertRaisesRegex(ValueError, "contract changed"):
            validate_repairability_source_cohort_config(config)

    def test_runtime_source_contract_is_uncapped_by_selector_logic(self) -> None:
        runtime = json.loads(
            (
                self._root()
                / "configs"
                / "stride_repairability_source_runtime.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(runtime["formal"])
        self.assertEqual(runtime["split"], "balanced_wall_clock")
        self.assertEqual(runtime["solver_seeds"], [1, 2])
        self.assertEqual(
            runtime["policies"], ["official_adaptive", "realized_dynamic"]
        )
        self.assertEqual(runtime["max_decisions"], 12)
        self.assertEqual(runtime["metric_iteration_budget"], 12)
        self.assertNotIn("remaining_time_guard", runtime)
        self.assertNotIn("selector_time_limit", runtime)
        self.assertTrue(runtime["deterministic_pp_replay"])
        self.assertEqual(runtime["dataset_design"]["map_count"], 22)
        self.assertEqual(runtime["dataset_design"]["instance_count"], 44)

    def _selection_pool(self) -> tuple[list[dict], dict]:
        design = json.loads(
            (
                self._root()
                / "configs"
                / "stride_repairability_data_design.json"
            ).read_text(encoding="utf-8")
        )
        cohort = self._config()
        effective_split = {
            map_id: split
            for split, map_ids in cohort["effective_map_split"].items()
            for map_id in map_ids
        }
        ratios = design["static_low_degree_cell_ratio_by_map"]
        rows = []
        for policy in design["source_policies"]:
            for map_id in sorted(effective_split):
                for episode_index in range(4):
                    episode_id = f"{policy}-{map_id}-{episode_index}"
                    for decision_index, conflicts in ((0, 5), (4, 50), (8, 150)):
                        rows.append(
                            {
                                "schema": "lns2.stride.state_selection.v1",
                                "state_id": (
                                    f"state-{policy}-{map_id}-{episode_index}-"
                                    f"{decision_index}"
                                ),
                                "map_id": map_id,
                                "task_id": f"task-{map_id}-{episode_index}",
                                "split": "balanced_wall_clock",
                                "source_policy": policy,
                                "decision_stage": (
                                    "early"
                                    if decision_index < 4
                                    else "middle"
                                    if decision_index < 8
                                    else "late"
                                ),
                                "source_root": "/source",
                                "episode_id": episode_id,
                                "before_fingerprint": (
                                    f"fingerprint-{policy}-{map_id}-"
                                    f"{episode_index}-{decision_index}"
                                ),
                                "before_conflicts": conflicts,
                                "solver_seed": 1 + episode_index % 2,
                                "decision_index": decision_index,
                                "agent_count": 100 + episode_index,
                                "agent_band": "low_mid",
                                "prefix_actions": [],
                                "static_low_degree_cell_ratio": ratios[map_id],
                                "topology_group": (
                                    "boundary_relevant"
                                    if ratios[map_id]
                                    >= design["topology_boundary_threshold"]
                                    else "control"
                                ),
                                "conflict_band": (
                                    "1_10"
                                    if conflicts <= 10
                                    else "11_100"
                                    if conflicts <= 100
                                    else "101_500"
                                ),
                                "research_split": effective_split[map_id],
                            }
                        )
        return rows, design

    def test_result_blind_source_selector_meets_all_registered_floors(self) -> None:
        pool, design = self._selection_pool()
        selected, reports = _select_repairability_source_pool(pool, design)
        self.assertEqual(len(selected), 240)
        self.assertEqual(
            Counter(row["source_policy"] for row in selected),
            Counter({"official_adaptive": 120, "v2-full": 120}),
        )
        map_counts = Counter(row["map_id"] for row in selected)
        self.assertEqual(len(map_counts), 22)
        self.assertTrue(all(count >= 8 for count in map_counts.values()))
        map_policies: defaultdict[str, set[str]] = defaultdict(set)
        for row in selected:
            map_policies[row["map_id"]].add(row["source_policy"])
        self.assertTrue(
            all(
                policies == {"official_adaptive", "v2-full"}
                for policies in map_policies.values()
            )
        )
        topology = Counter(row["topology_group"] for row in selected)
        self.assertGreaterEqual(topology["control"], 96)
        self.assertGreaterEqual(topology["boundary_relevant"], 96)
        self.assertGreaterEqual(
            sum(row["before_conflicts"] >= 101 for row in selected), 24
        )
        self.assertEqual(
            {row["decision_stage"] for row in selected},
            {"early", "middle", "late"},
        )
        self.assertTrue(
            all(report["maximum_states_per_episode"] <= 2 for report in reports.values())
        )

    def test_result_blind_source_selector_rejects_outcome_fields(self) -> None:
        pool, design = self._selection_pool()
        pool[0]["ttf"] = 1.0
        with self.assertRaisesRegex(ValueError, "forbidden fields"):
            _select_repairability_source_pool(pool, design)


if __name__ == "__main__":
    unittest.main()
