from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.stride_robustaction_expansion import (
    prepare_robustaction_preflight_dataset,
    robustaction_source_adapter,
    topology_group,
    validate_robustaction_expansion_design,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_robustaction_expansion_design.json"
STRUCTPOOL_CONFIG = (
    ROOT / "configs" / "stride_robustaction_structpool_data_design.json"
)
LOAD_EXTENSION_CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_load_extension_design.json"
)
MAP_REPLACEMENT_CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_map_replacement_design.json"
)
DA2_SUPPLEMENT_CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_da2_supplement_design.json"
)


class RobustActionExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_design_is_valid(self) -> None:
        validate_robustaction_expansion_design(self.config)

    def test_structpool_rebalance_is_valid_and_keeps_legacy_design_frozen(self) -> None:
        structpool = json.loads(STRUCTPOOL_CONFIG.read_text(encoding="utf-8"))
        validate_robustaction_expansion_design(structpool)
        self.assertEqual(
            structpool["topology_group_targets"],
            {
                "dao_high_topology": 12,
                "dao_mid_topology": 6,
                "dao_low_topology_control": 2,
            },
        )
        self.assertEqual(self.config["topology_group_targets"], {
            "dao_high_topology": 7,
            "dao_mid_topology": 7,
            "dao_low_topology_control": 6,
        })
        self.assertEqual(
            structpool["post_collection_opportunity_gates"]
            ["minimum_positive_opportunity_maps_by_topology_group"]
            ["dao_low_topology_control"],
            2,
        )

    def test_structpool_source_adapter_has_rebalanced_dimensions(self) -> None:
        structpool = json.loads(STRUCTPOOL_CONFIG.read_text(encoding="utf-8"))
        adapter = robustaction_source_adapter(structpool)
        self.assertEqual(
            adapter["dataset_revision"],
            "stride-robustaction-structpool-preflight-v1",
        )
        self.assertEqual(adapter["expected_map_count"], 20)
        self.assertEqual(adapter["expected_instance_count"], 124)

    def test_structpool_load_extension_is_static_and_dimensioned(self) -> None:
        extension = json.loads(
            LOAD_EXTENSION_CONFIG.read_text(encoding="utf-8")
        )
        validate_robustaction_expansion_design(extension)
        adapter = robustaction_source_adapter(extension)
        self.assertEqual(adapter["expected_map_count"], 12)
        self.assertEqual(adapter["expected_instance_count"], 48)
        self.assertEqual(
            adapter["dataset_revision"],
            "stride-robustaction-structpool-load-extension-v2",
        )
        self.assertEqual(
            {row["id"] for row in adapter["benchmarks"]},
            {
                "brc200d", "brc201d", "den900d", "hrt001d", "lak106d",
                "lak308d", "lak526d", "lgt600d", "lgt604d", "orz601d",
                "ost001d", "oth000d",
            },
        )

    def test_structpool_map_replacement_is_static_and_dimensioned(self) -> None:
        replacement = json.loads(
            MAP_REPLACEMENT_CONFIG.read_text(encoding="utf-8")
        )
        validate_robustaction_expansion_design(replacement)
        adapter = robustaction_source_adapter(replacement)
        self.assertEqual(adapter["expected_map_count"], 4)
        self.assertEqual(adapter["expected_instance_count"], 24)
        self.assertEqual(
            adapter["dataset_revision"],
            "stride-robustaction-structpool-map-replacement-v3",
        )
        self.assertEqual(
            {row["id"] for row in adapter["benchmarks"]},
            {"orz201d", "lak203d", "ost101d", "rmtst"},
        )

    def test_structpool_da2_supplement_is_static_and_dimensioned(self) -> None:
        supplement = json.loads(
            DA2_SUPPLEMENT_CONFIG.read_text(encoding="utf-8")
        )
        validate_robustaction_expansion_design(supplement)
        adapter = robustaction_source_adapter(supplement)
        self.assertEqual(adapter["expected_map_count"], 8)
        self.assertEqual(adapter["expected_instance_count"], 48)
        self.assertEqual(
            adapter["dataset_revision"],
            "stride-robustaction-structpool-da2-supplement-v1",
        )
        self.assertEqual(
            {row["id"] for row in adapter["benchmarks"]},
            {
                "ca_cave",
                "ca_caverns2",
                "dr_primevalentrance",
                "ht_bartrand_n",
                "lt_hangedman",
                "lt_undercitydungeon",
                "lt_undercityserialkiller",
                "w_encounter3",
            },
        )

    def test_structpool_da2_supplement_rejects_outcome_selection(self) -> None:
        supplement = json.loads(
            DA2_SUPPLEMENT_CONFIG.read_text(encoding="utf-8")
        )
        changed = copy.deepcopy(supplement)
        changed["selection_boundary"]["allowed_inputs"].append(
            "candidate_runtime"
        )
        with self.assertRaisesRegex(ValueError, "outcome boundary"):
            validate_robustaction_expansion_design(changed)

    def test_structpool_design_rejects_impossible_low_group_gate(self) -> None:
        structpool = json.loads(STRUCTPOOL_CONFIG.read_text(encoding="utf-8"))
        changed = copy.deepcopy(structpool)
        changed["post_collection_opportunity_gates"][
            "minimum_positive_opportunity_maps_by_topology_group"
        ]["dao_low_topology_control"] = 3
        with self.assertRaisesRegex(ValueError, "opportunity gates"):
            validate_robustaction_expansion_design(changed)

    def test_topology_boundaries_are_deterministic(self) -> None:
        thresholds = [0.035, 0.06]
        self.assertEqual(topology_group(0.060, thresholds), "dao_high_topology")
        self.assertEqual(topology_group(0.035, thresholds), "dao_mid_topology")
        self.assertEqual(
            topology_group(0.034999999, thresholds),
            "dao_low_topology_control",
        )

    def test_generic_source_adapter_has_registered_dimensions(self) -> None:
        adapter = robustaction_source_adapter(self.config)
        self.assertEqual(adapter["expected_map_count"], 20)
        self.assertEqual(adapter["expected_instance_count"], 132)
        self.assertEqual(
            adapter["dataset_revision"], "stride-robustaction-preflight-v2"
        )
        self.assertEqual(adapter["task_seeds"], [307])
        self.assertEqual(
            sum(
                len(row["agent_counts"]) * len(adapter["task_variants"])
                for row in adapter["benchmarks"]
            ),
            132,
        )
        self.assertTrue(
            all(
                value % 2 == 0
                for row in adapter["benchmarks"]
                for value in row["agent_counts"]
            )
        )

    def test_forbidden_outcome_input_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["selection_boundary"]["allowed_inputs"].append(
            "candidate_repair_outcome"
        )
        with self.assertRaisesRegex(ValueError, "outcome boundary"):
            validate_robustaction_expansion_design(changed)

    def test_locked_map_overlap_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["formal_ood_map_ids"][0] = changed["benchmarks"][0]["id"]
        with self.assertRaisesRegex(ValueError, "locked evidence"):
            validate_robustaction_expansion_design(changed)

    def test_incomplete_destination_is_rejected_before_adapter_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "dataset-v2"
            output.mkdir()
            marker = output / "partial.txt"
            marker.write_text("preserve", encoding="utf-8")
            adapter = root / "dataset-v2.source_adapter.json"

            with self.assertRaisesRegex(ValueError, "non-empty but incomplete"):
                prepare_robustaction_preflight_dataset(
                    config_path=CONFIG,
                    fetched=root / "unused-source",
                    output=output,
                )

            self.assertFalse(adapter.exists())
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
