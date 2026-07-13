from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from generators.dataset import generate_dataset
from generators.io import write_instance_bundle, write_map_bundle
from generators.task_flows import generate_tasks
from generators.validation import validate_map, validate_task
from generators.visualization import ascii_preview, svg_preview
from generators.warehouse import generate_warehouse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = json.loads(
    (PROJECT_ROOT / "configs" / "stage1_example.json").read_text(
        encoding="utf-8"
    )
)
MAP_CONFIG = EXAMPLE_CONFIG["map"]
TASK_CONFIG = EXAMPLE_CONFIG["task"]
ACTIVE_LAYOUTS = (
    "regular_beltway",
    "compartmentalized",
    "dead_end_aisles",
)


class WarehouseGeneratorTests(unittest.TestCase):
    def test_active_maps_are_deterministic_connected_and_fixed_size(
        self,
    ) -> None:
        for index, mode in enumerate(ACTIVE_LAYOUTS):
            config = {**MAP_CONFIG, "layout_mode": mode}
            first = generate_warehouse(
                config, 1200 + index, f"active_{mode}"
            )
            second = generate_warehouse(
                config, 1200 + index, f"active_{mode}"
            )
            validate_map(first)
            self.assertEqual(first.grid, second.grid)
            self.assertEqual(first.metadata, second.metadata)
            self.assertEqual((first.rows, first.cols), (28, 39))
            parameters = first.metadata["sampled_parameters"]
            self.assertEqual(parameters["horizontal_aisle_width"], 2)
            self.assertEqual(parameters["vertical_aisle_width"], 2)
            self.assertEqual(parameters["outer_beltway_width"], 2)
            self.assertEqual(parameters["station_count"], 2)
            self.assertTrue(first.metadata["zones"]["top_storage"])
            self.assertTrue(first.metadata["zones"]["bottom_storage"])
            station_columns = sorted(
                station["cell"][1]
                for station in first.metadata["stations"]
            )
            self.assertLess(station_columns[0], first.cols // 2)
            self.assertGreater(station_columns[1], first.cols // 2)

    def test_active_layout_geometry(self) -> None:
        expected_orientations = {
            "cross_four_gate": ["horizontal", "vertical"],
            "double_horizontal": ["horizontal", "horizontal"],
            "double_vertical": ["vertical", "vertical"],
        }
        for index, (template, expected) in enumerate(
            expected_orientations.items()
        ):
            compartment = generate_warehouse(
                {
                    **MAP_CONFIG,
                    "layout_mode": "compartmentalized",
                    "divider_template": template,
                },
                2201 + index,
                template,
            )
            dividers = compartment.metadata["structural_changes"][
                "compartment_gates"
            ]
            self.assertEqual(len(dividers), 2)
            self.assertEqual(
                sorted(item["orientation"] for item in dividers),
                expected,
            )
            self.assertEqual(
                sum(len(item["gate_cells"]) for item in dividers),
                4,
            )
            for divider in dividers:
                self.assertEqual(len(divider["gate_cells"]), 2)
                for gate_row, gate_col in divider["gate_cells"]:
                    self.assertTrue(
                        compartment.traversable((gate_row, gate_col))
                    )

        dead_end = generate_warehouse(
            {**MAP_CONFIG, "layout_mode": "dead_end_aisles"},
            2202,
            "two_dead_ends",
        )
        changes = dead_end.metadata["structural_changes"]
        self.assertEqual(len(changes["dead_end_caps"]), 4)
        self.assertEqual(
            sorted(cap["orientation"] for cap in changes["dead_end_caps"]),
            ["horizontal", "horizontal", "vertical", "vertical"],
        )
        obstacle_layer = dead_end.metadata["obstacle_type_layer"]
        for cap in changes["dead_end_caps"]:
            if cap["orientation"] == "vertical":
                left_row, left_col = cap["left_shelf_cell"]
                right_row, right_col = cap["right_shelf_cell"]
                self.assertEqual(obstacle_layer[left_row][left_col], "S")
                self.assertEqual(obstacle_layer[right_row][right_col], "S")
            else:
                top_row, top_col = cap["top_shelf_cell"]
                bottom_row, bottom_col = cap["bottom_shelf_cell"]
                self.assertEqual(obstacle_layer[top_row][top_col], "S")
                self.assertEqual(
                    obstacle_layer[bottom_row][bottom_col], "S"
                )
            self.assertTrue(cap["cells"])

    def test_active_task_scenarios_are_valid_and_deterministic(self) -> None:
        map_data = generate_warehouse(
            MAP_CONFIG, 3300, "task_map"
        )
        balanced = generate_tasks(
            map_data,
            {
                **TASK_CONFIG,
                "scenario_type": "balanced_bidirectional",
            },
            3301,
            "balanced",
        )
        repeated = generate_tasks(
            map_data,
            {
                **TASK_CONFIG,
                "scenario_type": "balanced_bidirectional",
            },
            3301,
            "balanced",
        )
        validate_task(map_data, balanced)
        self.assertEqual(balanced.starts, repeated.starts)
        self.assertEqual(balanced.goals, repeated.goals)
        self.assertEqual(
            balanced.metadata["realized_flow_counts"],
            {"left_to_right": 18, "right_to_left": 18},
        )
        dense = generate_tasks(
            map_data,
            {
                **TASK_CONFIG,
                "agent_count": 60,
                "scenario_type": "balanced_bidirectional",
            },
            3302,
            "dense",
        )
        validate_task(map_data, dense)
        self.assertEqual(
            dense.metadata["realized_flow_counts"],
            {"left_to_right": 30, "right_to_left": 30},
        )
        clustered = generate_tasks(
            map_data,
            {
                **TASK_CONFIG,
                "agent_count": 48,
                "scenario_type": "balanced_bidirectional",
                "hotspot_skew": 0.7,
                "origin_cluster_count": 2,
                "goal_cluster_count": 2,
                "cluster_radius": 4,
            },
            3303,
            "clustered",
        )
        validate_task(map_data, clustered)
        self.assertEqual(
            clustered.metadata["realized_flow_counts"],
            {"left_to_right": 24, "right_to_left": 24},
        )
        self.assertEqual(clustered.metadata["hotspot_skew"], 0.7)
        uniform = generate_tasks(
            map_data,
            {**TASK_CONFIG, "scenario_type": "uniform_random"},
            3304,
            "uniform",
        )
        validate_task(map_data, uniform)
        self.assertEqual(uniform.agent_count, 36)
        self.assertEqual(len(set(uniform.starts)), 36)
        self.assertEqual(len(set(uniform.goals)), 36)

    def test_cross_zone_exchange_uses_exact_six_pair_quotas(self) -> None:
        map_data = generate_warehouse(MAP_CONFIG, 3600, "cross_zone_map")
        config = {
            **TASK_CONFIG,
            "agent_count": 60,
            "scenario_type": "cross_zone_exchange",
        }
        task = generate_tasks(map_data, config, 3601, "cross_zone")
        repeated = generate_tasks(map_data, config, 3601, "cross_zone")
        validate_task(map_data, task)
        self.assertEqual(task.starts, repeated.starts)
        self.assertEqual(task.goals, repeated.goals)
        expected = {
            "left_to_center": 10,
            "center_to_left": 10,
            "center_to_right": 10,
            "right_to_center": 10,
            "left_to_right": 10,
            "right_to_left": 10,
        }
        self.assertEqual(task.metadata["od_quota_counts"], expected)
        self.assertEqual(task.metadata["realized_flow_counts"], expected)

        zones = {
            name.removesuffix("_storage"): {
                tuple(cell) for cell in values
            }
            for name, values in map_data.metadata["zones"].items()
            if name.endswith("_storage")
        }
        for start, goal, assignment in zip(
            task.starts, task.goals, task.metadata["flow_assignments"]
        ):
            origin, destination = assignment.split("_to_", 1)
            self.assertNotEqual(origin, destination)
            self.assertIn(start, zones[origin])
            self.assertIn(goal, zones[destination])

    def test_generic_od_matrix_still_overrides_builtin_scenarios(self) -> None:
        map_data = generate_warehouse(MAP_CONFIG, 3650, "od_override_map")
        task = generate_tasks(
            map_data,
            {
                **TASK_CONFIG,
                "agent_count": 12,
                "scenario_type": "cross_zone_exchange",
                "od_matrix": {"left->right": 1.0},
            },
            3651,
            "od_override",
        )
        validate_task(map_data, task)
        self.assertIsNone(task.metadata["od_quota_counts"])
        self.assertEqual(
            task.metadata["realized_flow_counts"], {"left->right": 12}
        )

    def test_exact_od_rejects_insufficient_zone_capacity(self) -> None:
        map_data = generate_warehouse(MAP_CONFIG, 3660, "small_zone_map")
        map_data = copy.deepcopy(map_data)
        map_data.metadata["zones"]["left_storage"] = map_data.metadata[
            "zones"
        ]["left_storage"][:1]
        with self.assertRaisesRegex(
            ValueError, "exact OD schedule requires .* starts in left"
        ):
            generate_tasks(
                map_data,
                {
                    **TASK_CONFIG,
                    "agent_count": 60,
                    "scenario_type": "cross_zone_exchange",
                },
                3661,
                "insufficient_zone",
            )

    def test_intersection_crossing_is_four_way_and_hard_constrained(self) -> None:
        map_data = generate_warehouse(MAP_CONFIG, 3700, "intersection_map")
        config = {
            **TASK_CONFIG,
            "agent_count": 100,
            "scenario_type": "intersection_crossing",
            "required_intersection_crossing_ratio": 0.6,
            "target_intersection_count": 2,
        }
        task = generate_tasks(map_data, config, 3701, "intersection")
        validate_task(map_data, task)
        self.assertEqual(
            task.metadata["od_quota_counts"],
            {
                "left_to_right": 25,
                "right_to_left": 25,
                "top_to_bottom": 25,
                "bottom_to_top": 25,
            },
        )
        self.assertEqual(
            task.metadata["realized_flow_counts"],
            task.metadata["od_quota_counts"],
        )
        self.assertEqual(
            len(task.metadata["selected_intersection_components"]), 2
        )
        required = task.metadata["required_intersections"]
        self.assertEqual(sum(cell is not None for cell in required), 60)
        self.assertEqual(
            task.metadata["realized_intersection_crossing_ratio"], 0.6
        )
        semantic = map_data.metadata["semantic_cell_types"]
        for cell in required:
            if cell is not None:
                self.assertEqual(semantic[cell[0]][cell[1]], "X")

    def test_intersection_constraints_fail_instead_of_degrading(self) -> None:
        map_data = generate_warehouse(MAP_CONFIG, 3800, "no_intersections")
        map_data = copy.deepcopy(map_data)
        map_data.metadata["semantic_cell_types"] = [
            row.replace("X", "H")
            for row in map_data.metadata["semantic_cell_types"]
        ]
        config = {
            **TASK_CONFIG,
            "agent_count": 40,
            "scenario_type": "intersection_crossing",
            "required_intersection_crossing_ratio": 0.6,
            "target_intersection_count": 2,
        }
        with self.assertRaisesRegex(
            ValueError, "requires 2 feasible intersection components"
        ):
            generate_tasks(map_data, config, 3801, "must_fail")
        with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
            generate_tasks(
                map_data,
                {**config, "required_intersection_crossing_ratio": 1.1},
                3802,
                "invalid_ratio",
            )

    def test_dormant_layouts_have_compatibility_smoke_coverage(self) -> None:
        dormant_modes = (
            "partial_beltway",
            "wall_shelves",
            "partial_cross_aisles",
            "mixed_width",
            "asymmetric",
            "station_centric",
        )
        compatibility = {
            **MAP_CONFIG,
            "outer_beltway_width": 3,
            "beltway_narrow_segment_count": 1,
            "beltway_blocked_segment_count": 1,
            "wall_shelf_extension_count": 1,
            "cross_aisle_closure_count": 1,
            "cross_aisle_span_blocks": 1,
            "variable_vertical_aisle_width": [1, 3],
            "asymmetric_trim_width": 1,
            "station_centric_placement": "clustered",
        }
        for index, mode in enumerate(dormant_modes):
            map_data = generate_warehouse(
                {**compatibility, "layout_mode": mode},
                4400 + index,
                f"dormant_{mode}",
            )
            validate_map(map_data)
            self.assertEqual(
                map_data.metadata["sampled_parameters"]["layout_mode"],
                mode,
            )

    def test_export_and_previews(self) -> None:
        map_data = generate_warehouse(MAP_CONFIG, 5500, "export_map")
        task_data = generate_tasks(
            map_data,
            {**TASK_CONFIG, "agent_count": 5},
            5501,
            "export_task",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_map_bundle(output, map_data)
            write_instance_bundle(output, map_data, task_data)
            mapf = (output / "export_task.mapf").read_text(
                encoding="utf-8"
            )
            self.assertTrue(mapf.startswith("28 39\n"))
            self.assertIn("\n5\n", mapf)
            moving_map = (output / "export_map.map").read_text(
                encoding="utf-8"
            )
            self.assertTrue(
                moving_map.startswith("type octile\nheight 28\nwidth 39\nmap\n")
            )
            scenario_lines = (output / "export_task.scen").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(scenario_lines[0], "version 1")
            self.assertEqual(len(scenario_lines), 6)
            fields = scenario_lines[1].split("\t")
            start = task_data.starts[0]
            goal = task_data.goals[0]
            self.assertEqual(fields[1], "export_map.map")
            self.assertEqual((int(fields[4]), int(fields[5])), (start[1], start[0]))
            self.assertEqual((int(fields[6]), int(fields[7])), (goal[1], goal[0]))
            self.assertGreater(int(fields[8]), 0)
            map_sidecar = json.loads(
                (output / "export_map.json").read_text(encoding="utf-8")
            )
            task_sidecar = json.loads(
                (output / "export_task.json").read_text(encoding="utf-8")
            )
            self.assertEqual(map_sidecar["schema_version"], 2)
            self.assertEqual(task_sidecar["schema_version"], 2)
            self.assertEqual(
                task_sidecar["metadata"]["task_semantics_version"], 2
            )
            self.assertIn("<svg", svg_preview(map_data, task_data))
            self.assertNotIn("#e8f5e9", svg_preview(map_data, task_data))
            self.assertNotIn("high prior", svg_preview(map_data, task_data))
            self.assertIn(
                "high prior",
                svg_preview(map_data, task_data, diagnostic=True),
            )
            preview = ascii_preview(map_data, task_data)
            self.assertIn("s", preview)
            self.assertIn("g", preview)

    def test_dataset_uses_exact_layout_and_scenario_quotas(self) -> None:
        config = {
            **EXAMPLE_CONFIG,
            "splits": {
                "train": {
                    "layout_counts": {
                        "regular_beltway": 1,
                        "compartmentalized": 3,
                        "dead_end_aisles": 1,
                    }
                },
                "validation": {
                    "layout_counts": {"regular_beltway": 1}
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            summary = generate_dataset(config, directory)
            self.assertEqual(summary["schema_version"], 2)
            self.assertEqual(summary["task_semantics_version"], 2)
            self.assertEqual(len(summary["configuration_fingerprint"]), 64)
            train = summary["splits"]["train"]
            self.assertEqual(train["map_count"], 5)
            self.assertEqual(train["instance_count"], 20)
            self.assertEqual(
                train["layout_counts"],
                {
                    "compartmentalized": 3,
                    "dead_end_aisles": 1,
                    "regular_beltway": 1,
                },
            )
            self.assertEqual(
                train["scenario_counts"],
                {
                    "balanced_bidirectional": 15,
                    "uniform_random": 5,
                },
            )
            self.assertEqual(
                train["layout_variant_counts"],
                {
                    "cross_four_gate": 1,
                    "double_horizontal": 1,
                    "double_vertical": 1,
                },
            )
            self.assertEqual(
                train["task_variant_counts"],
                {
                    "balanced_base": 5,
                    "balanced_clustered": 5,
                    "balanced_dense": 5,
                    "uniform_control": 5,
                },
            )
            seeds = [
                seed
                for split in summary["splits"].values()
                for seed in split["map_seeds"]
            ]
            self.assertEqual(len(seeds), len(set(seeds)))
            manifest_text = (
                Path(directory) / "train" / "manifest.jsonl"
            ).read_text(encoding="utf-8")
            self.assertEqual(len(manifest_text.splitlines()), 20)
            first_row = json.loads(manifest_text.splitlines()[0])
            self.assertTrue(first_row["map_file"].endswith(".map"))
            self.assertTrue(first_row["scenario_file"].endswith(".scen"))
            self.assertTrue(first_row["map_metadata_file"].endswith(".json"))
            self.assertEqual(first_row["task_schema_version"], 2)
            self.assertEqual(first_row["task_semantics_version"], 2)
            self.assertTrue(
                (Path(directory) / "train" / first_row["map_file"]).is_file()
            )
            self.assertTrue(
                (Path(directory) / "train" / first_row["scenario_file"]).is_file()
            )


if __name__ == "__main__":
    unittest.main()
