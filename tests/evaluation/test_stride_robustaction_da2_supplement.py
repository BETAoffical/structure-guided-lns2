from __future__ import annotations

import unittest

from experiments.stride_robustaction_da2_supplement import (
    select_static_da2_maps,
)


def _row(map_id: str, component: int, ratio: float) -> dict[str, object]:
    return {
        "map_id": map_id,
        "member": f"{map_id}.map",
        "member_sha256": "0" * 64,
        "free_cell_count": component,
        "largest_four_connected_component": component,
        "static_low_degree_cell_ratio": ratio,
        "static_obstacle_ratio": 0.5,
    }


class RobustActionDA2SupplementTests(unittest.TestCase):
    def test_static_selection_filters_scenes_and_deduplicates_families(self) -> None:
        static = {
            "minimum_largest_component": 3000,
            "maximum_largest_component": 30000,
            "target_largest_component": 8000,
            "blocked_scene_prefixes": ["old_scene"],
            "topology_thresholds": [0.035, 0.06],
            "selected_group_counts": {
                "dao_high_topology": 1,
                "dao_mid_topology": 1,
                "dao_low_topology_control": 1,
            },
        }
        rows = [
            _row("aa_best", 7900, 0.07),
            _row("aa_worse_same_family", 8050, 0.061),
            _row("bb_high", 8200, 0.08),
            _row("cc_mid", 7950, 0.04),
            _row("dd_low", 8100, 0.02),
            _row("old_scene_variant", 8000, 0.10),
            _row("too_small", 2999, 0.10),
        ]

        candidates, selected = select_static_da2_maps(rows, static)

        self.assertEqual(len(candidates), 5)
        self.assertEqual(
            [row["map_id"] for row in selected],
            ["aa_worse_same_family", "cc_mid", "dd_low"],
        )
        self.assertNotIn(
            "old_scene_variant", {row["map_id"] for row in candidates}
        )

    def test_static_selection_is_repeatable_for_ties(self) -> None:
        static = {
            "minimum_largest_component": 3000,
            "maximum_largest_component": 30000,
            "target_largest_component": 8000,
            "blocked_scene_prefixes": [],
            "topology_thresholds": [0.035, 0.06],
            "selected_group_counts": {
                "dao_high_topology": 1,
                "dao_mid_topology": 0,
                "dao_low_topology_control": 0,
            },
        }
        rows = [_row("zz_tie", 8000, 0.07), _row("aa_tie", 8000, 0.07)]

        _candidates, selected = select_static_da2_maps(rows, static)

        self.assertEqual([row["map_id"] for row in selected], ["aa_tie"])


if __name__ == "__main__":
    unittest.main()
