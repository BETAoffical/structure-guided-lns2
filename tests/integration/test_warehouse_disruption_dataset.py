from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from generators.dataset import generate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "stride_warehouse_disruption_recovery_dataset_v1.json"
)

EXPECTED_SPLITS = {
    "development": {"maps": 4, "tasks": 16},
    "controller_held_out": {"maps": 2, "tasks": 8},
}
EXPECTED_DENSITIES = {
    "station_rush_d10": 0.1,
    "station_release_d10": 0.1,
    "station_dominant_d125": 0.125,
    "balanced_od_d125": 0.125,
}
EXPECTED_OD_MATRICES = {
    "station_dominant_d125": {
        "storage->station": 0.55,
        "station->storage": 0.25,
        "left->right": 0.1,
        "right->left": 0.1,
    },
    "balanced_od_d125": {
        "storage->station": 0.35,
        "station->storage": 0.35,
        "left->right": 0.15,
        "right->left": 0.15,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_station_centric_disruption_dataset_contract() -> None:
    config = _read_json(CONFIG_PATH)
    assert "map_id_prefix" not in config
    assert config["map"]["rows"] == 48
    assert config["map"]["cols"] == 72
    assert config["map"]["layout_mode"] == "station_centric"
    assert config["task"]["density_reference"] == "free_cells"

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        summary = generate_dataset(config, output)

        rows: list[dict[str, Any]] = []
        for split, expected in EXPECTED_SPLITS.items():
            split_summary = summary["splits"][split]
            assert split_summary["map_count"] == expected["maps"]
            assert split_summary["instance_count"] == expected["tasks"]
            assert split_summary["tasks_per_map"] == 4
            assert split_summary["layout_counts"] == {
                "station_centric": expected["maps"]
            }
            assert split_summary["task_variant_counts"] == {
                name: expected["maps"] for name in EXPECTED_DENSITIES
            }

            split_rows = _read_jsonl(output / split / "manifest.jsonl")
            assert len(split_rows) == expected["tasks"]
            assert {row["split"] for row in split_rows} == {split}
            rows.extend(split_rows)

        assert len(rows) == 24
        assert len({row["task_id"] for row in rows}) == 24
        assert {row["task_variant"] for row in rows} == set(
            EXPECTED_DENSITIES
        )

        map_rows: dict[str, dict[str, Any]] = {}
        for row in rows:
            existing = map_rows.setdefault(row["map_id"], row)
            assert existing["split"] == row["split"]
            assert existing["map_file"] == row["map_file"]

        assert len(map_rows) == 6
        assert len({row["map_id"] for row in map_rows.values()}) == 6
        assert {
            split: sum(row["split"] == split for row in map_rows.values())
            for split in EXPECTED_SPLITS
        } == {"development": 4, "controller_held_out": 2}

        map_hashes: set[str] = set()
        map_documents: dict[str, dict[str, Any]] = {}
        for map_id, row in map_rows.items():
            split_root = output / row["split"]
            map_path = split_root / row["map_file"]
            map_metadata_path = split_root / row["map_metadata_file"]
            map_hashes.add(_sha256(map_path))

            document = _read_json(map_metadata_path)
            map_documents[map_id] = document
            metadata = document["metadata"]
            parameters = metadata["sampled_parameters"]
            station_approach = metadata["zones"]["station_approach"]

            assert document["rows"] == 48
            assert document["cols"] == 72
            assert parameters["layout_mode"] == "station_centric"
            assert parameters["station_placement"] == "clustered"
            assert parameters["station_count"] == 6
            assert parameters["station_queue_depth"] == 8
            assert parameters["station_queue_width"] == 5
            assert parameters["station_demand_distribution"] == "zipf"
            assert len(metadata["stations"]) == 6
            assert len(metadata["station_zones"]) == 6
            assert station_approach
            assert len(station_approach) >= round(
                0.1 * metadata["free_cell_count"]
            )

        assert len(map_hashes) == 6

        for row in rows:
            split_root = output / row["split"]
            task = _read_json(split_root / row["task_file"])
            metadata = task["metadata"]
            map_metadata = map_documents[row["map_id"]]["metadata"]
            free_cell_count = int(map_metadata["free_cell_count"])
            variant = row["task_variant"]
            expected_density = EXPECTED_DENSITIES[variant]
            expected_agent_count = round(expected_density * free_cell_count)

            assert metadata["agent_count"] == expected_agent_count
            assert len(task["starts"]) == expected_agent_count
            assert len(task["goals"]) == expected_agent_count
            assert len({tuple(cell) for cell in task["starts"]}) == (
                expected_agent_count
            )
            assert len({tuple(cell) for cell in task["goals"]}) == (
                expected_agent_count
            )
            assert abs(
                metadata["agent_density_free_cells"]
                - expected_agent_count / free_cell_count
            ) <= 1e-6

            station_cells = {
                tuple(cell)
                for cell in map_metadata["zones"]["station_approach"]
            }
            flow_counts = metadata["realized_flow_counts"]
            assert sum(flow_counts.values()) == expected_agent_count

            if variant == "station_rush_d10":
                assert metadata["od_matrix"] is None
                assert flow_counts == {
                    "storage_to_station": expected_agent_count
                }
                assert all(tuple(goal) in station_cells for goal in task["goals"])
            elif variant == "station_release_d10":
                assert metadata["od_matrix"] is None
                assert flow_counts == {
                    "station_to_storage": expected_agent_count
                }
                assert all(
                    tuple(start) in station_cells for start in task["starts"]
                )
            else:
                assert metadata["od_matrix"] == EXPECTED_OD_MATRICES[variant]
                assert set(flow_counts) == set(EXPECTED_OD_MATRICES[variant])
                assert flow_counts["storage->station"] > 0
                assert flow_counts["station->storage"] > 0
