#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import sha256_file  # noqa: E402
from experiments.balanced_wall_clock import _write_jsonl_atomic  # noqa: E402
from experiments.repair_collection import _read_json, _write_json  # noqa: E402


SCHEMA = "lns2.stride.structpool_map_swap_raw_source.v1"


def materialize(config_path: Path, output: Path) -> dict[str, object]:
    config_path = config_path.resolve()
    output = output.resolve()
    config = _read_json(config_path)
    if (
        config.get("schema") != SCHEMA
        or config.get("scientific_status")
        != "input_only_map_selection_before_new_resets"
        or config.get("selection_basis") != "static_map_scale_and_topology_only"
        or set(config.get("forbidden_selection_inputs") or ())
        != {"initial_conflicts", "repair_outcomes", "controller_outcomes", "ttf"}
    ):
        raise ValueError("map-swap raw source contract changed")
    rows = []
    maps_dir = output / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for raw in config.get("maps") or ():
        row = dict(raw)
        source = (PROJECT_ROOT / str(row["source_path"])).resolve()
        if not source.is_file() or sha256_file(source) != str(row["sha256"]):
            raise ValueError(f"map-swap source changed: {source}")
        destination = maps_dir / f"{row['id']}.map"
        shutil.copy2(source, destination)
        if sha256_file(destination) != str(row["sha256"]):
            raise ValueError(f"map-swap copy changed: {destination}")
        rows.append(
            {
                "id": str(row["id"]),
                "layout_family": str(row["layout_family"]),
                "map_file": f"maps/{destination.name}",
                "map_sha256": str(row["sha256"]),
                "agent_counts": list(map(int, row["agent_counts"])),
                "selection_basis": str(row["input_only_reason"]),
            }
        )
    rows.sort(key=lambda value: value["id"])
    _write_jsonl_atomic(output / "manifest.jsonl", rows)
    report = {
        "schema": "lns2.stride.structpool_map_swap_raw_source_report.v1",
        "config_sha256": sha256_file(config_path),
        "map_count": len(rows),
        "map_ids": [row["id"] for row in rows],
        "outcomes_read": False,
        "manifest_sha256": sha256_file(output / "manifest.jsonl"),
    }
    _write_json(output / "source_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize pinned map-only swap sources.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = materialize(Path(arguments.config), Path(arguments.output))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
