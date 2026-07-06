from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generators.config import load_json  # noqa: E402
from generators.io import write_instance_bundle, write_map_bundle  # noqa: E402
from generators.task_flows import generate_tasks  # noqa: E402
from generators.validation import validate_map, validate_task  # noqa: E402
from generators.warehouse import generate_warehouse  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate one structured warehouse MAPF instance."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()

    config = load_json(arguments.config)
    seed = (
        arguments.seed
        if arguments.seed is not None
        else int(config.get("master_seed", 2026))
    )
    map_id = str(config.get("map_id", "warehouse_example"))
    task_id = str(config.get("task_id", f"{map_id}__task_0000"))
    map_data = generate_warehouse(config["map"], seed, map_id)
    task_data = generate_tasks(
        map_data, config["task"], seed + 1, task_id
    )
    validate_map(map_data)
    validate_task(map_data, task_data)
    output = Path(arguments.output)
    write_map_bundle(output / "maps", map_data)
    write_instance_bundle(output / "instances", map_data, task_data)
    print(output / "instances" / f"{task_id}.mapf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
