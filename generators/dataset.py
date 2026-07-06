from __future__ import annotations

import json
import random
import collections
from pathlib import Path
from typing import Any

from .config import merge_dicts
from .io import write_instance_bundle, write_map_bundle
from .task_flows import generate_tasks
from .validation import validate_map, validate_task
from .warehouse import generate_warehouse


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _layout_schedule(split_config: dict[str, Any]) -> list[str | None]:
    layout_counts_value = split_config.get("layout_counts")
    if layout_counts_value is None:
        map_count = int(split_config.get("maps", 1))
        if map_count <= 0:
            raise ValueError("maps must be positive")
        return [None] * map_count

    layout_counts = dict(layout_counts_value)
    if not layout_counts:
        raise ValueError("layout_counts must not be empty")
    schedule: list[str | None] = []
    for layout_mode, count_value in layout_counts.items():
        count = int(count_value)
        if count <= 0:
            raise ValueError("layout_counts values must be positive")
        schedule.extend([str(layout_mode)] * count)
    if "maps" in split_config and int(split_config["maps"]) != len(schedule):
        raise ValueError("maps must equal the sum of layout_counts")
    return schedule


def _task_schedule(
    config: dict[str, Any],
    split_config: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks_per_map = int(
        split_config.get(
            "tasks_per_map",
            config.get("tasks_per_map", 1),
        )
    )
    if tasks_per_map <= 0:
        raise ValueError("tasks_per_map must be positive")
    variants_value = split_config.get(
        "task_variants", config.get("task_variants")
    )
    if variants_value is not None:
        variants = [dict(item) for item in list(variants_value)]
        if len(variants) != tasks_per_map:
            raise ValueError(
                "task_variants length must equal tasks_per_map"
            )
        names = [str(item.get("name", "")) for item in variants]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                "task_variants must have unique non-empty names"
            )
        for item in variants:
            item["name"] = str(item["name"])
            item["task"] = dict(item.get("task", {}))
        return variants

    scenario_value = split_config.get(
        "task_scenarios", config.get("task_scenarios")
    )
    if scenario_value is None:
        return [
            {"name": f"task_{index:04d}", "task": {}}
            for index in range(tasks_per_map)
        ]
    scenarios = [str(item) for item in list(scenario_value)]
    if len(scenarios) != tasks_per_map:
        raise ValueError(
            "task_scenarios length must equal tasks_per_map"
        )
    return [
        {
            "name": scenario,
            "task": {"scenario_type": scenario},
        }
        for scenario in scenarios
    ]


def _variant_schedules(
    config: dict[str, Any],
    split_configs: dict[str, Any],
    master_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    layout_totals: collections.Counter[str] = collections.Counter()
    for split_value in split_configs.values():
        for layout_mode in _layout_schedule(dict(split_value)):
            if layout_mode is not None:
                layout_totals[layout_mode] += 1

    schedules: dict[str, list[dict[str, Any]]] = {}
    rng = random.Random(master_seed ^ 0x4C41594F)
    for layout_mode, variants_value in dict(
        config.get("layout_variants", {})
    ).items():
        variants = [dict(item) for item in list(variants_value)]
        if not variants:
            raise ValueError("layout variant lists must not be empty")
        names = [str(item.get("name", "")) for item in variants]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                "layout variants must have unique non-empty names"
            )
        total = layout_totals.get(layout_mode, 0)
        if total % len(variants) != 0:
            raise ValueError(
                f"{layout_mode} count must be divisible by its "
                "layout variant count"
            )
        schedule = []
        for item in variants:
            normalized = {
                "name": str(item["name"]),
                "map": dict(item.get("map", {})),
            }
            schedule.extend([normalized] * (total // len(variants)))
        rng.shuffle(schedule)
        schedules[str(layout_mode)] = schedule
    return schedules


def generate_dataset(
    config: dict[str, Any], output_override: str | Path | None = None
) -> dict[str, Any]:
    output_root = Path(
        output_override
        if output_override is not None
        else config.get("output_dir", "data/stage1")
    )
    master_seed = int(config.get("master_seed", 2026))
    seed_rng = random.Random(master_seed)
    base_map_config = dict(config.get("map", {}))
    base_task_config = dict(config.get("task", {}))
    split_configs = dict(config.get("splits", {}))
    if not split_configs:
        raise ValueError("dataset config must define at least one split")

    all_map_seeds: set[int] = set()
    variant_schedules = _variant_schedules(
        config, split_configs, master_seed
    )
    variant_indices: collections.Counter[str] = collections.Counter()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "master_seed": master_seed,
        "output_dir": str(output_root),
        "splits": {},
    }

    for split_name, split_config_value in split_configs.items():
        split_config = dict(split_config_value)
        layout_schedule = _layout_schedule(split_config)
        task_schedule = _task_schedule(config, split_config)
        map_count = len(layout_schedule)
        tasks_per_map = len(task_schedule)
        map_config = merge_dicts(
            base_map_config, split_config.get("map")
        )
        task_config = merge_dicts(
            base_task_config, split_config.get("task")
        )
        split_root = output_root / split_name
        manifest: list[dict[str, Any]] = []
        split_map_seeds: list[int] = []

        layout_indices: dict[str, int] = {}
        for map_index, layout_mode in enumerate(layout_schedule):
            map_seed = seed_rng.randrange(1, 2**31)
            while map_seed in all_map_seeds:
                map_seed = seed_rng.randrange(1, 2**31)
            all_map_seeds.add(map_seed)
            split_map_seeds.append(map_seed)
            current_map_config = dict(map_config)
            layout_variant: str | None = None
            if layout_mode is None:
                map_id = f"{split_name}_warehouse_{map_index:04d}"
            else:
                layout_index = layout_indices.get(layout_mode, 0)
                layout_indices[layout_mode] = layout_index + 1
                current_map_config["layout_mode"] = layout_mode
                variant_schedule = variant_schedules.get(layout_mode)
                if variant_schedule is not None:
                    variant_index = variant_indices[layout_mode]
                    variant_indices[layout_mode] += 1
                    variant = variant_schedule[variant_index]
                    layout_variant = str(variant["name"])
                    current_map_config = merge_dicts(
                        current_map_config, variant["map"]
                    )
                    map_id = (
                        f"{split_name}_{layout_mode}_"
                        f"{layout_variant}_{layout_index:04d}"
                    )
                else:
                    map_id = (
                        f"{split_name}_{layout_mode}_{layout_index:04d}"
                    )
            map_data = generate_warehouse(
                current_map_config, map_seed, map_id
            )
            validate_map(map_data)
            write_map_bundle(split_root / "maps", map_data)
            structural_changes = map_data.metadata[
                "structural_changes"
            ]
            dividers = structural_changes["compartment_gates"]
            dead_end_orientations: collections.Counter[str] = (
                collections.Counter(
                    item.get("orientation", "vertical")
                    for item in structural_changes["dead_end_caps"]
                )
            )

            for task_index, task_variant in enumerate(task_schedule):
                task_seed = seed_rng.randrange(1, 2**31)
                task_id = f"{map_id}__task_{task_index:04d}"
                current_task_config = merge_dicts(
                    task_config, task_variant["task"]
                )
                task_data = generate_tasks(
                    map_data, current_task_config, task_seed, task_id
                )
                validate_task(map_data, task_data)
                write_instance_bundle(
                    split_root / "instances", map_data, task_data
                )
                manifest.append(
                    {
                        "split": split_name,
                        "map_id": map_id,
                        "task_id": task_id,
                        "map_seed": map_seed,
                        "task_seed": task_seed,
                        "map_file": f"maps/{map_id}.json",
                        "instance_file": f"instances/{task_id}.mapf",
                        "task_file": f"instances/{task_id}.json",
                        "map_parameters": map_data.metadata[
                            "sampled_parameters"
                        ],
                        "layout_mode": map_data.metadata[
                            "sampled_parameters"
                        ]["layout_mode"],
                        "layout_variant": layout_variant,
                        "divider_wall_count": len(dividers),
                        "gate_count": sum(
                            len(item["gate_cells"]) for item in dividers
                        ),
                        "horizontal_dead_end_count": (
                            dead_end_orientations["horizontal"]
                        ),
                        "vertical_dead_end_count": (
                            dead_end_orientations["vertical"]
                        ),
                        "topology_metrics": map_data.metadata[
                            "topology_metrics"
                        ],
                        "flow_type": task_data.metadata["flow_type"],
                        "scenario_type": task_data.metadata[
                            "scenario_type"
                        ],
                        "task_variant": task_variant["name"],
                        "dominant_flow_ratio": task_data.metadata[
                            "dominant_flow_ratio"
                        ],
                        "hotspot_skew": task_data.metadata["hotspot_skew"],
                        "required_bottleneck_crossing_ratio": (
                            task_data.metadata[
                                "required_bottleneck_crossing_ratio"
                            ]
                        ),
                        "agent_count": task_data.agent_count,
                        "mean_shortest_distance": task_data.metadata[
                            "mean_shortest_distance"
                        ],
                    }
                )

        _write_jsonl(split_root / "manifest.jsonl", manifest)
        summary["splits"][split_name] = {
            "map_count": map_count,
            "instance_count": len(manifest),
            "tasks_per_map": tasks_per_map,
            "map_seeds": split_map_seeds,
            "layout_counts": {
                layout: sum(
                    row["layout_mode"] == layout for row in manifest
                )
                // tasks_per_map
                for layout in sorted(
                    {row["layout_mode"] for row in manifest}
                )
            },
            "scenario_counts": {
                scenario: sum(
                    row["scenario_type"] == scenario for row in manifest
                )
                for scenario in sorted(
                    {row["scenario_type"] for row in manifest}
                )
            },
            "layout_variant_counts": {
                variant: sum(
                    row["layout_variant"] == variant for row in manifest
                )
                // tasks_per_map
                for variant in sorted(
                    {
                        row["layout_variant"]
                        for row in manifest
                        if row["layout_variant"] is not None
                    }
                )
            },
            "task_variant_counts": {
                variant: sum(
                    row["task_variant"] == variant for row in manifest
                )
                for variant in sorted(
                    {row["task_variant"] for row in manifest}
                )
            },
            "agent_count_min": min(row["agent_count"] for row in manifest),
            "agent_count_max": max(row["agent_count"] for row in manifest),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
