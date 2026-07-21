from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.closed_loop_confirmation import run_closed_loop_collection  # noqa: E402
from experiments.high_load_rescue import collect_high_load_rescue_data  # noqa: E402
from experiments.high_load_rescue_training import (  # noqa: E402
    train_high_load_rescue_controller,
)
from experiments.repair_collection import _read_json, _read_jsonl, _write_json  # noqa: E402
from generators.config import load_json  # noqa: E402
from generators.dataset import generate_dataset  # noqa: E402


def _source_config(split: str, output_root: Path, dataset: Path) -> Path:
    base = _read_json(PROJECT_ROOT / "configs" / "closed_loop_multiseed_collection.json")
    rows = _read_jsonl(dataset / split / "manifest.jsonl")
    map_layouts = {
        str(row["map_id"]): str(row["layout_mode"])
        for row in rows
    }
    task_counts = collections.Counter(str(row["map_id"]) for row in rows)
    tasks_per_map = set(task_counts.values())
    if len(tasks_per_map) != 1:
        raise ValueError(f"{split} does not use a constant tasks-per-map design")
    base.update(
        {
            "formal": False,
            "split": split,
            "solver_seeds": [0],
            "policies": ["official_adaptive", "realized_dynamic"],
            "dataset_design": {
                "map_count": len(map_layouts),
                "tasks_per_map": next(iter(tasks_per_map)),
                "task_variants": sorted(
                    {str(row["task_variant"]) for row in rows}
                ),
                "layout_counts": dict(collections.Counter(map_layouts.values())),
            },
            "environment": {
                "time_limit": 120.0,
                "max_repair_iterations": 30,
                "neighborhood_size": 8,
                "replan_algorithm": "PP",
                "use_sipp": True,
            },
            "qualification": {
                "minimum_nonzero_states": 1,
                "minimum_nonzero_states_per_layout": 0,
                "minimum_active_maps": 1,
            },
            "max_decisions": 30,
            "metric_iteration_budget": 30,
            "wall_time_budget_seconds": 120.0,
            "episode_process_timeout_seconds": 180.0,
            "workers": 4,
            "reference_datasets": [],
        }
    )
    path = output_root / "protocol" / f"source_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, base)
    return path


def _pilot_tasks(dataset: Path, split: str) -> list[str]:
    rows = _read_jsonl(dataset / split / "manifest.jsonl")
    by_layout: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        by_layout.setdefault(str(row["layout_mode"]), {}).setdefault(
            str(row["map_id"]), []
        ).append(row)
    selected: list[str] = []
    maps_per_layout = 4 if split == "policy_train" else 2
    for layout in sorted(by_layout):
        for map_id in sorted(by_layout[layout])[:maps_per_layout]:
            tasks = sorted(by_layout[layout][map_id], key=lambda row: str(row["task_id"]))
            # Source episodes are cheap compared with paired PP arms. Running
            # all four dense 400/600 variants gives the pilot enough distinct
            # natural no-progress states while the collection still caps the
            # expensive paired phase at 48 train and 12 validation states.
            selected.extend(str(task["task_id"]) for task in tasks)
    return selected


def _write_status(root: Path, **values: Any) -> None:
    _write_json(
        root / "status.json",
        {"schema": "lns2.high_load_rescue_pipeline.v1", **values},
    )


def run_pipeline(
    *,
    mode: str,
    output: Path,
    resume: bool,
    workers: int,
    dataset_override: Path | None = None,
    dataset_config: Path | None = None,
    neighborhood_sizes: tuple[int, ...] = (4, 8, 12, 16),
) -> dict[str, Any]:
    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be pilot or full")
    if output.is_dir() and any(output.iterdir()) and not resume:
        raise ValueError("output already exists and is non-empty; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    dataset = dataset_override.resolve() if dataset_override is not None else output / "dataset"
    _write_status(output, phase="dataset", status="running")
    if dataset_override is not None and not (dataset / "dataset_summary.json").is_file():
        raise FileNotFoundError(f"dataset override is incomplete: {dataset}")
    if dataset_override is None and not (dataset / "dataset_summary.json").is_file():
        config_path = (
            dataset_config.resolve()
            if dataset_config is not None
            else PROJECT_ROOT
            / "configs"
            / (
                "high_load_rescue_dataset_full.json"
                if mode == "full"
                else "high_load_rescue_dataset.json"
            )
        )
        generate_dataset(load_json(config_path), dataset)

    source_roots: dict[str, Path] = {}
    for split in ("policy_train", "policy_validation"):
        _write_status(output, phase=f"source-{split}", status="running")
        config = _source_config(split, output, dataset)
        source = output / "sources" / split
        task_ids = _pilot_tasks(dataset, split) if mode == "pilot" else None
        common = {
            "workers": workers,
            "task_ids": task_ids,
            "controller": "v2-full",
            "feature_backend": "native",
            "controller_bundle": PROJECT_ROOT
            / "artifacts"
            / "initlns-closed-loop-controller-v2",
            "controller_runtime": "optimized",
            "verification_profile": "deployment",
            "stopping_rule": "historical",
        }
        run_closed_loop_collection(
            dataset,
            config,
            source,
            phase="qualify",
            resume=resume or (source / "run_config.json").is_file(),
            **common,
        )
        run_closed_loop_collection(
            dataset,
            config,
            source,
            phase="realized_dynamic",
            resume=True,
            **common,
        )
        source_roots[split] = source

    collection = output / "collection"
    maximum_states = (
        {"policy_train": 48, "policy_validation": 12}
        if mode == "pilot"
        else {"policy_train": 800, "policy_validation": 200}
    )
    _write_status(output, phase="paired-rescue-collection", status="running")
    collection_report = collect_high_load_rescue_data(
        source_roots=source_roots,
        output=collection,
        controller_bundle=PROJECT_ROOT
        / "artifacts"
        / "initlns-closed-loop-controller-v2",
        maximum_states=maximum_states,
        neighborhood_sizes=neighborhood_sizes,
        initial_trials=2,
        maximum_trials=2 if mode == "pilot" else 6,
        workers=workers,
        resume=resume or (collection / "run_config.json").is_file(),
    )
    if not bool(collection_report["complete"]):
        _write_status(output, phase="paired-rescue-collection", status="error")
        return {"complete": False, "collection": collection_report}
    observed_states = dict(collection_report["state_count_by_split"])
    if any(
        int(observed_states.get(split, 0)) < minimum
        for split, minimum in maximum_states.items()
    ):
        report = {
            "schema": "lns2.high_load_rescue_pipeline.v1",
            "mode": mode,
            "complete": False,
            "status": "insufficient_failure_states",
            "required_state_count_by_split": maximum_states,
            "observed_state_count_by_split": observed_states,
            "collection": collection_report,
        }
        _write_json(output / "pipeline_report.json", report)
        _write_status(output, phase="paired-rescue-collection", status="insufficient")
        return report

    _write_status(output, phase="training", status="running")
    training = train_high_load_rescue_controller(
        feature_index=collection / "feature_index.jsonl",
        trial_manifest=collection / "trial_manifest.jsonl",
        controller_bundle=PROJECT_ROOT
        / "artifacts"
        / "initlns-closed-loop-controller-v2",
        output=output / "controller",
    )
    report = {
        "schema": "lns2.high_load_rescue_pipeline.v1",
        "mode": mode,
        "complete": True,
        "collection": collection_report,
        "training": training,
        # A failed size-12 gate drops only that experimental branch. It must
        # not block the requested 4/8/16 high-load rescue model.
        "rescue_full_collection_recommended": True,
        "size12_full_collection_recommended": bool(
            collection_report["size12_pilot_gate"]["passed"]
        ),
        "recommended_full_neighborhood_sizes": (
            [4, 8, 12, 16]
            if bool(collection_report["size12_pilot_gate"]["passed"])
            else [4, 8, 16]
        ),
    }
    _write_json(output / "pipeline_report.json", report)
    _write_status(output, phase="complete", status="complete")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the synthetic 400/600-agent rescue pilot or full collection."
    )
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument(
        "--output", default="build/initlns-high-load-rescue-pilot-v1"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-config")
    parser.add_argument(
        "--neighborhood-sizes",
        default="4,8,12,16",
        help="sorted comma-separated rescue candidate sizes",
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    report = run_pipeline(
        mode=arguments.mode,
        output=output.resolve(),
        resume=arguments.resume,
        workers=arguments.workers,
        dataset_override=(
            (PROJECT_ROOT / arguments.dataset).resolve()
            if arguments.dataset and not Path(arguments.dataset).is_absolute()
            else Path(arguments.dataset).resolve()
            if arguments.dataset
            else None
        ),
        dataset_config=(
            (PROJECT_ROOT / arguments.dataset_config).resolve()
            if arguments.dataset_config
            and not Path(arguments.dataset_config).is_absolute()
            else Path(arguments.dataset_config).resolve()
            if arguments.dataset_config
            else None
        ),
        neighborhood_sizes=tuple(
            int(value.strip())
            for value in str(arguments.neighborhood_sizes).split(",")
            if value.strip()
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["complete"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
