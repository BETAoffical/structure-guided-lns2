from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.closed_loop_confirmation import run_closed_loop_collection  # noqa: E402
from experiments.repair_collection import _read_jsonl  # noqa: E402
from experiments.stride_maze_tail_state_collection import (  # noqa: E402
    _fused_controller_kwargs,
)
from experiments.stride_onpolicy_controller_attribution import (  # noqa: E402
    _episode_override,
    _parent_from_exact,
    _qualification_roots,
    prepare_tasks,
)
from lns2_selector.runtime.hybridstructpool import (  # noqa: E402
    hybridstructpool_runtime_augmentation,
)


DEFAULT_CONFIG = (
    ROOT / "configs/stride_onpolicy_controller_attribution_v1_r2_registration.json"
)


def run_smoke(config_path: Path, output: Path, *, task_index: int) -> dict:
    loaded, tasks = prepare_tasks(config_path)
    _path, root, _config, inputs, exact_loaded, _all_tasks = loaded
    if not 0 <= int(task_index) < len(tasks):
        raise ValueError("task index is outside the registered cohort")
    task = dict(tasks[int(task_index)])
    parent = _parent_from_exact(exact_loaded)
    source_cases = list(exact_loaded[-1])
    key = (str(task["task_id"]), int(task["solver_seed"]))
    all_keys = {
        (str(row["task_id"]), int(row["solver_seed"])) for row in tasks
    }
    override = _episode_override(task, trial_index=0, source_cases=source_cases)
    dataset = (root / str(parent["cohort"]["dataset"])).resolve()
    controller_kwargs = _fused_controller_kwargs(root, parent, "v2-full")
    qualification_source = _qualification_roots(inputs)["slot"]
    summaries: dict[str, dict] = {}
    for arm in ("v2_full", "hybridstructpool_full"):
        collection = output / arm
        common = {
            "cohort_job_keys": all_keys,
            "job_keys": {key},
            "episode_overrides": {key: override},
            "qualification_source": qualification_source,
            "use_global_collection_lock": False,
            "repair_seed_policy": "episode_stream",
            "deterministic_pp_replay": False,
            **controller_kwargs,
        }
        if arm == "hybridstructpool_full":
            common["hybridstructpool_augmentation"] = (
                hybridstructpool_runtime_augmentation()
            )
        run_closed_loop_collection(
            dataset,
            inputs["runtime_config"],
            collection,
            phase="qualify",
            workers=1,
            resume=collection.joinpath("run_config.json").is_file(),
            cohort_job_keys=all_keys,
            job_keys=all_keys,
            episode_overrides={key: override},
            qualification_source=qualification_source,
            use_global_collection_lock=False,
            repair_seed_policy="episode_stream",
            deterministic_pp_replay=False,
            **controller_kwargs,
            **(
                {
                    "hybridstructpool_augmentation": (
                        hybridstructpool_runtime_augmentation()
                    )
                }
                if arm == "hybridstructpool_full"
                else {}
            ),
        )
        run_closed_loop_collection(
            dataset,
            inputs["runtime_config"],
            collection,
            phase="realized_dynamic",
            workers=1,
            resume=True,
            **common,
        )
        manifests = _read_jsonl(collection / "realized_dynamic_manifest.jsonl")
        matches = [
            row
            for row in manifests
            if str(row.get("task_id")) == key[0]
            and int(row.get("solver_seed", -1)) == key[1]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{arm} smoke manifest is missing or ambiguous")
        manifest = dict(matches[0])
        summary = dict(manifest["summary"])
        totals = dict(summary["controller_totals"])
        summaries[arm] = {
            "status": str(manifest["status"]),
            "success": bool(summary["success"]),
            "repair_iterations": int(summary["repair_iterations"]),
            "normalized_fixed_budget_conflict_auc": float(
                summary["normalized_fixed_budget_conflict_auc"]
            ),
            "capped_wall_time_to_feasible": float(
                summary["capped_wall_time_to_feasible"]
            ),
            "controller_seconds_before_repair": float(
                totals["controller_seconds_before_repair"]
            ),
            "repair_wall_seconds": float(summary["repair_wall_seconds"]),
            "selected_family_counts": dict(summary["selected_family_counts"]),
            "trace_sha256": str(manifest["trace_sha256"]),
        }
    result = {
        "schema": "lns2.stride.hybridstructpool_runtime_smoke.v1",
        "task_index": int(task_index),
        "task_id": key[0],
        "solver_seed": key[1],
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "rescue_enabled": False,
        "arms": summaries,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    arguments = parser.parse_args()
    report = run_smoke(
        arguments.config.resolve(),
        arguments.output.resolve(),
        task_index=arguments.task_index,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
