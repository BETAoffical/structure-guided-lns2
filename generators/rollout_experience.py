from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .candidate_experience import (
    _read_json,
    _read_jsonl,
    _validate_candidate,
    _validate_conflict_events,
    _validate_path,
    candidate_raw_features,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path, rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _aggregate_rollouts(
    outcomes: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for outcome in outcomes:
        for rollout in outcome["rollouts"]:
            grouped.setdefault(int(rollout["horizon"]), []).append(
                {
                    **rollout,
                    "candidate_valid": outcome["candidate_valid"],
                    "one_step_conflict_reduction": outcome[
                        "one_step_conflict_reduction"
                    ],
                    "one_step_cost_improvement": outcome[
                        "one_step_cost_improvement"
                    ],
                    "conflicting_pairs_before": outcome[
                        "conflicting_pairs_before"
                    ],
                    "sum_of_costs_before": outcome["sum_of_costs_before"],
                }
            )
    aggregated: dict[int, dict[str, Any]] = {}
    for horizon, rows in grouped.items():
        count = len(rows)
        valid_count = sum(row["candidate_valid"] for row in rows)

        def expected(name: str) -> float:
            return sum(
                float(row[name]) if row["candidate_valid"] else 0.0
                for row in rows
            ) / count

        def valid_mean(name: str) -> float | None:
            values = [
                float(row[name])
                for row in rows
                if row["candidate_valid"]
            ]
            if not values:
                return None
            return sum(values) / len(values)

        before_conflicts = int(rows[0]["conflicting_pairs_before"])
        before_cost = int(rows[0]["sum_of_costs_before"])
        after_conflicts = valid_mean("conflicting_pairs_after")
        after_cost = valid_mean("sum_of_costs_after")
        aggregated[horizon] = {
            "candidate_valid": valid_count > 0,
            "valid_probability": valid_count / count,
            "order_trial_count": count,
            "valid_order_trial_count": valid_count,
            "horizon": horizon,
            "conflicting_pairs_before": before_conflicts,
            "conflicting_pairs_after": after_conflicts,
            "conflict_reduction": (
                before_conflicts - after_conflicts
                if after_conflicts is not None
                else 0.0
            ),
            "sum_of_costs_before": before_cost,
            "sum_of_costs_after": after_cost,
            "cost_improvement": (
                before_cost - after_cost if after_cost is not None else 0.0
            ),
            "solved_probability": sum(
                1.0 if row["candidate_valid"] and row["solved"] else 0.0
                for row in rows
            )
            / count,
            "rollout_iterations": valid_mean("iterations"),
            "rollout_accepted_iterations": valid_mean(
                "accepted_iterations"
            ),
            "replan_runtime_ms": 0.0,
            "total_runtime_ms": sum(
                float(row["runtime_ms"]) for row in rows
            )
            / count,
            "one_step_conflict_reduction": sum(
                float(row["one_step_conflict_reduction"])
                if row["candidate_valid"]
                else 0.0
                for row in rows
            )
            / count,
            "one_step_cost_improvement": sum(
                float(row["one_step_cost_improvement"])
                if row["candidate_valid"]
                else 0.0
                for row in rows
            )
            / count,
        }
    return aggregated


def build_rollout_candidate_experience(
    dataset: str | Path,
    collection: str | Path,
    output: str | Path,
    split: str,
) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError(
            "rollout candidate experience is limited to Train and Validation"
        )
    usage = "memory" if split == "train" else "evaluation"
    dataset_root = Path(dataset).resolve()
    collection_root = Path(collection).resolve()
    output_root = Path(output).resolve()
    dataset_rows = _read_jsonl(dataset_root / split / "manifest.jsonl")
    collection_rows = _read_jsonl(
        collection_root / "collection_manifest.jsonl"
    )
    task_rows = {str(row["task_id"]): row for row in dataset_rows}
    cases: list[dict[str, Any]] = []
    trace_schema_versions: set[int] = set()
    horizon_counts: dict[int, int] = {}
    map_cache: dict[str, dict[str, Any]] = {}
    task_cache: dict[str, dict[str, Any]] = {}

    for run in collection_rows:
        if run["split"] != split:
            raise ValueError("collection crosses the requested split")
        if run.get("status") == "error" or run.get("result") is None:
            raise ValueError("collection contains an invalid run")
        task_id = str(run["task_id"])
        if task_id not in task_rows:
            raise ValueError(f"unknown task: {task_id}")
        manifest_row = task_rows[task_id]
        map_id = str(manifest_row["map_id"])
        map_cache.setdefault(
            map_id,
            _read_json(
                dataset_root / split / str(manifest_row["map_file"])
            ),
        )
        task_cache.setdefault(
            task_id,
            _read_json(
                dataset_root / split / str(manifest_row["task_file"])
            ),
        )
        map_document = map_cache[map_id]
        task_document = task_cache[task_id]
        agent_count = int(task_document["metadata"]["agent_count"])
        trace = _read_jsonl(Path(run["trace_file"]))
        trace_schema_versions.update(
            int(row.get("schema_version", -1)) for row in trace
        )
        if (
            not trace
            or trace[-1].get("event_type") != "summary"
            or any(row.get("schema_version") != 6 for row in trace)
            or trace[-1].get("candidate_mode") != "collect"
        ):
            raise ValueError("rollout candidate experience requires Trace V6")
        run_id = f"{task_id}__seed_{int(run['solver_seed']):04d}"
        for iteration in (
            row for row in trace if row["event_type"] == "iteration"
        ):
            paths = iteration["paths_before"]
            events = iteration["conflict_events_before"]
            if len(paths) != agent_count:
                raise ValueError("Trace V6 omitted full current paths")
            _validate_conflict_events(events, map_document, agent_count)
            for agent, path in enumerate(paths):
                _validate_path(
                    path,
                    task_document["starts"][agent],
                    task_document["goals"][agent],
                    map_document["grid"],
                )
            candidates = iteration["candidate_trials"]
            expected_count = int(run.get("candidate_count", 8))
            if expected_count <= 0 or len(candidates) != expected_count:
                raise ValueError(
                    "state does not contain the expected candidate count"
                )
            state_id = (
                f"{run_id}__iteration_"
                f"{int(iteration['iteration']):04d}"
            )
            for candidate in candidates:
                _validate_candidate(
                    candidate,
                    iteration["seed_conflict"],
                    agent_count,
                    int(run["neighborhood_size"]),
                )
                raw_order_trials = candidate.get("order_trials") or []
                if not raw_order_trials:
                    raise ValueError("Trace V6 candidate omitted order trials")
                order_outcomes = []
                for order_trial in raw_order_trials:
                    _validate_candidate(
                        {
                            **candidate,
                            "replan_order": order_trial["replan_order"],
                            "trial_performed": order_trial[
                                "trial_performed"
                            ],
                        },
                        iteration["seed_conflict"],
                        agent_count,
                        int(run["neighborhood_size"]),
                    )
                    valid = bool(order_trial["candidate_valid"])
                    if valid and not order_trial.get("rollouts"):
                        raise ValueError(
                            "valid Trace V6 candidate omitted rollouts"
                        )
                    order_outcomes.append(
                        {
                            "candidate_valid": valid,
                            "conflicting_pairs_before": int(
                                iteration["conflicting_pairs_before"]
                            ),
                            "sum_of_costs_before": int(
                                iteration["sum_of_costs_before"]
                            ),
                            "one_step_conflict_reduction": (
                                int(iteration["conflicting_pairs_before"])
                                - int(
                                    order_trial["conflicting_pairs_after"]
                                )
                                if valid
                                else 0.0
                            ),
                            "one_step_cost_improvement": (
                                int(iteration["sum_of_costs_before"])
                                - int(order_trial["sum_of_costs_after"])
                                if valid
                                else 0.0
                            ),
                            "rollouts": order_trial.get("rollouts", []),
                        }
                    )
                features = candidate_raw_features(
                    map_document,
                    task_document,
                    manifest_row,
                    events,
                    paths,
                    iteration["seed_conflict"],
                    candidate,
                )
                for horizon, outcome in _aggregate_rollouts(
                    order_outcomes
                ).items():
                    horizon_counts[horizon] = horizon_counts.get(horizon, 0) + 1
                    rollout_features = {
                        **features,
                        "candidate.one_step_conflict_reduction": float(
                            outcome["one_step_conflict_reduction"]
                        ),
                        "candidate.one_step_cost_improvement": float(
                            outcome["one_step_cost_improvement"]
                        ),
                        "rollout.horizon": float(horizon),
                    }
                    cases.append(
                        {
                            "schema_version": 1,
                            "usage": usage,
                            "split": split,
                            "label_source": "rollout",
                            "case_id": (
                                f"{state_id}__horizon_{horizon:04d}"
                                f"__candidate_"
                                f"{int(candidate['candidate_index']):02d}"
                            ),
                            "state_id": (
                                f"{state_id}__horizon_{horizon:04d}"
                            ),
                            "source_state_id": state_id,
                            "run_id": run_id,
                            "map_id": map_id,
                            "task_id": task_id,
                            "layout_mode": manifest_row["layout_mode"],
                            "layout_variant": manifest_row.get(
                                "layout_variant"
                            ),
                            "task_variant": manifest_row.get(
                                "task_variant"
                            ),
                            "solver_seed": int(run["solver_seed"]),
                            "iteration": int(iteration["iteration"]),
                            "candidate_index": int(
                                candidate["candidate_index"]
                            ),
                            "generator": str(candidate["generator"]),
                            "seed_conflict": iteration["seed_conflict"],
                            "agents": candidate["agents"],
                            "replan_order": candidate["replan_order"],
                            "features": rollout_features,
                            "outcome": outcome,
                        }
                    )

    cases.sort(key=lambda row: str(row["case_id"]))
    _write_jsonl(output_root / "rollout_candidate_cases.jsonl", cases)
    _write_jsonl(output_root / "candidate_cases.jsonl", cases)
    summary = {
        "schema_version": 1,
        "source_trace_schema_versions": sorted(trace_schema_versions),
        "split": split,
        "usage": usage,
        "label_source": "rollout",
        "collection_run_count": len(collection_rows),
        "state_count": len({row["state_id"] for row in cases}),
        "source_state_count": len({row["source_state_id"] for row in cases}),
        "candidate_case_count": len(cases),
        "candidate_count_per_state": (
            len({int(row["candidate_index"]) for row in cases})
            if cases
            else 0
        ),
        "candidate_generator_profiles": sorted(
            {
                str(row.get("candidate_generator_profile", "full8"))
                for row in collection_rows
                if row.get("candidate_trials")
            }
        ),
        "horizons": sorted(horizon_counts),
        "horizon_case_counts": {
            str(horizon): count
            for horizon, count in sorted(horizon_counts.items())
        },
    }
    _write_json(output_root / "rollout_candidate_summary.json", summary)
    _write_json(output_root / "candidate_summary.json", summary)
    return summary
