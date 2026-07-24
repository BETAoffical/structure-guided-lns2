from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.high_load_rescue_training import _balanced_map_folds  # noqa: E402
from experiments.repair_collection import _read_json, _read_jsonl, _write_json  # noqa: E402
from experiments.v3_controller import load_v3_controller_bundle  # noqa: E402
from experiments.v3_horizon_training import (  # noqa: E402
    _fit,
    _h1_rows,
    _h3_rows,
    _predict,
    _selected,
    _states,
)
from experiments.v3_training import _index_predictions  # noqa: E402


CONTROLLERS = ("v3-h3", "v2", "adaptive")


def _mean(rows: Iterable[float]) -> float:
    values = list(rows)
    return statistics.fmean(values) if values else 0.0


def _sum(rows: Iterable[float]) -> float:
    return math.fsum(rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _final_model_predictions(
    bundle: Any, rows: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, list[float]]]:
    model = [row for row in rows if str(row["route"]) == "model"]
    dense = [
        {
            "feature_profile": "realized_dynamic",
            "feature_names": tuple(bundle.manifest["feature_names"]),
            "feature_values": tuple(row["features"]),
        }
        for row in model
    ]
    return _index_predictions(model, bundle.predict(dense))


def _oof_predictions(
    h1_rows: list[dict[str, Any]], h3_rows: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, list[float]]]:
    result: dict[tuple[str, str], dict[str, list[float]]] = {}
    for fold in _balanced_map_folds(h3_rows):
        held_maps = set(fold["validation_maps"])
        models = _fit(
            [row for row in h1_rows if row["map_id"] not in held_maps],
            [row for row in h3_rows if row["map_id"] not in held_maps],
        )
        held = [
            row
            for row in h3_rows
            if row["map_id"] in held_maps and str(row["route"]) == "model"
        ]
        indexed = _index_predictions(held, _predict(models, held))
        if set(indexed) & set(result):
            raise ValueError("OOF prediction overlap while building detailed report")
        result.update(indexed)
    return result


def _state_selection_rows(
    states: list[dict[str, Any]],
    trials: dict[tuple[str, str], list[dict[str, Any]]],
    thresholds: dict[str, float],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in states:
        for controller, kind in (
            ("v3-h3", "h3"),
            ("v2", "v2"),
            ("adaptive", "adaptive"),
        ):
            arm = _selected(state, kind, thresholds)
            selected_trials = trials[(str(state["state_id"]), str(arm["candidate_id"]))]
            h3_rows = [dict(row["h3"]) for row in selected_trials]
            h1_rows = [dict(row["h1"]) for row in selected_trials]
            executed = [int(row["executed_steps"]) for row in selected_trials]
            initial = [int(row["conflict_trajectory"][0]) for row in selected_trials]
            final = [int(row["conflict_trajectory"][-1]) for row in selected_trials]
            controller_seconds = _mean(float(row["controller_seconds"]) for row in h3_rows)
            pp_seconds = _mean(float(row["pp_replan_seconds"]) for row in h3_rows)
            repair_seconds = _mean(float(row["repair_seconds"]) for row in h3_rows)
            total_seconds = _mean(float(row["total_seconds"]) for row in h3_rows)
            reduction = _mean(float(row["conflict_reduction"]) for row in h3_rows)
            rows.append(
                {
                    "split": split,
                    "map_id": str(state["map_id"]),
                    "layout_mode": str(state["layout_mode"]),
                    "agent_count": int(state["agent_count"]),
                    "state_id": str(state["state_id"]),
                    "controller": controller,
                    "candidate_id": str(arm["candidate_id"]),
                    "actual_size": int(arm["actual_size"]),
                    "v2_base_selected": bool(arm["base_selected"]),
                    "trial_count": len(selected_trials),
                    "mean_executed_repairs": _mean(map(float, executed)),
                    "mean_initial_conflicts": _mean(map(float, initial)),
                    "mean_final_conflicts": _mean(map(float, final)),
                    "mean_conflict_reduction": reduction,
                    "mean_h1_conflict_reduction": _mean(
                        float(row["conflict_reduction"]) for row in h1_rows
                    ),
                    "mean_controller_seconds": controller_seconds,
                    "mean_pp_seconds": pp_seconds,
                    "mean_repair_seconds": repair_seconds,
                    "mean_non_pp_repair_seconds": repair_seconds - pp_seconds,
                    "mean_total_seconds": total_seconds,
                    "conflict_reduction_per_total_second": reduction
                    / max(1e-12, total_seconds),
                    "h1_effective_rate": _mean(
                        float(bool(row["effective_progress"])) for row in h1_rows
                    ),
                    "h1_no_progress_rate": _mean(
                        float(bool(row["no_progress"])) for row in h1_rows
                    ),
                    "h3_no_progress_rate": _mean(
                        float(bool(row["no_progress"])) for row in h3_rows
                    ),
                    "h3_feasible_rate": _mean(
                        float(bool(row["feasible"])) for row in h3_rows
                    ),
                    "mean_generated": _mean(float(row["generated"]) for row in h3_rows),
                    "mean_expanded": _mean(float(row["expanded"]) for row in h3_rows),
                    "mean_reopened": _mean(float(row["reopened"]) for row in h3_rows),
                }
            )
    return rows


def _aggregate(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in group_fields)].append(row)
    output = []
    for key, selected in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        total_seconds = _sum(float(row["mean_total_seconds"]) for row in selected)
        reduction = _sum(float(row["mean_conflict_reduction"]) for row in selected)
        output.append(
            {
                **dict(zip(group_fields, key)),
                "state_count": len(selected),
                "mean_actual_size": _mean(float(row["actual_size"]) for row in selected),
                "mean_executed_repairs": _mean(
                    float(row["mean_executed_repairs"]) for row in selected
                ),
                "mean_initial_conflicts": _mean(
                    float(row["mean_initial_conflicts"]) for row in selected
                ),
                "mean_final_conflicts": _mean(
                    float(row["mean_final_conflicts"]) for row in selected
                ),
                "mean_conflict_reduction": _mean(
                    float(row["mean_conflict_reduction"]) for row in selected
                ),
                "sum_conflict_reduction": reduction,
                "mean_controller_seconds": _mean(
                    float(row["mean_controller_seconds"]) for row in selected
                ),
                "mean_pp_seconds": _mean(
                    float(row["mean_pp_seconds"]) for row in selected
                ),
                "mean_repair_seconds": _mean(
                    float(row["mean_repair_seconds"]) for row in selected
                ),
                "mean_non_pp_repair_seconds": _mean(
                    float(row["mean_non_pp_repair_seconds"]) for row in selected
                ),
                "mean_total_seconds": _mean(
                    float(row["mean_total_seconds"]) for row in selected
                ),
                "sum_total_seconds": total_seconds,
                "conflict_reduction_per_total_second": reduction
                / max(1e-12, total_seconds),
                "h1_effective_rate": _mean(
                    float(row["h1_effective_rate"]) for row in selected
                ),
                "h1_no_progress_rate": _mean(
                    float(row["h1_no_progress_rate"]) for row in selected
                ),
                "h3_no_progress_rate": _mean(
                    float(row["h3_no_progress_rate"]) for row in selected
                ),
                "h3_feasible_rate": _mean(
                    float(row["h3_feasible_rate"]) for row in selected
                ),
                "mean_generated": _mean(float(row["mean_generated"]) for row in selected),
                "mean_expanded": _mean(float(row["mean_expanded"]) for row in selected),
                "mean_reopened": _mean(float(row["mean_reopened"]) for row in selected),
            }
        )
    return output


def _raw_trial_rows(
    horizon_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    metadata = {
        str(row["state_id"]): row for row in state_rows if row["controller"] == "v2"
    }
    result = []
    for row in horizon_rows:
        state = metadata[str(row["state_id"])]
        h1 = dict(row["h1"])
        h3 = dict(row["h3"])
        trajectory = list(map(int, row["conflict_trajectory"]))
        result.append(
            {
                "split": str(row["split"]),
                "map_id": str(state["map_id"]),
                "layout_mode": str(state["layout_mode"]),
                "agent_count": int(state["agent_count"]),
                "state_id": str(row["state_id"]),
                "candidate_id": str(row["candidate_id"]),
                "route": str(row["route"]),
                "actual_size": int(row["actual_size"]),
                "trial_index": int(row["trial_index"]),
                "executed_repairs": int(row["executed_steps"]),
                "initial_conflicts": trajectory[0],
                "final_conflicts": trajectory[-1],
                "conflict_trajectory": json.dumps(trajectory, separators=(",", ":")),
                "h1_effective_progress": bool(h1["effective_progress"]),
                "h1_no_progress": bool(h1["no_progress"]),
                "h1_conflict_reduction": float(h1["conflict_reduction"]),
                "h1_pp_seconds": float(h1["pp_replan_seconds"]),
                "h3_conflict_reduction": float(h3["conflict_reduction"]),
                "h3_best_conflict_reduction": float(h3["best_conflict_reduction"]),
                "h3_no_progress": bool(h3["no_progress"]),
                "h3_feasible": bool(h3["feasible"]),
                "controller_seconds": float(h3["controller_seconds"]),
                "pp_seconds": float(h3["pp_replan_seconds"]),
                "repair_seconds": float(h3["repair_seconds"]),
                "non_pp_repair_seconds": float(h3["repair_seconds"])
                - float(h3["pp_replan_seconds"]),
                "total_seconds": float(h3["total_seconds"]),
                "conflict_reduction_per_total_second": float(
                    h3["conflict_reduction"]
                )
                / max(1e-12, float(h3["total_seconds"])),
                "generated": int(h3["generated"]),
                "expanded": int(h3["expanded"]),
                "reopened": int(h3["reopened"]),
            }
        )
    return result


def _pairwise_map_rows(map_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(dict)
    for row in map_rows:
        grouped[
            (row["split"], row["map_id"], row.get("agent_count", "all"))
        ][row["controller"]] = row
    output = []
    for (split, map_id, agent_count), values in sorted(grouped.items()):
        if set(values) != set(CONTROLLERS):
            raise ValueError(f"incomplete controller coverage for {map_id}")
        h3, v2, adaptive = values["v3-h3"], values["v2"], values["adaptive"]
        output.append(
            {
                "split": split,
                "map_id": map_id,
                "layout_mode": h3["layout_mode"],
                "agent_count": agent_count,
                "state_count": h3["state_count"],
                "v3_h3_mean_total_seconds": h3["mean_total_seconds"],
                "v2_mean_total_seconds": v2["mean_total_seconds"],
                "adaptive_mean_total_seconds": adaptive["mean_total_seconds"],
                "v3_h3_vs_v2_time_ratio": float(h3["mean_total_seconds"])
                / max(1e-12, float(v2["mean_total_seconds"])),
                "v3_h3_vs_v2_time_delta_seconds": float(h3["mean_total_seconds"])
                - float(v2["mean_total_seconds"]),
                "v3_h3_mean_pp_seconds": h3["mean_pp_seconds"],
                "v2_mean_pp_seconds": v2["mean_pp_seconds"],
                "adaptive_mean_pp_seconds": adaptive["mean_pp_seconds"],
                "v3_h3_mean_conflict_reduction": h3["mean_conflict_reduction"],
                "v2_mean_conflict_reduction": v2["mean_conflict_reduction"],
                "adaptive_mean_conflict_reduction": adaptive[
                    "mean_conflict_reduction"
                ],
                "v3_h3_vs_v2_reduction_ratio": float(
                    h3["mean_conflict_reduction"]
                )
                / max(1e-12, float(v2["mean_conflict_reduction"])),
                "v3_h3_efficiency": h3["conflict_reduction_per_total_second"],
                "v2_efficiency": v2["conflict_reduction_per_total_second"],
                "adaptive_efficiency": adaptive[
                    "conflict_reduction_per_total_second"
                ],
                "v3_h3_vs_v2_efficiency_ratio": float(
                    h3["conflict_reduction_per_total_second"]
                )
                / max(1e-12, float(v2["conflict_reduction_per_total_second"])),
                "v3_h3_no_progress_rate": h3["h3_no_progress_rate"],
                "v2_no_progress_rate": v2["h3_no_progress_rate"],
                "adaptive_no_progress_rate": adaptive["h3_no_progress_rate"],
            }
        )
    return output


def _reconcile(
    overall: list[dict[str, Any]], training_report: dict[str, Any]
) -> dict[str, Any]:
    checks = []
    for split, expected in (
        ("policy_train", training_report["calibration"]["selected"]["gate"]),
        ("policy_validation", training_report["diagnostic_gate"]),
    ):
        lookup = {
            str(row["controller"]): row for row in overall if row["split"] == split
        }
        for controller, expected_name in (
            ("v3-h3", "h3"),
            ("v2", "v2"),
            ("adaptive", "adaptive"),
        ):
            actual = lookup[controller]
            target = expected[expected_name]
            for actual_name, target_name in (
                ("mean_conflict_reduction", "mean_conflict_reduction"),
                ("mean_total_seconds", "mean_total_seconds"),
                (
                    "conflict_reduction_per_total_second",
                    "conflict_reduction_per_total_second",
                ),
                ("h3_no_progress_rate", "no_progress_rate"),
            ):
                delta = abs(float(actual[actual_name]) - float(target[target_name]))
                checks.append(
                    {
                        "split": split,
                        "controller": controller,
                        "metric": actual_name,
                        "delta": delta,
                        "passed": delta <= 1e-12,
                    }
                )
    return {
        "checks": checks,
        "maximum_delta": max(float(row["delta"]) for row in checks),
        "passed": all(bool(row["passed"]) for row in checks),
    }


def build_report(source: Path, pilot: Path, output: Path) -> dict[str, Any]:
    feature_rows = _read_jsonl(source / "collection" / "feature_index.jsonl")
    one_step_rows = _read_jsonl(source / "collection" / "trial_manifest.jsonl")
    horizon_rows = _read_jsonl(
        pilot / "horizon_collection" / "horizon_manifest.jsonl"
    )
    training_report = _read_json(pilot / "controller" / "training_report.json")
    bundle = load_v3_controller_bundle(pilot / "controller")
    thresholds = dict(bundle.thresholds)
    trials: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in horizon_rows:
        trials[(str(row["state_id"]), str(row["candidate_id"]))].append(row)

    train_h1 = _h1_rows(feature_rows, one_step_rows, "policy_train")
    train_h3 = _h3_rows(feature_rows, horizon_rows, "policy_train")
    diagnostic_h3 = _h3_rows(feature_rows, horizon_rows, "policy_validation")
    train_states = _states(train_h3, _oof_predictions(train_h1, train_h3))
    diagnostic_states = _states(
        diagnostic_h3, _final_model_predictions(bundle, diagnostic_h3)
    )
    state_rows = _state_selection_rows(
        train_states, trials, thresholds, "policy_train"
    ) + _state_selection_rows(
        diagnostic_states, trials, thresholds, "policy_validation"
    )
    raw_trials = _raw_trial_rows(horizon_rows, state_rows)
    map_rows = _aggregate(
        state_rows,
        ("split", "map_id", "layout_mode", "agent_count", "controller"),
    )
    map_overall_rows = _aggregate(
        state_rows, ("split", "map_id", "layout_mode", "controller")
    )
    cell_rows = _aggregate(
        state_rows, ("split", "layout_mode", "agent_count", "controller")
    )
    overall = _aggregate(state_rows, ("split", "controller"))
    pairwise = _pairwise_map_rows(map_overall_rows)
    map_agent_pairwise = _pairwise_map_rows(map_rows)
    reconciliation = _reconcile(overall, training_report)
    if not reconciliation["passed"]:
        raise ValueError("detailed report does not reconcile with the pilot gate")
    payload = {
        "schema": "lns2.v3_horizon_detailed_report.v1",
        "pilot_decision": training_report["decision"],
        "thresholds": thresholds,
        "collection": _read_json(
            pilot / "horizon_collection" / "collection_report.json"
        ),
        "training_checks": training_report["pilot_checks"],
        "portable_maximum_delta": training_report["portable_maximum_delta"],
        "reconciliation": reconciliation,
        "overall": overall,
        "map_summary": map_rows,
        "map_overall_summary": map_overall_rows,
        "map_pairwise": pairwise,
        "map_agent_pairwise": map_agent_pairwise,
        "cell_summary": cell_rows,
        "state_detail": state_rows,
        "raw_trial_detail": raw_trials,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "detailed_report.json", payload)
    _write_csv(output / "overall_summary.csv", overall)
    _write_csv(output / "map_timing_summary.csv", map_rows)
    _write_csv(output / "map_pairwise_comparison.csv", pairwise)
    _write_csv(output / "map_agent_pairwise_comparison.csv", map_agent_pairwise)
    _write_csv(output / "cell_timing_summary.csv", cell_rows)
    _write_csv(output / "state_selection_detail.csv", state_rows)
    _write_csv(output / "raw_horizon_trials.csv", raw_trials)
    lines = [
        "# v3-H3 pilot detailed timing report",
        "",
        f"Decision: `{training_report['decision']}`",
        "",
        "All timing metrics describe a paired window of at most three repairs, not a complete solver episode.",
        "",
        f"States: {len(state_rows) // 3}; maps: {len({row['map_id'] for row in map_rows})}; selected-controller rows: {len(state_rows)}.",
        f"Reconciliation maximum delta: {reconciliation['maximum_delta']:.3g}.",
        "",
    ]
    (output / "detailed_timing_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate map-level and state-level timing data for v3-H3."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = build_report(
        resolve_cli_path(PROJECT_ROOT, arguments.source),
        resolve_cli_path(PROJECT_ROOT, arguments.pilot),
        resolve_cli_path(PROJECT_ROOT, arguments.output),
    )
    print(
        json.dumps(
            {
                "pilot_decision": report["pilot_decision"],
                "state_count": len(report["state_detail"]) // 3,
                "map_count": len({row["map_id"] for row in report["map_summary"]}),
                "reconciliation": report["reconciliation"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
