from __future__ import annotations

import collections
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments.closed_loop_confirmation import CLOSED_LOOP_SCHEMA, POLICIES
from experiments.repair_collection import SCHEMA_VERSION, _read_json, _read_jsonl, _write_json


def _mean(values: Iterable[float | int]) -> float:
    numbers = [float(value) for value in values]
    return statistics.fmean(numbers) if numbers else 0.0


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _relative_improvement(baseline: float, primary: float) -> float:
    if baseline == 0.0:
        return 0.0 if primary == 0.0 else -float("inf")
    return (baseline - primary) / baseline


def summarize_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if str(row.get("status")) in {"ok", "resumed"}]
    repairable = [row for row in valid if bool(row["summary"]["repairable"])]
    size_counts: collections.Counter[str] = collections.Counter()
    family_counts: collections.Counter[str] = collections.Counter()
    for row in repairable:
        size_counts.update(dict(row["summary"].get("selected_size_counts", {})))
        family_counts.update(dict(row["summary"].get("selected_family_counts", {})))
    low_level_keys = ("expanded", "generated", "reopened", "runs")
    return {
        "episode_count": len(rows),
        "valid_count": len(valid),
        "error_count": len(rows) - len(valid),
        "success_count": sum(bool(row["summary"]["success"]) for row in valid),
        "repairable_count": len(repairable),
        "repairable_success_count": sum(bool(row["summary"]["success"]) for row in repairable),
        "mean_fixed_budget_conflict_auc": _mean(
            row["summary"]["fixed_budget_conflict_auc"] for row in repairable
        ),
        "mean_capped_wall_time_to_feasible": _mean(
            row["summary"]["capped_wall_time_to_feasible"] for row in repairable
        ),
        "mean_raw_conflict_auc": _mean(row["summary"]["conflict_auc"] for row in repairable),
        "mean_repair_iterations": _mean(
            row["summary"]["repair_iterations"] for row in repairable
        ),
        "mean_final_low_level": {
            key: _mean(
                row["summary"].get("final_low_level", {}).get(key, 0)
                for row in repairable
            )
            for key in low_level_keys
        },
        "selected_size_counts": dict(sorted(size_counts.items())),
        "selected_family_counts": dict(sorted(family_counts.items())),
        "invalid_action_count": sum(
            int(row["summary"].get("invalid_action_count", 0)) for row in valid
        ),
        "fingerprint_mismatch_count": sum(
            int(row["summary"].get("fingerprint_mismatch_count", 0)) for row in valid
        ),
        "external_timeout_count": sum(
            bool(row["summary"].get("external_timeout")) for row in valid
        ),
        "controller_proposal_seconds": sum(
            float(row["summary"].get("controller_totals", {}).get("proposal_seconds", 0.0))
            for row in valid
        ),
        "controller_feature_seconds": sum(
            float(row["summary"].get("controller_totals", {}).get("feature_seconds", 0.0))
            for row in valid
        ),
        "controller_inference_seconds": sum(
            float(row["summary"].get("controller_totals", {}).get("inference_seconds", 0.0))
            for row in valid
        ),
        "mean_repair_wall_seconds": _mean(
            row.get("trace_timing", {}).get("repair_wall_seconds", 0.0)
            for row in repairable
        ),
        "mean_controller_before_repair_seconds": _mean(
            row.get("trace_timing", {}).get("controller_before_repair_seconds", 0.0)
            for row in repairable
        ),
        "mean_other_wall_seconds": _mean(
            max(
                0.0,
                float(row["summary"]["capped_wall_time_to_feasible"])
                - float(row.get("trace_timing", {}).get("repair_wall_seconds", 0.0))
                - float(
                    row.get("trace_timing", {}).get(
                        "controller_before_repair_seconds", 0.0
                    )
                ),
            )
            for row in repairable
        ),
        "mean_selected_feature_outside_fraction": _mean(
            row["summary"].get("mean_selected_feature_outside_fraction", 0.0)
            for row in repairable
        ),
    }


def _paired_rows(
    baseline: list[dict[str, Any]], primary: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in baseline
        if str(row.get("status")) in {"ok", "resumed"}
    }
    right = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in primary
        if str(row.get("status")) in {"ok", "resumed"}
    }
    keys = sorted(set(left) & set(right))
    return [
        (left[key], right[key])
        for key in keys
        if bool(left[key]["summary"]["repairable"])
        and bool(right[key]["summary"]["repairable"])
    ]


def _metric_value(row: dict[str, Any], metric: str) -> float:
    return float(row["summary"][metric])


def map_paired_bootstrap(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metric: str,
    samples: int,
    *,
    seed: int = 20261219,
) -> dict[str, Any]:
    by_map: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for baseline, primary in pairs:
        if str(baseline["map_id"]) != str(primary["map_id"]):
            raise ValueError("paired rows disagree on map id")
        by_map[str(baseline["map_id"])].append(
            (_metric_value(baseline, metric), _metric_value(primary, metric))
        )
    map_ids = sorted(by_map)
    if not map_ids:
        return {"map_count": 0, "improvement_95_ci": [0.0, 0.0]}
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        selected = [rng.choice(map_ids) for _ in map_ids]
        baseline_values = []
        primary_values = []
        for map_id in selected:
            baseline_values.extend(value[0] for value in by_map[map_id])
            primary_values.extend(value[1] for value in by_map[map_id])
        values.append(
            _relative_improvement(_mean(baseline_values), _mean(primary_values))
        )
    return {
        "map_count": len(map_ids),
        "samples": samples,
        "improvement_95_ci": [_quantile(values, 0.025), _quantile(values, 0.975)],
    }


def compare_policies(
    baseline: list[dict[str, Any]],
    primary: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    pairs = _paired_rows(baseline, primary)
    metrics = (
        "fixed_budget_conflict_auc",
        "capped_wall_time_to_feasible",
    )
    comparisons = {}
    for metric in metrics:
        baseline_mean = _mean(_metric_value(left, metric) for left, _ in pairs)
        primary_mean = _mean(_metric_value(right, metric) for _, right in pairs)
        maps: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
        for left, right in pairs:
            maps[str(left["map_id"])].append(
                (_metric_value(left, metric), _metric_value(right, metric))
            )
        maps_no_worse = sum(
            _mean(value[1] for value in values) <= _mean(value[0] for value in values)
            for values in maps.values()
        )
        comparisons[metric] = {
            "baseline_mean": baseline_mean,
            "primary_mean": primary_mean,
            "relative_improvement": _relative_improvement(baseline_mean, primary_mean),
            "maps_no_worse": maps_no_worse,
            "map_count": len(maps),
            "bootstrap": map_paired_bootstrap(
                pairs, metric, bootstrap_samples, seed=20261219
            ),
        }
    return {"paired_repairable_count": len(pairs), "metrics": comparisons}


def _initial_fingerprint_integrity(
    manifests: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    fingerprints: dict[tuple[str, int], dict[str, str]] = collections.defaultdict(dict)
    for policy, rows in manifests.items():
        for row in rows:
            if str(row.get("status")) not in {"ok", "resumed"}:
                continue
            key = (str(row["task_id"]), int(row["solver_seed"]))
            fingerprints[key][policy] = str(row["summary"]["initial_fingerprint"])
    mismatches = []
    incomplete = []
    for key, values in sorted(fingerprints.items()):
        if set(values) != set(POLICIES):
            incomplete.append({"task_seed": list(key), "policies": sorted(values)})
        elif len(set(values.values())) != 1:
            mismatches.append({"task_seed": list(key), "fingerprints": values})
    return {
        "passed": not mismatches and not incomplete,
        "paired_count": len(fingerprints),
        "mismatches": mismatches,
        "incomplete": incomplete,
    }


def closed_loop_acceptance(
    qualification: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    integrity: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    adaptive = summaries["official_adaptive"]
    realized = summaries["realized_dynamic"]
    minimum_improvement = float(thresholds["minimum_metric_improvement"])
    minimum_maps = int(thresholds["minimum_maps_no_worse"])
    qualifying_metrics = []
    for name, row in comparison["metrics"].items():
        if (
            float(row["relative_improvement"]) >= minimum_improvement
            and int(row["maps_no_worse"]) >= minimum_maps
            and float(row["bootstrap"]["improvement_95_ci"][1]) >= 0.0
        ):
            qualifying_metrics.append(name)
    policy_errors = sum(summary["error_count"] for summary in summaries.values())
    gates = {
        "qualification": bool(qualification["passed"]),
        "all_policy_episodes_valid": policy_errors == 0,
        "initial_fingerprints_match": bool(integrity["passed"]),
        "success_not_below_adaptive": int(realized["success_count"])
        >= int(adaptive["success_count"]),
        "metric_improvement": bool(qualifying_metrics),
        "no_invalid_actions": int(realized["invalid_action_count"]) == 0,
        "no_fingerprint_mismatch": int(realized["fingerprint_mismatch_count"]) == 0,
    }
    passed = all(gates.values())
    return {
        "passed": passed,
        "gates": gates,
        "qualifying_metrics": qualifying_metrics,
        "decision": (
            "advance_to_policy_visited_data_and_rl_warm_start"
            if passed
            else "keep_rl_paused_and_diagnose_closed_loop_shift"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summaries = report["policy_summaries"]
    comparison = report["comparisons"]["realized_dynamic_vs_official_adaptive"]
    lines = [
        "# InitLNS frozen realized-neighborhood closed-loop confirmation",
        "",
        f"Decision: `{report['acceptance']['decision']}`",
        "",
        "## Cohort",
        "",
        f"- Valid resets: {report['qualification']['valid_count']}/24",
        f"- Initial feasible: {report['qualification']['initial_feasible_count']}",
        f"- Repairable: {report['qualification']['nonzero_state_count']}",
        "",
        "## Policies",
        "",
    ]
    for policy in POLICIES:
        row = summaries[policy]
        lines.append(
            f"- `{policy}`: success {row['success_count']}/{row['valid_count']}, "
            f"fixed AUC {row['mean_fixed_budget_conflict_auc']:.3f}, "
            f"capped wall time {row['mean_capped_wall_time_to_feasible']:.3f}s"
        )
    lines.extend(["", "## Primary comparison", ""])
    for name, row in comparison["metrics"].items():
        lines.append(
            f"- `{name}`: improvement {row['relative_improvement']:.1%}, "
            f"maps no worse {row['maps_no_worse']}/{row['map_count']}, "
            f"bootstrap 95% CI {row['bootstrap']['improvement_95_ci']}"
        )
    lines.extend(
        [
            "",
            "## Registered gates",
            "",
            *[
                f"- {name}: {'PASS' if value else 'FAIL'}"
                for name, value in report["acceptance"]["gates"].items()
            ],
            "",
            "Static context is not evaluated by this stage. No confirmation label was used for training.",
            "",
        ]
    )
    return "\n".join(lines)


def run_closed_loop_analysis(
    collection: str | Path,
    config_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    root = Path(collection).resolve()
    config = _read_json(Path(config_path).resolve())
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported closed-loop analysis config")
    run_config = _read_json(root / "run_config.json")
    if str(run_config.get("schema")) != CLOSED_LOOP_SCHEMA:
        raise ValueError("collection is not a closed-loop confirmation run")
    if not bool(run_config.get("formal")):
        raise ValueError("Pilot data cannot be used for formal closed-loop analysis")
    qualification = _read_json(root / "qualification_report.json")
    manifests = {
        policy: _read_jsonl(root / f"{policy}_manifest.jsonl") for policy in POLICIES
    }
    for rows in manifests.values():
        for row in rows:
            if str(row.get("status")) not in {"ok", "resumed"}:
                continue
            events = _read_jsonl(root / str(row["trace_file"]))
            transitions = [event for event in events if event.get("event") == "transition"]
            row["trace_timing"] = {
                "repair_wall_seconds": sum(
                    float(event.get("repair_wall_seconds", 0.0)) for event in transitions
                ),
                "controller_before_repair_seconds": sum(
                    float(
                        event.get("controller", {}).get(
                            "controller_seconds_before_repair", 0.0
                        )
                    )
                    for event in transitions
                ),
            }
    summaries = {policy: summarize_policy(rows) for policy, rows in manifests.items()}
    expected_episodes = int(config["expected_policy_episodes"])
    for policy, summary in summaries.items():
        if int(summary["episode_count"]) != expected_episodes:
            raise ValueError(f"{policy} does not contain {expected_episodes} episodes")
    integrity = _initial_fingerprint_integrity(manifests)
    bootstrap_samples = int(config["bootstrap_samples"])
    primary = compare_policies(
        manifests["official_adaptive"], manifests["realized_dynamic"], bootstrap_samples
    )
    proposal = compare_policies(
        manifests["official_adaptive"], manifests["proposal_dynamic"], bootstrap_samples
    )
    thresholds = dict(config["thresholds"])
    acceptance = closed_loop_acceptance(
        qualification, summaries, primary, integrity, thresholds
    )
    report = {
        "schema": CLOSED_LOOP_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_config["run_fingerprint"],
        "pre_registration": {
            "models_retrained": False,
            "confirmation_labels_seen": False,
            "primary_policy": "realized_dynamic",
            "primary_baseline": "official_adaptive",
            "proposal_dynamic_role": "representation_ablation",
            "static_context_role": "excluded",
        },
        "qualification": qualification,
        "frozen_models": run_config["frozen_models"],
        "integrity": integrity,
        "policy_summaries": summaries,
        "comparisons": {
            "realized_dynamic_vs_official_adaptive": primary,
            "proposal_dynamic_vs_official_adaptive": proposal,
        },
        "acceptance": acceptance,
    }
    output_root = Path(output).resolve()
    _write_json(output_root / "closed_loop_confirmation.json", report)
    markdown_path = output_root / "closed_loop_confirmation.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report


__all__ = [
    "closed_loop_acceptance",
    "compare_policies",
    "map_paired_bootstrap",
    "render_markdown",
    "run_closed_loop_analysis",
    "summarize_policy",
]
