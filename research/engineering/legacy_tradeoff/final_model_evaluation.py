from __future__ import annotations

import csv
import html
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments.closed_loop_confirmation import validate_closed_loop_trace
from experiments.compact_controller_model import _file_sha256, load_controller_bundle
from experiments.repair_collection import _read_json, _read_jsonl, _write_json


REPORT_SCHEMA = "lns2.final_model_evaluation.v2"
PRIMARY_POLICY = "realized_dynamic"
BASELINES = (
    "official_adaptive",
    "fixed_target",
    "fixed_collision",
    "fixed_random",
)
POLICY_ORDER = (*BASELINES, PRIMARY_POLICY)
POLICY_LABELS = {
    "official_adaptive": "Adaptive",
    "fixed_target": "Fixed Target",
    "fixed_collision": "Fixed Collision",
    "fixed_random": "Fixed Random",
    "realized_dynamic": "Final model",
}


def _mean(values: Iterable[float]) -> float | None:
    numbers = list(values)
    return statistics.fmean(numbers) if numbers else None


def _episode_row(row: dict[str, Any]) -> dict[str, Any]:
    summary = dict(row.get("summary") or {})
    controller = dict(summary.get("controller_totals") or {})
    return {
        "episode_id": row.get("episode_id"),
        "policy": row.get("policy"),
        "layout_family": row.get("layout_mode"),
        "map_id": row.get("map_id"),
        "task_id": row.get("task_id"),
        "agent_count": row.get("agent_count"),
        "solver_seed": row.get("solver_seed"),
        "status": row.get("status"),
        "success": summary.get("success"),
        "repairable": summary.get("repairable"),
        "initial_conflicts": summary.get("initial_conflicts"),
        "final_conflicts": summary.get("final_conflicts"),
        "fixed_budget_conflict_auc": summary.get("fixed_budget_conflict_auc"),
        "raw_conflict_auc": summary.get("conflict_auc"),
        "repair_iterations": summary.get("repair_iterations"),
        "capped_wall_time_seconds": summary.get("capped_wall_time_to_feasible"),
        "wall_time_to_feasible_seconds": summary.get("wall_time_to_feasible"),
        "final_sum_of_costs": summary.get("final_sum_of_costs"),
        "controller_mode": summary.get("controller_mode"),
        "feature_backend": summary.get("feature_backend"),
        "candidate_count_before_pruning": controller.get(
            "candidate_count_before_pruning"
        ),
        "candidate_count_after_pruning": controller.get(
            "candidate_count_after_pruning"
        ),
        "candidate_reduction_fraction": summary.get(
            "candidate_reduction_fraction"
        ),
        "pruner_fallback_fraction": summary.get("pruner_fallback_fraction"),
        "pruner_ood_fallback_fraction": summary.get(
            "pruner_ood_fallback_fraction"
        ),
        "controller_seconds": controller.get("controller_seconds_before_repair"),
        "repair_wall_seconds": summary.get("repair_wall_seconds"),
        "state_analysis_seconds": controller.get("state_analysis_seconds"),
        "conflict_update_seconds": controller.get("conflict_update_seconds"),
        "proposal_feature_seconds": controller.get("proposal_feature_seconds"),
        "pruner_seconds": controller.get("pruner_seconds"),
        "realized_feature_seconds": controller.get("realized_feature_seconds"),
        "ranking_inference_seconds": controller.get("inference_seconds"),
        "v1_v2_shadow_validation_count": controller.get(
            "shadow_validation_count"
        ),
        "v1_v2_shadow_max_score_delta": controller.get(
            "shadow_score_max_delta"
        ),
        "trace_format": row.get("trace_format"),
        "trace_bytes": row.get("trace_bytes"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(
    episodes: list[dict[str, Any]],
    controller_performance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in episodes:
        if str(row["status"]) not in {"ok", "resumed"}:
            continue
        for family in ("all", str(row["layout_family"])):
            groups.setdefault((str(row["policy"]), family), []).append(row)
    result = []
    for policy in POLICY_ORDER:
        families = [family for candidate, family in groups if candidate == policy]
        for family in sorted(set(families), key=lambda value: (value != "all", value)):
            rows = groups[(policy, family)]
            repairable = [row for row in rows if bool(row["repairable"])]
            controller_rows = [
                row for row in rows if row.get("controller_seconds") is not None
            ]
            before = sum(
                int(row.get("candidate_count_before_pruning") or 0)
                for row in controller_rows
            )
            after = sum(
                int(row.get("candidate_count_after_pruning") or 0)
                for row in controller_rows
            )
            attach_performance = (
                controller_performance is not None
                and policy == PRIMARY_POLICY
                and family == "all"
            )
            result.append(
                {
                    "policy": policy,
                    "layout_family": family,
                    "episode_count": len(rows),
                    "success_count": sum(bool(row["success"]) for row in rows),
                    "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
                    "repairable_count": len(repairable),
                    "mean_fixed_budget_conflict_auc": _mean(
                        float(row["fixed_budget_conflict_auc"]) for row in repairable
                    ),
                    "mean_capped_wall_time_seconds": _mean(
                        float(row["capped_wall_time_seconds"]) for row in repairable
                    ),
                    "mean_repair_iterations": _mean(
                        float(row["repair_iterations"]) for row in repairable
                    ),
                    "candidate_reduction_fraction": (
                        1.0 - after / before if before else None
                    ),
                    "mean_pruner_fallback_fraction": _mean(
                        float(row["pruner_fallback_fraction"])
                        for row in controller_rows
                    ),
                    "mean_pruner_ood_fallback_fraction": _mean(
                        float(row["pruner_ood_fallback_fraction"])
                        for row in controller_rows
                    ),
                    "mean_controller_seconds": _mean(
                        float(row["controller_seconds"]) for row in controller_rows
                    ),
                    "v1_v2_shadow_validation_count": sum(
                        int(row.get("v1_v2_shadow_validation_count") or 0)
                        for row in controller_rows
                    ),
                    "v1_v2_shadow_max_score_delta": max(
                        (
                            float(row["v1_v2_shadow_max_score_delta"])
                            for row in controller_rows
                            if row.get("v1_v2_shadow_max_score_delta") is not None
                        ),
                        default=None,
                    ),
                    "fixed_suite_feature_time_reduction": (
                        controller_performance.get("feature_time_reduction")
                        if attach_performance
                        else None
                    ),
                    "fixed_suite_feature_speedup": (
                        controller_performance.get("feature_speedup")
                        if attach_performance
                        else None
                    ),
                    "estimated_controller_time_reduction": (
                        controller_performance.get(
                            "estimated_controller_time_reduction"
                        )
                        if attach_performance
                        else None
                    ),
                    "estimated_controller_speedup": (
                        controller_performance.get("estimated_controller_speedup")
                        if attach_performance
                        else None
                    ),
                    "estimated_end_to_end_time_reduction": (
                        controller_performance.get(
                            "estimated_end_to_end_time_reduction"
                        )
                        if attach_performance
                        else None
                    ),
                    "estimated_end_to_end_speedup": (
                        controller_performance.get("estimated_end_to_end_speedup")
                        if attach_performance
                        else None
                    ),
                    "speedup_measurement": (
                        controller_performance.get("measurement")
                        if attach_performance
                        else None
                    ),
                    "mean_repair_wall_seconds": _mean(
                        float(row["repair_wall_seconds"])
                        for row in rows
                        if row.get("repair_wall_seconds") is not None
                    ),
                    **{
                        f"mean_{name}": _mean(
                            float(row[name])
                            for row in controller_rows
                            if row.get(name) is not None
                        )
                        for name in (
                            "state_analysis_seconds",
                            "conflict_update_seconds",
                            "proposal_feature_seconds",
                            "pruner_seconds",
                            "realized_feature_seconds",
                            "ranking_inference_seconds",
                        )
                    },
                }
            )
    return result


def _controller_performance_evidence(
    run_config: dict[str, Any], collection_root: Path
) -> dict[str, Any] | None:
    if str(run_config.get("controller", "v1-full")) == "v1-full":
        return None
    configuration = dict(run_config.get("configuration") or {})
    raw_bundle = configuration.get("controller_bundle")
    if not raw_bundle:
        raise ValueError("controller-v2 run is missing its controller bundle path")
    bundle_root = Path(str(raw_bundle))
    if not bundle_root.is_absolute():
        bundle_root = (collection_root / bundle_root).resolve()
    loaded = load_controller_bundle(bundle_root)
    row = dict(loaded.promotion_report.get("performance_benchmark") or {})
    if not row:
        raise ValueError("controller-v2 bundle has no fixed-suite performance evidence")
    benchmark_path = Path(str(row.get("file", "")))
    if not benchmark_path.is_absolute():
        benchmark_path = bundle_root / benchmark_path
    expected_sha = str(row.get("sha256", "")).lower()
    if not benchmark_path.is_file() or _file_sha256(benchmark_path) != expected_sha:
        raise ValueError("controller performance benchmark SHA256 mismatch")
    benchmark = _read_json(benchmark_path)
    overall = dict(benchmark.get("overall") or {})
    gate = dict(benchmark.get("performance_gate") or {})
    return {
        "benchmark_file": str(benchmark_path),
        "benchmark_sha256": expected_sha,
        "feature_backend": benchmark.get("feature_backend"),
        "feature_time_reduction": overall.get("feature_time_reduction"),
        "feature_speedup": overall.get("speedup"),
        "estimated_controller_time_reduction": overall.get(
            "estimated_controller_time_reduction"
        ),
        "estimated_controller_speedup": overall.get(
            "estimated_controller_speedup"
        ),
        "estimated_end_to_end_time_reduction": overall.get(
            "estimated_end_to_end_time_reduction"
        ),
        "estimated_end_to_end_speedup": overall.get(
            "estimated_end_to_end_speedup"
        ),
        "maximum_feature_delta": max(
            (
                float(case.get("maximum_feature_delta", 0.0))
                for case in benchmark.get("cases", [])
            ),
            default=0.0,
        ),
        "measurement": overall.get("end_to_end_measurement"),
        "performance_gate_passed": bool(gate.get("passed")),
    }


def _paired_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (str(row["policy"]), str(row["task_id"]), int(row["solver_seed"])): row
        for row in episodes
        if str(row["status"]) in {"ok", "resumed"}
    }
    result = []
    for baseline in BASELINES:
        pairs = []
        for key, model in index.items():
            policy, task_id, solver_seed = key
            if policy != PRIMARY_POLICY or not bool(model["repairable"]):
                continue
            reference = index.get((baseline, task_id, solver_seed))
            if reference is None or not bool(reference["repairable"]):
                continue
            pairs.append((reference, model))
        baseline_auc = _mean(float(left["fixed_budget_conflict_auc"]) for left, _ in pairs)
        model_auc = _mean(float(right["fixed_budget_conflict_auc"]) for _, right in pairs)
        baseline_wall = _mean(float(left["capped_wall_time_seconds"]) for left, _ in pairs)
        model_wall = _mean(float(right["capped_wall_time_seconds"]) for _, right in pairs)
        result.append(
            {
                "baseline": baseline,
                "primary_policy": PRIMARY_POLICY,
                "paired_repairable_count": len(pairs),
                "baseline_mean_fixed_budget_conflict_auc": baseline_auc,
                "model_mean_fixed_budget_conflict_auc": model_auc,
                "auc_relative_improvement": (
                    (baseline_auc - model_auc) / baseline_auc
                    if baseline_auc not in {None, 0.0} and model_auc is not None
                    else None
                ),
                "baseline_mean_capped_wall_time_seconds": baseline_wall,
                "model_mean_capped_wall_time_seconds": model_wall,
                "wall_time_relative_improvement": (
                    (baseline_wall - model_wall) / baseline_wall
                    if baseline_wall not in {None, 0.0} and model_wall is not None
                    else None
                ),
                "model_successes": sum(bool(right["success"]) for _, right in pairs),
                "baseline_successes": sum(bool(left["success"]) for left, _ in pairs),
            }
        )
    return result


def _bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    *,
    value_format: str = ".1f",
    percent: bool = False,
    lower_is_better: bool = False,
) -> None:
    width = 960
    height = 520
    margin_left = 95
    margin_right = 40
    margin_top = 80
    margin_bottom = 145
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    maximum = max([0.0, *values])
    minimum = min([0.0, *values])
    if maximum == minimum:
        maximum = minimum + 1.0
    span = maximum - minimum
    zero_y = margin_top + plot_height * maximum / span
    bar_slot = plot_width / max(1, len(values))
    bar_width = bar_slot * 0.62
    colors = ["#64748b"] * max(0, len(values) - 1) + ["#0f766e"]
    displays = [
        f"{value * 100:{value_format}}%" if percent else f"{value:{value_format}}"
        for value in values
    ]
    description = "; ".join(
        f"{label}: {display}" for label, display in zip(labels, displays)
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="chart-title chart-description" style="max-width:100%;height:auto">',
        f'<title id="chart-title">{html.escape(title)}</title>',
        f'<desc id="chart-description">{html.escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="500">{html.escape(title)}</text>',
        f'<text x="{width/2}" y="62" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">{"Lower is better" if lower_is_better else "Higher is better"}</text>',
        f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{margin_left+plot_width}" y2="{zero_y:.2f}" stroke="#94a3b8"/>',
    ]
    for index, (label, value, display) in enumerate(zip(labels, values, displays)):
        x = margin_left + index * bar_slot + (bar_slot - bar_width) / 2
        value_y = margin_top + plot_height * (maximum - value) / span
        y = min(zero_y, value_y)
        bar_height = abs(value_y - zero_y)
        color = "#b91c1c" if value < 0 else colors[index]
        label_y = max(margin_top + 15, y - 8) if value >= 0 else min(
            margin_top + plot_height - 4, y + bar_height + 16
        )
        parts.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="5" fill="{color}"/>',
                f'<text x="{x+bar_width/2:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="500">{html.escape(display)}</text>',
                f'<text transform="translate({x+bar_width/2:.2f},{margin_top+plot_height+18}) rotate(35)" text-anchor="start" font-family="sans-serif" font-size="12">{html.escape(label)}</text>',
            ]
        )
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def _render_markdown(
    report: dict[str, Any], *, formal: bool, official_report: dict[str, Any] | None
) -> str:
    aggregate = {
        str(row["policy"]): row
        for row in report["aggregates"]
        if row["layout_family"] == "all"
    }
    lines = [
        "# Final frozen-model evaluation",
        "",
        (
            "**Formal registered evaluation.**"
            if formal
            else "**非正式试跑，不可作为正式结论。**"
        ),
        "",
        f"- Trace format: `{report['trace_format']}`",
        f"- Controller: `{report.get('controller', 'v1-full')}`",
        f"- Feature backend: `{report.get('feature_backend', 'reference-v1')}`",
        f"- Paired v1/v2 shadow validation: `{bool(report.get('feature_shadow_validation'))}`",
        f"- Valid traces: {report['valid_trace_count']}/{report['expected_trace_count']}",
        "",
        "## Policy comparison",
        "",
        "| Policy | Success | Success rate | Mean fixed-budget AUC | Mean capped wall time |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy in POLICY_ORDER:
        row = aggregate.get(policy)
        if row is None:
            continue
        auc = row["mean_fixed_budget_conflict_auc"]
        wall = row["mean_capped_wall_time_seconds"]
        auc_text = "n/a" if auc is None else f"{auc:.3f}"
        wall_text = "n/a" if wall is None else f"{wall:.3f}s"
        lines.append(
            f"| {POLICY_LABELS[policy]} | {row['success_count']}/{row['episode_count']} | "
            f"{row['success_rate']:.1%} | {auc_text} | {wall_text} |"
        )
    lines.extend(["", "## Final model versus baselines", ""])
    for row in report["paired_comparisons"]:
        improvement = row["auc_relative_improvement"]
        value = "n/a" if improvement is None else f"{improvement:.2%}"
        lines.append(
            f"- vs `{row['baseline']}`: paired repairable episodes "
            f"{row['paired_repairable_count']}, fixed-budget AUC improvement {value}."
        )
    primary = aggregate.get(PRIMARY_POLICY)
    if primary is not None and primary.get("candidate_reduction_fraction") is not None:
        lines.extend(
            [
                "",
                "## Controller diagnostics",
                "",
                f"- Candidate reduction: {primary['candidate_reduction_fraction']:.2%}",
                f"- Mean fallback rate: {primary['mean_pruner_fallback_fraction']:.2%}",
                f"- Mean OOD fallback rate: {primary['mean_pruner_ood_fallback_fraction']:.2%}",
                f"- Mean controller time per episode: {primary['mean_controller_seconds']:.3f}s",
            ]
        )
    performance = report.get("controller_performance")
    if performance is not None:
        lines.extend(
            [
                "",
                "## Controller acceleration evidence",
                "",
                "- Fixed-suite feature extraction: "
                f"{float(performance['feature_speedup']):.2f}x speedup "
                f"({float(performance['feature_time_reduction']):.2%} less time).",
                "- Estimated controller total: "
                f"{float(performance['estimated_controller_speedup']):.2f}x speedup "
                f"({float(performance['estimated_controller_time_reduction']):.2%} less time).",
                "- Estimated controller + repair wall time: "
                f"{float(performance['estimated_end_to_end_speedup']):.2f}x speedup "
                f"({float(performance['estimated_end_to_end_time_reduction']):.2%} less time).",
                "- The feature result is a paired measurement. Controller and end-to-end "
                "figures are projections from recorded v1 timings; the quick/formal run "
                "provides the final observed wall-time evidence.",
            ]
        )
    if primary is not None and primary.get("v1_v2_shadow_validation_count"):
        lines.extend(
            [
                "",
                "## Paired v1/v2 quick audit",
                "",
                f"- Shadow-validated decisions: {primary['v1_v2_shadow_validation_count']}",
                "- Maximum score delta: "
                f"{float(primary['v1_v2_shadow_max_score_delta'] or 0.0):.3g}",
                "- All candidate rankings and selected candidates matched; a mismatch "
                "would have terminated collection.",
            ]
        )
    if official_report is not None:
        acceptance = dict(official_report.get("acceptance", {}))
        lines.extend(
            [
                "",
                "## Registered conclusion",
                "",
                f"- Passed: `{bool(acceptance.get('passed'))}`",
                f"- Decision: `{acceptance.get('decision')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `episodes.csv`: episode-level results",
            "- `aggregates.csv`: policy and layout-family summaries",
            "- `paired_comparisons.csv`: final-model paired comparisons",
            "- `charts/*.svg`: conclusion figures",
            "",
        ]
    )
    return "\n".join(lines)


def generate_evaluation_artifacts(
    collection: str | Path,
    output: str | Path,
    *,
    formal: bool,
    validate_traces: bool = True,
    official_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(collection).resolve()
    output_root = Path(output).resolve()
    run_config = _read_json(root / "run_config.json")
    configuration = dict(run_config["configuration"])
    policies = tuple(map(str, configuration.get("policies", POLICY_ORDER)))
    if set(policies) != set(POLICY_ORDER):
        raise ValueError("final evaluation requires exactly the registered five policies")
    manifests = {
        policy: _read_jsonl(root / f"{policy}_manifest.jsonl") for policy in policies
    }
    valid_trace_count = 0
    if validate_traces:
        for rows in manifests.values():
            for row in rows:
                if str(row.get("status")) not in {"ok", "resumed"}:
                    continue
                validated = validate_closed_loop_trace(
                    root / str(row["trace_file"]),
                    str(run_config["run_fingerprint"]),
                    expected_episode_id=str(row["episode_id"]),
                    expected_policy=str(row["policy"]),
                    expected_solver_seed=int(row["solver_seed"]),
                    metric_iteration_budget=int(configuration["metric_iteration_budget"]),
                    collection_root=root,
                )
                if validated["summary"] != row.get("summary"):
                    raise ValueError(f"manifest summary mismatch: {row['episode_id']}")
                valid_trace_count += 1
    else:
        valid_trace_count = sum(
            str(row.get("status")) in {"ok", "resumed"}
            for rows in manifests.values()
            for row in rows
        )
    episodes = [_episode_row(row) for policy in policies for row in manifests[policy]]
    if valid_trace_count != len(episodes):
        raise ValueError(
            "final evaluation is incomplete: "
            f"validated {valid_trace_count} of {len(episodes)} episode traces"
        )
    episodes.sort(
        key=lambda row: (
            POLICY_ORDER.index(str(row["policy"])),
            str(row["task_id"]),
            int(row["solver_seed"]),
        )
    )
    controller_performance = _controller_performance_evidence(run_config, root)
    aggregates = _aggregate_rows(episodes, controller_performance)
    paired = _paired_rows(episodes)
    report = {
        "schema": REPORT_SCHEMA,
        "formal": formal,
        "formal_conclusion_allowed": formal,
        "notice": None if formal else "非正式试跑，不可作为正式结论。",
        "run_fingerprint": run_config["run_fingerprint"],
        "trace_format": run_config.get("trace_format", "full-v1"),
        "controller": run_config.get("controller", "v1-full"),
        "feature_backend": run_config.get("feature_backend", "reference-v1"),
        "feature_shadow_validation": bool(
            configuration.get("feature_shadow_validation", False)
        ),
        "controller_performance": controller_performance,
        "expected_trace_count": len(episodes),
        "valid_trace_count": valid_trace_count,
        "aggregates": aggregates,
        "paired_comparisons": paired,
        "official_acceptance": (
            official_report.get("acceptance") if official_report is not None else None
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "episodes.csv", episodes)
    _write_csv(output_root / "aggregates.csv", aggregates)
    _write_csv(output_root / "paired_comparisons.csv", paired)
    _write_json(output_root / "evaluation_summary.json", report)
    aggregate_all = {
        str(row["policy"]): row for row in aggregates if row["layout_family"] == "all"
    }
    chart_labels = [POLICY_LABELS[policy] for policy in POLICY_ORDER]
    _bar_chart(
        output_root / "charts" / "success_rate.svg",
        "Success rate by policy",
        chart_labels,
        [float(aggregate_all[policy]["success_rate"]) for policy in POLICY_ORDER],
        value_format=".1f",
        percent=True,
    )
    _bar_chart(
        output_root / "charts" / "fixed_budget_auc.svg",
        "Mean fixed-budget conflict AUC",
        chart_labels,
        [
            float(aggregate_all[policy]["mean_fixed_budget_conflict_auc"] or 0.0)
            for policy in POLICY_ORDER
        ],
        lower_is_better=True,
    )
    _bar_chart(
        output_root / "charts" / "auc_improvement.svg",
        "Final-model AUC improvement versus baselines",
        [POLICY_LABELS[str(row["baseline"])] for row in paired],
        [float(row["auc_relative_improvement"] or 0.0) for row in paired],
        value_format=".1f",
        percent=True,
    )
    primary_rows = [
        row for row in aggregates if str(row["policy"]) == PRIMARY_POLICY
    ]
    candidate_rows = [
        row
        for row in primary_rows
        if row.get("candidate_reduction_fraction") is not None
    ]
    if candidate_rows:
        _bar_chart(
            output_root / "charts" / "candidate_reduction.svg",
            "Final-controller candidate reduction",
            [str(row["layout_family"]) for row in candidate_rows],
            [float(row["candidate_reduction_fraction"]) for row in candidate_rows],
            value_format=".1f",
            percent=True,
        )
    primary_all = aggregate_all.get(PRIMARY_POLICY)
    if primary_all is not None and primary_all.get("mean_controller_seconds") is not None:
        stage_names = (
            "state_analysis_seconds",
            "proposal_feature_seconds",
            "pruner_seconds",
            "realized_feature_seconds",
            "ranking_inference_seconds",
        )
        _bar_chart(
            output_root / "charts" / "controller_feature_stages.svg",
            "Mean controller stage time per episode",
            [name.removesuffix("_seconds").replace("_", " ") for name in stage_names],
            [float(primary_all.get(f"mean_{name}") or 0.0) for name in stage_names],
            lower_is_better=True,
        )
    if controller_performance is not None:
        _bar_chart(
            output_root / "charts" / "controller_speedups.svg",
            "Controller-v2 fixed-suite speedups",
            ["feature extraction", "controller (estimated)", "controller + repair (estimated)"],
            [
                float(controller_performance["feature_speedup"]),
                float(controller_performance["estimated_controller_speedup"]),
                float(controller_performance["estimated_end_to_end_speedup"]),
            ],
            value_format=".2f",
        )
    (output_root / "conclusion.md").write_text(
        _render_markdown(report, formal=formal, official_report=official_report),
        encoding="utf-8",
    )
    return report


__all__ = ["generate_evaluation_artifacts"]
