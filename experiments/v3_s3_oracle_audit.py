from __future__ import annotations

import collections
import csv
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file
from experiments.repair_collection import _read_jsonl, _write_json
from experiments.v3_s3 import (
    S3ActionTemplate,
    V3_S3_FULL_FEATURE_NAMES,
    load_v3_s3_bundle,
    rank_s3_sequences,
)
from experiments.v3_s3_training import (
    _runtime_reachable_steps,
    _sequence_rows,
    _state_groups,
)


V3_S3_ORACLE_AUDIT_SCHEMA = "lns2.v3_s3_oracle_audit.v1"
QUALITY_RETENTION = 0.98
RUNTIME_REPORT_SCHEMA = "lns2.v3_s3_runtime_comparison_report.v1"


def _atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write an empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({name for row in materialized for name in row})
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(1e-12, float(denominator))


def _runtime_prefix(row: dict[str, Any]) -> dict[str, float]:
    trajectory = list(map(int, row["conflict_trajectory"]))
    initial_conflicts = int(trajectory[0])
    final_conflicts = initial_conflicts
    total_seconds = 0.0
    selection_seconds = 0.0
    positive_progress = False
    feasible = False
    for step in sorted(row["steps"], key=lambda value: int(value["step"])):
        total_seconds += float(step["total_seconds"])
        selection_seconds += float(step.get("selection_seconds", 0.0))
        final_conflicts = int(step["conflicts_after"])
        reduction = int(step["conflicts_before"]) - final_conflicts
        positive_progress = positive_progress or reduction > 0
        outcome = str(step.get("repair_outcome"))
        feasible = feasible or outcome == "feasible" or final_conflicts == 0
        if outcome in {"hard_failure", "accepted_noop"}:
            break
    return {
        "conflict_reduction": float(initial_conflicts - final_conflicts),
        "total_seconds": float(total_seconds),
        "selection_seconds": float(selection_seconds),
        "no_progress": float(not positive_progress),
        "feasible": float(feasible),
    }


def _first_step_prefix(row: dict[str, Any]) -> dict[str, float]:
    steps = list(sorted(row["steps"], key=lambda value: int(value["step"])))
    if not steps:
        raise ValueError("baseline trial contains no repair step")
    step = dict(steps[0])
    conflicts_before = int(step["conflicts_before"])
    conflicts_after = int(step["conflicts_after"])
    outcome = str(step.get("repair_outcome"))
    reduction = float(conflicts_before - conflicts_after)
    return {
        "conflict_reduction": reduction,
        "total_seconds": float(step["total_seconds"]),
        "selection_seconds": float(step.get("selection_seconds", 0.0)),
        "no_progress": float(
            outcome in {"hard_failure", "accepted_noop"} or reduction <= 0.0
        ),
        "feasible": float(outcome == "feasible" or conflicts_after == 0),
    }


def _mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot aggregate empty metrics")
    return {
        name: statistics.fmean(float(row[name]) for row in materialized)
        for name in (
            "conflict_reduction",
            "total_seconds",
            "selection_seconds",
            "no_progress",
            "feasible",
        )
    }


def _sequence_metrics(row: dict[str, Any]) -> dict[str, float]:
    actual = dict(row["actual"])
    return {
        "conflict_reduction": float(
            actual["runtime_prefix_net_conflict_reduction"]
        ),
        "total_seconds": float(actual["runtime_prefix_total_seconds"]),
        "selection_seconds": float(
            actual["runtime_prefix_selection_seconds"]
        ),
        "no_progress": float(actual["runtime_prefix_no_progress_rate"]),
        "feasible": float(actual["runtime_prefix_feasible_rate"]),
    }


def _sequence_first_step_metrics(row: dict[str, Any]) -> dict[str, float]:
    prefixes = []
    for trial in row["actual"]["trials"]:
        reachable = [
            dict(step)
            for step in _runtime_reachable_steps(trial)
            if bool(step.get("executed"))
        ]
        if not reachable:
            raise ValueError(
                f"sequence has no executable first step: {row['sequence_id']}"
            )
        step = reachable[0]
        outcome = str(step.get("repair_outcome"))
        reduction = float(step["conflict_reduction"])
        prefixes.append(
            {
                "conflict_reduction": reduction,
                "total_seconds": float(step["total_seconds"]),
                "selection_seconds": float(
                    step.get("selection_seconds", 0.0)
                ),
                "no_progress": float(
                    outcome in {"hard_failure", "accepted_noop"}
                    or reduction <= 0.0
                ),
                "feasible": float(
                    outcome == "feasible"
                    or int(step["conflicts_after"]) == 0
                ),
            }
        )
    return _mean_metrics(prefixes)


def _metric_order_key(
    sequence_id: str,
    metrics: dict[str, float],
    *,
    objective: str,
) -> tuple[Any, ...]:
    reduction = float(metrics["conflict_reduction"])
    seconds = float(metrics["total_seconds"])
    efficiency = _safe_ratio(reduction, seconds)
    no_progress = float(metrics["no_progress"])
    feasible = float(metrics["feasible"])
    if objective == "efficiency":
        return (
            -round(efficiency, 12),
            -round(reduction, 12),
            round(no_progress, 12),
            -round(feasible, 12),
            round(seconds, 12),
            sequence_id,
        )
    if objective == "reduction":
        return (
            -round(reduction, 12),
            round(no_progress, 12),
            -round(feasible, 12),
            round(seconds, 12),
            sequence_id,
        )
    if objective == "time":
        return (
            round(seconds, 12),
            round(no_progress, 12),
            -round(feasible, 12),
            -round(reduction, 12),
            sequence_id,
        )
    raise ValueError(f"unsupported oracle objective: {objective}")


def select_oracle_row(
    rows: list[dict[str, Any]],
    *,
    objective: str,
    quality_retention: float = QUALITY_RETENTION,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("oracle selection requires at least one sequence")
    eligible = list(rows)
    if objective == "quality_time":
        maximum = max(
            float(_sequence_metrics(row)["conflict_reduction"])
            for row in rows
        )
        floor = float(quality_retention) * maximum
        eligible = [
            row
            for row in rows
            if float(_sequence_metrics(row)["conflict_reduction"])
            + 1e-12
            >= floor
        ]
        objective = "time"
    return min(
        eligible,
        key=lambda row: _metric_order_key(
            str(row["sequence_id"]),
            _sequence_metrics(row),
            objective=objective,
        ),
    )


def _first_template_key(row: dict[str, Any]) -> str:
    templates = list(row["templates"])
    if not templates:
        raise ValueError("S3 sequence has no templates")
    return str(dict(templates[0])["template_key"])


def select_first_action_oracle(
    rows: list[dict[str, Any]],
) -> tuple[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    for row in rows:
        grouped[_first_template_key(row)].append(
            _sequence_first_step_metrics(row)
        )
    choices = [
        (template, _mean_metrics(values))
        for template, values in sorted(grouped.items())
    ]
    return min(
        choices,
        key=lambda item: _metric_order_key(
            item[0], item[1], objective="efficiency"
        ),
    )


def _dense_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "feature_profile": "v3_s3",
            "feature_names": V3_S3_FULL_FEATURE_NAMES,
            "feature_values": tuple(map(float, row["features"])),
        }
        for row in rows
    ]


def _model_choice(
    rows: list[dict[str, Any]],
    bundle: Any,
) -> tuple[dict[str, Any], bool]:
    predictions = bundle.predict(_dense_prediction_rows(rows))
    sequences = [
        tuple(
            S3ActionTemplate.from_payload(value)
            for value in row["templates"]
        )
        for row in rows
    ]
    order = rank_s3_sequences(sequences, predictions, bundle.thresholds)
    risk_relaxed = False
    if not order:
        order = rank_s3_sequences(
            sequences,
            predictions,
            bundle.thresholds,
            allow_risk_relaxation=True,
        )
        risk_relaxed = bool(order)
    if not order:
        raise ValueError(
            f"model has no selectable sequence for state {rows[0]['state_id']}"
        )
    return rows[order[0]], risk_relaxed


def _baseline_index(
    path: Path,
) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = collections.defaultdict(list)
    for row in _read_jsonl(path):
        key = (
            str(row["split"]),
            str(row["state_id"]),
            str(row["controller"]),
        )
        grouped[key].append(row)
    result = {}
    for key, rows in grouped.items():
        indices = sorted(int(row["trial_index"]) for row in rows)
        if indices != [0, 1]:
            raise ValueError(
                f"baseline does not contain paired trials 0/1: {key}"
            )
        result[key] = {
            **{
                f"h3_{name}": value
                for name, value in _mean_metrics(
                    _runtime_prefix(row) for row in rows
                ).items()
            },
            **{
                f"h1_{name}": value
                for name, value in _mean_metrics(
                    _first_step_prefix(row) for row in rows
                ).items()
            },
        }
    return result


def _strategy_summary(
    state_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for split in sorted({str(row["split"]) for row in state_rows}):
        split_rows = [row for row in state_rows if row["split"] == split]
        agent_cells: list[int | str] = [
            "all",
            *sorted({int(row["agent_count"]) for row in split_rows}),
        ]
        for agents in agent_cells:
            subset = [
                row
                for row in split_rows
                if agents == "all" or int(row["agent_count"]) == agents
            ]
            for strategy in (
                "v2_full",
                "model_s3",
                "oracle_s3_efficiency",
                "oracle_s3_quality_time",
                "oracle_s3_reduction",
                "oracle_h1_efficiency",
                "v2_h1",
            ):
                prefix = strategy
                reductions = [
                    float(row[f"{prefix}_conflict_reduction"])
                    for row in subset
                ]
                seconds = [
                    float(row[f"{prefix}_total_seconds"])
                    for row in subset
                ]
                summaries.append(
                    {
                        "split": split,
                        "agent_count": agents,
                        "strategy": strategy,
                        "state_count": len(subset),
                        "mean_conflict_reduction": statistics.fmean(
                            reductions
                        ),
                        "mean_total_seconds": statistics.fmean(seconds),
                        "pooled_conflict_reduction_per_second": _safe_ratio(
                            math.fsum(reductions), math.fsum(seconds)
                        ),
                        "mean_no_progress_rate": statistics.fmean(
                            float(row[f"{prefix}_no_progress"])
                            for row in subset
                        ),
                        "mean_feasible_rate": statistics.fmean(
                            float(row[f"{prefix}_feasible"])
                            for row in subset
                        ),
                    }
                )
    return summaries


def _runtime_evidence(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = dict(read_json(path))
        if str(payload.get("schema")) != RUNTIME_REPORT_SCHEMA:
            raise ValueError(
                f"unexpected v3-S3 runtime report schema: {path}"
            )
        if not bool(payload.get("diagnostic_only")):
            raise ValueError("oracle audit only accepts diagnostic runtime reports")
        aggregate = dict(payload["aggregate"])
        controllers = dict(aggregate["controllers"])
        v2 = dict(controllers["v2-full"])
        v3 = dict(controllers["v3-s3"])
        rows.append(
            {
                "report": f"{path.parent.name}/{path.name}",
                "sha256": sha256_file(path),
                "paired_episode_count": int(
                    aggregate["paired_episode_count"]
                ),
                "common_success_count": int(
                    aggregate["common_success_count"]
                ),
                "pooled_time_change_fraction": float(
                    aggregate[
                        "pooled_v3_s3_time_change_fraction_common_success"
                    ]
                ),
                "mean_iteration_change_fraction": _safe_ratio(
                    float(v3["mean_repair_iterations"])
                    - float(v2["mean_repair_iterations"]),
                    float(v2["mean_repair_iterations"]),
                ),
                "normalized_wall_auc_change_fraction": _safe_ratio(
                    float(v3["mean_normalized_wall_clock_conflict_auc"])
                    - float(v2["mean_normalized_wall_clock_conflict_auc"]),
                    float(v2["mean_normalized_wall_clock_conflict_auc"]),
                ),
                "v3_faster_pair_count": int(
                    aggregate["v3_s3_faster_common_success_count"]
                ),
            }
        )
    return rows


def _summary_lookup(
    rows: list[dict[str, Any]],
    *,
    split: str,
    strategy: str,
    agents: int | str = "all",
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if str(row["split"]) == split
        and str(row["strategy"]) == strategy
        and str(row["agent_count"]) == str(agents)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"missing summary cell: {split}/{agents}/{strategy}"
        )
    return matches[0]


def _report_markdown(report: dict[str, Any]) -> str:
    diagnostic = dict(report["diagnostic"])
    runtime = list(report["runtime_evidence"])
    lines = [
        "# v3-S3 Oracle upper-bound and regret audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        (
            "This is a retrospective diagnostic over the existing paired S3 "
            "collection. Oracle rows use observed labels and are not deployable "
            "models or independent validation."
        ),
        "",
        "| Strategy | Reduction | Seconds | Reduction / second | No progress | Feasible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "v2_full",
        "model_s3",
        "oracle_s3_efficiency",
        "oracle_s3_quality_time",
        "oracle_s3_reduction",
    ):
        row = dict(diagnostic[name])
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    f"{float(row['mean_conflict_reduction']):.4f}",
                    f"{float(row['mean_total_seconds']):.6f}",
                    f"{float(row['pooled_conflict_reduction_per_second']):.4f}",
                    f"{float(row['mean_no_progress_rate']):.3%}",
                    f"{float(row['mean_feasible_rate']):.3%}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decomposition",
            "",
            (
                "- Model exact sequence match with the efficiency Oracle: "
                f"{float(diagnostic['model_oracle_exact_match_rate']):.3%}."
            ),
            (
                "- Model first-template match with the efficiency Oracle: "
                f"{float(diagnostic['model_oracle_first_template_match_rate']):.3%}."
            ),
            (
                "- Observed Oracle efficiency headroom over v2: "
                f"{float(diagnostic['oracle_efficiency_headroom_vs_v2']):+.2%}."
            ),
            (
                "- Model captured fraction of positive Oracle-v2 headroom: "
                f"{float(diagnostic['model_headroom_capture_fraction']):.3%}."
            ),
            (
                "- Quality-constrained Oracle reduction retention versus v2: "
                f"{float(diagnostic['quality_oracle_reduction_retention_vs_v2']):.3%}."
            ),
            "",
            "## Closed-loop evidence",
            "",
        ]
    )
    if runtime:
        lines.extend(
            [
                "| Report | Pairs | Time change | Iteration change | Wall AUC change |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in runtime:
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(row["report"]),
                        str(row["paired_episode_count"]),
                        f"{float(row['pooled_time_change_fraction']):+.2%}",
                        f"{float(row['mean_iteration_change_fraction']):+.2%}",
                        f"{float(row['normalized_wall_auc_change_fraction']):+.2%}",
                    )
                )
                + " |"
            )
    else:
        lines.append("- No closed-loop runtime report was supplied.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "- The Oracle is optimistic because it selects after seeing the "
                "paired outcomes."
            ),
            (
                "- The collection observes at most the registered three-step "
                "prefix. It does not contain true remaining time-to-feasible "
                "labels."
            ),
            (
                "- Therefore this audit can diagnose selection regret and a "
                "local-versus-closed-loop contradiction, but cannot train or "
                "validate a long-horizon value controller."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def audit_v3_s3_oracle(
    *,
    source: str | Path,
    output: str | Path,
    runtime_reports: Iterable[str | Path] = (),
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    collection_root = source_root / "collection"
    controller_root = source_root / "controller"
    required = (
        collection_root / "sequence_features.jsonl",
        collection_root / "sequence_trials.jsonl",
        collection_root / "external_baselines.jsonl",
        controller_root / "v3_s3_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v3-S3 audit source is incomplete: {missing}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"v3-S3 Oracle audit output is not empty: {output_root}"
        )

    train, diagnostic = _sequence_rows(required[0], required[1])
    bundle = load_v3_s3_bundle(controller_root)
    planner_overhead = float(
        dict(dict(bundle.report.get("diagnostic", {})).get("v3_s3", {})).get(
            "planner_inference_seconds_per_state", 0.0
        )
    )
    if not math.isfinite(planner_overhead) or planner_overhead < 0.0:
        raise ValueError("v3-S3 bundle reports invalid planner overhead")
    baselines = _baseline_index(required[2])
    state_rows = []
    risk_relaxed_count = 0

    for split, rows in (("policy_train", train), ("policy_validation", diagnostic)):
        for indices in _state_groups(rows):
            registered = [
                rows[index]
                for index in indices
                if bool(rows[index]["runtime_registered"])
            ]
            if not registered:
                raise ValueError(
                    f"state has no runtime-registered sequence: {rows[indices[0]]['state_id']}"
                )
            state_id = str(registered[0]["state_id"])
            model, relaxed = _model_choice(registered, bundle)
            risk_relaxed_count += int(relaxed)
            oracle_efficiency = select_oracle_row(
                registered, objective="efficiency"
            )
            oracle_quality = select_oracle_row(
                registered,
                objective="quality_time",
                quality_retention=QUALITY_RETENTION,
            )
            oracle_reduction = select_oracle_row(
                registered, objective="reduction"
            )
            h1_template, h1_metrics = select_first_action_oracle(registered)
            baseline_key = (split, state_id, "v2-full")
            if baseline_key not in baselines:
                raise ValueError(f"state lacks v2 baseline: {state_id}")
            baseline = baselines[baseline_key]
            model_metrics = _sequence_metrics(model)
            model_metrics["total_seconds"] += planner_overhead
            model_metrics["selection_seconds"] += planner_overhead
            oracle_efficiency_metrics = _sequence_metrics(
                oracle_efficiency
            )
            oracle_quality_metrics = _sequence_metrics(oracle_quality)
            oracle_reduction_metrics = _sequence_metrics(oracle_reduction)
            row = {
                "split": split,
                "state_id": state_id,
                "map_id": str(registered[0]["map_id"]),
                "layout_mode": str(registered[0]["layout_mode"]),
                "agent_count": int(registered[0]["agent_count"]),
                "registered_sequence_count": len(registered),
                "model_sequence_id": str(model["sequence_id"]),
                "model_first_template": _first_template_key(model),
                "model_risk_relaxed": bool(relaxed),
                "model_planner_overhead_seconds": planner_overhead,
                "oracle_s3_efficiency_sequence_id": str(
                    oracle_efficiency["sequence_id"]
                ),
                "oracle_s3_efficiency_first_template": _first_template_key(
                    oracle_efficiency
                ),
                "oracle_s3_quality_time_sequence_id": str(
                    oracle_quality["sequence_id"]
                ),
                "oracle_s3_reduction_sequence_id": str(
                    oracle_reduction["sequence_id"]
                ),
                "oracle_h1_efficiency_first_template": h1_template,
                "model_oracle_exact_match": str(model["sequence_id"])
                == str(oracle_efficiency["sequence_id"]),
                "model_oracle_first_template_match": _first_template_key(
                    model
                )
                == _first_template_key(oracle_efficiency),
            }
            for prefix, metrics in (
                (
                    "v2_full",
                    {
                        name: float(baseline[f"h3_{name}"])
                        for name in (
                            "conflict_reduction",
                            "total_seconds",
                            "selection_seconds",
                            "no_progress",
                            "feasible",
                        )
                    },
                ),
                (
                    "v2_h1",
                    {
                        name: float(baseline[f"h1_{name}"])
                        for name in (
                            "conflict_reduction",
                            "total_seconds",
                            "selection_seconds",
                            "no_progress",
                            "feasible",
                        )
                    },
                ),
                ("model_s3", model_metrics),
                ("oracle_s3_efficiency", oracle_efficiency_metrics),
                ("oracle_s3_quality_time", oracle_quality_metrics),
                ("oracle_s3_reduction", oracle_reduction_metrics),
                ("oracle_h1_efficiency", h1_metrics),
            ):
                for name, value in metrics.items():
                    row[f"{prefix}_{name}"] = float(value)
                row[f"{prefix}_efficiency"] = _safe_ratio(
                    metrics["conflict_reduction"],
                    metrics["total_seconds"],
                )
            row["model_utility_regret"] = (
                float(row["oracle_s3_efficiency_efficiency"])
                - float(row["model_s3_efficiency"])
            )
            state_rows.append(row)

    summaries = _strategy_summary(state_rows)
    diagnostic_cells = {
        strategy: _summary_lookup(
            summaries,
            split="policy_validation",
            strategy=strategy,
        )
        for strategy in (
            "v2_full",
            "model_s3",
            "oracle_s3_efficiency",
            "oracle_s3_quality_time",
            "oracle_s3_reduction",
            "oracle_h1_efficiency",
            "v2_h1",
        )
    }
    diagnostic_states = [
        row for row in state_rows if row["split"] == "policy_validation"
    ]
    v2_efficiency = float(
        diagnostic_cells["v2_full"][
            "pooled_conflict_reduction_per_second"
        ]
    )
    model_efficiency = float(
        diagnostic_cells["model_s3"][
            "pooled_conflict_reduction_per_second"
        ]
    )
    oracle_efficiency = float(
        diagnostic_cells["oracle_s3_efficiency"][
            "pooled_conflict_reduction_per_second"
        ]
    )
    positive_headroom = max(0.0, oracle_efficiency - v2_efficiency)
    runtime_paths = [Path(path).resolve() for path in runtime_reports]
    runtime = _runtime_evidence(runtime_paths)
    local_model_better = model_efficiency > v2_efficiency
    runtime_consistently_worse = bool(runtime) and all(
        float(row["pooled_time_change_fraction"]) > 0.0
        and float(row["normalized_wall_auc_change_fraction"]) > 0.0
        and float(row["mean_iteration_change_fraction"]) > 0.0
        for row in runtime
    )
    oracle_has_headroom = oracle_efficiency >= 1.10 * v2_efficiency
    decision = (
        "proceed_to_long_horizon_value_pilot"
        if oracle_has_headroom
        and local_model_better
        and runtime_consistently_worse
        else (
            "improve_s3_prediction_before_closed_loop"
            if oracle_has_headroom
            and model_efficiency < 0.90 * oracle_efficiency
            else "retire_s3_without_new_collection"
        )
    )
    diagnostic_report = {
        **diagnostic_cells,
        "model_oracle_exact_match_rate": statistics.fmean(
            float(row["model_oracle_exact_match"])
            for row in diagnostic_states
        ),
        "model_oracle_first_template_match_rate": statistics.fmean(
            float(row["model_oracle_first_template_match"])
            for row in diagnostic_states
        ),
        "mean_model_utility_regret": statistics.fmean(
            float(row["model_utility_regret"])
            for row in diagnostic_states
        ),
        "oracle_efficiency_headroom_vs_v2": _safe_ratio(
            oracle_efficiency - v2_efficiency, v2_efficiency
        ),
        "model_efficiency_change_vs_v2": _safe_ratio(
            model_efficiency - v2_efficiency, v2_efficiency
        ),
        "model_headroom_capture_fraction": (
            (model_efficiency - v2_efficiency) / positive_headroom
            if positive_headroom > 0.0
            else 0.0
        ),
        "quality_oracle_reduction_retention_vs_v2": _safe_ratio(
            float(
                diagnostic_cells["oracle_s3_quality_time"][
                    "mean_conflict_reduction"
                ]
            ),
            float(
                diagnostic_cells["v2_full"]["mean_conflict_reduction"]
            ),
        ),
    }
    report = {
        "schema": V3_S3_ORACLE_AUDIT_SCHEMA,
        "decision": decision,
        "diagnostic_only": True,
        "source": {
            "collection_schema": str(
                read_json(collection_root / "collection_report.json")[
                    "schema"
                ]
            ),
            "sequence_features_sha256": sha256_file(required[0]),
            "sequence_trials_sha256": sha256_file(required[1]),
            "external_baselines_sha256": sha256_file(required[2]),
            "bundle_source_fingerprint": str(
                bundle.manifest["source_fingerprint"]
            ),
            "model_planner_overhead_seconds_per_state": planner_overhead,
        },
        "coverage": {
            "state_count": len(state_rows),
            "training_state_count": sum(
                row["split"] == "policy_train" for row in state_rows
            ),
            "diagnostic_state_count": len(diagnostic_states),
            "risk_relaxed_state_count": risk_relaxed_count,
            "runtime_report_count": len(runtime),
            "runtime_paired_episode_count": sum(
                int(row["paired_episode_count"]) for row in runtime
            ),
        },
        "diagnostic": diagnostic_report,
        "runtime_evidence": runtime,
        "decision_checks": {
            "oracle_has_at_least_10pct_local_headroom": oracle_has_headroom,
            "model_local_efficiency_beats_v2": local_model_better,
            "closed_loop_repeats_consistently_worse": runtime_consistently_worse,
            "true_remaining_time_to_feasible_labels_available": False,
        },
        "limitations": [
            "Oracle selection observes paired outcomes and is optimistic.",
            "Policy-validation maps are diagnostic, not untouched promotion data.",
            "Registered S3 sequences do not enumerate every possible 18^3 sequence.",
            "The collection ends after the registered three-step prefix and has no true remaining time-to-feasible target.",
            "Closed-loop evidence is a four-task, one-seed repeated diagnostic subset.",
        ],
        "recommended_next_step": (
            "Collect a small map-isolated variable-horizon pilot and train an "
            "independent actual-candidate cost-to-go controller; keep v2-full "
            "as the default and archive v3-S3."
            if decision == "proceed_to_long_horizon_value_pilot"
            else "Follow the decision-specific branch before collecting more data."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(output_root / "oracle_state_comparison.csv", state_rows)
    _atomic_write_csv(output_root / "oracle_strategy_summary.csv", summaries)
    if runtime:
        _atomic_write_csv(output_root / "runtime_evidence.csv", runtime)
    _write_json(output_root / "v3_s3_oracle_audit_report.json", report)
    (output_root / "v3_s3_oracle_audit_report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    return report
