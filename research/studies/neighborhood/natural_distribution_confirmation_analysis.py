from __future__ import annotations

import collections
import math
import time
from pathlib import Path
from typing import Any

from research.studies.neighborhood.natural_distribution_confirmation import (
    _number_summary,
    conflict_density,
    conflict_severity,
)
from research.studies.neighborhood.realized_neighborhood_ranking_audit import (
    _beats_simple_baseline,
    _oracle_support,
    _uniform_summary,
    build_ranking_index,
    compare_records,
    evaluate_model,
    feature_diagnostics,
    internal_coverage_records,
    map_bootstrap,
    oracle_records,
    pairwise_accuracy,
    summarize_records,
    uniform_random_records,
)
from research.studies.neighborhood.realized_ranking_confirmation_analysis import (
    PRIMARY_PROFILES,
    _configuration,
    _load_frozen_models,
    freeze_confirmation_models,
)
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl


SCHEMA_VERSION = 1


def _baseline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if str(row.get("status")) in {"ok", "resumed"}]
    summaries = [dict(row["summary"]) for row in valid]
    return {
        "episode_count": len(rows),
        "valid_count": len(valid),
        "error_count": len(rows) - len(valid),
        "initial_feasible_count": sum(int(row["initial_conflicts"]) == 0 for row in summaries),
        "repairable_count": sum(bool(row["repairable"]) for row in summaries),
        "success_count": sum(bool(row["success"]) for row in summaries),
        "success_rate": (
            sum(bool(row["success"]) for row in summaries) / len(summaries)
            if summaries
            else None
        ),
        "initial_conflicts": _number_summary(
            [float(row["initial_conflicts"]) for row in summaries]
        ),
        "final_conflicts": _number_summary(
            [float(row["final_conflicts"]) for row in summaries]
        ),
        "conflict_auc": _number_summary([float(row["conflict_auc"]) for row in summaries]),
        "time_to_feasible": _number_summary(
            [
                float(row["time_to_feasible"])
                for row in summaries
                if row.get("time_to_feasible") is not None
            ]
        ),
        "repair_iterations": _number_summary(
            [float(row["repair_iterations"]) for row in summaries]
        ),
    }


def _grouped_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = {}
    for field in ("layout_mode", "task_variant", "agent_count"):
        values: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            values[str(row.get(field, "unknown"))].append(row)
        grouped[field] = {
            name: _baseline_summary(group) for name, group in sorted(values.items())
        }
    return grouped


def _annotate_index(
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    source_conflicts: dict[str, int],
) -> dict[str, str]:
    state_severity = {}
    for row in rows:
        state_id = str(row["state_id"])
        if state_id not in source_conflicts:
            raise ValueError(f"missing source conflict count for {state_id}")
        conflicts = int(source_conflicts[state_id])
        agents = int(row["agent_count"])
        density = conflict_density(conflicts, agents)
        severity = conflict_severity(density, thresholds)
        row["initial_conflicts"] = conflicts
        row["conflict_density"] = density
        row["conflict_severity"] = severity
        if state_id in state_severity and state_severity[state_id] != severity:
            raise ValueError("candidate rows disagree on source conflict severity")
        state_severity[state_id] = severity
    return state_severity


def _selector_summary(
    name: str, records: dict[str, dict[str, Any]], pairwise: float | None = None
) -> dict[str, Any]:
    if not records:
        return {"state_count": 0}
    if name == "uniform_random":
        return _uniform_summary(records)
    return summarize_records(records, pairwise)


def _severity_summaries(
    records: dict[str, dict[str, dict[str, Any]]],
    state_severity: dict[str, str],
) -> dict[str, Any]:
    result = {}
    for severity in ("low", "medium", "high"):
        selectors = {}
        for name, values in records.items():
            subset = {
                state_id: row
                for state_id, row in values.items()
                if state_severity.get(state_id) == severity
            }
            selectors[name] = _selector_summary(name, subset)
        result[severity] = {
            "state_count": sum(value == severity for value in state_severity.values()),
            "selectors": selectors,
        }
    return result


def natural_acceptance(
    summaries: dict[str, Any],
    comparison: dict[str, Any],
    bootstrap: dict[str, Any],
    oracle_support: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    realized = summaries["realized_dynamic"]
    simple = {
        name: _beats_simple_baseline(realized, summaries[name])
        for name in ("uniform_random", "internal_conflict_coverage")
    }
    map_count = int(comparison["map_count"])
    minimum_maps = math.ceil(
        float(thresholds["minimum_map_fraction_no_worse"]) * map_count
    )
    gates = {
        "minimum_top1_gain": comparison["pareto_top1_gain"]
        >= float(thresholds["minimum_top1_gain"]),
        "minimum_conflict_regret_reduction": comparison[
            "relative_conflict_regret_reduction"
        ]
        >= float(thresholds["minimum_conflict_regret_reduction"]),
        "no_significant_bootstrap_degradation": bootstrap["hit_gain_95_ci"][1]
        >= 0.0
        and bootstrap["conflict_improvement_95_ci"][1] >= 0.0,
        "maps_no_worse": int(comparison["maps_no_worse"]) >= minimum_maps,
        "beats_uniform_random": simple["uniform_random"],
        "beats_internal_coverage": simple["internal_conflict_coverage"],
        "no_unsupported_size_collapse": not (
            oracle_support["multiple_sizes_supported"]
            and float(realized["maximum_size_share"])
            > float(thresholds["maximum_size_share"])
        ),
    }
    passed = all(gates.values())
    return {
        "passed": passed,
        "gates": gates,
        "eligible_map_count": map_count,
        "minimum_maps_no_worse": minimum_maps,
        "decision": (
            "proceed_to_fresh_closed_loop_confirmation"
            if passed
            else "keep_rl_paused_and_reconsider_candidate_or_repair_order"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    natural = report["natural_distribution"]
    comparison = report["comparisons"]["realized_dynamic_vs_proposal_dynamic"]
    summaries = report["summaries"]
    lines = [
        "# InitLNS natural-distribution independent confirmation",
        "",
        f"Decision: `{report['acceptance']['decision']}`",
        "",
        "## Natural cohort",
        "",
        f"- Valid resets: {report['qualification']['valid_count']}/48",
        f"- Initial feasible: {report['qualification']['initial_feasible_count']}",
        f"- Nonzero conflict states: {report['qualification']['nonzero_state_count']}",
        f"- Adaptive success: {natural['baseline']['success_count']}/{natural['baseline']['valid_count']}",
        "",
        "## Frozen ranking",
        "",
        f"- Proposal top-1: {summaries['proposal_dynamic']['pareto_top1_hit_rate']:.1%}",
        f"- Realized top-1: {summaries['realized_dynamic']['pareto_top1_hit_rate']:.1%}",
        f"- Top-1 gain: {comparison['pareto_top1_gain']:+.1%}",
        f"- Conflict-regret reduction: {comparison['relative_conflict_regret_reduction']:.1%}",
        f"- Maps no worse: {comparison['maps_no_worse']}/{comparison['map_count']}",
        "",
        "## Gates",
        "",
    ]
    for name, passed in report["acceptance"]["gates"].items():
        lines.append(f"- {name}: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "Zero-conflict tasks remain in the natural cohort but have no ranking label. "
            "Static context and low/medium/high severity results are exploratory only.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_natural_confirmation_analysis(
    collection: str | Path,
    config_path: str | Path,
    frozen_models: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, _ = _configuration(config_path)
    root = Path(collection).resolve()
    qualification = _read_json(root / "qualification_report.json")
    if not bool(qualification.get("formal")) or not bool(qualification.get("passed")):
        raise ValueError("formal natural qualification did not pass")
    baseline_rows = _read_jsonl(root / "baseline_manifest.jsonl")
    if len(baseline_rows) != 48 or any(
        str(row.get("status")) not in {"ok", "resumed"} for row in baseline_rows
    ):
        raise ValueError("formal Adaptive baseline is incomplete")
    frozen_root = Path(frozen_models).resolve()
    freeze_manifest = _read_json(frozen_root / "freeze_manifest.json")
    if freeze_manifest["development_index_sha256"] != str(
        config["expected_development_index_sha256"]
    ).lower():
        raise ValueError("frozen model development index differs from registration")
    if bool(freeze_manifest.get("confirmation_labels_seen")):
        raise ValueError("frozen model manifest has seen confirmation labels")
    models = _load_frozen_models(frozen_root, freeze_manifest)
    rows, integrity = build_ranking_index(
        root,
        expected_states=None,
        expected_candidates=None,
        expected_outcomes=None,
        expected_trials=int(config["evaluation_trials"]),
        expected_maps=None,
        expected_split=str(config["expected_split"]),
    )
    if integrity["state_count"] < int(config["minimum_nonzero_states"]):
        raise ValueError("ranking index contains too few nonzero states")
    if integrity["map_count"] < int(config["minimum_active_maps"]):
        raise ValueError("ranking index contains too few active maps")
    candidate_sources = _read_jsonl(root / "candidates.jsonl")
    source_conflicts = {
        str(row["state_id"]): int(row["state"]["num_of_colliding_pairs"])
        for row in candidate_sources
    }
    state_severity = _annotate_index(
        rows, dict(config["severity_thresholds"]), source_conflicts
    )
    learned_records = {
        profile: evaluate_model(rows, models[profile], profile)
        for profile in PRIMARY_PROFILES
    }
    records = {
        "uniform_random": uniform_random_records(rows),
        "internal_conflict_coverage": internal_coverage_records(rows),
        "oracle": oracle_records(rows),
        **learned_records,
    }
    summaries = {
        "uniform_random": _uniform_summary(records["uniform_random"]),
        "internal_conflict_coverage": summarize_records(
            records["internal_conflict_coverage"]
        ),
        "oracle": summarize_records(records["oracle"]),
        **{
            profile: summarize_records(
                records[profile], pairwise_accuracy(rows, models[profile])
            )
            for profile in PRIMARY_PROFILES
        },
    }
    comparisons = {
        "realized_dynamic_vs_proposal_dynamic": compare_records(
            records["proposal_dynamic"], records["realized_dynamic"]
        ),
        "realized_context_vs_realized_dynamic_exploratory": compare_records(
            records["realized_dynamic"], records["realized_context"]
        ),
    }
    bootstrap = map_bootstrap(
        records["proposal_dynamic"],
        records["realized_dynamic"],
        int(config["bootstrap_samples"]),
    )
    oracle_support = _oracle_support(rows)
    acceptance = natural_acceptance(
        summaries,
        comparisons["realized_dynamic_vs_proposal_dynamic"],
        bootstrap,
        oracle_support,
        dict(config["thresholds"]),
    )
    prediction_rows = []
    for selector, values in records.items():
        for state_id, row in sorted(values.items()):
            prediction_rows.append(
                {
                    **dict(row),
                    "selector": selector,
                    "state_id": state_id,
                    "conflict_severity": state_severity[state_id],
                }
            )
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "natural_confirmation_index.jsonl", rows)
    _write_jsonl(output_root / "predictions.jsonl", prediction_rows)
    baseline = _baseline_summary(baseline_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "qualification": qualification,
        "integrity": integrity,
        "frozen_models": freeze_manifest,
        "natural_distribution": {
            "qualification": qualification["natural_distribution"],
            "baseline": baseline,
            "baseline_grouped": _grouped_baseline(baseline_rows),
        },
        "feature_diagnostics": feature_diagnostics(rows),
        "summaries": summaries,
        "severity_summaries": _severity_summaries(records, state_severity),
        "comparisons": comparisons,
        "map_bootstrap": bootstrap,
        "oracle_support": oracle_support,
        "acceptance": acceptance,
        "pre_registration": {
            "primary_profile": "realized_dynamic",
            "baseline_profile": "proposal_dynamic",
            "static_context_role": "exploratory only",
            "severity_role": "exploratory only",
            "zero_conflict_role": "natural cohort only; no ranking label",
            "confirmation_model_training": False,
        },
        "timings_seconds": {"total": time.perf_counter() - started},
    }
    _write_json(output_root / "natural_distribution_confirmation.json", report)
    (output_root / "natural_distribution_confirmation.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


__all__ = [
    "freeze_confirmation_models",
    "natural_acceptance",
    "run_natural_confirmation_analysis",
]
