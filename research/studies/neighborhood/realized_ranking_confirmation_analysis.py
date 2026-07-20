from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

from experiments._common import sha256_file as _sha256
from research.studies.neighborhood.realized_neighborhood_ranking_audit import (
    _beats_simple_baseline,
    _oracle_support,
    _read_json,
    _read_jsonl,
    _uniform_summary,
    _write_json,
    _write_jsonl,
    build_ranking_index,
    compare_records,
    evaluate_model,
    feature_diagnostics,
    internal_coverage_records,
    map_bootstrap,
    oracle_records,
    pairwise_accuracy,
    summarize_records,
    train_pairwise_model,
    uniform_random_records,
)


SCHEMA_VERSION = 1
PRIMARY_PROFILES = ("proposal_dynamic", "realized_dynamic", "realized_context")


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _configuration(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    config = _read_json(path)
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported confirmation analysis config")
    return config, path.parents[1]


def freeze_confirmation_models(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config, project_root = _configuration(config_path)
    audit_root = _resolve(project_root, str(config["development_audit"]))
    index_path = audit_root / "ranking_index.jsonl"
    report_path = audit_root / "realized_neighborhood_ranking_audit.json"
    index_sha = _sha256(index_path)
    report_sha = _sha256(report_path)
    if index_sha != str(config["expected_development_index_sha256"]).lower():
        raise ValueError("development ranking index SHA256 mismatch")
    if report_sha != str(config["expected_development_report_sha256"]).lower():
        raise ValueError("development audit report SHA256 mismatch")
    report = _read_json(report_path)
    if not bool(report.get("acceptance", {}).get("passed")):
        raise ValueError("development realized-ranking gate did not pass")
    rows = _read_jsonl(index_path)
    if len(rows) != 412 or len({str(row["state_id"]) for row in rows}) != 23:
        raise ValueError("development index is not the registered 23-state/412-candidate set")
    if {str(row["split"]) for row in rows} != {"probe"}:
        raise ValueError("development index contains a forbidden split")
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    models = {}
    model_rows = []
    for profile in PRIMARY_PROFILES:
        model, pairs = train_pairwise_model(rows, profile, dict(config["model"]))
        path = output_root / "models" / f"pairwise__{profile}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(model, stream)
        models[profile] = model
        model_rows.append(
            {
                "profile": profile,
                "model_file": path.relative_to(output_root).as_posix(),
                "model_sha256": _sha256(path),
                "feature_count": len(model.feature_names),
                "dominance_pair_count": pairs,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "development_index": str(index_path),
        "development_index_sha256": index_sha,
        "development_report": str(report_path),
        "development_report_sha256": report_sha,
        "development_state_count": 23,
        "development_map_count": 6,
        "development_candidate_count": 412,
        "model_parameters": config["model"],
        "models": model_rows,
        "feature_profiles": list(PRIMARY_PROFILES),
        "confirmation_labels_seen": False,
    }
    _write_json(output_root / "freeze_manifest.json", manifest)
    return manifest


def _load_frozen_models(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    models = {}
    for row in manifest["models"]:
        path = root / str(row["model_file"])
        if _sha256(path) != str(row["model_sha256"]):
            raise ValueError(f"frozen model SHA256 mismatch: {row['profile']}")
        with path.open("rb") as stream:
            model = pickle.load(stream)
        if str(model.profile) != str(row["profile"]):
            raise ValueError("frozen model profile mismatch")
        models[str(row["profile"])] = model
    if set(models) != set(PRIMARY_PROFILES):
        raise ValueError("frozen model set is incomplete")
    return models


def _acceptance(
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
        "maps_no_worse": comparison["maps_no_worse"]
        >= int(thresholds["minimum_maps_no_worse"]),
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
        "decision": (
            "proceed_to_fresh_closed_loop_confirmation"
            if passed
            else "keep_rl_paused_and_redesign_realized_candidate_ranking"
        ),
        "requirement": (
            "frozen realized_dynamic vs proposal_dynamic: top-1 +5pp, conflict regret "
            "-5%, no significant map-bootstrap degradation, >=8/12 maps no worse, "
            "beat both simple baselines, and no unsupported >80% size collapse"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summaries = report["summaries"]
    comparison = report["comparisons"]["realized_dynamic_vs_proposal_dynamic"]
    lines = [
        "# InitLNS independent realized-neighborhood ranking confirmation",
        "",
        f"Decision: `{report['acceptance']['decision']}`",
        "",
        "## Coverage",
        "",
        f"- Maps: {report['integrity']['map_count']}",
        f"- States: {report['integrity']['state_count']}",
        f"- Candidates: {report['integrity']['candidate_count']}",
        f"- PP-order outcomes: {report['integrity']['outcome_count']}",
        "",
        "## Frozen-model result",
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
            "The models were frozen on the earlier six-map development index before "
            "confirmation outcomes were loaded. Static context remains exploratory and "
            "does not control this decision. Generated nodes and runtime are sensitivity "
            "metrics only.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_confirmation_analysis(
    collection: str | Path,
    config_path: str | Path,
    frozen_models: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, _ = _configuration(config_path)
    frozen_root = Path(frozen_models).resolve()
    freeze_manifest = _read_json(frozen_root / "freeze_manifest.json")
    if freeze_manifest["development_index_sha256"] != str(
        config["expected_development_index_sha256"]
    ).lower():
        raise ValueError("frozen model development index differs from the registered index")
    models = _load_frozen_models(frozen_root, freeze_manifest)
    rows, integrity = build_ranking_index(
        collection,
        expected_states=None,
        expected_candidates=None,
        expected_outcomes=None,
        expected_trials=8,
        expected_maps=12,
        expected_split="confirmation",
    )
    if integrity["state_count"] < 36:
        raise ValueError("confirmation index contains fewer than 36 registered states")
    records = {
        profile: evaluate_model(rows, models[profile], profile)
        for profile in PRIMARY_PROFILES
    }
    random_records = uniform_random_records(rows)
    coverage_records = internal_coverage_records(rows)
    perfect_records = oracle_records(rows)
    summaries = {
        "uniform_random": _uniform_summary(random_records),
        "internal_conflict_coverage": summarize_records(coverage_records),
        "oracle": summarize_records(perfect_records),
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
        int(config["evaluation"]["bootstrap_samples"]),
    )
    oracle_support = _oracle_support(rows)
    acceptance = _acceptance(
        summaries,
        comparisons["realized_dynamic_vs_proposal_dynamic"],
        bootstrap,
        oracle_support,
        dict(config["thresholds"]),
    )
    prediction_rows = []
    for selector, values in {
        "uniform_random": random_records,
        "internal_conflict_coverage": coverage_records,
        "oracle": perfect_records,
        **records,
    }.items():
        prediction_rows.extend(dict(row) for _, row in sorted(values.items()))
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "confirmation_index.jsonl", rows)
    _write_jsonl(output_root / "predictions.jsonl", prediction_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "integrity": integrity,
        "frozen_models": freeze_manifest,
        "feature_diagnostics": feature_diagnostics(rows),
        "summaries": summaries,
        "comparisons": comparisons,
        "map_bootstrap": bootstrap,
        "oracle_support": oracle_support,
        "acceptance": acceptance,
        "pre_registration": {
            "primary_profile": "realized_dynamic",
            "baseline_profile": "proposal_dynamic",
            "static_context_role": "exploratory only",
            "generated_nodes_role": "compute-aware sensitivity only",
            "runtime_role": "machine diagnostic only",
            "bootstrap_unit": "map_id",
            "confirmation_model_training": False,
        },
        "timings_seconds": {"total": time.perf_counter() - started},
    }
    _write_json(output_root / "independent_ranking_confirmation.json", report)
    (output_root / "independent_ranking_confirmation.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


__all__ = [
    "freeze_confirmation_models",
    "run_confirmation_analysis",
]
