from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from experiments._common import mean as _mean
from experiments.closed_loop_confirmation_analysis import (
    _paired_rows,
    compare_policies,
    run_closed_loop_analysis,
)
from experiments.repair_collection import SCHEMA_VERSION, _read_json, _read_jsonl, _write_json


SCHEMA = "lns2.movingai_ood_closed_loop.v1"
FIXED_POLICIES = ("fixed_target", "fixed_collision", "fixed_random")


def family_auc_comparison(
    baseline: list[dict[str, Any]], primary: list[dict[str, Any]]
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for left, right in _paired_rows(baseline, primary):
        family = str(left["layout_mode"])
        if family != str(right["layout_mode"]):
            raise ValueError("paired OOD rows disagree on layout family")
        grouped[family].append(
            (
                float(left["summary"]["fixed_budget_conflict_auc"]),
                float(right["summary"]["fixed_budget_conflict_auc"]),
            )
        )
    families = {}
    for name, values in sorted(grouped.items()):
        baseline_mean = _mean(left for left, _ in values)
        primary_mean = _mean(right for _, right in values)
        families[name] = {
            "episode_count": len(values),
            "baseline_mean": baseline_mean,
            "primary_mean": primary_mean,
            "relative_improvement": (
                (baseline_mean - primary_mean) / baseline_mean
                if baseline_mean
                else (0.0 if primary_mean == 0.0 else -float("inf"))
            ),
            "no_worse": primary_mean <= baseline_mean,
        }
    return {
        "family_count": len(families),
        "families_no_worse": sum(bool(row["no_worse"]) for row in families.values()),
        "families": families,
    }


def movingai_ood_acceptance(
    base: dict[str, Any],
    manifests: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = dict(config["ood_thresholds"])
    summaries = dict(base["policy_summaries"])
    comparison = base["comparisons"]["realized_dynamic_vs_official_adaptive"]
    auc = comparison["metrics"]["fixed_budget_conflict_auc"]
    families = family_auc_comparison(
        manifests["official_adaptive"], manifests["realized_dynamic"]
    )
    realized = summaries["realized_dynamic"]
    adaptive = summaries["official_adaptive"]
    error_count = sum(int(row["error_count"]) for row in summaries.values())
    gates = {
        "qualification": bool(base["qualification"]["passed"]),
        "all_policy_episodes_valid": error_count == 0,
        "initial_fingerprints_match": bool(base["integrity"]["passed"]),
        "success_not_below_adaptive": int(realized["success_count"])
        >= int(adaptive["success_count"]),
        "auc_improvement": float(auc["relative_improvement"])
        >= float(thresholds["minimum_auc_improvement"]),
        "bootstrap_lower_bound": float(auc["bootstrap"]["improvement_95_ci"][0])
        >= float(thresholds["bootstrap_lower_bound"]),
        "maps_no_worse": int(auc["maps_no_worse"])
        >= int(thresholds["minimum_maps_no_worse"]),
        "layout_families_no_worse": int(families["families_no_worse"])
        >= int(thresholds["minimum_layout_families_no_worse"]),
        "no_invalid_actions": int(realized["invalid_action_count"]) == 0,
        "no_fingerprint_mismatch": int(realized["fingerprint_mismatch_count"]) == 0,
    }
    passed = all(gates.values())
    return {
        "passed": passed,
        "gates": gates,
        "layout_family_comparison": families,
        "decision": (
            "confirm_dynamic_realized_neighborhood_cross_layout_generalization"
            if passed
            else "stop_cross_layout_claim_and_consolidate_results"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    auc = report["comparisons"]["realized_dynamic_vs_official_adaptive"]["metrics"][
        "fixed_budget_conflict_auc"
    ]
    lines = [
        "# Frozen V1 MovingAI OOD closed-loop confirmation",
        "",
        f"Decision: `{report['acceptance']['decision']}`",
        "",
        f"- Qualification: {report['qualification']['valid_count']}/"
        f"{report['qualification']['expected_reset_count']} valid resets",
        f"- Repairable episodes: {report['qualification']['nonzero_state_count']}",
        f"- Fixed-budget AUC improvement: {auc['relative_improvement']:.2%}",
        f"- Maps no worse: {auc['maps_no_worse']}/{auc['map_count']}",
        f"- Map bootstrap 95% CI: {auc['bootstrap']['improvement_95_ci']}",
        "",
        "## Registered gates",
        "",
        *[
            f"- {name}: {'PASS' if value else 'FAIL'}"
            for name, value in report["acceptance"]["gates"].items()
        ],
        "",
        "The frozen model was not retrained. Static map, OD, and density context was excluded.",
        "Wall-clock time is diagnostic and is not an acceptance gate.",
        "",
    ]
    return "\n".join(lines)


def run_movingai_ood_analysis(
    collection: str | Path, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    collection_root = Path(collection).resolve()
    config_path = Path(config_path).resolve()
    output_root = Path(output).resolve()
    config = _read_json(config_path)
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported MovingAI OOD analysis config")
    base = run_closed_loop_analysis(collection_root, config_path, output_root / "base")
    policies = tuple(map(str, base["pre_registration"]["policies"]))
    required = {"official_adaptive", *FIXED_POLICIES, "realized_dynamic"}
    if set(policies) != required:
        raise ValueError("MovingAI OOD analysis requires exactly five registered policies")
    manifests = {
        policy: _read_jsonl(collection_root / f"{policy}_manifest.jsonl")
        for policy in policies
    }
    fixed = {
        policy: compare_policies(
            manifests["official_adaptive"],
            manifests[policy],
            int(config["bootstrap_samples"]),
            100,
        )
        for policy in FIXED_POLICIES
    }
    acceptance = movingai_ood_acceptance(base, manifests, config)
    report = {
        **base,
        "schema": SCHEMA,
        "pre_registration": {
            **base["pre_registration"],
            "study": "frozen_v1_movingai_cross_layout_ood",
            "models_retrained": False,
            "static_context_role": "excluded",
            "wall_clock_role": "diagnostic_only",
        },
        "comparisons": {
            **base["comparisons"],
            **{
                f"{policy}_vs_official_adaptive": value
                for policy, value in fixed.items()
            },
        },
        "acceptance": acceptance,
    }
    _write_json(output_root / "movingai_ood_confirmation.json", report)
    markdown = output_root / "movingai_ood_confirmation.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(report), encoding="utf-8")
    return report


__all__ = [
    "family_auc_comparison",
    "movingai_ood_acceptance",
    "run_movingai_ood_analysis",
]
