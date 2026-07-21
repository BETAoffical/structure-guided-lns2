from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _fingerprint, _read_json, _write_json
from experiments.rescue_lite_confirmation import (
    FROZEN_POLICY_ID,
    LAYOUTS,
    AGENT_COUNTS,
    _run_trials,
    analyze_confirmation,
    select_confirmation_states,
)


BALANCED_DIAGNOSTIC_SCHEMA = "lns2.rescue_lite_balanced_diagnostic.v1"
DEFAULT_QUOTA_PER_CELL = 4
DEFAULT_TRIAL_COUNT = 4


def _cell(layout: str, agent_count: int) -> str:
    return f"{layout}__agents_{int(agent_count)}"


def _write_status(root: Path, *, phase: str, status: str, **values: Any) -> None:
    _write_json(
        root / "status.json",
        {
            "schema": BALANCED_DIAGNOSTIC_SCHEMA,
            "phase": phase,
            "status": status,
            **values,
        },
    )


def select_balanced_diagnostic_states(
    prepared: list[dict[str, Any]], *, quota_per_cell: int = DEFAULT_QUOTA_PER_CELL
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected, counts = select_confirmation_states(
        prepared, quota_per_cell=quota_per_cell
    )
    required = {
        _cell(layout, agent_count): quota_per_cell
        for layout in LAYOUTS
        for agent_count in AGENT_COUNTS
    }
    if counts != required:
        raise ValueError(
            f"balanced diagnostic source cannot fill quotas: {counts} != {required}"
        )
    task_counts: collections.Counter[str] = collections.Counter(
        str(row["state"]["task_id"]) for row in selected
    )
    if task_counts and max(task_counts.values()) > 2:
        raise ValueError("balanced diagnostic exceeds two states per task")
    if len(selected) != quota_per_cell * len(required):
        raise ValueError("balanced diagnostic state count is inconsistent")
    return selected, counts


def diagnostic_decision(raw_decision: str) -> tuple[str, str]:
    mapping = {
        "rescue_lite_confirmed": (
            "diagnostic_supports_fixed_rescue",
            "continue_rescue_lite_research_without_promotion",
        ),
        "learned_rescue_reconsidered": (
            "diagnostic_supports_learned_rescue",
            "reconsider_learned_rescue_without_promotion",
        ),
        "proceed_to_v3": (
            "diagnostic_supports_v3",
            "proceed_to_v3_design",
        ),
        "inconclusive_collect_more": (
            "diagnostic_inconclusive",
            "keep_v2_full_and_do_not_promote_rescue",
        ),
    }
    if raw_decision not in mapping:
        raise ValueError(f"unexpected confirmation analysis decision: {raw_decision}")
    return mapping[raw_decision]


def _load_prepared(source: Path) -> list[dict[str, Any]]:
    prepared_root = source / "prepared"
    rows = [_read_json(path) for path in sorted(prepared_root.glob("*.json"))]
    if not rows:
        raise ValueError("locked source has no prepared states")
    return rows


def run_balanced_rescue_diagnostic(
    *,
    project_root: str | Path,
    source: str | Path,
    output: str | Path,
    workers: int = 4,
    resume: bool = False,
    quota_per_cell: int = DEFAULT_QUOTA_PER_CELL,
    trial_count: int = DEFAULT_TRIAL_COUNT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    if workers <= 0 or quota_per_cell <= 0 or trial_count <= 0:
        raise ValueError("workers, quota and trial count must be positive")
    source_report = _read_json(source_root / "locked_confirmation_report.json")
    source_status = _read_json(source_root / "status.json")
    source_run = _read_json(source_root / "run_config.json")
    if str(source_report.get("decision")) != "locked_confirmation_insufficient_states":
        raise ValueError("diagnostic source is not an insufficient locked confirmation")
    if bool(source_report.get("branch_trials_started")):
        raise ValueError("diagnostic source already ran branch trials")
    if str(source_status.get("status")) != "insufficient":
        raise ValueError("diagnostic source status is not insufficient")

    prepared = _load_prepared(source_root)
    source_fingerprint = str(source_run["run_fingerprint"])
    invalid = [
        row
        for row in prepared
        if not bool(row.get("complete"))
        or not bool(row.get("valid"))
        or str(row.get("run_fingerprint")) != source_fingerprint
    ]
    if invalid:
        raise ValueError("diagnostic source contains invalid prepared states")
    selected, counts = select_balanced_diagnostic_states(
        prepared, quota_per_cell=quota_per_cell
    )
    selection = [
        {
            "state_id": str(row["state"]["state_id"]),
            "cell": str(row["state"]["cell"]),
            "map_id": str(row["state"]["map_id"]),
            "task_id": str(row["state"]["task_id"]),
            "decision_index": int(row["state"]["decision_index"]),
            "top_candidate_by_size": dict(row["top_candidate_by_size"]),
            "learned_candidate_id": str(row["learned_candidate_id"]),
        }
        for row in selected
    ]
    identity = {
        "schema": BALANCED_DIAGNOSTIC_SCHEMA,
        "source_run_config_sha256": sha256_file(source_root / "run_config.json"),
        "source_report_sha256": sha256_file(
            source_root / "locked_confirmation_report.json"
        ),
        "source_run_fingerprint": source_fingerprint,
        "quota_per_cell": int(quota_per_cell),
        "trial_count": int(trial_count),
        "selection": selection,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "confirmation_helper_sha256": sha256_file(
            root / "experiments" / "rescue_lite_confirmation.py"
        ),
        "promotion_eligible": False,
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output_root / "run_config.json"
    if output_root.is_dir() and any(output_root.iterdir()):
        if not resume:
            raise ValueError("diagnostic output exists; pass --resume")
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("diagnostic resume fingerprint mismatch")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    _write_json(
        output_root / "selection_manifest.json",
        {
            "schema": BALANCED_DIAGNOSTIC_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "state_count_by_cell": counts,
            "states": selection,
        },
    )

    _write_status(
        output_root,
        phase="paired-trials",
        status="running",
        state_count=len(selected),
        promotion_eligible=False,
    )
    results = _run_trials(
        selected=selected,
        output=output_root,
        run_fingerprint=run_fingerprint,
        trial_count=trial_count,
        workers=workers,
    )
    _write_status(output_root, phase="analysis", status="running")
    raw = analyze_confirmation(results=results, output=output_root)
    raw_decision = str(raw["decision"])
    decision, recommendation = diagnostic_decision(raw_decision)
    report = {
        **raw,
        "schema": BALANCED_DIAGNOSTIC_SCHEMA,
        "decision": decision,
        "raw_analysis_decision": raw_decision,
        "recommendation": recommendation,
        "evidence_tier": "diagnostic_only",
        "promotion_eligible": False,
        "deployment_promoted": False,
        "default_controller_changed": False,
        "source": str(source_root),
        "source_run_fingerprint": source_fingerprint,
        "run_fingerprint": run_fingerprint,
        "state_count_by_cell": counts,
        "balanced_state_count": len(selected),
        "paired_trial_count": int(raw["coverage"]["branch_trial_count"]),
        "complete_episode_evaluation": False,
        "controller_overhead_in_primary_efficiency": False,
        "quick_formal_v3_started": False,
    }
    _write_json(output_root / "diagnostic_report.json", report)
    # Replace the generic analyzer report so no file in this diagnostic output
    # can be mistaken for promotion-eligible confirmation evidence.
    _write_json(output_root / "confirmation_report.json", report)
    fixed = dict(report["fixed_metrics"])
    adaptive = dict(report["adaptive_metrics"])
    learned = dict(report["learned_metrics"])
    markdown = [
        "# Balanced rescue diagnostic (not promotion eligible)",
        "",
        f"- Decision: `{decision}`",
        f"- Recommendation: `{recommendation}`",
        f"- Raw analyzer decision: `{raw_decision}`",
        f"- Balanced states: {len(selected)} ({quota_per_cell} per cell)",
        f"- Paired branch trials: {report['paired_trial_count']}",
        "- Promotion eligible: false",
        "- Default controller changed: false",
        "",
        "## Repair-only comparison",
        "",
        (
            f"`{FROZEN_POLICY_ID}`: escape rate "
            f"{float(fixed['state_escape_rate']):.3f}, conflict reduction/second "
            f"{float(fixed['conflict_reduction_per_second']):.3f}."
        ),
        (
            f"Adaptive: escape rate {float(adaptive['state_escape_rate']):.3f}, "
            f"conflict reduction/second "
            f"{float(adaptive['conflict_reduction_per_second']):.3f}."
        ),
        (
            "Learned rescue reference: escape rate "
            f"{float(learned['state_escape_rate']):.3f}, conflict reduction/second "
            f"{float(learned['conflict_reduction_per_second']):.3f}."
        ),
        "",
        "These are same-state repair diagnostics. They exclude complete-episode",
        "quality and the primary v2 candidate/feature computation overhead.",
        "",
    ]
    (output_root / "diagnostic_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    (output_root / "confirmation_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    _write_status(
        output_root,
        phase="complete",
        status="complete",
        decision=decision,
        promotion_eligible=False,
    )
    return report


__all__ = [
    "BALANCED_DIAGNOSTIC_SCHEMA",
    "diagnostic_decision",
    "run_balanced_rescue_diagnostic",
    "select_balanced_diagnostic_states",
]
