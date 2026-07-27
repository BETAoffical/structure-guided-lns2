from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    read_trace_events,
)
from experiments.critical_conflicts import (
    CRITICAL_CONFIG_SCHEMA,
    CRITICAL_PROFILES,
    select_critical_seed_agents,
    update_edge_ages,
)
from experiments.receding_q_pilot import _atomic_write_csv
from experiments.receding_q_risk_audit import resolve_persisted_source_path
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.trace_replay import _initial_state


CRITICAL_AUDIT_SCHEMA = "lns2.v2_critical_conflict_audit.v1"
MARGIN_THRESHOLDS = (0.0, 0.05, 0.10, 0.20)


def _state_files(root: Path) -> dict[str, Path]:
    result = {}
    for path in sorted((root / "collection" / "states").glob("**/*.json")):
        payload = _read_json(path)
        state_id = str(dict(payload.get("state") or {}).get("state_id", ""))
        if not state_id or state_id in result:
            raise ValueError("v3 pilot has missing or duplicate state ids")
        result[state_id] = path
    return result


def _outcome_winner(payload: dict[str, Any]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for trial in payload.get("trials", []):
        candidate_id = str(trial.get("candidate_id", ""))
        if candidate_id and candidate_id != "official_adaptive":
            grouped[candidate_id].append(dict(trial["outcome"]))
    if not grouped:
        raise ValueError("critical audit state has no candidate outcomes")

    def key(candidate_id: str) -> tuple[float, ...]:
        values = grouped[candidate_id]
        return (
            statistics.fmean(float(row.get("feasible", False)) for row in values),
            statistics.fmean(
                float(int(row.get("conflict_reduction", 0)) > 0) for row in values
            ),
            statistics.fmean(float(row.get("conflict_reduction", 0)) for row in values),
            -statistics.fmean(float(row.get("repair_seconds", 0.0)) for row in values),
        )

    return min(grouped, key=lambda value: (tuple(-x for x in key(value)), value))


def _episode_snapshots(
    source_root: Path, manifest: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    trace_path = source_root / str(manifest["trace_file"])
    events = read_trace_events(trace_path)
    state = _initial_state(source_root, trace_path, events[0])
    ages = update_edge_ages({}, state)
    result = {}
    for event in events[1:-1]:
        decision_index = int(event["decision_index"])
        controller = dict(event.get("controller") or {})
        result[decision_index] = {
            "state": state,
            "edge_ages": dict(ages),
            "candidate_pool": list(controller.get("candidate_pool") or []),
            "selected_candidate_id": str(
                controller.get("selected_candidate_id", "")
            ),
        }
        if str(event.get("schema")) == EPISODE_SCHEMA_V2:
            after = apply_state_delta(state, event["state_delta"])
            after.update(apply_extras_delta(state, event["state_extras_delta"]))
        else:
            after = dict(event["after"])
        state = after
        ages = update_edge_ages(ages, state)
    return result


def _retained(candidate: dict[str, Any], seeds: set[int]) -> bool:
    return bool(seeds & set(map(int, candidate.get("seed_agents", []))))


def _load_states(source: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selections = _read_jsonl(source / "collection" / "state_selection.jsonl")
    files = _state_files(source)
    source_cache: dict[str, tuple[Path, dict[str, dict[str, Any]]]] = {}
    episode_cache: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    rows = []
    errors: collections.Counter[str] = collections.Counter()
    for selection in selections:
        state_id = str(selection["state_id"])
        state_file = files.get(state_id)
        if state_file is None:
            errors["missing_state_file"] += 1
            continue
        source_text = str(selection["source_root"])
        if source_text not in source_cache:
            source_root = resolve_persisted_source_path(
                source_text, sibling_root=source.parent / "initlns-high-load-rescue-pilot-dense-v2" / "sources"
            )
            manifests = {
                str(row["episode_id"]): row
                for row in _read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
            }
            source_cache[source_text] = (source_root, manifests)
        source_root, manifests = source_cache[source_text]
        episode_id = str(selection["episode_id"])
        manifest = manifests.get(episode_id)
        if manifest is None:
            errors["missing_source_manifest"] += 1
            continue
        cache_key = (source_text, episode_id)
        if cache_key not in episode_cache:
            episode_cache[cache_key] = _episode_snapshots(source_root, manifest)
        snapshot = episode_cache[cache_key].get(int(selection["decision_index"]))
        if snapshot is None:
            errors["missing_source_decision"] += 1
            continue
        payload = _read_json(state_file)
        candidate_pool = list(snapshot["candidate_pool"])
        candidate_ids = {str(row["candidate_id"]) for row in candidate_pool}
        regenerated_ids = {
            str(row["candidate_id"])
            for row in payload.get("candidates", [])
            if str(row.get("candidate_id")) != "official_adaptive"
        }
        if candidate_ids != regenerated_ids:
            errors["candidate_coverage_mismatch"] += 1
            continue
        selected_id = str(snapshot["selected_candidate_id"])
        if selected_id not in candidate_ids:
            errors["missing_v2_winner"] += 1
            continue
        outcome_id = _outcome_winner(payload)
        if outcome_id not in candidate_ids:
            errors["missing_outcome_winner"] += 1
            continue
        rows.append(
            {
                "state_id": state_id,
                "split": str(selection["split"]),
                "map_id": str(selection["map_id"]),
                "layout_mode": str(selection["layout_mode"]),
                "agent_count": int(selection["agent_count"]),
                "state_hash": str(selection["before_fingerprint"]),
                "state": snapshot["state"],
                "edge_ages": snapshot["edge_ages"],
                "candidate_pool": candidate_pool,
                "v2_winner": selected_id,
                "outcome_winner": outcome_id,
            }
        )
    return rows, dict(errors)


def _evaluate(
    states: list[dict[str, Any]], profile: str, margin: float
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for state in states:
        raw, raw_diagnostic = select_critical_seed_agents(
            state["state"],
            profile=profile,
            margin_threshold=-0.0,
            minimum_seeds=2,
            maximum_seeds=2,
            state_hash=state["state_hash"],
            edge_ages=state["edge_ages"],
        )
        selected, diagnostic = select_critical_seed_agents(
            state["state"],
            profile=profile,
            margin_threshold=margin,
            minimum_seeds=2,
            maximum_seeds=4,
            state_hash=state["state_hash"],
            edge_ages=state["edge_ages"],
        )
        pool_by_id = {
            str(value["candidate_id"]): value for value in state["candidate_pool"]
        }
        raw_set, selected_set = set(raw), set(selected)
        rows.append(
            {
                "state_id": state["state_id"],
                "split": state["split"],
                "map_id": state["map_id"],
                "layout_mode": state["layout_mode"],
                "agent_count": state["agent_count"],
                "profile": profile,
                "margin_threshold": margin,
                "legacy_seed_count": len(diagnostic["legacy_seed_agents"]),
                "raw_seed_count": len(raw),
                "selected_seed_count": len(selected),
                "raw_v2_winner_recalled": int(
                    _retained(pool_by_id[state["v2_winner"]], raw_set)
                ),
                "v2_winner_recalled": int(
                    _retained(pool_by_id[state["v2_winner"]], selected_set)
                ),
                "raw_outcome_winner_recalled": int(
                    _retained(pool_by_id[state["outcome_winner"]], raw_set)
                ),
                "outcome_winner_recalled": int(
                    _retained(pool_by_id[state["outcome_winner"]], selected_set)
                ),
                "selected_seed_agents": ";".join(map(str, selected)),
                "ranked_seed_agents": ";".join(
                    map(str, raw_diagnostic["ranked_seed_agents"])
                ),
            }
        )
    metrics = {
        "state_count": float(len(rows)),
        "raw_v2_winner_recall": statistics.fmean(
            row["raw_v2_winner_recalled"] for row in rows
        ),
        "v2_winner_recall": statistics.fmean(
            row["v2_winner_recalled"] for row in rows
        ),
        "raw_outcome_winner_recall": statistics.fmean(
            row["raw_outcome_winner_recalled"] for row in rows
        ),
        "outcome_winner_recall": statistics.fmean(
            row["outcome_winner_recalled"] for row in rows
        ),
        "mean_selected_seeds": statistics.fmean(
            row["selected_seed_count"] for row in rows
        ),
        "full_fallback_fraction": statistics.fmean(
            row["selected_seed_count"] == row["legacy_seed_count"] for row in rows
        ),
    }
    return metrics, rows


def run_critical_conflict_audit(
    source: str | Path, output: str | Path
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    states, errors = _load_states(source_root)
    train = [row for row in states if row["split"] == "policy_train"]
    diagnostic = [row for row in states if row["split"] != "policy_train"]
    if not train or not diagnostic:
        raise ValueError("critical audit requires training and diagnostic states")
    grid = []
    detailed: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for profile in CRITICAL_PROFILES:
        for margin in MARGIN_THRESHOLDS:
            train_metrics, train_rows = _evaluate(train, profile, margin)
            diagnostic_metrics, diagnostic_rows = _evaluate(
                diagnostic, profile, margin
            )
            qualified = (
                train_metrics["raw_v2_winner_recall"] >= 0.90
                and train_metrics["v2_winner_recall"] >= 0.99
                and train_metrics["mean_selected_seeds"] <= 3.0
                and diagnostic_metrics["v2_winner_recall"] >= 0.97
                and diagnostic_metrics["mean_selected_seeds"] <= 3.0
            )
            row = {
                "profile": profile,
                "margin_threshold": margin,
                "qualified": qualified,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"diagnostic_{key}": value
                    for key, value in diagnostic_metrics.items()
                },
            }
            grid.append(row)
            detailed[(profile, margin, "train")] = train_rows
            detailed[(profile, margin, "diagnostic")] = diagnostic_rows
    qualified_rows = [row for row in grid if row["qualified"]]
    best_observed = max(
        grid,
        key=lambda row: (
            min(
                float(row["train_v2_winner_recall"]),
                float(row["diagnostic_v2_winner_recall"]),
            ),
            -float(row["train_mean_selected_seeds"]),
            -float(row["diagnostic_mean_selected_seeds"]),
            -CRITICAL_PROFILES.index(str(row["profile"])),
            -float(row["margin_threshold"]),
        ),
    )
    selected = (
        min(
            qualified_rows,
            key=lambda row: (
                float(row["train_mean_selected_seeds"]),
                -float(row["diagnostic_v2_winner_recall"]),
                CRITICAL_PROFILES.index(str(row["profile"])),
                float(row["margin_threshold"]),
            ),
        )
        if qualified_rows and not errors
        else None
    )
    report_rows_for = selected if selected is not None else best_observed
    selected_rows = (
        detailed[
            (
                str(report_rows_for["profile"]),
                float(report_rows_for["margin_threshold"]),
                "train",
            )
        ]
        + detailed[
            (
                str(report_rows_for["profile"]),
                float(report_rows_for["margin_threshold"]),
                "diagnostic",
            )
        ]
    )
    _atomic_write_csv(output_root / "critical_config_grid.csv", grid)
    _atomic_write_csv(output_root / "critical_state_rows.csv", selected_rows)
    promoted = selected is not None
    config = None
    config_path = output_root / "v2_critical_config.json"
    if promoted:
        config = {
            "schema": CRITICAL_CONFIG_SCHEMA,
            "deployment_promoted": True,
            "profile": str(selected["profile"]),
            "margin_threshold": float(selected["margin_threshold"]),
            "minimum_seeds": 2,
            "maximum_seeds": 4,
            "source": {
                "audit_schema": CRITICAL_AUDIT_SCHEMA,
                "pilot_state_selection_sha256": sha256_file(
                    source_root / "collection" / "state_selection.jsonl"
                ),
                "pilot_trial_manifest_sha256": sha256_file(
                    source_root / "collection" / "trial_manifest.jsonl"
                ),
            },
        }
        _write_json(config_path, config)
    elif config_path.exists():
        # Never leave a previously promoted runtime config beside a failed
        # rerun of the same audit output.
        config_path.unlink()
    report = {
        "schema": CRITICAL_AUDIT_SCHEMA,
        "source": str(source_root),
        "state_count": len(states),
        "training_state_count": len(train),
        "diagnostic_state_count": len(diagnostic),
        "errors": errors,
        "grid": grid,
        "best_observed": best_observed,
        "selected": selected,
        "deployment_promoted": promoted,
        "decision": "v2_critical_candidate" if promoted else "keep_v2_full",
        "config": config,
        "evidence_boundary": (
            "synthetic offline candidate-retention audit; one-step outcome winner is "
            "diagnostic and does not establish wall-clock solver improvement"
        ),
    }
    _write_json(output_root / "critical_conflict_audit_report.json", report)
    markdown = [
        "# v2 critical-conflict audit",
        "",
        f"- states: {len(states)} ({len(train)} train, {len(diagnostic)} diagnostic)",
        f"- errors: {sum(errors.values())}",
        f"- decision: `{report['decision']}`",
    ]
    if selected is not None:
        markdown.extend(
            [
                f"- profile: `{selected['profile']}`",
                f"- margin: {float(selected['margin_threshold']):.2f}",
                f"- train v2-winner recall: {float(selected['train_v2_winner_recall']):.3%}",
                f"- diagnostic v2-winner recall: {float(selected['diagnostic_v2_winner_recall']):.3%}",
                f"- mean retained seeds: {float(selected['train_mean_selected_seeds']):.3f}",
            ]
        )
    else:
        markdown.extend(
            [
                f"- best observed profile: `{best_observed['profile']}`",
                f"- best observed margin: {float(best_observed['margin_threshold']):.2f}",
                f"- train v2-winner recall: {float(best_observed['train_v2_winner_recall']):.3%}",
                f"- diagnostic v2-winner recall: {float(best_observed['diagnostic_v2_winner_recall']):.3%}",
                f"- train mean retained seeds: {float(best_observed['train_mean_selected_seeds']):.3f}",
                "- no runtime config was emitted because the promotion gates failed",
            ]
        )
    markdown.extend(["", f"> {report['evidence_boundary']}", ""])
    (output_root / "critical_conflict_audit_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    return report
