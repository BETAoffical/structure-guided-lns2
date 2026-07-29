from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from experiments._common import atomic_write_csv, sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.lns2_bottleneck import validate_manifest_trace
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.stall_escape_qualification import normalize_episode_decisions
from experiments.trace_replay import decision_rows


STALL_ESCAPE_CONTINUATION_SCHEMA = "lns2.stall_escape_continuation.v1"


def resolve_unresolved_state(
    decisions: list[dict[str, Any]],
    qualification: dict[str, Any],
    *,
    future_observation_decisions: int,
) -> dict[str, Any]:
    if future_observation_decisions <= 0:
        raise ValueError("future observation decisions must be positive")
    anchor_index = int(qualification["anchor_decision_index"])
    threshold = int(qualification["threshold"])
    fingerprint = str(qualification["before_repair_fingerprint"])
    if threshold <= 0 or anchor_index < 0 or not fingerprint:
        raise ValueError("unresolved qualification anchor is invalid")
    if anchor_index >= len(decisions):
        raise ValueError("continued episode is shorter than the unresolved anchor")
    anchor = decisions[anchor_index]
    if str(anchor["before_repair_fingerprint"]) != fingerprint:
        raise ValueError("continued episode does not reproduce the unresolved state")
    run_length = 0
    index = anchor_index
    recovery_outcome: str | None = None
    while index < len(decisions):
        row = decisions[index]
        if str(row["before_repair_fingerprint"]) != fingerprint:
            break
        if bool(row["no_progress"]):
            run_length += 1
            index += 1
            continue
        recovery_outcome = str(row["repair_outcome"])
        break
    if run_length < threshold:
        raise ValueError("continued episode no longer reproduces the trigger prefix")
    observed_after_trigger = run_length - threshold
    if recovery_outcome is not None and observed_after_trigger < future_observation_decisions:
        resolution = "natural_recovery"
        recovery_delay = observed_after_trigger + 1
    elif observed_after_trigger >= future_observation_decisions:
        resolution = "confirmed_long_stall"
        recovery_delay = None
    else:
        resolution = "unresolved_stall"
        recovery_delay = None
    return {
        "resolution": resolution,
        "run_length": run_length,
        "observed_after_trigger": observed_after_trigger,
        "recovery_delay_decisions": recovery_delay,
        "recovery_outcome": recovery_outcome,
    }


def _controller_signature(event: dict[str, Any]) -> dict[str, Any]:
    controller = dict(event.get("controller") or {})
    metrics = dict(event.get("metrics") or {})
    raw_pool = controller.get("candidate_pool")
    candidate_pool = (
        [
            {
                name: candidate.get(name)
                for name in (
                    "actual_size",
                    "agents",
                    "candidate_id",
                    "feature_out_of_range_fraction",
                    "proposal_count_by_family",
                    "proposal_seeds",
                    "retained",
                    "score",
                    "seed_agents",
                    "selection_families",
                )
            }
            for candidate in raw_pool
        ]
        if isinstance(raw_pool, list)
        and all(isinstance(candidate, dict) for candidate in raw_pool)
        else raw_pool
    )
    requested_pp_seed = metrics.get("requested_pp_random_seed")
    return {
        "decision_index": event.get("decision_index"),
        "before_fingerprint": event.get("before_fingerprint"),
        "action": event.get("action"),
        "controller": {
            name: candidate_pool if name == "candidate_pool" else controller.get(name)
            for name in (
                "base_selected_candidate_id",
                "base_selected_score",
                "candidate_pool",
                "inference_backend",
                "route",
                "selected_candidate_id",
                "selected_score",
            )
        },
        "metrics": {
            name: (
                -1
                if name == "requested_pp_random_seed"
                and requested_pp_seed is None
                else metrics.get(name)
            )
            for name in (
                "requested_mode",
                "requested_random_seed",
                "requested_pp_random_seed",
                "requested_repair_order",
            )
        },
    }


def _post_repair_signature(
    row: dict[str, Any], event: dict[str, Any], *, include_full_fingerprint: bool
) -> dict[str, Any]:
    metrics = dict(event.get("metrics") or {})
    applied_pp_seed = metrics.get("applied_pp_random_seed")
    result = {
        "after_repair_fingerprint": row.get("after_repair_fingerprint"),
        "low_level_delta": event.get("low_level_delta"),
        "metrics": {
            name: (
                -1
                if name == "applied_pp_random_seed" and applied_pp_seed is None
                else metrics.get(name)
            )
            for name in (
                "action_valid",
                "applied_heuristic",
                "applied_pp_random_seed",
                "conflicts_before",
                "conflicts_after",
                "conflict_delta",
                "iteration",
                "neighborhood",
                "repair_order",
                "replan_success",
                "sum_of_costs_before",
                "sum_of_costs_after",
            )
        },
    }
    if include_full_fingerprint:
        result["after_fingerprint"] = event.get("after_fingerprint")
    return result


def compare_continuation_prefix(
    source_rows: list[dict[str, Any]],
    source_events: list[dict[str, Any]],
    continued_rows: list[dict[str, Any]],
    continued_events: list[dict[str, Any]],
) -> dict[str, Any]:
    source_transitions = [
        event for event in source_events if str(event.get("event")) == "transition"
    ]
    continued_transitions = [
        event
        for event in continued_events
        if str(event.get("event")) == "transition"
    ]
    if len(source_rows) != len(source_transitions) or len(continued_rows) != len(
        continued_transitions
    ):
        raise ValueError("continuation trace views have inconsistent lengths")
    if len(continued_rows) < len(source_rows):
        raise ValueError("continued episode is shorter than its source")
    mismatches: list[int] = []
    budget_boundary_exclusions = 0
    for index, (source_row, source_event) in enumerate(
        zip(source_rows, source_transitions)
    ):
        continued_row = continued_rows[index]
        continued_event = continued_transitions[index]
        boundary = index == len(source_rows) - 1 and bool(source_event.get("truncated"))
        if boundary:
            budget_boundary_exclusions += 1
        if (
            _controller_signature(source_event)
            != _controller_signature(continued_event)
            or _post_repair_signature(
                source_row,
                source_event,
                include_full_fingerprint=not boundary,
            )
            != _post_repair_signature(
                continued_row,
                continued_event,
                include_full_fingerprint=not boundary,
            )
        ):
            mismatches.append(index)
    return {
        "source_transition_count": len(source_rows),
        "continued_transition_count": len(continued_rows),
        "semantic_mismatch_count": len(mismatches),
        "mismatch_decision_indices": mismatches,
        "budget_boundary_exclusion_count": budget_boundary_exclusions,
        "passed": not mismatches,
    }


def _derived_config(source_root: Path, split: str, output: Path, future: int) -> Path:
    source_path = source_root.parent.parent / "protocol" / f"source_{split}.json"
    source = _read_json(source_path)
    original_decisions = int(source["max_decisions"])
    extended_decisions = original_decisions + future
    source["max_decisions"] = extended_decisions
    source["metric_iteration_budget"] = extended_decisions
    environment = dict(source["environment"])
    environment["max_repair_iterations"] = extended_decisions
    source["environment"] = environment
    destination = output / "protocol" / f"continuation_{split}.json"
    _write_json(destination, source)
    return destination


def run_stall_escape_continuation(
    qualification_root: str | Path,
    output: str | Path,
    *,
    future_observation_decisions: int = 3,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    qualification_path = Path(qualification_root).resolve()
    output_root = Path(output).resolve()
    if future_observation_decisions <= 0 or workers <= 0:
        raise ValueError("future decisions and workers must be positive")
    if output_root.is_dir() and any(output_root.iterdir()) and not resume:
        raise ValueError("continuation output is non-empty; pass resume")
    qualification_report = _read_json(
        qualification_path / "qualification_report.json"
    )
    if (
        not bool(qualification_report.get("complete"))
        or str(qualification_report.get("decision"))
        != "extend_unresolved_v2_prefixes_before_oracle"
    ):
        raise ValueError("qualification does not authorize continuation")
    qualifications = [
        row
        for row in _read_jsonl(qualification_path / "qualification_states.jsonl")
        if str(row.get("qualification_class")) == "unresolved_stall"
    ]
    if not qualifications:
        raise ValueError("qualification contains no unresolved states")
    by_source: dict[Path, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in qualifications:
        by_source[Path(str(row["source_root"])).resolve()].append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    collection_roots: dict[str, str] = {}
    for source_root, rows in sorted(by_source.items(), key=lambda item: str(item[0])):
        source_run = _read_json(source_root / "run_config.json")
        source_configuration = dict(source_run["configuration"])
        split = str(source_configuration["split"])
        config = _derived_config(
            source_root,
            split,
            output_root,
            future_observation_decisions,
        )
        collection = output_root / "collections" / split
        task_ids = sorted({str(row["task_id"]) for row in rows})
        job_keys = {
            (str(row["task_id"]), int(row["solver_seed"])) for row in rows
        }
        common = {
            "workers": workers,
            "task_ids": task_ids,
            "controller": "v2-full",
            "feature_backend": str(source_configuration["feature_backend"]),
            "controller_bundle": str(source_configuration["controller_bundle"]),
            "controller_runtime": str(source_configuration["controller_runtime"]),
            "verification_profile": str(source_configuration["verification_profile"]),
            "job_keys": job_keys,
            "cohort_job_keys": job_keys,
            "stopping_rule": "historical",
        }
        run_closed_loop_collection(
            str(source_run["dataset"]),
            config,
            collection,
            phase="qualify",
            resume=resume or (collection / "run_config.json").is_file(),
            **common,
        )
        run_closed_loop_collection(
            str(source_run["dataset"]),
            config,
            collection,
            phase="realized_dynamic",
            resume=True,
            **common,
        )
        collection_roots[split] = str(collection)

    result_rows: list[dict[str, Any]] = []
    for qualification in qualifications:
        source_root = Path(str(qualification["source_root"])).resolve()
        source_run = _read_json(source_root / "run_config.json")
        split = str(qualification["split"])
        continued_root = Path(collection_roots[split])
        continued_run = _read_json(continued_root / "run_config.json")
        source_manifests = {
            str(row["episode_id"]): row
            for row in _read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
        }
        continued_manifests = {
            str(row["episode_id"]): row
            for row in _read_jsonl(
                continued_root / "realized_dynamic_manifest.jsonl"
            )
        }
        episode_id = str(qualification["episode_id"])
        if episode_id not in source_manifests or episode_id not in continued_manifests:
            raise ValueError("continuation episode coverage is incomplete")
        source_manifest = source_manifests[episode_id]
        continued_manifest = continued_manifests[episode_id]
        _path, source_events, _blob = validate_manifest_trace(
            source_root,
            source_manifest,
            run_fingerprint=str(source_run["run_fingerprint"]),
            expected_policy="realized_dynamic",
        )
        _path, continued_events, _blob = validate_manifest_trace(
            continued_root,
            continued_manifest,
            run_fingerprint=str(continued_run["run_fingerprint"]),
            expected_policy="realized_dynamic",
        )
        source_rows, _events = decision_rows(source_root, source_manifest)
        continued_rows, _events = decision_rows(continued_root, continued_manifest)
        prefix = compare_continuation_prefix(
            source_rows,
            source_events,
            continued_rows,
            continued_events,
        )
        if bool(prefix["passed"]):
            normalized = normalize_episode_decisions(continued_rows, continued_events)
            resolution = resolve_unresolved_state(
                normalized,
                qualification,
                future_observation_decisions=future_observation_decisions,
            )
        else:
            resolution = {
                "resolution": "prefix_mismatch",
                "run_length": None,
                "observed_after_trigger": None,
                "recovery_delay_decisions": None,
                "recovery_outcome": None,
            }
        result_rows.append(
            {
                "split": split,
                "map_id": qualification["map_id"],
                "task_id": qualification["task_id"],
                "solver_seed": qualification["solver_seed"],
                "episode_id": episode_id,
                "anchor_decision_index": qualification["anchor_decision_index"],
                "threshold": qualification["threshold"],
                **resolution,
                **prefix,
            }
        )
    counts = collections.Counter(str(row["resolution"]) for row in result_rows)
    semantic_mismatch_count = sum(
        int(row["semantic_mismatch_count"]) for row in result_rows
    )
    report = {
        "schema": STALL_ESCAPE_CONTINUATION_SCHEMA,
        "complete": semantic_mismatch_count == 0,
        "training_started": False,
        "controller_actions_changed": False,
        "qualification_report_sha256": sha256_file(
            qualification_path / "qualification_report.json"
        ),
        "future_observation_decisions": future_observation_decisions,
        "episode_count": len(result_rows),
        "semantic_mismatch_count": semantic_mismatch_count,
        "budget_boundary_exclusion_count": sum(
            int(row["budget_boundary_exclusion_count"]) for row in result_rows
        ),
        "resolution_counts": dict(sorted(counts.items())),
        "collections": collection_roots,
        "decision": (
            "prefix_mismatch_requires_exact_replay"
            if semantic_mismatch_count
            else "collect_exact_oracle_for_confirmed_stalls"
            if counts["confirmed_long_stall"]
            else "extend_or_investigate_remaining_unresolved"
            if counts["unresolved_stall"]
            else "no_confirmed_stall_in_source_cohort_do_not_train"
        ),
    }
    atomic_write_csv(output_root / "continuation_results.csv", result_rows)
    _write_json(output_root / "continuation_report.json", report)
    lines = [
        "# Stall escape continuation",
        "",
        f"- Episodes: `{len(result_rows)}`; semantic mismatches: `{report['semantic_mismatch_count']}`.",
        f"- Budget-boundary exclusions: `{report['budget_boundary_exclusion_count']}`.",
        f"- Resolutions: `{dict(sorted(counts.items()))}`.",
        "- Training started: `false`; controller actions changed: `false`.",
        f"- Decision: `{report['decision']}`.",
        "",
    ]
    (output_root / "continuation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = [
    "STALL_ESCAPE_CONTINUATION_SCHEMA",
    "compare_continuation_prefix",
    "resolve_unresolved_state",
    "run_stall_escape_continuation",
]
