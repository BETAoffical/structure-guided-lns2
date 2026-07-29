from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, sha256_file
from experiments.lns2_bottleneck import validate_manifest_trace
from experiments.repair_aware import classify_repair_outcome
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stall_shadow import neighborhood_key
from experiments.trace_replay import decision_rows


STALL_ESCAPE_QUALIFICATION_SCHEMA = "lns2.stall_escape_qualification.v1"
QUALIFICATION_CLASSES = (
    "confirmed_long_stall",
    "natural_recovery",
    "unresolved_stall",
    "ordinary_progress",
)
NO_PROGRESS_OUTCOMES = {"hard_failure", "accepted_noop"}


def _strict_positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _normalized_thresholds(values: Iterable[int]) -> tuple[int, ...]:
    thresholds = tuple(values)
    if (
        not thresholds
        or any(type(value) is not int or value <= 0 for value in thresholds)
        or tuple(sorted(set(thresholds))) != thresholds
    ):
        raise ValueError("stall thresholds must be unique positive ascending integers")
    return thresholds


def _candidate_pool(event: dict[str, Any]) -> list[dict[str, Any]]:
    controller = event.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("transition is missing controller evidence")
    raw_pool = controller.get("candidate_pool")
    if not isinstance(raw_pool, list) or not raw_pool:
        raise ValueError("transition is missing its complete candidate pool")
    rows: list[dict[str, Any]] = []
    for raw in raw_pool:
        if not isinstance(raw, dict):
            raise ValueError("candidate pool contains a non-object")
        candidate_id = str(raw.get("candidate_id") or "")
        agents = raw.get("agents")
        actual_size = raw.get("actual_size")
        score = raw.get("score")
        if (
            not candidate_id
            or not isinstance(agents, list)
            or any(type(agent) is not int or agent < 0 for agent in agents)
            or len(agents) != len(set(agents))
            or type(actual_size) is not int
            or actual_size != len(agents)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("candidate pool contains invalid candidate evidence")
        rows.append(
            {
                "candidate_id": candidate_id,
                "agents": list(map(int, agents)),
                "actual_size": int(actual_size),
                "score": float(score),
                "rank": 0,
            }
        )
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("candidate pool contains duplicate candidate ids")
    ranked = sorted(rows, key=lambda row: (-round(row["score"], 12), row["candidate_id"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    selected = str(controller.get("selected_candidate_id") or "")
    if selected not in {row["candidate_id"] for row in ranked}:
        raise ValueError("selected candidate is absent from the recorded pool")
    return ranked


def _attempt_key(row: dict[str, Any]) -> str:
    action = dict(row["actual_action"])
    metrics = dict(row["actual_metrics"])
    agents = metrics.get("neighborhood", action.get("agents"))
    repair_order = metrics.get("repair_order")
    if not isinstance(agents, list) or not agents:
        raise ValueError("repair attempt is missing its actual neighborhood")
    if not isinstance(repair_order, list) or sorted(map(int, repair_order)) != sorted(
        map(int, agents)
    ):
        raise ValueError("repair attempt is missing a valid repair order")
    return _fingerprint(
        {
            "neighborhood": neighborhood_key(map(int, agents)),
            "repair_order": list(map(int, repair_order)),
            "step_random_seed": action.get("random_seed"),
            "requested_pp_random_seed": metrics.get("requested_pp_random_seed"),
            "applied_pp_random_seed": metrics.get("applied_pp_random_seed"),
        }
    )


def normalize_episode_decisions(
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    transitions = [event for event in events if str(event.get("event")) == "transition"]
    if len(rows) != len(transitions):
        raise ValueError("decision rows and transition events have different lengths")
    normalized: list[dict[str, Any]] = []
    for row, event in zip(rows, transitions):
        outcome = dict(row["actual_lns2"])["outcome"]
        metrics = dict(row["actual_metrics"])
        if type(metrics.get("replan_success")) is not bool:
            raise ValueError("transition lacks a boolean PP result")
        repair_outcome = classify_repair_outcome(
            before_fingerprint=str(row["before_repair_fingerprint"]),
            after_fingerprint=str(row["after_repair_fingerprint"]),
            replan_success=metrics["replan_success"],
            conflicts_before=int(outcome["conflicts_before"]),
            conflicts_after=int(outcome["conflicts_after"]),
            feasible=bool(outcome["success"]),
        )
        pool = _candidate_pool(event)
        controller = dict(event["controller"])
        normalized.append(
            {
                **row,
                "repair_outcome": repair_outcome,
                "no_progress": repair_outcome in NO_PROGRESS_OUTCOMES,
                "attempt_key": _attempt_key(row),
                "actual_neighborhood_key": neighborhood_key(
                    map(int, metrics["neighborhood"])
                ),
                "candidate_pool": pool,
                "selected_candidate_id": str(controller["selected_candidate_id"]),
            }
        )
    return normalized


def classify_stall_sequences(
    decisions: list[dict[str, Any]],
    *,
    thresholds: Iterable[int] = (3, 4, 6),
    future_observation_decisions: int = 3,
    minimum_distinct_attempts: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checked_thresholds = _normalized_thresholds(thresholds)
    future = _strict_positive_int(
        future_observation_decisions, field="future observation decisions"
    )
    minimum_attempts = _strict_positive_int(
        minimum_distinct_attempts, field="minimum distinct attempts"
    )
    sequences: list[dict[str, Any]] = []
    ordinary: list[dict[str, Any]] = []
    previous_no_progress = False
    index = 0
    while index < len(decisions):
        decision = decisions[index]
        if not bool(decision["no_progress"]):
            if not previous_no_progress:
                ordinary.append(decision)
            previous_no_progress = False
            index += 1
            continue
        state_fingerprint = str(decision["before_repair_fingerprint"])
        start = index
        while (
            index < len(decisions)
            and bool(decisions[index]["no_progress"])
            and str(decisions[index]["before_repair_fingerprint"])
            == state_fingerprint
        ):
            index += 1
        run = decisions[start:index]
        next_decision = decisions[index] if index < len(decisions) else None
        recovered = bool(
            next_decision is not None
            and not bool(next_decision["no_progress"])
            and str(next_decision["before_repair_fingerprint"]) == state_fingerprint
        )
        eligible = [
            threshold
            for threshold in checked_thresholds
            if len(run) >= threshold
            and len({row["attempt_key"] for row in run[:threshold]})
            >= minimum_attempts
        ]
        if eligible:
            threshold = max(eligible)
            observed_after_trigger = len(run) - threshold
            if observed_after_trigger >= future:
                qualification_class = "confirmed_long_stall"
                resolution = "no_change_in_complete_future_window"
                recovery_delay = None
            elif recovered:
                qualification_class = "natural_recovery"
                resolution = "v2_self_recovered"
                recovery_delay = observed_after_trigger + 1
            else:
                qualification_class = "unresolved_stall"
                resolution = "trace_ended_before_complete_future_window"
                recovery_delay = None
            trigger = run[threshold - 1]
            sequences.append(
                {
                    "qualification_class": qualification_class,
                    "resolution": resolution,
                    "state_fingerprint": state_fingerprint,
                    "run_start": run[0],
                    "trigger": trigger,
                    "next_changed_decision": next_decision if recovered else None,
                    "threshold": threshold,
                    "run_length": len(run),
                    "observed_after_trigger": observed_after_trigger,
                    "recovery_delay_decisions": recovery_delay,
                    "distinct_attempt_count": len(
                        {row["attempt_key"] for row in run}
                    ),
                    "distinct_neighborhood_count": len(
                        {row["actual_neighborhood_key"] for row in run}
                    ),
                    "history": run,
                }
            )
        previous_no_progress = True
    return sequences, ordinary


def validate_split_map_isolation(maps_by_split: dict[str, set[str]]) -> None:
    if set(maps_by_split) != {"policy_train", "policy_validation"}:
        raise ValueError("qualification requires policy_train and policy_validation")
    overlap = maps_by_split["policy_train"] & maps_by_split["policy_validation"]
    if overlap:
        raise ValueError(
            "training and diagnostic maps overlap: " + ", ".join(sorted(overlap))
        )


def load_oracle_index(
    oracle_root: str | Path | None,
    *,
    allowed_sizes: tuple[int, ...] = (4, 8, 16),
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if oracle_root is None:
        return {}, None
    root = Path(oracle_root).resolve()
    run_path = root / "run_config.json"
    run = _read_json(run_path)
    run_fingerprint = str(run.get("run_fingerprint") or "")
    if str(run.get("schema")) != "lns2.high_load_rescue_collection.v1" or not run_fingerprint:
        raise ValueError("Oracle collection is not a registered rescue collection")
    coverage_path = root.parent / "coverage_validation.json"
    if not coverage_path.is_file():
        coverage_path = root / "coverage_validation.json"
    coverage = _read_json(coverage_path)
    if not bool(coverage.get("passed")):
        raise ValueError("Oracle collection coverage did not pass")
    index: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "states").glob("*/*.json")):
        payload = _read_json(path)
        if not bool(payload.get("complete")) or str(payload.get("run_fingerprint")) != run_fingerprint:
            raise ValueError(f"Oracle state is incomplete or has wrong identity: {path}")
        state = payload.get("state")
        candidates = payload.get("candidates")
        trials = payload.get("trials")
        if not isinstance(state, dict) or not isinstance(candidates, list) or not isinstance(trials, list):
            raise ValueError(f"Oracle state payload is malformed: {path}")
        repair_fingerprint = str(state.get("before_repair_fingerprint") or "")
        if not repair_fingerprint or repair_fingerprint in index:
            raise ValueError("Oracle state fingerprints must be non-empty and unique")
        model_candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("route")) == "model"
            and int(candidate.get("actual_size", 0)) in allowed_sizes
        ]
        candidate_ids = [str(candidate.get("candidate_id") or "") for candidate in model_candidates]
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Oracle model candidate ids are missing or duplicated")
        counts = collections.Counter(
            str(trial.get("candidate_id") or "") for trial in trials
        )
        if any(counts[candidate_id] <= 0 for candidate_id in candidate_ids):
            raise ValueError("Oracle candidate is missing trial evidence")
        index[repair_fingerprint] = {
            "path": str(path),
            "state_id": str(state.get("state_id") or ""),
            "candidate_ids": sorted(candidate_ids),
            "candidate_trial_counts": {
                candidate_id: counts[candidate_id] for candidate_id in candidate_ids
            },
            "candidate_count": len(candidate_ids),
            "trial_count": sum(counts[candidate_id] for candidate_id in candidate_ids),
        }
    metadata = {
        "root": str(root),
        "run_config_sha256": sha256_file(run_path),
        "coverage_sha256": sha256_file(coverage_path),
        "run_fingerprint": run_fingerprint,
        "state_count": len(index),
    }
    return index, metadata


def _round_robin_limit(
    rows: list[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["map_id"])].append(row)
    for values in grouped.values():
        values.sort(
            key=lambda row: (
                str(row["task_id"]),
                int(row["solver_seed"]),
                int(row["anchor_decision_index"]),
            )
        )
    selected: list[dict[str, Any]] = []
    while len(selected) < maximum and any(grouped.values()):
        for map_id in sorted(grouped):
            if grouped[map_id] and len(selected) < maximum:
                selected.append(grouped[map_id].pop(0))
    return selected


def _qualification_row(
    *,
    source_root: Path,
    source_run_fingerprint: str,
    manifest: dict[str, Any],
    qualification_class: str,
    anchor: dict[str, Any],
    threshold: int,
    run_length: int,
    observed_after_trigger: int,
    distinct_attempt_count: int,
    distinct_neighborhood_count: int,
    resolution: str,
    recovery_delay: int | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    if qualification_class not in QUALIFICATION_CLASSES:
        raise ValueError("unknown stall qualification class")
    pool = list(anchor["candidate_pool"])
    return {
        "schema": STALL_ESCAPE_QUALIFICATION_SCHEMA,
        "qualification_class": qualification_class,
        "split": str(manifest["split"]),
        "map_id": str(manifest["map_id"]),
        "layout_mode": str(manifest.get("layout_mode", "unknown")),
        "agent_count": int(manifest.get("agent_count", 0)),
        "task_id": str(manifest["task_id"]),
        "solver_seed": int(manifest["solver_seed"]),
        "episode_id": str(manifest["episode_id"]),
        "source_root": str(source_root),
        "source_run_fingerprint": source_run_fingerprint,
        "source_trace_file": str(manifest["trace_file"]),
        "source_trace_sha256": str(manifest["trace_sha256"]),
        "anchor_decision_index": int(anchor["decision_index"]),
        "before_fingerprint": str(anchor["before_fingerprint"]),
        "before_repair_fingerprint": str(anchor["before_repair_fingerprint"]),
        "prefix_actions": list(anchor["prefix_actions"]),
        "candidate_pool": pool,
        "candidate_count": len(pool),
        "selected_candidate_id": str(anchor["selected_candidate_id"]),
        "threshold": int(threshold),
        "run_length": int(run_length),
        "observed_after_trigger": int(observed_after_trigger),
        "distinct_attempt_count": int(distinct_attempt_count),
        "distinct_neighborhood_count": int(distinct_neighborhood_count),
        "resolution": resolution,
        "recovery_delay_decisions": recovery_delay,
        "history_attempts": [
            {
                "decision_index": int(row["decision_index"]),
                "attempt_key": str(row["attempt_key"]),
                "neighborhood_key": str(row["actual_neighborhood_key"]),
                "selected_candidate_id": str(row["selected_candidate_id"]),
                "repair_outcome": str(row["repair_outcome"]),
            }
            for row in history
        ],
    }


def _attach_oracle_plan(
    row: dict[str, Any],
    oracle_index: dict[str, dict[str, Any]],
    *,
    required_trials_per_candidate: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = oracle_index.get(str(row["before_repair_fingerprint"]))
    requires_oracle = str(row["qualification_class"]) in {
        "confirmed_long_stall",
        "unresolved_stall",
    }
    source_ids = sorted(
        str(candidate["candidate_id"])
        for candidate in row["candidate_pool"]
        if int(candidate["actual_size"]) in {4, 8, 16}
    )
    exact_pool = bool(evidence and source_ids == evidence["candidate_ids"])
    existing_counts = (
        dict(evidence["candidate_trial_counts"]) if exact_pool and evidence else {}
    )
    missing = (
        sum(
            max(
                0,
                required_trials_per_candidate
                - int(existing_counts.get(candidate_id, 0)),
            )
            for candidate_id in source_ids
        )
        if requires_oracle
        else 0
    )
    attached = {
        **row,
        "oracle_state_available": evidence is not None,
        "oracle_exact_candidate_pool": exact_pool,
        "oracle_state_file": str(evidence["path"]) if evidence else None,
        "existing_oracle_trial_count": sum(existing_counts.values()),
        "required_oracle_trial_count": (
            len(source_ids) * required_trials_per_candidate
            if requires_oracle
            else 0
        ),
        "missing_oracle_trial_count": missing,
    }
    plan = {
        "split": row["split"],
        "qualification_class": row["qualification_class"],
        "map_id": row["map_id"],
        "task_id": row["task_id"],
        "solver_seed": row["solver_seed"],
        "anchor_decision_index": row["anchor_decision_index"],
        "before_repair_fingerprint": row["before_repair_fingerprint"],
        "candidate_count": len(source_ids),
        "exact_oracle_pool_available": exact_pool,
        "existing_trial_count": sum(existing_counts.values()),
        "required_trial_count": (
            len(source_ids) * required_trials_per_candidate
            if requires_oracle
            else 0
        ),
        "missing_trial_count": missing,
        "action": (
            "shadow_control_no_pp_branches"
            if not requires_oracle
            else "reuse_exact_oracle"
            if missing == 0
            else "collect_missing_pp_branches"
        ),
    }
    return attached, plan


def prepare_stall_escape_qualification(
    sources: Iterable[str | Path],
    output: str | Path,
    *,
    oracle_root: str | Path | None = None,
    thresholds: Iterable[int] = (3, 4, 6),
    future_observation_decisions: int = 3,
    minimum_distinct_attempts: int = 2,
    maximum_per_class_per_split: int = 48,
    required_trials_per_candidate: int = 4,
) -> dict[str, Any]:
    source_roots = [Path(source).resolve() for source in sources]
    if len(source_roots) != 2 or len(set(source_roots)) != 2:
        raise ValueError("qualification requires exactly two distinct source roots")
    output_root = Path(output).resolve()
    if output_root.is_dir() and any(output_root.iterdir()):
        raise ValueError("qualification output is non-empty")
    checked_thresholds = _normalized_thresholds(thresholds)
    maximum = _strict_positive_int(
        maximum_per_class_per_split, field="maximum states per class and split"
    )
    required_trials = _strict_positive_int(
        required_trials_per_candidate, field="required trials per candidate"
    )
    oracle_index, oracle_metadata = load_oracle_index(oracle_root)

    maps_by_split: dict[str, set[str]] = {}
    source_metadata: list[dict[str, Any]] = []
    class_rows: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    episode_count = 0
    decision_count = 0
    legacy_negative_step_runtime_count = 0
    for root in source_roots:
        run_path = root / "run_config.json"
        manifest_path = root / "realized_dynamic_manifest.jsonl"
        run = _read_json(run_path)
        configuration = run.get("configuration")
        if (
            str(run.get("controller")) != "v2-full"
            or not isinstance(configuration, dict)
            or str(configuration.get("split")) not in {"policy_train", "policy_validation"}
        ):
            raise ValueError("qualification sources must be v2-full policy sources")
        split = str(configuration["split"])
        if split in maps_by_split:
            raise ValueError(f"duplicate source split: {split}")
        run_fingerprint = str(run.get("run_fingerprint") or "")
        if not run_fingerprint:
            raise ValueError("qualification source lacks a run fingerprint")
        manifests = _read_jsonl(manifest_path)
        maps_by_split[split] = set()
        seen_episode_ids: set[str] = set()
        for manifest in manifests:
            if str(manifest.get("status")) not in {"ok", "resumed"}:
                raise ValueError("qualification source contains an incomplete episode")
            if str(manifest.get("split")) != split:
                raise ValueError("manifest split differs from its source configuration")
            episode_id = str(manifest.get("episode_id") or "")
            if not episode_id or episode_id in seen_episode_ids:
                raise ValueError("manifest episode ids must be non-empty and unique")
            seen_episode_ids.add(episode_id)
            maps_by_split[split].add(str(manifest["map_id"]))
            _trace_path, validated_events, _blob = validate_manifest_trace(
                root,
                manifest,
                run_fingerprint=run_fingerprint,
                expected_policy="realized_dynamic",
            )
            legacy_negative_step_runtime_count += sum(
                str(event.get("native_timing_schema")) == "lns2.repair_timing.v1"
                and isinstance(event.get("metrics"), dict)
                and isinstance(event["metrics"].get("step_runtime"), (int, float))
                and not isinstance(event["metrics"].get("step_runtime"), bool)
                and math.isfinite(float(event["metrics"]["step_runtime"]))
                and float(event["metrics"]["step_runtime"]) < 0.0
                for event in validated_events
            )
            rows, replay_events = decision_rows(root, manifest)
            if validated_events != replay_events:
                raise ValueError("validated and replay trace views differ")
            normalized = normalize_episode_decisions(rows, replay_events)
            sequences, ordinary = classify_stall_sequences(
                normalized,
                thresholds=checked_thresholds,
                future_observation_decisions=future_observation_decisions,
                minimum_distinct_attempts=minimum_distinct_attempts,
            )
            for sequence in sequences:
                row = _qualification_row(
                    source_root=root,
                    source_run_fingerprint=run_fingerprint,
                    manifest=manifest,
                    qualification_class=str(sequence["qualification_class"]),
                    anchor=sequence["run_start"],
                    threshold=int(sequence["threshold"]),
                    run_length=int(sequence["run_length"]),
                    observed_after_trigger=int(sequence["observed_after_trigger"]),
                    distinct_attempt_count=int(sequence["distinct_attempt_count"]),
                    distinct_neighborhood_count=int(sequence["distinct_neighborhood_count"]),
                    resolution=str(sequence["resolution"]),
                    recovery_delay=sequence["recovery_delay_decisions"],
                    history=list(sequence["history"]),
                )
                class_rows[(split, row["qualification_class"])].append(row)
            for decision in ordinary:
                row = _qualification_row(
                    source_root=root,
                    source_run_fingerprint=run_fingerprint,
                    manifest=manifest,
                    qualification_class="ordinary_progress",
                    anchor=decision,
                    threshold=0,
                    run_length=0,
                    observed_after_trigger=0,
                    distinct_attempt_count=1,
                    distinct_neighborhood_count=1,
                    resolution=str(decision["repair_outcome"]),
                    recovery_delay=None,
                    history=[],
                )
                class_rows[(split, "ordinary_progress")].append(row)
            episode_count += 1
            decision_count += len(normalized)
        source_metadata.append(
            {
                "root": str(root),
                "split": split,
                "run_fingerprint": run_fingerprint,
                "run_config_sha256": sha256_file(run_path),
                "manifest_sha256": sha256_file(manifest_path),
                "episode_count": len(manifests),
                "map_count": len(maps_by_split[split]),
            }
        )
    validate_split_map_isolation(maps_by_split)

    selected: list[dict[str, Any]] = []
    for split in ("policy_train", "policy_validation"):
        for qualification_class in QUALIFICATION_CLASSES:
            selected.extend(
                _round_robin_limit(
                    class_rows[(split, qualification_class)], maximum
                )
            )
    seen_state_keys: set[tuple[str, str, str]] = set()
    attached_rows: list[dict[str, Any]] = []
    branch_plan: list[dict[str, Any]] = []
    for row in selected:
        key = (
            str(row["split"]),
            str(row["qualification_class"]),
            str(row["before_repair_fingerprint"]),
        )
        if key in seen_state_keys:
            continue
        seen_state_keys.add(key)
        attached, plan = _attach_oracle_plan(
            row,
            oracle_index,
            required_trials_per_candidate=required_trials,
        )
        attached_rows.append(attached)
        branch_plan.append(plan)
    if not attached_rows:
        raise ValueError("qualification found no usable states")

    counts = collections.Counter(
        (str(row["split"]), str(row["qualification_class"]))
        for row in attached_rows
    )
    exact_oracle_count = sum(bool(row["oracle_exact_candidate_pool"]) for row in attached_rows)
    missing_trials = sum(int(row["missing_oracle_trial_count"]) for row in attached_rows)
    confirmed_count = sum(
        str(row["qualification_class"]) == "confirmed_long_stall"
        for row in attached_rows
    )
    unresolved_count = sum(
        str(row["qualification_class"]) == "unresolved_stall"
        for row in attached_rows
    )
    report = {
        "schema": STALL_ESCAPE_QUALIFICATION_SCHEMA,
        "complete": True,
        "training_started": False,
        "controller_actions_changed": False,
        "movingai_or_formal_labels_seen": False,
        "sources": sorted(source_metadata, key=lambda row: row["split"]),
        "oracle": oracle_metadata,
        "thresholds": list(checked_thresholds),
        "future_observation_decisions": int(future_observation_decisions),
        "minimum_distinct_attempts": int(minimum_distinct_attempts),
        "required_trials_per_candidate": required_trials,
        "source_episode_count": episode_count,
        "source_decision_count": decision_count,
        "legacy_v1_negative_step_runtime_count": (
            legacy_negative_step_runtime_count
        ),
        "map_isolation_passed": True,
        "qualification_state_count": len(attached_rows),
        "state_count_by_split_and_class": {
            f"{split}:{qualification_class}": counts[(split, qualification_class)]
            for split in ("policy_train", "policy_validation")
            for qualification_class in QUALIFICATION_CLASSES
        },
        "exact_oracle_pool_state_count": exact_oracle_count,
        "confirmed_long_stall_state_count": confirmed_count,
        "unresolved_stall_state_count": unresolved_count,
        "missing_oracle_trial_count": missing_trials,
        "decision": (
            "extend_unresolved_v2_prefixes_before_oracle"
            if unresolved_count and not confirmed_count
            else "collect_missing_exact_pp_branches"
            if confirmed_count and missing_trials
            else "qualification_ready_for_stall_model_design"
            if confirmed_count
            else "no_confirmed_stall_labels_do_not_train"
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "qualification_states.jsonl", attached_rows)
    atomic_write_csv(
        output_root / "qualification_states.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"prefix_actions", "candidate_pool", "history_attempts"}
            }
            for row in attached_rows
        ],
    )
    atomic_write_csv(output_root / "missing_pp_branches.csv", branch_plan)
    _write_json(output_root / "qualification_report.json", report)
    lines = [
        "# Stall escape qualification",
        "",
        f"- Source episodes: `{episode_count}`; decisions: `{decision_count}`.",
        f"- Qualified states: `{len(attached_rows)}`; exact existing Oracle pools: `{exact_oracle_count}`.",
        f"- Authenticated legacy-v1 negative step_runtime fields viewed through native timing compatibility: `{legacy_negative_step_runtime_count}`.",
        f"- Confirmed long stalls: `{confirmed_count}`; unresolved terminal stalls: `{unresolved_count}`.",
        f"- Missing paired PP trials for confirmed/unresolved states at {required_trials} trials/candidate: `{missing_trials}`.",
        "- Map isolation: `passed`; MovingAI/formal labels used: `false`.",
        "- Training started: `false`; controller actions changed: `false`.",
        f"- Decision: `{report['decision']}`.",
        "",
        "A no-conflict-reduction transition is never treated as failure when the repair state changes. A terminal trace without the full future observation window remains unresolved rather than becoming a positive stall label.",
        "",
    ]
    (output_root / "qualification_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = [
    "STALL_ESCAPE_QUALIFICATION_SCHEMA",
    "classify_stall_sequences",
    "load_oracle_index",
    "normalize_episode_decisions",
    "prepare_stall_escape_qualification",
    "validate_split_map_isolation",
]
