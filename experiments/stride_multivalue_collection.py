from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments._common import contained_file, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine, TopologyAnalysisCache
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_closurepool_longtail import reconstruct_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.paretopool import (
    PARETOPOOL_BUDGETS,
    generate_paretopool_candidates,
)
from lns2_selector.runtime.temporal_state import (
    TemporalHistoryContext,
    temporal_history_context,
    temporal_state_identity,
)


CONFIG_SCHEMA = "lns2.stride.multivalue_collection_registration.v1"
STATE_MANIFEST_SCHEMA = "lns2.stride.multivalue_state_manifest.v1"
PREPARATION_REPORT_SCHEMA = "lns2.stride.multivalue_preparation_report.v1"


def _registered(root: Path, specification: Mapping[str, Any]) -> Path:
    return registered_input(root, dict(specification), label="MultiValue collection")


def load_multivalue_collection_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_outcome_enriched_label_stability_pilot"
        or config.get("experiment_id") != "stride-multivalue-pilot-v1"
    ):
        raise ValueError("MultiValue pilot identity changed")
    inputs = {
        name: _registered(root, specification)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "root_checkpoints",
        "paretopool_gap_report",
        "state_collection_config",
        "runtime_config",
        "qualification_report",
    }:
        raise ValueError("MultiValue pilot input registry changed")
    rollout = dict(config.get("rollout") or {})
    if (
        tuple(map(int, rollout.get("horizons") or ())) != (1, 8, 32, 128)
        or tuple(map(str, rollout.get("teachers") or ()))
        != ("v2-full", "official-adaptive")
        or int(rollout.get("initial_paired_seed_count", -1)) != 8
        or int(rollout.get("extended_paired_seed_count", -1)) != 16
        or int(rollout.get("maximum_repair_decisions", -1)) != 128
        or float(rollout.get("wall_time_fuse_seconds", -1)) != 300.0
        or int(rollout.get("state_attempt_timeout_seconds", -1)) != 1800
        or int(rollout.get("maximum_state_attempts", -1)) != 4
        or int(rollout.get("no_progress_attempt_limit", -1)) != 2
    ):
        raise ValueError("MultiValue pilot rollout or recovery contract changed")
    if tuple(map(int, config["paretopool"]["candidate_budgets"])) != PARETOPOOL_BUDGETS:
        raise ValueError("MultiValue pilot candidate budgets changed")
    return path, root, config, inputs


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    first, second = set(left), set(right)
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _transition_neighborhood(transition: Mapping[str, Any]) -> tuple[int, ...]:
    metrics = transition.get("metrics")
    raw = metrics.get("neighborhood") if isinstance(metrics, Mapping) else None
    if not isinstance(raw, list) or not raw:
        action = transition.get("action")
        raw = action.get("agents") if isinstance(action, Mapping) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError("MultiValue history transition has no neighborhood")
    result = tuple(sorted(set(map(int, raw))))
    if len(result) != len(raw):
        raise ValueError("MultiValue history neighborhood contains duplicates")
    return result


def _edge_set(state: Mapping[str, Any]) -> set[tuple[int, int]]:
    return {tuple(sorted(map(int, edge))) for edge in state["conflict_edges"]}


def history_before_decision(
    states: Sequence[dict[str, Any]],
    transitions: Sequence[dict[str, Any]],
    *,
    decision_index: int,
    history_limit: int = 8,
) -> TemporalHistoryContext:
    target = int(decision_index)
    if target < 0 or target >= len(states) or target > len(transitions):
        raise ValueError("MultiValue target decision is outside its source trace")
    prefix_transitions = list(transitions[:target])
    prefix_states = list(states[: target + 1])
    neighborhoods = [_transition_neighborhood(row) for row in prefix_transitions]
    exact_repeat = []
    max_jaccard = []
    for index, neighborhood in enumerate(neighborhoods):
        prior = neighborhoods[:index]
        exact_repeat.append(neighborhood in prior)
        max_jaccard.append(
            max((_jaccard(neighborhood, row) for row in prior), default=0.0)
        )
    current_edges = _edge_set(prefix_states[-1])
    previous_edges = _edge_set(prefix_states[-2]) if len(prefix_states) > 1 else set()
    older_edges = set().union(*map(_edge_set, prefix_states[:-1]))
    streak = 1
    for state in reversed(prefix_states[:-1]):
        if _edge_set(state) != current_edges:
            break
        streak += 1
    repair_counts: collections.Counter[int] = collections.Counter(
        agent for neighborhood in neighborhoods for agent in neighborhood
    )
    candidate_ids = []
    pp_history = []
    for index, transition in enumerate(prefix_transitions):
        controller = transition.get("controller")
        candidate_ids.append(
            str(controller.get("selected_candidate_id", ""))
            if isinstance(controller, Mapping)
            else ""
        )
        metrics = dict(transition.get("metrics") or {})
        changed = repair_structure_fingerprint(prefix_states[index]) != (
            repair_structure_fingerprint(prefix_states[index + 1])
        )
        pp_history.append(
            (bool(metrics.get("replan_success")), not changed, changed)
        )
    return temporal_history_context(
        recent_neighborhoods=neighborhoods[-history_limit:],
        recent_neighborhood_exact_repeat=exact_repeat[-history_limit:],
        recent_neighborhood_max_jaccard=max_jaccard[-history_limit:],
        persistent_conflict_edges=current_edges & previous_edges,
        new_conflict_edges=current_edges - previous_edges,
        disappeared_conflict_edges=previous_edges - current_edges,
        reappeared_conflict_edges=current_edges & (older_edges - previous_edges),
        conflict_signature_streak=streak,
        agent_repair_counts=repair_counts,
        recent_candidate_ids=candidate_ids[-history_limit:],
        recent_pp_history=pp_history[-history_limit:],
    )


def _base_anchor(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    base = [
        dict(candidate)
        for candidate in checkpoint["candidate_pool"]
        if not candidate.get("structpool_family_groups")
    ]
    if not base or any(candidate.get("score") is None for candidate in base):
        raise ValueError("MultiValue checkpoint has no scored V2 base pool")
    anchor = max(
        base,
        key=lambda candidate: (
            float(candidate["score"]), str(candidate["candidate_id"])
        ),
    )
    anchor["multivalue_role"] = "v2_anchor"
    anchor["paretopool_budget_membership"] = []
    return anchor


def select_pilot_occurrences(
    checkpoints: Sequence[dict[str, Any]], *, per_map: int
) -> list[dict[str, Any]]:
    by_map_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for checkpoint in checkpoints:
        by_map_task.setdefault(str(checkpoint["map_id"]), {}).setdefault(
            str(checkpoint["task_id"]), []
        ).append(dict(checkpoint))
    selected: list[dict[str, Any]] = []
    for map_id, tasks in sorted(by_map_task.items()):
        ordered = {
            task_id: sorted(
                rows,
                key=lambda row: _fingerprint(
                    {
                        "purpose": "multivalue-pilot-outcome-blind-selection",
                        "map_id": map_id,
                        "task_id": task_id,
                        "case_id": str(row["case_id"]),
                        "decision_index": int(row["decision_index"]),
                    }
                ),
            )
            for task_id, rows in tasks.items()
        }
        map_rows: list[dict[str, Any]] = []
        depth = 0
        while len(map_rows) < per_map:
            added = False
            for task_id in sorted(ordered):
                if depth < len(ordered[task_id]):
                    map_rows.append(ordered[task_id][depth])
                    added = True
                    if len(map_rows) == per_map:
                        break
            if not added:
                raise ValueError(f"MultiValue map has too few pilot states: {map_id}")
            depth += 1
        selected.extend(map_rows)
    return sorted(
        selected,
        key=lambda row: (
            str(row["map_id"]),
            str(row["task_id"]),
            str(row["case_id"]),
            int(row["decision_index"]),
        ),
    )


def _source_collection(root: Path, checkpoint: Mapping[str, Any]) -> Path:
    return (
        root
        / "build"
        / "stride-tailswitch-v1"
        / "states"
        / _fingerprint({"state_id": str(checkpoint["state_id"])})[:20]
        / str(checkpoint["treatment_policy"])
    )


def _source_trace(
    root: Path, checkpoint: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source = _source_collection(root, checkpoint)
    manifests = _read_jsonl(source / "realized_dynamic_manifest.jsonl")
    if len(manifests) != 1 or manifests[0].get("status") != "ok":
        raise ValueError("MultiValue source trace manifest is invalid")
    manifest = dict(manifests[0])
    trace = reconstruct_trace(source, manifest)
    return source, manifest, trace


def _candidate_union(
    state: dict[str, Any],
    analysis: Any,
    anchor: dict[str, Any],
    history: TemporalHistoryContext,
    *,
    maximum_neighborhood_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {str(anchor["candidate_id"]): dict(anchor)}
    attempts: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, Any] = {}
    for budget in PARETOPOOL_BUDGETS:
        result = generate_paretopool_candidates(
            state,
            analysis,
            v2_anchor_agents=anchor["agents"],
            history=history,
            maximum_candidates=budget,
            maximum_neighborhood_size=maximum_neighborhood_size,
        )
        counts[str(budget)] = {
            "raw_candidate_count": result.raw_candidate_count,
            "selected_candidate_count": len(result.candidates),
            "history_candidate_count": result.history_candidate_count,
        }
        attempts[str(budget)] = result.attempts
        for candidate in result.candidates:
            identity = str(candidate["candidate_id"])
            existing = by_id.get(identity)
            if existing is None:
                existing = {**candidate, "multivalue_role": "paretopool_challenger"}
                existing["paretopool_budget_membership"] = []
                by_id[identity] = existing
            elif existing["agents"] != candidate["agents"]:
                raise ValueError("MultiValue candidate identity collision")
            existing["paretopool_budget_membership"].append(budget)
    candidates = [by_id[str(anchor["candidate_id"])] ] + sorted(
        [
            candidate
            for identity, candidate in by_id.items()
            if identity != str(anchor["candidate_id"])
        ],
        key=lambda candidate: str(candidate["candidate_id"]),
    )
    return candidates, {"budget_counts": counts, "attempts": attempts}


def prepare_multivalue_pilot(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, root, config, inputs = load_multivalue_collection_config(config_path)
    checkpoints = _read_jsonl(inputs["root_checkpoints"])
    selected = select_pilot_occurrences(
        checkpoints, per_map=int(config["cohort"]["per_map_occurrences"])
    )
    expected = int(config["cohort"]["state_occurrence_count"])
    if len(selected) != expected:
        raise ValueError("MultiValue pilot state count changed")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for checkpoint in selected:
        state_path = contained_file(
            inputs["root_checkpoints"].parent,
            checkpoint["state_blob"],
            field="MultiValue state blob",
        )
        if sha256_file(state_path) != str(checkpoint["state_blob_sha256"]):
            raise ValueError("MultiValue checkpoint state blob hash changed")
        state = read_state_blob(state_path)
        if state_fingerprint(state) != str(checkpoint["state_fingerprint"]):
            raise ValueError("MultiValue checkpoint state fingerprint changed")
        source, manifest, trace = _source_trace(root, checkpoint)
        target = int(checkpoint["decision_index"])
        if target >= len(trace["states"]):
            raise ValueError("MultiValue source trace ended before target")
        source_state = trace["states"][target]
        if state_fingerprint(source_state) != state_fingerprint(state):
            raise ValueError("MultiValue source trace and state blob differ")
        history = history_before_decision(
            trace["states"], trace["transitions"], decision_index=target
        )
        temporal = temporal_state_identity(
            episode_id=str(manifest["episode_id"]),
            decision_index=target,
            state_fingerprint=str(checkpoint["state_fingerprint"]),
            history=history,
        )
        cache = TopologyAnalysisCache(state, backend="native")
        if cache.analysis is None:
            raise RuntimeError("MultiValue native topology analysis is missing")
        anchor = _base_anchor(checkpoint)
        candidates, pool_diagnostic = _candidate_union(
            state,
            cache.analysis,
            anchor,
            history,
            maximum_neighborhood_size=int(
                config["paretopool"]["maximum_neighborhood_size"]
            ),
        )
        engine = OnlineFeatureEngine(
            state,
            backend="native",
            required_features={
                "realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]
            },
        )
        if cache.last_native_prepared is not None:
            engine.prepare(state, prepared_native_analysis=cache.last_native_prepared)
        feature_rows, feature_metrics = engine.realized_rows(
            candidates, state_hash=str(checkpoint["state_fingerprint"])
        )
        features = {
            str(row["candidate_id"]): dict(row["features"]["realized_dynamic"])
            for row in feature_rows
        }
        enriched = []
        for position, candidate in enumerate(candidates):
            identity = str(candidate["candidate_id"])
            values = features[identity]
            if set(values) != set(PROFILE_FEATURE_NAMES["realized_dynamic"]):
                raise RuntimeError("MultiValue feature schema changed")
            enriched.append(
                {
                    **candidate,
                    "candidate_position": position,
                    "feature_schema": "lns2.realized_features.v2",
                    "feature_count": len(values),
                    "features": values,
                    "feature_sha256": _fingerprint(values),
                }
            )
        rows.append(
            {
                "schema": STATE_MANIFEST_SCHEMA,
                "state_occurrence_id": temporal["state_occurrence_id"],
                "temporal_identity": temporal,
                "case_id": str(checkpoint["case_id"]),
                "map_id": str(checkpoint["map_id"]),
                "task_id": str(checkpoint["task_id"]),
                "solver_seed": int(checkpoint["solver_seed"]),
                "state_id": str(checkpoint["state_id"]),
                "decision_index": target,
                "before_conflicts": int(checkpoint["conflict_pair_count"]),
                "state_blob": str(state_path),
                "state_blob_sha256": str(checkpoint["state_blob_sha256"]),
                "source_collection": str(source),
                "source_manifest": manifest,
                "source_trace_sha256": str(manifest["trace_sha256"]),
                "v2_anchor_candidate_id": str(anchor["candidate_id"]),
                "candidate_count": len(enriched),
                "candidates": enriched,
                "candidate_pool_sha256": _fingerprint(
                    [
                        {
                            "candidate_id": row["candidate_id"],
                            "agents": row["agents"],
                            "budgets": row["paretopool_budget_membership"],
                            "feature_sha256": row["feature_sha256"],
                        }
                        for row in enriched
                    ]
                ),
                "pool_diagnostic": pool_diagnostic,
                "feature_metrics": feature_metrics,
                "future_outcome_used": False,
            }
        )
    rows.sort(key=lambda row: str(row["state_occurrence_id"]))
    manifest_path = output / "state_manifest.jsonl"
    _write_jsonl(manifest_path, rows)
    candidate_count = sum(int(row["candidate_count"]) for row in rows)
    initial_trials = int(config["rollout"]["initial_paired_seed_count"])
    teacher_count = len(config["rollout"]["teachers"])
    report = {
        "schema": PREPARATION_REPORT_SCHEMA,
        "scientific_status": "prepared_outcome_enriched_label_stability_pilot",
        "state_occurrence_count": len(rows),
        "unique_state_fingerprint_count": len(
            {str(row["temporal_identity"]["state_fingerprint"]) for row in rows}
        ),
        "map_counts": dict(collections.Counter(row["map_id"] for row in rows)),
        "task_count": len({str(row["task_id"]) for row in rows}),
        "candidate_count": candidate_count,
        "anchor_count": len(rows),
        "challenger_count": candidate_count - len(rows),
        "initial_trial_count": candidate_count * initial_trials * teacher_count,
        "maximum_extended_trial_count": candidate_count
        * int(config["rollout"]["extended_paired_seed_count"])
        * teacher_count,
        "state_manifest": str(manifest_path),
        "state_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(path),
        "input_sha256": {
            name: sha256_file(value) for name, value in sorted(inputs.items())
        },
        "integrity": {
            "state_count": len(rows) == expected,
            "map_balance": set(collections.Counter(row["map_id"] for row in rows).values())
            == {int(config["cohort"]["per_map_occurrences"])},
            "all_candidate_features_124": all(
                int(candidate["feature_count"]) == 124
                for row in rows
                for candidate in row["candidates"]
            ),
            "one_anchor_per_state": all(
                sum(
                    candidate["multivalue_role"] == "v2_anchor"
                    for candidate in row["candidates"]
                )
                == 1
                for row in rows
            ),
            "future_outcome_not_used": all(
                row["future_outcome_used"] is False for row in rows
            ),
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    report["integrity_passed"] = all(report["integrity"].values())
    _write_json(output / "preparation_report.json", report)
    return report


__all__ = [
    "history_before_decision",
    "load_multivalue_collection_config",
    "prepare_multivalue_pilot",
    "select_pilot_occurrences",
]
