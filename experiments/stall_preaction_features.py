from __future__ import annotations

import collections
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    atomic_write_csv,
    contained_file,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    read_trace_events,
)
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import _fingerprint, state_fingerprint
from experiments.run_output_guard import prepare_run_output
from experiments.stall_escape_qualification import _candidate_pool
from experiments.stall_guard import repair_structure_fingerprint
from experiments.stall_shadow import neighborhood_key
from experiments.trace_replay import _initial_state


STALL_PREACTION_FEATURE_SCHEMA = "lns2.stall_preaction_features.v1"

# Frozen before model fitting.  These are all available before the target action.
COMPACT_CANONICAL_FEATURE_NAMES = (
    "state.agent_count",
    "state.colliding_pairs",
    "state.conflict_event_count",
    "state.conflicting_agent_ratio",
    "state.component_count",
    "state.largest_component_ratio",
    "state.degree_max",
    "state.conflict_time_mean",
    "state.conflict_time_std",
    "proposal.actual_size_ratio_agents",
    "proposal.seed_component_size_mean",
    "proposal.seed_conflict_degree_max",
    "proposal.seed_conflict_degree_mean",
    "proposal.seed_delay_mean",
    "realized.component_coverage_max",
    "realized.conflict_degree_max",
    "realized.conflict_degree_mean",
    "realized.conflicting_agent_coverage",
    "realized.incident_conflict_coverage",
    "realized.incident_event_coverage",
    "realized.internal_conflict_coverage",
    "realized.path_articulation_ratio",
    "realized.path_low_degree_ratio",
)

DERIVED_FEATURE_NAMES = (
    "v2.rank_fraction",
    "v2.score",
    "v2.score_gap_to_rank1",
    "history.available_steps",
    "history.recent_no_progress_length",
    "history.last_conflict_reduction_ratio",
    "history.mean3_conflict_reduction_ratio",
    "history.state_change_rate3",
    "history.previous_size_ratio_agents",
    "history.same_state_no_progress_attempts",
    "history.same_state_distinct_neighborhoods",
    "history.same_state_distinct_pp_orders",
    "candidate.overlap_last_failed",
    "candidate.overlap_failed_union",
    "candidate.exact_prior_no_progress_count",
    "candidate.persistent_edge_coverage",
    "candidate.persistent_agent_coverage",
    "state.persistent_edge_fraction",
)

FEATURE_NAMES = (*COMPACT_CANONICAL_FEATURE_NAMES, *DERIVED_FEATURE_NAMES)


def _bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in {"True", "true", "1"}:
        return True
    if value in {"False", "false", "0"}:
        return False
    raise ValueError(f"{field} is not a strict boolean")


def _source_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists() or os.name != "nt":
        return path.resolve()
    text = str(value).replace("\\", "/")
    if text.startswith("/mnt/") and len(text) > 7 and text[6] == "/":
        mapped = Path(f"{text[5].upper()}:{text[6:]}")
        if mapped.exists():
            return mapped.resolve()
    return path.resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _edge_set(state: dict[str, Any]) -> set[tuple[int, int]]:
    result = set()
    for edge in state.get("conflict_edges") or ():
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("state contains an invalid conflict edge")
        left, right = sorted(map(int, edge))
        result.add((left, right))
    return result


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _overlap(left: set[int], right: set[int]) -> float:
    return _ratio(len(left & right), len(left | right))


def temporal_candidate_features(
    *,
    history: list[dict[str, Any]],
    current_repair_fingerprint: str,
    current_conflicts: int,
    agent_count: int,
    candidate_agents: Iterable[int],
    candidate_rank: int,
    candidate_score: float,
    rank1_score: float,
    current_edges: set[tuple[int, int]],
    previous_edge_sets: list[set[tuple[int, int]]],
) -> dict[str, float]:
    agents = set(map(int, candidate_agents))
    recent = history[-3:]
    no_progress_length = 0
    for row in reversed(history):
        if not bool(row["no_progress"]):
            break
        no_progress_length += 1
    same_state: list[dict[str, Any]] = []
    for row in reversed(history):
        if (
            str(row["before_repair_fingerprint"]) != current_repair_fingerprint
            or not bool(row["no_progress"])
        ):
            break
        same_state.append(row)
    failed_sets = [set(map(int, row["agents"])) for row in same_state]
    failed_union = set().union(*failed_sets) if failed_sets else set()
    last_failed = failed_sets[0] if failed_sets else set()
    candidate_key = neighborhood_key(sorted(agents))
    previous_size = len(history[-1]["agents"]) if history else 0
    reductions = [float(row["conflict_reduction"]) for row in recent]
    persistent_edges = set(current_edges)
    for edges in previous_edge_sets[-2:]:
        persistent_edges &= set(edges)
    if not previous_edge_sets:
        persistent_edges.clear()
    persistent_agents = {agent for edge in persistent_edges for agent in edge}
    covered_edges = {
        edge for edge in persistent_edges if edge[0] in agents or edge[1] in agents
    }
    return {
        "v2.rank_fraction": _ratio(candidate_rank, 8),
        "v2.score": float(candidate_score),
        "v2.score_gap_to_rank1": float(rank1_score) - float(candidate_score),
        "history.available_steps": float(len(recent)),
        "history.recent_no_progress_length": float(no_progress_length),
        "history.last_conflict_reduction_ratio": _ratio(
            reductions[-1] if reductions else 0.0, current_conflicts
        ),
        "history.mean3_conflict_reduction_ratio": _ratio(
            math.fsum(reductions) / len(reductions) if reductions else 0.0,
            current_conflicts,
        ),
        "history.state_change_rate3": (
            math.fsum(float(row["state_changed"]) for row in recent) / len(recent)
            if recent
            else 0.0
        ),
        "history.previous_size_ratio_agents": _ratio(previous_size, agent_count),
        "history.same_state_no_progress_attempts": float(len(same_state)),
        "history.same_state_distinct_neighborhoods": float(
            len({str(row["neighborhood_key"]) for row in same_state})
        ),
        "history.same_state_distinct_pp_orders": float(
            len(
                {
                    (
                        int(row["pp_random_seed"]),
                        tuple(map(int, row["repair_order"])),
                    )
                    for row in same_state
                }
            )
        ),
        "candidate.overlap_last_failed": _overlap(agents, last_failed),
        "candidate.overlap_failed_union": _overlap(agents, failed_union),
        "candidate.exact_prior_no_progress_count": float(
            sum(str(row["neighborhood_key"]) == candidate_key for row in same_state)
        ),
        "candidate.persistent_edge_coverage": _ratio(
            len(covered_edges), len(persistent_edges)
        ),
        "candidate.persistent_agent_coverage": _ratio(
            len(agents & persistent_agents), len(persistent_agents)
        ),
        "state.persistent_edge_fraction": _ratio(
            len(persistent_edges), len(current_edges)
        ),
    }


def _feature_dict(row: dict[str, Any]) -> dict[str, float]:
    if "feature_names" in row:
        names = tuple(map(str, row["feature_names"]))
        values = tuple(map(float, row["feature_values"]))
        return dict(zip(names, values))
    return {
        str(name): float(value)
        for name, value in dict(row["features"]["realized_dynamic"]).items()
    }


def _history_row(
    state: dict[str, Any], after: dict[str, Any], event: dict[str, Any]
) -> dict[str, Any]:
    metrics = dict(event.get("metrics") or {})
    agents = list(map(int, metrics.get("neighborhood") or event["action"]["agents"]))
    before_repair = repair_structure_fingerprint(state)
    after_repair = repair_structure_fingerprint(after)
    return {
        "before_repair_fingerprint": before_repair,
        "after_repair_fingerprint": after_repair,
        "state_changed": before_repair != after_repair,
        "no_progress": before_repair == after_repair,
        "conflict_reduction": int(state["num_of_colliding_pairs"])
        - int(after["num_of_colliding_pairs"]),
        "agents": agents,
        "neighborhood_key": neighborhood_key(agents),
        "pp_random_seed": int(metrics.get("applied_pp_random_seed", -1)),
        "repair_order": list(map(int, metrics.get("repair_order") or ())),
    }


def _extract_target(
    *,
    selected: dict[str, Any],
    state: dict[str, Any],
    event: dict[str, Any],
    history: list[dict[str, Any]],
    previous_edge_sets: list[set[tuple[int, int]]],
    feature_backend: str,
    state_label: dict[str, str],
    candidate_labels: dict[tuple[str, str], dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    full_fingerprint = state_fingerprint(state)
    repair_fingerprint = repair_structure_fingerprint(state)
    if (
        full_fingerprint != str(selected["before_fingerprint"])
        or repair_fingerprint != str(selected["before_repair_fingerprint"])
    ):
        raise ValueError("pre-action feature state fingerprint mismatch")
    pool = _candidate_pool(event)
    registered = [dict(row) for row in selected["candidate_rows"]]
    if len(pool) < len(registered):
        raise ValueError("pre-action feature source lost registered candidates")
    candidates = pool[: len(registered)]
    for index, (actual, expected) in enumerate(zip(candidates, registered), 1):
        if (
            str(actual.get("candidate_id")) != str(expected["candidate_id"])
            or neighborhood_key(actual["agents"]) != str(expected["neighborhood_key"])
            or int(actual["actual_size"]) != int(expected["actual_size"])
            or abs(float(actual["score"]) - float(expected["score"])) > 1e-12
            or int(expected["rank"]) != index
        ):
            raise ValueError("pre-action candidate pool differs from registration")
    engine = OnlineFeatureEngine(
        state,
        backend=feature_backend,
        shadow_validation=False,
        required_features={"realized_dynamic": COMPACT_CANONICAL_FEATURE_NAMES},
        dense_output=feature_backend == "native",
    )
    feature_rows, _metrics = engine.realized_rows(
        candidates, state_hash=full_fingerprint
    )
    if [str(row["candidate_id"]) for row in feature_rows] != [
        str(row["candidate_id"]) for row in candidates
    ]:
        raise ValueError("pre-action feature candidate order mismatch")
    current_edges = _edge_set(state)
    rank1_score = float(candidates[0]["score"])
    output_candidates = []
    for rank, (candidate, raw_features) in enumerate(
        zip(candidates, feature_rows), 1
    ):
        canonical = _feature_dict(raw_features)
        if set(canonical) != set(COMPACT_CANONICAL_FEATURE_NAMES):
            raise ValueError("compact pre-action feature schema mismatch")
        derived = temporal_candidate_features(
            history=history,
            current_repair_fingerprint=repair_fingerprint,
            current_conflicts=int(state["num_of_colliding_pairs"]),
            agent_count=int(selected["agent_count"]),
            candidate_agents=candidate["agents"],
            candidate_rank=rank,
            candidate_score=float(candidate["score"]),
            rank1_score=rank1_score,
            current_edges=current_edges,
            previous_edge_sets=previous_edge_sets,
        )
        features = {**canonical, **derived}
        if set(features) != set(FEATURE_NAMES) or not all(
            math.isfinite(float(value)) for value in features.values()
        ):
            raise ValueError("pre-action compact features are incomplete or non-finite")
        label = candidate_labels[(repair_fingerprint, str(candidate["candidate_id"]))]
        output_candidates.append(
            {
                "state_key": str(selected["state_key"]),
                "map_id": str(selected["map_id"]),
                "map_fold": int(selected["map_fold"]),
                "cohort_role": str(selected["cohort_role"]),
                "before_repair_fingerprint": repair_fingerprint,
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_rank": rank,
                "stable_escape": _bool(label["stable_escape"], field="stable escape"),
                "stable_failure": _bool(label["stable_failure"], field="stable failure"),
                "pp_order_sensitive": _bool(
                    label["pp_order_sensitive"], field="PP order sensitive"
                ),
                "mean_conflict_delta": float(label["mean_conflict_delta"]),
                "mean_total_decision_seconds": float(
                    label["mean_total_decision_seconds"]
                ),
                **{f"feature:{name}": float(features[name]) for name in FEATURE_NAMES},
            }
        )
    rank1 = output_candidates[0]
    state_output = {
        "state_key": str(selected["state_key"]),
        "map_id": str(selected["map_id"]),
        "map_fold": int(selected["map_fold"]),
        "layout_mode": str(selected["layout_mode"]),
        "agent_count": int(selected["agent_count"]),
        "cohort_role": str(selected["cohort_role"]),
        "before_repair_fingerprint": repair_fingerprint,
        "target_rescuable_selector_failure": _bool(
            state_label["target_rescuable_selector_failure"], field="trigger target"
        ),
        **{f"feature:{name}": rank1[f"feature:{name}"] for name in FEATURE_NAMES},
    }
    return state_output, output_candidates, engine.backend


def build_stall_preaction_features(
    cohort: str | Path,
    labels: str | Path,
    output: str | Path,
    *,
    feature_backend: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    cohort_root = Path(cohort).resolve()
    label_root = Path(labels).resolve()
    output_root = Path(output).resolve()
    selected_path = cohort_root / "selected_states.jsonl"
    label_report_path = label_root / "stall_preaction_label_report.json"
    state_label_path = label_root / "stall_trigger_labels.csv"
    candidate_label_path = label_root / "stall_rescue_candidate_labels.csv"
    label_report = dict(read_json(label_report_path))
    if label_report.get("complete") is not True or label_report.get(
        "complete_coverage"
    ) is not True:
        raise ValueError("pre-action feature labels are incomplete")
    selected = [dict(row) for row in read_jsonl(selected_path)]
    state_labels = {
        str(row["before_repair_fingerprint"]): row
        for row in _read_csv(state_label_path)
    }
    candidate_labels = {
        (str(row["before_repair_fingerprint"]), str(row["candidate_id"])): row
        for row in _read_csv(candidate_label_path)
    }
    if len(state_labels) != len(selected):
        raise ValueError("pre-action feature state label coverage mismatch")
    identity = {
        "schema": STALL_PREACTION_FEATURE_SCHEMA,
        "selected_states_sha256": sha256_file(selected_path),
        "label_report_sha256": sha256_file(label_report_path),
        "state_labels_sha256": sha256_file(state_label_path),
        "candidate_labels_sha256": sha256_file(candidate_label_path),
        "feature_names": list(FEATURE_NAMES),
        "feature_backend": feature_backend,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    prepare_run_output(output_root, resume=resume, identity=identity)

    by_episode: dict[tuple[str, str, int], dict[int, dict[str, Any]]] = (
        collections.defaultdict(dict)
    )
    for row in selected:
        key = (str(row["source"]), str(row["task_id"]), int(row["solver_seed"]))
        decision_index = int(row["decision_index"])
        if decision_index in by_episode[key]:
            raise ValueError("pre-action feature selection repeats one decision")
        by_episode[key][decision_index] = row

    state_outputs: list[dict[str, Any]] = []
    candidate_outputs: list[dict[str, Any]] = []
    backends: collections.Counter[str] = collections.Counter()
    source_hashes: dict[str, str] = {}
    for (source_text, task_id, solver_seed), targets in sorted(by_episode.items()):
        source_root = _source_path(source_text)
        run = dict(read_json(source_root / "run_config.json"))
        if str(run.get("controller")) != "v2-full":
            raise ValueError("pre-action features require frozen v2-full sources")
        manifests = [
            dict(row)
            for row in read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
            if str(row.get("task_id")) == task_id
            and int(row.get("solver_seed", -1)) == solver_seed
            and str(row.get("status")) in {"ok", "resumed"}
        ]
        if len(manifests) != 1:
            raise ValueError("pre-action feature source episode is ambiguous")
        manifest = manifests[0]
        trace_path = contained_file(
            source_root, manifest.get("trace_file"), field="trace_file"
        )
        actual_sha = sha256_file(trace_path)
        if actual_sha != str(manifest.get("trace_sha256")):
            raise ValueError("pre-action feature source trace SHA256 mismatch")
        expected_sha = {str(row["source_trace_sha256"]) for row in targets.values()}
        if expected_sha != {actual_sha}:
            raise ValueError("pre-action registration/source trace mismatch")
        source_hashes[str(trace_path)] = actual_sha
        events = read_trace_events(trace_path)
        state = _initial_state(source_root, trace_path, events[0])
        history: list[dict[str, Any]] = []
        previous_edge_sets: list[set[tuple[int, int]]] = []
        seen = set()
        for event in events[1:-1]:
            decision_index = int(event["decision_index"])
            if decision_index in targets:
                selected_row = targets[decision_index]
                repair_fingerprint = str(selected_row["before_repair_fingerprint"])
                state_output, candidate_rows, backend = _extract_target(
                    selected=selected_row,
                    state=state,
                    event=event,
                    history=history,
                    previous_edge_sets=previous_edge_sets,
                    feature_backend=feature_backend,
                    state_label=state_labels[repair_fingerprint],
                    candidate_labels=candidate_labels,
                )
                state_outputs.append(state_output)
                candidate_outputs.extend(candidate_rows)
                backends[backend] += 1
                seen.add(decision_index)
            if str(event.get("schema")) == EPISODE_SCHEMA_V2:
                after = apply_state_delta(state, event["state_delta"])
                after.update(apply_extras_delta(state, event["state_extras_delta"]))
            else:
                after = dict(event["after"])
            history.append(_history_row(state, after, event))
            previous_edge_sets.append(_edge_set(state))
            previous_edge_sets = previous_edge_sets[-2:]
            state = after
        if seen != set(targets):
            raise ValueError("pre-action feature source missed registered decisions")

    if len(state_outputs) != len(selected) or len(candidate_outputs) != 8 * len(
        selected
    ):
        raise ValueError("pre-action feature output coverage mismatch")
    state_outputs.sort(key=lambda row: str(row["state_key"]))
    candidate_outputs.sort(
        key=lambda row: (str(row["state_key"]), int(row["candidate_rank"]))
    )
    atomic_write_csv(output_root / "stall_trigger_features.csv", state_outputs)
    atomic_write_csv(output_root / "stall_rescue_features.csv", candidate_outputs)
    report = {
        "schema": STALL_PREACTION_FEATURE_SCHEMA,
        "complete": True,
        "evidence_level": "pre-action trace-reconstructed compact features",
        "state_count": len(state_outputs),
        "candidate_count": len(candidate_outputs),
        "candidate_count_per_state": 8,
        "feature_count": len(FEATURE_NAMES),
        "canonical_feature_count": len(COMPACT_CANONICAL_FEATURE_NAMES),
        "derived_feature_count": len(DERIVED_FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "actual_feature_backends": dict(sorted(backends.items())),
        "source_trace_count": len(source_hashes),
        "source_trace_sha256": source_hashes,
        "future_observational_class_used_as_feature": False,
        "map_id_used_as_feature": False,
        "measured_wall_time_used_as_feature": False,
        "fingerprint_mismatch_count": 0,
        "candidate_mismatch_count": 0,
        "training_started": False,
        "controller_actions_changed": False,
        "deployment_promoted": False,
    }
    write_json(output_root / "stall_preaction_feature_report.json", report)
    return report


__all__ = [
    "COMPACT_CANONICAL_FEATURE_NAMES",
    "DERIVED_FEATURE_NAMES",
    "FEATURE_NAMES",
    "STALL_PREACTION_FEATURE_SCHEMA",
    "build_stall_preaction_features",
    "temporal_candidate_features",
]
