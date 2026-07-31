from __future__ import annotations

import collections
import concurrent.futures
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    producer_identity,
    sha256_file,
    strict_nonnegative_int as _strict_nonnegative_int,
)
from experiments.closed_loop_confirmation import (
    generate_online_candidates,
    score_online_candidates,
)
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.parallel_runtime import (
    candidate_lane_counts,
    initialize_isolated_worker,
    isolated_lane_cpu_sets,
    parallel_runtime_metadata,
    select_parallel_lane_count,
)
from experiments.repair_aware import classify_repair_outcome
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _low_level_delta,
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stall_guard import repair_structure_fingerprint
from experiments.trace_replay import (
    TRACE_REPLAY_CONTRACT,
    decision_rows,
    replay_prefix,
)
from experiments.v3_s3 import (
    S3_ACTION_TEMPLATES,
    S3_HORIZON,
    S3ActionTemplate,
    V3_S3_FULL_FEATURE_NAMES,
    V3_S3_PROFILE,
    all_runtime_sequences,
    balanced_sequence_templates,
    candidate_template_indices,
    s3_temporal_context,
    sequence_feature_row,
    sequence_id,
)


V3_S3_COLLECTION_SCHEMA = "lns2.v3_s3_collection.v3"
V3_S3_COLLECTION_VERSION = 3
V3_S3_CANDIDATE_GENERATION_SCHEMA = "lns2.v3_s3_candidate_generation.v1"
V3_S3_STOP_REASONS = frozenset(
    {
        "feasible",
        "deadline_or_iteration_truncation",
        "template_unavailable",
        "horizon_complete",
    }
)
V3_S3_COLLECTION_PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/closed_loop_confirmation.py",
    "experiments/closed_loop_trace_storage.py",
    "experiments/compact_controller_model.py",
    "experiments/context_audit.py",
    "experiments/feature_schema_v2.py",
    "experiments/feature_schema_v3.py",
    "experiments/neighborhood_candidates.py",
    "experiments/neighborhood_features.py",
    "experiments/online_feature_engine.py",
    "experiments/parallel_runtime.py",
    "experiments/repair_aware.py",
    "experiments/repair_aware_training.py",
    "experiments/repair_collection.py",
    "experiments/stall_guard.py",
    "experiments/trace_replay.py",
    "experiments/v3_s3.py",
    "experiments/v3_s3_collection.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/BasicLNS.h",
    "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/inc/SIPP.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)
S3_AGENT_COUNTS = (80, 100, 200, 400, 600)
S3_LAYOUTS = ("regular_beltway", "compartmentalized", "dead_end_aisles")
S3_SOURCE_POLICIES = ("fixed_random", "official_adaptive", "realized_dynamic")
S3_SPLIT_CELL_QUOTAS = {"policy_train": 20, "policy_validation": 5}
S3_SPLIT_TARGET_COUNTS = {
    split: quota * len(S3_LAYOUTS) * len(S3_AGENT_COUNTS)
    for split, quota in S3_SPLIT_CELL_QUOTAS.items()
}
S3_SPLIT_MINIMUM_COUNTS = {"policy_train": 100, "policy_validation": 30}
S3_TARGET_STATE_CAP = sum(S3_SPLIT_TARGET_COUNTS.values())
S3_STRATUM_QUOTAS = {
    "policy_train": {
        "ordinary_progress": 10,
        "high_cost_progress": 4,
        "no_progress": 3,
        "near_feasible": 3,
    },
    "policy_validation": {
        "ordinary_progress": 2,
        "high_cost_progress": 1,
        "no_progress": 1,
        "near_feasible": 1,
    },
}


_WORKER_CONTROLLER_BUNDLE: Any | None = None
_WORKER_SOURCE_CACHE: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}

try:
    import resource as _resource
except ImportError:  # pragma: no cover - Windows training does not expose it.
    _resource = None


def _stable_key(value: Any) -> str:
    return _fingerprint(value)


def _directory_content_fingerprint(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"fingerprinted directory is empty: {root}")
    return _fingerprint(
        [(path.relative_to(root).as_posix(), sha256_file(path)) for path in files]
    )


def _decision_stage(index: int) -> str:
    if int(index) < 4:
        return "early"
    if int(index) < 8:
        return "middle"
    return "late"


def _outcome_row(decision: dict[str, Any]) -> dict[str, Any]:
    actual = dict(decision.get("actual_lns2") or {})
    outcome = dict(actual.get("outcome") or {})
    metrics = dict(decision.get("actual_metrics") or actual.get("metrics") or {})
    before = int(decision.get("before_conflicts", outcome.get("conflicts_before", 0)))
    after = int(outcome.get("conflicts_after", metrics.get("conflicts_after", before)))
    changed = bool(
        decision.get(
            "repair_state_changed",
            str(decision.get("before_repair_fingerprint"))
            != str(decision.get("after_repair_fingerprint")),
        )
    )
    replan_success = bool(metrics.get("replan_success", True))
    no_progress = (not replan_success) or not changed
    action = dict(decision.get("actual_action") or actual.get("action") or {})
    actual_neighborhood = metrics.get("neighborhood", action.get("agents", ()))
    if not isinstance(actual_neighborhood, (list, tuple)):
        actual_neighborhood = ()
    return {
        "conflicts_before": before,
        "conflicts_after": after,
        "conflict_reduction": max(0, before - after),
        "repair_seconds": max(0.0, float(outcome.get("repair_seconds", 0.0))),
        "state_changed": changed,
        "no_progress": no_progress,
        "neighborhood_size": len(actual_neighborhood),
    }


def temporal_context(history: list[dict[str, Any]], agent_count: int) -> dict[str, float]:
    return s3_temporal_context(history, agent_count)


def source_decisions(
    source_roots: dict[str, list[Path]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split, roots in sorted(source_roots.items()):
        if split not in S3_SPLIT_CELL_QUOTAS:
            raise ValueError(f"unsupported v3-S3 split: {split}")
        for root in map(Path.resolve, roots):
            run = _read_json(root / "run_config.json")
            for policy in S3_SOURCE_POLICIES:
                manifest_path = root / f"{policy}_manifest.jsonl"
                if not manifest_path.is_file():
                    continue
                for manifest in _read_jsonl(manifest_path):
                    if str(manifest.get("status")) not in {"ok", "resumed"}:
                        continue
                    decisions, _events = decision_rows(root, manifest)
                    history: list[dict[str, Any]] = []
                    for decision in decisions:
                        fingerprint = str(decision["before_repair_fingerprint"])
                        dedup = f"{split}|{fingerprint}"
                        outcome = _outcome_row(decision)
                        if dedup not in seen and int(decision["before_conflicts"]) > 0:
                            seen.add(dedup)
                            agents = int(manifest["agent_count"])
                            result.append(
                                {
                                    **decision,
                                    "split": split,
                                    "source_root": str(root),
                                    "source_run_fingerprint": str(run["run_fingerprint"]),
                                    "source_policy": policy,
                                    "map_id": str(manifest["map_id"]),
                                    "layout_mode": str(manifest["layout_mode"]),
                                    "agent_count": agents,
                                    "task_id": str(manifest["task_id"]),
                                    "solver_seed": int(manifest["solver_seed"]),
                                    "episode_id": str(manifest["episode_id"]),
                                    "decision_stage": _decision_stage(
                                        int(decision["decision_index"])
                                    ),
                                    "source_repair_seconds": float(
                                        outcome["repair_seconds"]
                                    ),
                                    "source_no_progress": bool(outcome["no_progress"]),
                                    "temporal_context": temporal_context(
                                        history, agents
                                    ),
                                }
                            )
                        history.append(outcome)
    if not result:
        raise ValueError("v3-S3 source collections contain no repair decisions")
    return result


def assign_source_strata(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[
            (str(row["split"]), str(row["layout_mode"]), int(row["agent_count"]))
        ].append(row)
    for cell, values in grouped.items():
        no_progress = [row for row in values if bool(row["source_no_progress"])]
        progressing = [row for row in values if not bool(row["source_no_progress"])]
        for row in no_progress:
            row["source_stratum"] = "no_progress"
        progressing.sort(
            key=lambda row: (
                int(row["before_conflicts"]),
                _stable_key(
                    (row["map_id"], row["task_id"], row["decision_index"])
                ),
            )
        )
        near_count = min(
            len(progressing),
            max(1, math.ceil(0.20 * len(progressing))),
        )
        near_ids = {id(row) for row in progressing[:near_count]}
        remaining = [row for row in progressing if id(row) not in near_ids]
        remaining.sort(
            key=lambda row: (
                -float(row["source_repair_seconds"]),
                _stable_key(
                    (row["map_id"], row["task_id"], row["decision_index"])
                ),
            )
        )
        high_count = min(
            len(remaining),
            max(1, math.ceil(0.25 * len(remaining))),
        )
        high_ids = {id(row) for row in remaining[:high_count]}
        for row in progressing:
            row["source_stratum"] = (
                "near_feasible"
                if id(row) in near_ids
                else "high_cost_progress"
                if id(row) in high_ids
                else "ordinary_progress"
            )


def _balanced_take(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    remaining = list(rows)
    selected: list[dict[str, Any]] = []
    map_counts: collections.Counter[str] = collections.Counter()
    policy_counts: collections.Counter[str] = collections.Counter()
    stage_counts: collections.Counter[str] = collections.Counter()
    while remaining and len(selected) < int(count):
        chosen = min(
            remaining,
            key=lambda row: (
                map_counts[str(row["map_id"])],
                policy_counts[str(row["source_policy"])],
                stage_counts[str(row["decision_stage"])],
                _stable_key(
                    (
                        row["source_root"],
                        row["episode_id"],
                        row["decision_index"],
                    )
                ),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        map_counts[str(chosen["map_id"])] += 1
        policy_counts[str(chosen["source_policy"])] += 1
        stage_counts[str(chosen["decision_stage"])] += 1
    return selected


def _balanced_state_take(
    rows: list[dict[str, Any]], count: int, *, split: str
) -> list[dict[str, Any]]:
    """Select a deterministic globally balanced cohort from sparse source cells.

    Low-agent source episodes are often initially feasible, so exact quotas for
    every layout/load/stratum cell are not realistic.  This sampler balances the
    cells that actually exist and lets exhausted sparse cells contribute all of
    their rows before filling the remainder from denser cells.
    """

    if split not in S3_STRATUM_QUOTAS:
        raise ValueError(f"unsupported v3-S3 split: {split}")
    if count <= 0:
        return []
    remaining = list(rows)
    if count >= len(remaining):
        return sorted(
            remaining,
            key=lambda row: _stable_key(
                (row["source_root"], row["episode_id"], row["decision_index"])
            ),
        )

    layouts = sorted({str(row["layout_mode"]) for row in remaining})
    agents = sorted({int(row["agent_count"]) for row in remaining})
    strata = sorted({str(row["source_stratum"]) for row in remaining})
    cells = sorted(
        {
            (
                str(row["layout_mode"]),
                int(row["agent_count"]),
                str(row["source_stratum"]),
            )
            for row in remaining
        }
    )
    stratum_weights = {
        name: float(S3_STRATUM_QUOTAS[split][name])
        for name in strata
    }
    weight_total = sum(stratum_weights.values())
    stratum_weights = {
        name: value / weight_total for name, value in stratum_weights.items()
    }

    layout_counts: collections.Counter[str] = collections.Counter()
    agent_counts: collections.Counter[int] = collections.Counter()
    stratum_counts: collections.Counter[str] = collections.Counter()
    cell_counts: collections.Counter[tuple[str, int, str]] = collections.Counter()
    map_counts: collections.Counter[str] = collections.Counter()
    policy_counts: collections.Counter[str] = collections.Counter()
    stage_counts: collections.Counter[str] = collections.Counter()
    selected: list[dict[str, Any]] = []

    def pressure(value: int, target: float) -> float:
        return float(value) / max(target, 1e-12)

    while remaining and len(selected) < int(count):
        def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
            layout = str(row["layout_mode"])
            agent_count = int(row["agent_count"])
            stratum = str(row["source_stratum"])
            cell = (layout, agent_count, stratum)
            balances = (
                pressure(layout_counts[layout], float(count) / len(layouts)),
                pressure(agent_counts[agent_count], float(count) / len(agents)),
                pressure(
                    stratum_counts[stratum],
                    float(count) * stratum_weights[stratum],
                ),
                pressure(cell_counts[cell], float(count) / len(cells)),
            )
            return (
                max(balances),
                sum(balances),
                map_counts[str(row["map_id"])],
                policy_counts[str(row["source_policy"])],
                stage_counts[str(row["decision_stage"])],
                _stable_key(
                    (row["source_root"], row["episode_id"], row["decision_index"])
                ),
            )

        chosen = min(remaining, key=selection_key)
        remaining.remove(chosen)
        selected.append(chosen)
        layout = str(chosen["layout_mode"])
        agent_count = int(chosen["agent_count"])
        stratum = str(chosen["source_stratum"])
        layout_counts[layout] += 1
        agent_counts[agent_count] += 1
        stratum_counts[stratum] += 1
        cell_counts[(layout, agent_count, stratum)] += 1
        map_counts[str(chosen["map_id"])] += 1
        policy_counts[str(chosen["source_policy"])] += 1
        stage_counts[str(chosen["decision_stage"])] += 1
    return selected


def _adaptive_selection_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    errors = []
    required_strata = set(next(iter(S3_STRATUM_QUOTAS.values())))
    for split in S3_SPLIT_TARGET_COUNTS:
        split_rows = [row for row in rows if str(row["split"]) == split]
        layouts = sorted({str(row["layout_mode"]) for row in split_rows})
        agents = sorted({int(row["agent_count"]) for row in split_rows})
        strata = sorted({str(row["source_stratum"]) for row in split_rows})
        target_cap = int(S3_SPLIT_TARGET_COUNTS[split])
        target = min(target_cap, len(split_rows))
        split_errors = []
        if target < int(S3_SPLIT_MINIMUM_COUNTS[split]):
            split_errors.append(
                {
                    "reason": "insufficient-total-states",
                    "minimum": int(S3_SPLIT_MINIMUM_COUNTS[split]),
                    "available": len(split_rows),
                }
            )
        missing_layouts = sorted(set(S3_LAYOUTS) - set(layouts))
        if missing_layouts:
            split_errors.append(
                {"reason": "missing-layouts", "values": missing_layouts}
            )
        if len(agents) < 2:
            split_errors.append(
                {
                    "reason": "insufficient-agent-scales",
                    "minimum": 2,
                    "available": agents,
                }
            )
        missing_strata = sorted(required_strata - set(strata))
        if missing_strata:
            split_errors.append(
                {"reason": "missing-strata", "values": missing_strata}
            )
        errors.extend({"split": split, **error} for error in split_errors)
        available_cells = {
            (
                str(row["layout_mode"]),
                int(row["agent_count"]),
                str(row["source_stratum"]),
            )
            for row in split_rows
        }
        registered_cells = {
            (layout, agents_count, stratum)
            for layout in S3_LAYOUTS
            for agents_count in S3_AGENT_COUNTS
            for stratum in required_strata
        }
        by_split[split] = {
            "available_state_count": len(split_rows),
            "target_state_cap": target_cap,
            "target_state_count": target,
            "target_shortfall": target_cap - target,
            "minimum_state_count": int(S3_SPLIT_MINIMUM_COUNTS[split]),
            "available_layouts": layouts,
            "available_agent_counts": agents,
            "missing_agent_counts": sorted(set(S3_AGENT_COUNTS) - set(agents)),
            "available_strata": strata,
            "available_cell_count": len(available_cells),
            "missing_cell_count": len(registered_cells - available_cells),
            "errors": split_errors,
        }
    return {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "policy": "adaptive-global-backfill-v1",
        "target_state_cap": S3_TARGET_STATE_CAP,
        "target_state_count": sum(
            int(values["target_state_count"]) for values in by_split.values()
        ),
        "by_split": by_split,
        "errors": errors,
        "passed": not errors,
    }


def qualification_pool(
    rows: list[dict[str, Any]], *, reserve_multiplier: int = 2
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assign_source_strata(rows)
    if int(reserve_multiplier) <= 0:
        raise ValueError("v3-S3 reserve_multiplier must be positive")
    plan = _adaptive_selection_plan(rows)
    if not bool(plan["passed"]):
        raise ValueError(
            f"v3-S3 overall source coverage is insufficient: {plan['errors']}"
        )
    selected: list[dict[str, Any]] = []
    pool_by_split = {}
    for split, split_plan in dict(plan["by_split"]).items():
        available = [row for row in rows if str(row["split"]) == split]
        pool_count = min(
            len(available),
            int(split_plan["target_state_count"]) * int(reserve_multiplier),
        )
        values = _balanced_state_take(available, pool_count, split=split)
        selected.extend(values)
        pool_by_split[split] = len(values)
    for index, row in enumerate(
        sorted(
            selected,
            key=lambda value: (
                str(value["split"]),
                str(value["layout_mode"]),
                int(value["agent_count"]),
                str(value["source_stratum"]),
                _stable_key(value["before_repair_fingerprint"]),
            ),
        )
    ):
        row["state_id"] = (
            f"s3__{row['split']}__{row['episode_id']}__"
            f"decision_{int(row['decision_index']):04d}__{index:04d}"
        )
    return selected, {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "source_decision_count": len(rows),
        "qualification_pool_count": len(selected),
        "reserve_multiplier": int(reserve_multiplier),
        "pool_by_split": pool_by_split,
        "selection_plan": plan,
        "shortages": [],
        "passed": True,
    }


def _worker_initialize(
    cpu_sets: tuple[tuple[int, ...], ...], controller_bundle: str
) -> None:
    initialize_isolated_worker(cpu_sets)
    global _WORKER_CONTROLLER_BUNDLE
    _WORKER_CONTROLLER_BUNDLE = load_controller_bundle(Path(controller_bundle))
    _WORKER_SOURCE_CACHE.clear()


def _source_replay_job(decision: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = str(Path(decision["source_root"]).resolve())
    cached = _WORKER_SOURCE_CACHE.get(root)
    if cached is None:
        run = _read_json(Path(root) / "run_config.json")
        dataset_root = Path(str(run["dataset"])).resolve()
        lookup = {
            str(row["task_id"]): row
            for row in _load_dataset_rows(dataset_root, [str(decision["split"])])
        }
        cached = (run, lookup)
        _WORKER_SOURCE_CACHE[root] = cached
    run, lookup = cached
    configuration = dict(run["configuration"])
    environment = dict(configuration["environment"])
    environment["max_repair_iterations"] = max(
        int(environment.get("max_repair_iterations", 0)),
        len(decision["prefix_actions"]) + S3_HORIZON,
    )
    replay = {
        "dataset_root": str(Path(str(run["dataset"])).resolve()),
        "row": lookup[str(decision["task_id"])],
        "environment": environment,
        "solver_seed": int(decision["solver_seed"]),
        # Prefix actions are recorded neighborhoods, so replay is independent
        # of the source policy.  Adaptive here gives every counterfactual
        # baseline the same fresh official LNS2 state after the prefix.
        "replay_destroy_strategy": "Adaptive",
    }
    return replay, configuration


def _full_candidate_rows(
    environment: Any,
    state: dict[str, Any],
    decision: dict[str, Any],
    proposal: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    state_hash = state_fingerprint(state)
    full_proposal = dict(proposal)
    full_proposal["heuristics"] = ["target", "collision", "random"]
    full_proposal["neighborhood_sizes"] = [4, 8, 16]
    full_proposal["candidates_per_family"] = 2
    candidates, generation = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=full_proposal,
        state_hash=state_hash,
        verify_full_state=True,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features={
            "realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]
        },
        dense_output=False,
    )
    rows, feature_metrics = engine.realized_rows(candidates, state_hash=state_hash)
    elapsed = time.perf_counter() - started
    return candidates, rows, {
        "full_pool_seconds": elapsed,
        "generation": generation,
        "features": feature_metrics,
    }


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _finite_nonnegative(value: Any) -> bool:
    return _finite_number(value) and float(value) >= 0.0


def _stored_fingerprint(value: Any) -> str | None:
    return value if isinstance(value, str) and bool(value) else None


def _stored_agent_ids(
    value: Any,
    *,
    agent_count: int,
    allow_empty: bool = False,
) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if any(
        isinstance(agent, bool)
        or not isinstance(agent, int)
        or agent < 0
        or agent >= int(agent_count)
        for agent in value
    ):
        return None
    if len(set(value)) != len(value) or (not allow_empty and not value):
        return None
    return list(value)


def _candidate_semantics(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError("candidate is not an object")
    candidate_id = candidate.get("candidate_id")
    agents = candidate.get("agents")
    families = candidate.get("selection_families")
    ranks = candidate.get("selection_rank_by_family")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id is empty")
    if not isinstance(agents, list) or any(
        isinstance(agent, bool) or not isinstance(agent, int) or agent < 0
        for agent in agents
    ):
        raise ValueError("candidate agents are invalid")
    if len(set(agents)) != len(agents) or not agents:
        raise ValueError("candidate agents are empty or duplicated")
    if not isinstance(families, list) or any(
        not isinstance(name, str) or not name for name in families
    ):
        raise ValueError("candidate selection families are invalid")
    if not isinstance(ranks, dict):
        raise ValueError("candidate selection ranks are invalid")
    normalized_ranks: dict[str, int] = {}
    for name, rank in ranks.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
        ):
            raise ValueError("candidate selection rank is invalid")
        normalized_ranks[name] = rank
    actual_size = candidate.get("actual_size", len(agents))
    if not _strict_nonnegative_int(actual_size) or actual_size != len(agents):
        raise ValueError("candidate actual_size differs from agents")
    return {
        "candidate_id": candidate_id,
        "agents": list(agents),
        "actual_size": actual_size,
        "selection_families": list(families),
        "selection_rank_by_family": {
            name: normalized_ranks[name] for name in sorted(normalized_ranks)
        },
    }


def _candidate_generation_record(
    candidates: list[dict[str, Any]],
    template: S3ActionTemplate,
    *,
    before_fingerprint: str,
    decision_index: int,
    source: str,
    include_pool: bool,
) -> dict[str, Any]:
    """Bind each chosen continuation to the exact pool that produced it.

    The first step is already cross-bound to the qualification artifact, so its
    pool is referenced by fingerprint only.  Dynamic continuation pools are
    small and are stored in full, allowing resume validation to reconstruct the
    template mapping instead of trusting the selected candidate fields.
    """

    if source not in {"qualification_pool", "restricted_dynamic"}:
        raise ValueError("unsupported candidate-generation source")
    if not isinstance(before_fingerprint, str) or not before_fingerprint:
        raise ValueError("candidate-generation fingerprint is empty")
    if not _strict_nonnegative_int(decision_index):
        raise ValueError("candidate-generation decision index is invalid")
    expected_proposal_fingerprint = (
        _fingerprint(_restricted_proposal(template))
        if source == "restricted_dynamic"
        else ""
    )
    normalized = [_candidate_semantics(candidate) for candidate in candidates]
    indices = candidate_template_indices(normalized)
    selected_index = indices.get(template.key, -1)
    selected = normalized[selected_index] if selected_index >= 0 else None
    payload = {
        "schema": V3_S3_CANDIDATE_GENERATION_SCHEMA,
        "source": source,
        "before_fingerprint": before_fingerprint,
        "decision_index": decision_index,
        "template": template.payload(),
        "candidate_count": len(normalized),
        "candidate_pool_fingerprint": _fingerprint(normalized),
        "candidate_pool": normalized if include_pool else None,
        "selected_index": selected_index,
        "selected_candidate_fingerprint": (
            _fingerprint(selected) if selected is not None else ""
        ),
        "proposal_fingerprint": expected_proposal_fingerprint,
    }
    return {**payload, "fingerprint": _fingerprint(payload)}


def _restricted_proposal(template: S3ActionTemplate) -> dict[str, Any]:
    return {
        "heuristics": [template.family],
        "neighborhood_sizes": [int(template.requested_size)],
        "candidates_per_family": 2,
    }


def _artifact_header_errors(
    payload: Any,
    *,
    run_fingerprint: str | None,
    require_complete: bool = True,
) -> list[str]:
    if not isinstance(payload, dict):
        return ["artifact payload is not a JSON object"]
    errors = []
    if str(payload.get("schema")) != V3_S3_COLLECTION_SCHEMA:
        errors.append("artifact schema is not v3-S3 collection v3")
    schema_version = payload.get("schema_version")
    if (
        not _strict_nonnegative_int(schema_version)
        or schema_version != V3_S3_COLLECTION_VERSION
    ):
        errors.append("artifact schema_version is not 3")
    if run_fingerprint is not None and str(payload.get("run_fingerprint")) != str(
        run_fingerprint
    ):
        errors.append("artifact run_fingerprint differs")
    if require_complete and payload.get("complete") is not True:
        errors.append("artifact is not marked complete")
    return errors


def _qualification_artifact_errors(
    payload: Any,
    *,
    run_fingerprint: str | None = None,
    expected_decision: dict[str, Any] | None = None,
) -> list[str]:
    errors = _artifact_header_errors(
        payload, run_fingerprint=run_fingerprint, require_complete=True
    )
    if not isinstance(payload, dict):
        return errors
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        return [*errors, "qualification decision is not an object"]
    state_id = str(decision.get("state_id", ""))
    before_fingerprint = str(decision.get("before_fingerprint", ""))
    if not state_id:
        errors.append("qualification decision has no state_id")
    if not before_fingerprint:
        errors.append("qualification decision has no before_fingerprint")
    if not str(decision.get("before_repair_fingerprint", "")):
        errors.append("qualification decision has no before_repair_fingerprint")
    if expected_decision is not None and _fingerprint(decision) != _fingerprint(
        expected_decision
    ):
        errors.append("qualification decision differs from the scheduled decision")

    candidates = payload.get("candidates")
    rows = payload.get("candidate_rows")
    raw_indices = payload.get("template_indices")
    timing = payload.get("timing")
    if not isinstance(candidates, list) or not candidates:
        errors.append("qualification candidates are missing")
        candidates = []
    if not isinstance(rows, list) or len(rows) != len(candidates):
        errors.append("qualification candidate row coverage differs")
        rows = []
    if not isinstance(raw_indices, dict):
        errors.append("qualification template_indices is not an object")
        raw_indices = {}
    try:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_indices.values()
        ):
            raise ValueError("template index is not an integer")
        indices = {str(key): int(value) for key, value in raw_indices.items()}
    except (TypeError, ValueError):
        errors.append("qualification template_indices contains a non-integer index")
        indices = {}
    try:
        computed_indices = candidate_template_indices(candidates)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"qualification candidate/template mapping is invalid: {error}")
        computed_indices = {}
    expected_template_keys = {template.key for template in S3_ACTION_TEMPLATES}
    if set(indices) != expected_template_keys:
        errors.append("qualification does not cover every registered action template")
    if indices != computed_indices:
        errors.append("qualification template_indices differs from candidate mapping")

    required_features = set(PROFILE_FEATURE_NAMES["realized_dynamic"])
    for index, (candidate, row) in enumerate(zip(candidates, rows)):
        if not isinstance(candidate, dict) or not isinstance(row, dict):
            errors.append(f"qualification candidate {index} is not an object pair")
            continue
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or str(row.get("candidate_id", "")) != candidate_id:
            errors.append(f"qualification candidate {index} identity differs")
        agents = candidate.get("agents")
        if (
            not isinstance(agents, list)
            or not agents
            or any(isinstance(agent, bool) or not isinstance(agent, int) for agent in agents)
            or len(agents) != len(set(agents))
        ):
            errors.append(f"qualification candidate {index} agents are invalid")
        if str(row.get("candidate_key", "")) != candidate_id:
            errors.append(f"qualification candidate {index} key differs")
        if str(row.get("state_id", "")) != before_fingerprint:
            errors.append(f"qualification candidate {index} state fingerprint differs")
        features = row.get("features")
        profile = (
            features.get("realized_dynamic")
            if isinstance(features, dict)
            else None
        )
        if not isinstance(profile, dict) or set(map(str, profile)) != required_features:
            errors.append(f"qualification candidate {index} feature coverage differs")
        elif any(not _finite_number(value) for value in profile.values()):
            errors.append(f"qualification candidate {index} has a non-finite feature")
    if not isinstance(timing, dict) or not _finite_nonnegative(
        timing.get("full_pool_seconds") if isinstance(timing, dict) else None
    ):
        errors.append("qualification full-pool timing is invalid")
    return errors


def _resume_completed_artifact(
    path: Path,
    *,
    run_fingerprint: str,
    validator: Any,
    label: str,
    **validator_kwargs: Any,
) -> dict[str, Any] | None:
    """Return a verified completed artifact, or allow an incomplete rewrite.

    Existing malformed, legacy, foreign-run, or semantically invalid artifacts
    are deliberately left untouched and fail closed.  Only a current-schema
    object explicitly marked incomplete may be atomically replaced.
    """

    if not path.is_file():
        return None
    try:
        existing = _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{label} artifact is unreadable and was preserved: {error}"
        ) from error
    header_errors = _artifact_header_errors(
        existing,
        run_fingerprint=run_fingerprint,
        require_complete=False,
    )
    if header_errors:
        raise ValueError(
            f"{label} artifact is incompatible and was preserved: "
            + "; ".join(header_errors)
        )
    if existing.get("complete") is not True:
        return None
    errors = validator(
        existing,
        run_fingerprint=run_fingerprint,
        **validator_kwargs,
    )
    if errors:
        raise ValueError(
            f"{label} completed artifact is invalid and was preserved: "
            + "; ".join(errors[:5])
        )
    return existing


def _qualification_job(job: dict[str, Any]) -> dict[str, Any]:
    decision = dict(job["decision"])
    output = Path(job["state_file"])
    if bool(job["resume"]) and output.is_file():
        existing = _resume_completed_artifact(
            output,
            run_fingerprint=str(job["run_fingerprint"]),
            validator=_qualification_artifact_errors,
            label="v3-S3 qualification",
            expected_decision=decision,
        )
        if existing is not None:
            return {
                "state_id": str(decision["state_id"]),
                "state_file": str(output),
                "status": "resumed",
            }
    replay, configuration = _source_replay_job(decision)
    try:
        environment, state = replay_prefix(replay, decision["prefix_actions"])
    except RuntimeError as error:
        if str(error) != "prefix terminated before target state":
            raise
        return {
            "state_id": str(decision["state_id"]),
            "status": "rejected",
            "rejection_reason": str(error),
        }
    if state_fingerprint(state) != str(decision["before_fingerprint"]):
        return {
            "state_id": str(decision["state_id"]),
            "status": "rejected",
            "rejection_reason": "v3-S3 qualification replay fingerprint mismatch",
        }
    candidates, rows, timing = _full_candidate_rows(
        environment, state, decision, dict(configuration["proposal"])
    )
    templates = candidate_template_indices(candidates)
    missing = sorted(
        {template.key for template in S3_ACTION_TEMPLATES} - set(templates)
    )
    if missing:
        return {
            "state_id": str(decision["state_id"]),
            "status": "rejected",
            "rejection_reason": f"v3-S3 state lacks action templates: {missing}",
        }
    payload = {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "run_fingerprint": str(job["run_fingerprint"]),
        "complete": True,
        "decision": decision,
        "candidates": candidates,
        "candidate_rows": rows,
        "template_indices": templates,
        "timing": timing,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output)
    return {
        "state_id": str(decision["state_id"]),
        "state_file": str(output),
        "status": "ok",
    }


def _paired_seed(initial_fingerprint: str, trial_index: int, step: int) -> int:
    return int(
        _fingerprint(
            {
                "namespace": "v3-s3-paired-repair-v1",
                "state": str(initial_fingerprint),
                "trial": int(trial_index),
                "step": int(step),
            }
        )[:16],
        16,
    ) % (2**31)


def _paired_repair_action(
    mode: str, *, random_seed: int, agents: Iterable[int] = ()
) -> dict[str, Any]:
    """Build an action whose PP tie breaking is paired across controllers.

    ``random_seed`` is still applied at the beginning of the native step for
    neighborhood ordering/generation.  The independent PP seed is reapplied
    after that work so explicit, v2, and Adaptive branches start low-level
    search from the same RNG state.
    """

    seed = int(random_seed)
    if seed < 0:
        raise ValueError("paired repair seed must be non-negative")
    if mode == "official":
        return {
            "mode": "official",
            "random_seed": seed,
            "pp_random_seed": seed,
        }
    if mode != "explicit_neighborhood":
        raise ValueError(f"unsupported paired repair action mode: {mode}")
    neighborhood = list(map(int, agents))
    if not neighborhood:
        raise ValueError("paired explicit repair requires a non-empty neighborhood")
    return {
        "mode": "explicit_neighborhood",
        "agents": neighborhood,
        "random_seed": seed,
        "pp_random_seed": seed,
    }


def _parallel_probe_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one deterministic repair for the isolated-lane audit."""

    decision = dict(job["decision"])
    replay, configuration = _source_replay_job(decision)
    try:
        environment, state = replay_prefix(replay, decision["prefix_actions"])
    except RuntimeError as error:
        if str(error) != "prefix terminated before target state":
            raise
        return {
            "audit_key": str(job["audit_key"]),
            "state_id": str(job["state_id"]),
            "status": "rejected",
            "rejection_reason": str(error),
        }
    before = state_fingerprint(state)
    if before != str(decision["before_fingerprint"]):
        return {
            "audit_key": str(job["audit_key"]),
            "state_id": str(job["state_id"]),
            "status": "rejected",
            "rejection_reason": "v3-S3 parallel audit replay fingerprint mismatch",
        }
    template = S3ActionTemplate.from_payload(dict(job["template"]))
    candidate, selection_seconds = _restricted_candidate(
        environment,
        state,
        decision,
        dict(configuration["proposal"]),
        template,
        decision_index=int(decision["decision_index"]),
    )
    if candidate is None:
        return {
            "audit_key": str(job["audit_key"]),
            "state_id": str(job["state_id"]),
            "status": "rejected",
            "rejection_reason": f"parallel audit template is unavailable: {template.key}",
        }
    action = _paired_repair_action(
        "explicit_neighborhood",
        agents=candidate["agents"],
        random_seed=_paired_seed(
            repair_structure_fingerprint(state),
            int(job["trial_index"]),
            1,
        ),
    )
    started = time.perf_counter()
    transition = _plain(environment.step(action))
    repair_seconds = time.perf_counter() - started
    metrics = dict(transition["metrics"])
    after_state = dict(transition["observation"])
    peak = int(getattr(_resource, "getrusage", lambda *_: None)(_resource.RUSAGE_SELF).ru_maxrss * 1024) if _resource is not None else 0
    return {
        "audit_key": str(job["audit_key"]),
        "state_id": str(job["state_id"]),
        "status": "ok",
        "candidate_id": str(candidate["candidate_id"]),
        "agents": list(map(int, candidate["agents"])),
        "random_seed": int(action["random_seed"]),
        "pp_random_seed": int(action["pp_random_seed"]),
        "applied_pp_random_seed": int(
            metrics.get("applied_pp_random_seed", -1)
        ),
        "before_fingerprint": before,
        "after_fingerprint": state_fingerprint(after_state),
        "selection_seconds": selection_seconds,
        "repair_seconds": repair_seconds,
        "pp_replan_seconds": max(
            0.0, float(metrics.get("pp_replan_seconds", repair_seconds))
        ),
        "peak_memory_bytes": peak,
    }


def _audit_state_sample(
    rows: list[dict[str, Any]], *, count: int
) -> list[dict[str, Any]]:
    """Select a deterministic round-robin layout/agent audit sample."""

    cells: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        agents = int(row["agent_count"])
        if agents in {100, 400, 600}:
            cells[(str(row["layout_mode"]), agents)].append(row)
    ordered = {
        cell: _balanced_take(values, len(values))
        for cell, values in sorted(cells.items())
        if values
    }
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < int(count):
        added = False
        for cell in sorted(ordered):
            values = ordered[cell]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) >= int(count):
                    break
        if not added:
            break
        offset += 1
    return selected


def _rank_values(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        for index in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def _rank_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("rank correlation requires paired non-empty samples")
    left_rank = _rank_values(left)
    right_rank = _rank_values(right)
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    numerator = math.fsum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_rank, right_rank)
    )
    left_scale = math.sqrt(
        math.fsum((value - left_mean) ** 2 for value in left_rank)
    )
    right_scale = math.sqrt(
        math.fsum((value - right_mean) ** 2 for value in right_rank)
    )
    if left_scale <= 1e-15 or right_scale <= 1e-15:
        return 1.0 if left_rank == right_rank else 0.0
    return numerator / (left_scale * right_scale)


def audit_v3_s3_parallelism(
    *,
    source_roots: dict[str, list[Path]],
    output: str | Path,
    controller_bundle: str | Path,
    maximum_lanes: int | None = None,
) -> dict[str, Any]:
    """Select the highest isolated lane count that preserves paired PP costs."""

    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = source_decisions(source_roots)
    assign_source_strata(rows)
    controller = Path(controller_bundle).resolve()
    lane_values = candidate_lane_counts(maximum_lanes)
    audit_producer_identity = producer_identity(
        project_root=Path(__file__).resolve().parents[1],
        source_files=V3_S3_COLLECTION_PRODUCER_FILES,
        native_required=True,
        package_names=("numpy", "scikit-learn"),
        optional_package_names=("joblib",),
    )
    audit_identity = {
        "schema": "lns2.v3_s3_parallelism_audit_run.v2",
        "schema_version": V3_S3_COLLECTION_VERSION,
        "trace_replay_contract": TRACE_REPLAY_CONTRACT,
        "source_run_fingerprints": sorted(
            {str(row["source_run_fingerprint"]) for row in rows}
        ),
        "source_decision_fingerprint": _fingerprint(
            sorted(
                (
                    str(row["source_root"]),
                    str(row["episode_id"]),
                    int(row["decision_index"]),
                    str(row["before_fingerprint"]),
                )
                for row in rows
            )
        ),
        "controller_bundle": str(controller),
        "controller_bundle_fingerprint": _directory_content_fingerprint(controller),
        "producer_identity": audit_producer_identity,
        "producer_identity_fingerprint": _fingerprint(
            audit_producer_identity
        ),
        "lane_values": list(lane_values),
    }
    audit_run_fingerprint = _fingerprint(audit_identity)
    audit_run_path = output_root / "run_config.json"
    if audit_run_path.is_file():
        existing = _read_json(audit_run_path)
        if str(existing.get("run_fingerprint")) != audit_run_fingerprint:
            raise ValueError("v3-S3 parallel audit output belongs to another run")
    _write_json(
        audit_run_path,
        {**audit_identity, "run_fingerprint": audit_run_fingerprint},
    )
    report_path = output_root / "parallelism_audit_report.json"
    if report_path.is_file():
        previous = _read_json(report_path)
        if str(previous.get("run_fingerprint")) == audit_run_fingerprint:
            return previous
    reserve = _audit_state_sample(rows, count=18)
    if len(reserve) < 10:
        raise ValueError("v3-S3 parallel audit has too few stratified source states")
    templates = [
        S3ActionTemplate("collision", size, 0) for size in (4, 8, 16)
    ]
    reserve_jobs = []
    row_by_state = {}
    for row in reserve:
        audit_state_id = _fingerprint(
            (row["source_root"], row["episode_id"], row["decision_index"])
        )[:20]
        row_by_state[audit_state_id] = row
        for template in templates:
            key = _fingerprint((row["before_fingerprint"], template.key))[:20]
            reserve_jobs.append(
                {
                    "state_id": audit_state_id,
                    "audit_key": key,
                    "decision": row,
                    "template": template.payload(),
                    "trial_index": int(
                        _fingerprint(
                            ("parallel-audit", row["before_repair_fingerprint"])
                        )[:8],
                        16,
                    )
                    % 100000,
                }
            )
    strict_results, strict_errors = _run_jobs(
        reserve_jobs,
        workers=1,
        controller_bundle=controller,
        status_path=output_root / "status.json",
        phase="parallel-audit-1-preflight",
        job_function=_parallel_probe_job,
    )
    if strict_errors:
        raise RuntimeError(f"strict parallel audit failed: {strict_errors}")
    strict_ok = [row for row in strict_results if str(row.get("status")) == "ok"]
    ok_counts = collections.Counter(str(row["state_id"]) for row in strict_ok)
    successful_rows = [
        row_by_state[state_id]
        for state_id, count in ok_counts.items()
        if count == len(templates)
    ]
    selected = _audit_state_sample(successful_rows, count=10)
    if len(selected) != 10:
        rejected = [
            row for row in strict_results if str(row.get("status")) == "rejected"
        ]
        raise RuntimeError(
            "strict parallel audit has fewer than 10 replayable stratified states; "
            f"replayable={len(successful_rows)}, rejected_probes={len(rejected)}, "
            f"sample={rejected[:5]}"
        )
    selected_state_ids = {
        _fingerprint((row["source_root"], row["episode_id"], row["decision_index"]))[:20]
        for row in selected
    }
    base_jobs = [
        job for job in reserve_jobs if str(job["state_id"]) in selected_state_ids
    ]
    selected_keys = {str(job["audit_key"]) for job in base_jobs}
    strict_by_key = {
        str(row["audit_key"]): row
        for row in strict_ok
        if str(row["audit_key"]) in selected_keys
    }
    if len(strict_by_key) != len(base_jobs):
        raise RuntimeError("strict parallel audit replay coverage is incomplete")
    measurements = []
    for lanes in lane_values:
        if lanes == 1:
            results = list(strict_by_key.values())
            errors: list[dict[str, Any]] = []
        else:
            results, errors = _run_jobs(
                base_jobs,
                workers=int(lanes),
                controller_bundle=controller,
                status_path=output_root / "status.json",
                phase=f"parallel-audit-{lanes}",
                job_function=_parallel_probe_job,
            )
        current = {
            str(row["audit_key"]): row
            for row in results
            if str(row.get("status")) == "ok"
        }
        keys = sorted(strict_by_key)
        complete = not errors and set(current) == set(strict_by_key)
        strict_pp = [float(strict_by_key[key]["pp_replan_seconds"]) for key in keys]
        parallel_pp = [
            float(current[key]["pp_replan_seconds"])
            if key in current
            else max(1.0, 100.0 * float(strict_by_key[key]["pp_replan_seconds"]))
            for key in keys
        ]
        strict_controller = [
            float(strict_by_key[key]["selection_seconds"]) for key in keys
        ]
        parallel_controller = [
            float(current[key]["selection_seconds"])
            if key in current
            else max(1.0, 100.0 * float(strict_by_key[key]["selection_seconds"]))
            for key in keys
        ]
        semantic_match = complete and all(
            (
                current[key]["candidate_id"],
                current[key]["agents"],
                current[key]["random_seed"],
                current[key]["pp_random_seed"],
                current[key]["applied_pp_random_seed"],
                current[key]["before_fingerprint"],
                current[key]["after_fingerprint"],
            )
            == (
                strict_by_key[key]["candidate_id"],
                strict_by_key[key]["agents"],
                strict_by_key[key]["random_seed"],
                strict_by_key[key]["pp_random_seed"],
                strict_by_key[key]["applied_pp_random_seed"],
                strict_by_key[key]["before_fingerprint"],
                strict_by_key[key]["after_fingerprint"],
            )
            for key in keys
        )
        correlation = _rank_correlation(strict_pp, parallel_pp) if semantic_match else 0.0
        measurements.append(
            {
                "lanes": int(lanes),
                "strict_pp_seconds": strict_pp,
                "parallel_pp_seconds": parallel_pp,
                "strict_controller_seconds": strict_controller,
                "parallel_controller_seconds": parallel_controller,
                "cost_rank_correlation": correlation,
                "peak_memory_bytes": int(lanes) * max(
                    (int(row.get("peak_memory_bytes", 0)) for row in results),
                    default=0,
                ),
                "semantic_match": semantic_match,
                "error_count": len(errors),
                "errors": errors,
            }
        )
    report = select_parallel_lane_count(measurements)
    report.update(
        {
            "run_fingerprint": audit_run_fingerprint,
            "source_state_count": len(selected),
            "preflight_source_state_count": len(reserve),
            "preflight_rejected_probe_count": sum(
                str(row.get("status")) == "rejected" for row in strict_results
            ),
            "repair_probe_count_per_lane": len(base_jobs),
            "selected_by_layout": dict(
                collections.Counter(str(row["layout_mode"]) for row in selected)
            ),
            "selected_by_agent_count": dict(
                collections.Counter(int(row["agent_count"]) for row in selected)
            ),
            "selected_by_source_policy": dict(
                collections.Counter(str(row["source_policy"]) for row in selected)
            ),
            "semantic_checks_passed": all(
                bool(row["semantic_match"]) for row in measurements
            ),
        }
    )
    _write_json(report_path, report)
    return report


def _restricted_candidate(
    environment: Any,
    state: dict[str, Any],
    decision: dict[str, Any],
    proposal: dict[str, Any],
    template: S3ActionTemplate,
    *,
    decision_index: int,
) -> tuple[dict[str, Any] | None, float, dict[str, Any]]:
    restricted = {**proposal, **_restricted_proposal(template)}
    started = time.perf_counter()
    candidates, _metrics = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision_index),
        proposal_config=restricted,
        state_hash=state_fingerprint(state),
        verify_full_state=True,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    elapsed = time.perf_counter() - started
    generation = _candidate_generation_record(
        candidates,
        template,
        before_fingerprint=state_fingerprint(state),
        decision_index=decision_index,
        source="restricted_dynamic",
        include_pool=True,
    )
    index = int(generation["selected_index"])
    return (candidates[index] if index >= 0 else None), elapsed, generation


def _native_repair_evidence(
    metrics: dict[str, Any],
    *,
    expected_seed: int,
    agent_count: int,
    expected_agents: list[int] | None = None,
) -> tuple[list[int], list[int], int]:
    if metrics.get("step_applied") is not True:
        raise RuntimeError("v3-S3 native step ended before applying a repair")
    if not isinstance(metrics.get("replan_success"), bool):
        raise RuntimeError("v3-S3 native replan-success evidence is not boolean")
    requested_seed = metrics.get("requested_pp_random_seed")
    if (
        not _strict_nonnegative_int(requested_seed)
        or requested_seed != expected_seed
    ):
        raise RuntimeError("v3-S3 requested PP seed mismatch")
    neighborhood = _stored_agent_ids(
        metrics.get("neighborhood"), agent_count=agent_count
    )
    repair_order = _stored_agent_ids(
        metrics.get("repair_order"),
        agent_count=agent_count,
        allow_empty=True,
    )
    if neighborhood is None or repair_order is None:
        raise RuntimeError("v3-S3 native neighborhood evidence is invalid")
    if expected_agents is not None and (
        len(neighborhood) != len(expected_agents)
        or set(neighborhood) != set(expected_agents)
    ):
        raise RuntimeError("v3-S3 native neighborhood differs from action")
    if repair_order and set(repair_order) != set(neighborhood):
        raise RuntimeError("v3-S3 repair order differs from neighborhood")
    applied_seed = metrics.get("applied_pp_random_seed")
    if isinstance(applied_seed, bool) or not isinstance(applied_seed, int):
        raise RuntimeError("v3-S3 applied PP seed is invalid")
    expected_applied_seed = expected_seed if repair_order else -1
    if applied_seed != expected_applied_seed:
        raise RuntimeError("v3-S3 applied PP seed mismatch")
    return neighborhood, repair_order, applied_seed


def _terminal_flags(
    result: dict[str, Any], state: dict[str, Any]
) -> tuple[bool, bool, bool]:
    if "terminated" not in result or "truncated" not in result:
        raise RuntimeError("v3-S3 native step omitted terminal flags")
    if not isinstance(result["terminated"], bool) or not isinstance(
        result["truncated"], bool
    ):
        raise RuntimeError("v3-S3 native terminal flags are not boolean")
    if not isinstance(state.get("done"), bool) or not isinstance(
        state.get("feasible"), bool
    ):
        raise RuntimeError("v3-S3 native observation terminal flags are not boolean")
    terminated = result["terminated"]
    truncated = result["truncated"]
    done = state["done"]
    feasible = state["feasible"]
    if terminated != feasible or truncated != (done and not feasible):
        raise RuntimeError("v3-S3 native terminal flags disagree with observation")
    if done != (terminated or truncated):
        raise RuntimeError("v3-S3 native done flag is inconsistent")
    return terminated, truncated, done


def _state_stop_reason(state: dict[str, Any]) -> tuple[str, bool] | None:
    if not bool(state.get("done")):
        return None
    if bool(state.get("feasible")):
        return "feasible", False
    return "deadline_or_iteration_truncation", True


def _sequence_trial(
    qualified: dict[str, Any],
    templates: tuple[S3ActionTemplate, ...],
    trial_index: int,
) -> dict[str, Any]:
    decision = dict(qualified["decision"])
    replay, configuration = _source_replay_job(decision)
    environment, state = replay_prefix(replay, decision["prefix_actions"])
    if state_fingerprint(state) != str(decision["before_fingerprint"]):
        raise RuntimeError("v3-S3 trial replay fingerprint mismatch")
    initial_full = state_fingerprint(state)
    initial_repair = repair_structure_fingerprint(state)
    initial_conflicts = int(state["num_of_colliding_pairs"])
    trajectory = [initial_conflicts]
    total_seconds = 0.0
    total_pp_seconds = 0.0
    steps = []
    low_level_total = collections.Counter()
    initial_candidates = list(qualified["candidates"])
    initial_indices = {
        str(name): int(value)
        for name, value in dict(qualified["template_indices"]).items()
    }
    terminal = _state_stop_reason(state)
    stop_reason = terminal[0] if terminal is not None else "horizon_complete"
    truncated = terminal[1] if terminal is not None else False
    for offset, template in (() if terminal is not None else enumerate(templates)):
        before = state
        before_full = state_fingerprint(before)
        before_repair = repair_structure_fingerprint(before)
        conflicts_before = int(before["num_of_colliding_pairs"])
        if offset == 0:
            candidate = initial_candidates[initial_indices[template.key]]
            selection_seconds = float(qualified["timing"]["full_pool_seconds"])
            generation = _candidate_generation_record(
                initial_candidates,
                template,
                before_fingerprint=before_full,
                decision_index=int(decision["decision_index"]),
                source="qualification_pool",
                include_pool=False,
            )
        else:
            candidate, selection_seconds, generation = _restricted_candidate(
                environment,
                before,
                decision,
                dict(configuration["proposal"]),
                template,
                decision_index=int(decision["decision_index"]) + offset,
            )
        if candidate is None:
            steps.append(
                {
                    "step": offset + 1,
                    "template": template.payload(),
                    "template_valid": False,
                    "executed": False,
                    "selection_seconds": selection_seconds,
                    "candidate_generation": generation,
                }
            )
            total_seconds += selection_seconds
            stop_reason = "template_unavailable"
            truncated = False
            break
        seed = _paired_seed(initial_repair, trial_index, offset + 1)
        action = _paired_repair_action(
            "explicit_neighborhood",
            agents=candidate["agents"],
            random_seed=seed,
        )
        repair_started = time.perf_counter()
        result = _plain(environment.step(action))
        repair_seconds = time.perf_counter() - repair_started
        state = dict(result["observation"])
        metrics = dict(result["metrics"])
        _neighborhood, repair_order, applied_seed = _native_repair_evidence(
            metrics,
            expected_seed=seed,
            agent_count=int(decision["agent_count"]),
            expected_agents=list(map(int, candidate["agents"])),
        )
        terminated, step_truncated, done = _terminal_flags(result, state)
        conflicts_after = int(state["num_of_colliding_pairs"])
        after_repair = repair_structure_fingerprint(state)
        outcome = classify_repair_outcome(
            before_fingerprint=before_repair,
            after_fingerprint=after_repair,
            replan_success=bool(metrics.get("replan_success")),
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=bool(state.get("feasible")),
        )
        if not repair_order and (
            metrics["replan_success"] or outcome != "hard_failure"
        ):
            raise RuntimeError(
                "v3-S3 empty repair order is only valid for a hard failure"
            )
        low_level = _low_level_delta(before, state)
        for name in ("generated", "expanded", "reopened", "runs"):
            low_level_total[name] += int(low_level.get(name, 0))
        pp_seconds = max(0.0, float(metrics.get("pp_replan_seconds", 0.0)))
        step_total = selection_seconds + repair_seconds
        total_seconds += step_total
        total_pp_seconds += pp_seconds
        trajectory.append(conflicts_after)
        steps.append(
            {
                "step": offset + 1,
                "template": template.payload(),
                "template_valid": True,
                "executed": True,
                "candidate_id": str(candidate["candidate_id"]),
                "agents": list(map(int, candidate["agents"])),
                "candidate_generation": generation,
                "action": action,
                "repair_order": repair_order,
                "requested_pp_seed": seed,
                "applied_pp_seed": applied_seed,
                "step_applied": True,
                "selection_seconds": selection_seconds,
                "repair_seconds": repair_seconds,
                "pp_replan_seconds": pp_seconds,
                "total_seconds": step_total,
                "conflicts_before": conflicts_before,
                "conflicts_after": conflicts_after,
                "conflict_reduction": max(0, conflicts_before - conflicts_after),
                "repair_outcome": outcome,
                "replan_success": bool(metrics.get("replan_success")),
                "before_fingerprint": before_full,
                "after_fingerprint": state_fingerprint(state),
                "before_repair_fingerprint": before_repair,
                "after_repair_fingerprint": after_repair,
                "low_level_delta": low_level,
                "terminated": terminated,
                "truncated": step_truncated,
                "done": done,
            }
        )
        if done:
            stop_reason = (
                "feasible"
                if terminated
                else "deadline_or_iteration_truncation"
            )
            truncated = step_truncated
            break
    return {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "split": str(decision["split"]),
        "state_id": str(decision["state_id"]),
        "map_id": str(decision["map_id"]),
        "layout_mode": str(decision["layout_mode"]),
        "agent_count": int(decision["agent_count"]),
        "source_stratum": str(decision["source_stratum"]),
        "sequence_id": sequence_id(templates),
        "templates": [template.payload() for template in templates],
        "trial_index": int(trial_index),
        "initial_fingerprint": initial_full,
        "initial_repair_fingerprint": initial_repair,
        "final_fingerprint": state_fingerprint(state),
        "final_repair_fingerprint": repair_structure_fingerprint(state),
        "steps": steps,
        "executed_steps": sum(bool(step.get("executed")) for step in steps),
        "conflict_trajectory": trajectory,
        "conflict_reduction": max(0, initial_conflicts - int(trajectory[-1])),
        "best_conflict_reduction": max(0, initial_conflicts - min(trajectory)),
        "no_progress": min(trajectory) >= initial_conflicts,
        "feasible": bool(state.get("feasible")),
        "stop_reason": stop_reason,
        "truncated": truncated,
        "total_seconds": total_seconds,
        "pp_replan_seconds": total_pp_seconds,
        "generated": int(low_level_total["generated"]),
        "expanded": int(low_level_total["expanded"]),
        "reopened": int(low_level_total["reopened"]),
        "runs": int(low_level_total["runs"]),
        "complete": True,
    }


def _v2_action(
    environment: Any,
    state: dict[str, Any],
    decision: dict[str, Any],
    proposal: dict[str, Any],
    *,
    decision_index: int,
    random_seed: int,
) -> tuple[dict[str, Any], float, str]:
    if _WORKER_CONTROLLER_BUNDLE is None:
        raise RuntimeError("v3-S3 worker controller bundle is not initialized")
    started = time.perf_counter()
    candidates, rows, _timing = _full_candidate_rows(
        environment,
        state,
        {**decision, "decision_index": decision_index},
        proposal,
    )
    model = _WORKER_CONTROLLER_BUNDLE.main_models["realized_dynamic"]
    selected, _scores, _margin = score_online_candidates(rows, model)
    elapsed = time.perf_counter() - started
    candidate = candidates[selected]
    return (
        _paired_repair_action(
            "explicit_neighborhood",
            agents=candidate["agents"],
            random_seed=random_seed,
        ),
        elapsed,
        str(candidate["candidate_id"]),
    )


def _baseline_trial(
    qualified: dict[str, Any], controller: str, trial_index: int
) -> dict[str, Any]:
    if controller not in {"v2-full", "official_adaptive"}:
        raise ValueError("unknown v3-S3 external baseline")
    decision = dict(qualified["decision"])
    replay, configuration = _source_replay_job(decision)
    environment, state = replay_prefix(replay, decision["prefix_actions"])
    if state_fingerprint(state) != str(decision["before_fingerprint"]):
        raise RuntimeError("v3-S3 baseline replay fingerprint mismatch")
    initial_full = state_fingerprint(state)
    initial_repair = repair_structure_fingerprint(state)
    initial_conflicts = int(state["num_of_colliding_pairs"])
    trajectory = [initial_conflicts]
    steps = []
    total_seconds = 0.0
    total_pp_seconds = 0.0
    low_level_total = collections.Counter()
    terminal = _state_stop_reason(state)
    stop_reason = terminal[0] if terminal is not None else "horizon_complete"
    truncated = terminal[1] if terminal is not None else False
    for offset in (() if terminal is not None else range(S3_HORIZON)):
        before = state
        before_full = state_fingerprint(before)
        before_repair = repair_structure_fingerprint(before)
        conflicts_before = int(before["num_of_colliding_pairs"])
        seed = _paired_seed(initial_repair, trial_index, offset + 1)
        if controller == "v2-full":
            action, selection_seconds, candidate_id = _v2_action(
                environment,
                before,
                decision,
                dict(configuration["proposal"]),
                decision_index=int(decision["decision_index"]) + offset,
                random_seed=seed,
            )
        else:
            action = _paired_repair_action("official", random_seed=seed)
            selection_seconds = 0.0
            candidate_id = "official_adaptive"
        started = time.perf_counter()
        result = _plain(environment.step(action))
        repair_seconds = time.perf_counter() - started
        state = dict(result["observation"])
        metrics = dict(result["metrics"])
        neighborhood, repair_order, applied_seed = _native_repair_evidence(
            metrics,
            expected_seed=seed,
            agent_count=int(decision["agent_count"]),
            expected_agents=(
                list(map(int, action["agents"]))
                if controller == "v2-full"
                else None
            ),
        )
        terminated, step_truncated, done = _terminal_flags(result, state)
        conflicts_after = int(state["num_of_colliding_pairs"])
        after_repair = repair_structure_fingerprint(state)
        outcome = classify_repair_outcome(
            before_fingerprint=before_repair,
            after_fingerprint=after_repair,
            replan_success=bool(metrics.get("replan_success")),
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=bool(state.get("feasible")),
        )
        if not repair_order and (
            metrics["replan_success"] or outcome != "hard_failure"
        ):
            raise RuntimeError(
                "v3-S3 empty repair order is only valid for a hard failure"
            )
        step_total = selection_seconds + repair_seconds
        total_seconds += step_total
        pp_seconds = max(0.0, float(metrics.get("pp_replan_seconds", 0.0)))
        total_pp_seconds += pp_seconds
        low_level = _low_level_delta(before, state)
        for name in ("generated", "expanded", "reopened", "runs"):
            low_level_total[name] += int(low_level.get(name, 0))
        trajectory.append(conflicts_after)
        steps.append(
            {
                "step": offset + 1,
                "candidate_id": candidate_id,
                "agents": (
                    list(map(int, action["agents"]))
                    if controller == "v2-full"
                    else neighborhood
                ),
                "action": action,
                "repair_order": repair_order,
                "requested_pp_seed": seed,
                "applied_pp_seed": applied_seed,
                "step_applied": True,
                "selection_seconds": selection_seconds,
                "repair_seconds": repair_seconds,
                "pp_replan_seconds": pp_seconds,
                "total_seconds": step_total,
                "conflicts_before": conflicts_before,
                "conflicts_after": conflicts_after,
                "conflict_reduction": max(0, conflicts_before - conflicts_after),
                "repair_outcome": outcome,
                "replan_success": bool(metrics.get("replan_success")),
                "before_fingerprint": before_full,
                "after_fingerprint": state_fingerprint(state),
                "before_repair_fingerprint": before_repair,
                "after_repair_fingerprint": after_repair,
                "low_level_delta": low_level,
                "terminated": terminated,
                "truncated": step_truncated,
                "done": done,
            }
        )
        if done:
            stop_reason = (
                "feasible"
                if terminated
                else "deadline_or_iteration_truncation"
            )
            truncated = step_truncated
            break
    return {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "split": str(decision["split"]),
        "state_id": str(decision["state_id"]),
        "map_id": str(decision["map_id"]),
        "layout_mode": str(decision["layout_mode"]),
        "agent_count": int(decision["agent_count"]),
        "controller": controller,
        "trial_index": int(trial_index),
        "initial_fingerprint": initial_full,
        "initial_repair_fingerprint": initial_repair,
        "final_fingerprint": state_fingerprint(state),
        "final_repair_fingerprint": repair_structure_fingerprint(state),
        "steps": steps,
        "executed_steps": len(steps),
        "conflict_trajectory": trajectory,
        "conflict_reduction": max(0, initial_conflicts - int(trajectory[-1])),
        "best_conflict_reduction": max(0, initial_conflicts - min(trajectory)),
        "no_progress": min(trajectory) >= initial_conflicts,
        "feasible": bool(state.get("feasible")),
        "stop_reason": stop_reason,
        "truncated": truncated,
        "total_seconds": total_seconds,
        "pp_replan_seconds": total_pp_seconds,
        "generated": int(low_level_total["generated"]),
        "expanded": int(low_level_total["expanded"]),
        "reopened": int(low_level_total["reopened"]),
        "runs": int(low_level_total["runs"]),
        "complete": True,
    }


def _sequence_efficiency(rows: list[dict[str, Any]]) -> float:
    return statistics.fmean(
        float(row["conflict_reduction"]) / max(1e-9, float(row["total_seconds"]))
        for row in rows
    )


def _ambiguous_sequences(trials: list[dict[str, Any]]) -> bool:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trials:
        grouped[str(row["sequence_id"])].append(row)
    ranked = sorted(
        grouped.items(), key=lambda item: (-_sequence_efficiency(item[1]), item[0])
    )
    if len(ranked) < 2:
        return False
    first_efficiency = _sequence_efficiency(ranked[0][1])
    second_efficiency = _sequence_efficiency(ranked[1][1])
    close = first_efficiency <= 1.10 * max(1e-9, second_efficiency)
    by_trial = []
    for trial_index in (0, 1):
        values = []
        for sequence, rows in ranked[:2]:
            row = next(item for item in rows if int(item["trial_index"]) == trial_index)
            values.append(
                (
                    float(row["conflict_reduction"])
                    / max(1e-9, float(row["total_seconds"])),
                    sequence,
                )
            )
        by_trial.append(max(values)[1])
    return close or len(set(by_trial)) > 1


def _ambiguous_additional_sequences(
    qualified: dict[str, Any], state_id: str
) -> tuple[tuple[S3ActionTemplate, ...], ...]:
    base_ids = {
        sequence_id(templates) for templates in balanced_sequence_templates(state_id)
    }
    remaining = [
        templates
        for templates in all_runtime_sequences(
            dict(qualified["template_indices"]).keys()
        )
        if sequence_id(templates) not in base_ids
    ]
    remaining.sort(
        key=lambda templates: _stable_key((state_id, sequence_id(templates)))
    )
    additions = tuple(remaining[:12])
    if len(additions) != 12:
        raise ValueError(
            "v3-S3 ambiguous state does not provide 12 additional sequences"
        )
    return additions


def _sequence_feature(
    qualified: dict[str, Any], templates: tuple[S3ActionTemplate, ...]
) -> dict[str, Any]:
    decision = dict(qualified["decision"])
    first_index = int(qualified["template_indices"][templates[0].key])
    row = sequence_feature_row(
        qualified["candidate_rows"][first_index],
        dict(decision["temporal_context"]),
        templates,
        agent_count=int(decision["agent_count"]),
    )
    return {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "split": str(decision["split"]),
        "state_id": str(decision["state_id"]),
        "map_id": str(decision["map_id"]),
        "layout_mode": str(decision["layout_mode"]),
        "agent_count": int(decision["agent_count"]),
        "source_stratum": str(decision["source_stratum"]),
        "sequence_id": sequence_id(templates),
        "templates": [template.payload() for template in templates],
        "feature_profile": row["feature_profile"],
        "feature_names": list(row["feature_names"]),
        "feature_values": list(row["feature_values"]),
    }


def _float_matches(value: Any, expected: float) -> bool:
    return _finite_nonnegative(value) and math.isclose(
        float(value), float(expected), rel_tol=1e-9, abs_tol=1e-9
    )


def _row_identity_errors(
    row: dict[str, Any], decision: dict[str, Any], *, label: str
) -> list[str]:
    errors = []
    expected = {
        "split": str(decision["split"]),
        "state_id": str(decision["state_id"]),
        "map_id": str(decision["map_id"]),
        "layout_mode": str(decision["layout_mode"]),
        "agent_count": int(decision["agent_count"]),
    }
    for name, value in expected.items():
        if row.get(name) != value:
            errors.append(f"{label} {name} differs")
    return errors


def _candidate_generation_errors(
    record: Any,
    *,
    position: int,
    template: S3ActionTemplate,
    expected_before_fingerprint: str | None,
    decision: dict[str, Any],
    qualified: dict[str, Any],
    executed: bool,
    candidate_id: Any,
    agents: Any,
) -> list[str]:
    label = f"sequence trial step {position} candidate generation"
    if not isinstance(record, dict):
        return [f"{label} is missing"]
    errors: list[str] = []
    if record.get("schema") != V3_S3_CANDIDATE_GENERATION_SCHEMA:
        errors.append(f"{label} schema differs")
    stored_record_fingerprint = _stored_fingerprint(record.get("fingerprint"))
    payload = {name: value for name, value in record.items() if name != "fingerprint"}
    if (
        stored_record_fingerprint is None
        or stored_record_fingerprint != _fingerprint(payload)
    ):
        errors.append(f"{label} fingerprint differs")
    expected_source = "qualification_pool" if position == 1 else "restricted_dynamic"
    if record.get("source") != expected_source:
        errors.append(f"{label} source differs")
    if (
        expected_before_fingerprint is None
        or record.get("before_fingerprint") != expected_before_fingerprint
    ):
        errors.append(f"{label} state differs")
    if not _strict_nonnegative_int(record.get("decision_index")) or record.get(
        "decision_index"
    ) != int(decision["decision_index"]) + position - 1:
        errors.append(f"{label} decision index differs")
    if record.get("template") != template.payload():
        errors.append(f"{label} template differs")

    if position == 1:
        if record.get("candidate_pool") is not None:
            errors.append(f"{label} embeds the qualification pool")
        if record.get("proposal_fingerprint") != "":
            errors.append(f"{label} proposal fingerprint differs")
        raw_pool = qualified.get("candidates")
    else:
        raw_pool = record.get("candidate_pool")
        expected_proposal_fingerprint = _fingerprint(
            _restricted_proposal(template)
        )
        if record.get("proposal_fingerprint") != expected_proposal_fingerprint:
            errors.append(f"{label} proposal fingerprint differs")
    if not isinstance(raw_pool, list):
        return [*errors, f"{label} pool is invalid"]
    try:
        normalized_pool = [_candidate_semantics(value) for value in raw_pool]
    except (TypeError, ValueError) as error:
        return [*errors, f"{label} pool is invalid: {error}"]
    if position > 1 and normalized_pool != raw_pool:
        errors.append(f"{label} pool is not canonical")
    if not _strict_nonnegative_int(record.get("candidate_count")) or record.get(
        "candidate_count"
    ) != len(normalized_pool):
        errors.append(f"{label} candidate count differs")
    if record.get("candidate_pool_fingerprint") != _fingerprint(normalized_pool):
        errors.append(f"{label} pool fingerprint differs")
    try:
        selected_index = candidate_template_indices(normalized_pool).get(
            template.key, -1
        )
    except (TypeError, ValueError) as error:
        return [*errors, f"{label} mapping is invalid: {error}"]
    if record.get("selected_index") != selected_index:
        errors.append(f"{label} selected index differs")
    selected = normalized_pool[selected_index] if selected_index >= 0 else None
    expected_selected_fingerprint = (
        _fingerprint(selected) if selected is not None else ""
    )
    if record.get("selected_candidate_fingerprint") != expected_selected_fingerprint:
        errors.append(f"{label} selected candidate fingerprint differs")
    if executed != (selected is not None):
        errors.append(f"{label} availability differs")
    if executed and selected is not None:
        if candidate_id != selected["candidate_id"]:
            errors.append(f"{label} candidate_id differs")
        if agents != selected["agents"]:
            errors.append(f"{label} agents differ")
    return errors


def _sequence_trial_errors(
    row: Any,
    *,
    decision: dict[str, Any],
    qualified: dict[str, Any],
    expected_templates: tuple[S3ActionTemplate, ...],
) -> list[str]:
    errors = _artifact_header_errors(
        row, run_fingerprint=None, require_complete=True
    )
    if not isinstance(row, dict):
        return errors
    errors.extend(_row_identity_errors(row, decision, label="sequence trial"))
    try:
        templates = tuple(
            S3ActionTemplate.from_payload(dict(value)) for value in row["templates"]
        )
    except (KeyError, TypeError, ValueError) as error:
        return [*errors, f"sequence trial templates are invalid: {error}"]
    if templates != expected_templates:
        errors.append("sequence trial templates differ from feature registration")
    expected_sequence_id = sequence_id(expected_templates)
    if str(row.get("sequence_id")) != expected_sequence_id:
        errors.append("sequence trial sequence_id differs")
    trial_index = row.get("trial_index")
    if not _strict_nonnegative_int(trial_index):
        trial_index = -1
        errors.append("sequence trial index is invalid")
    if _stored_fingerprint(row.get("initial_fingerprint")) != decision[
        "before_fingerprint"
    ]:
        errors.append("sequence trial initial fingerprint differs")
    if _stored_fingerprint(row.get("initial_repair_fingerprint")) != decision[
        "before_repair_fingerprint"
    ]:
        errors.append("sequence trial initial repair fingerprint differs")

    steps = row.get("steps")
    trajectory = row.get("conflict_trajectory")
    if not isinstance(steps, list) or len(steps) > S3_HORIZON:
        return [*errors, "sequence trial step coverage is invalid"]
    if not isinstance(trajectory, list) or not trajectory:
        return [*errors, "sequence trial conflict trajectory is invalid"]
    if any(not _strict_nonnegative_int(value) for value in trajectory):
        return [*errors, "sequence trial conflict trajectory is non-integral"]
    conflicts = list(trajectory)
    executed = 0
    selection_total = 0.0
    repair_total = 0.0
    pp_total = 0.0
    low_level_total = collections.Counter()
    previous_full = _stored_fingerprint(row.get("initial_fingerprint"))
    previous_repair = _stored_fingerprint(row.get("initial_repair_fingerprint"))
    terminal_seen = False
    unavailable_seen = False
    for position, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            errors.append(f"sequence trial step {position} is not an object")
            continue
        if not _strict_nonnegative_int(step.get("step")) or step.get("step") != position:
            errors.append(f"sequence trial step {position} index differs")
        expected_template = expected_templates[position - 1]
        if step.get("template") != expected_template.payload():
            errors.append(f"sequence trial step {position} template differs")
        selection = step.get("selection_seconds")
        if not _finite_nonnegative(selection):
            errors.append(f"sequence trial step {position} selection time is invalid")
            selection = 0.0
        selection_total += float(selection)
        is_executed = step.get("executed") is True
        if not isinstance(step.get("executed"), bool):
            errors.append(f"sequence trial step {position} executed flag is invalid")
        errors.extend(
            _candidate_generation_errors(
                step.get("candidate_generation"),
                position=position,
                template=expected_template,
                expected_before_fingerprint=previous_full,
                decision=decision,
                qualified=qualified,
                executed=is_executed,
                candidate_id=step.get("candidate_id"),
                agents=step.get("agents"),
            )
        )
        if not is_executed:
            unavailable_seen = True
            if step.get("template_valid") is not False or position != len(steps):
                errors.append(
                    f"sequence trial step {position} invalid-template boundary differs"
                )
            continue
        executed += 1
        if unavailable_seen or terminal_seen or step.get("template_valid") is not True:
            errors.append(f"sequence trial step {position} executes after a stop")
        for name in ("repair_seconds", "pp_replan_seconds", "total_seconds"):
            if not _finite_nonnegative(step.get(name)):
                errors.append(f"sequence trial step {position} {name} is invalid")
        repair_seconds = (
            float(step.get("repair_seconds", 0.0))
            if _finite_nonnegative(step.get("repair_seconds"))
            else 0.0
        )
        pp_seconds = (
            float(step.get("pp_replan_seconds", 0.0))
            if _finite_nonnegative(step.get("pp_replan_seconds"))
            else 0.0
        )
        repair_total += repair_seconds
        pp_total += pp_seconds
        if not _float_matches(
            step.get("total_seconds"), float(selection) + repair_seconds
        ):
            errors.append(f"sequence trial step {position} total time differs")
        before_conflicts = step.get("conflicts_before")
        after_conflicts = step.get("conflicts_after")
        if not _strict_nonnegative_int(before_conflicts) or not _strict_nonnegative_int(
            after_conflicts
        ):
            errors.append(f"sequence trial step {position} conflicts are invalid")
            continue
        if before_conflicts != conflicts[executed - 1]:
            errors.append(f"sequence trial step {position} before conflicts differs")
        if len(conflicts) <= executed or after_conflicts != conflicts[executed]:
            errors.append(f"sequence trial step {position} after conflicts differs")
        if not _strict_nonnegative_int(step.get("conflict_reduction")) or step.get("conflict_reduction") != max(
            0, before_conflicts - after_conflicts
        ):
            errors.append(f"sequence trial step {position} reduction differs")
        if _stored_fingerprint(step.get("before_fingerprint")) != previous_full:
            errors.append(f"sequence trial step {position} fingerprint chain differs")
        if _stored_fingerprint(step.get("before_repair_fingerprint")) != previous_repair:
            errors.append(
                f"sequence trial step {position} repair fingerprint chain differs"
            )
        previous_full = _stored_fingerprint(step.get("after_fingerprint"))
        previous_repair = _stored_fingerprint(step.get("after_repair_fingerprint"))
        if previous_full is None or previous_repair is None:
            errors.append(f"sequence trial step {position} final fingerprint is empty")
        action = step.get("action")
        agents = step.get("agents")
        stored_agents = _stored_agent_ids(
            agents, agent_count=int(decision["agent_count"])
        )
        if (
            not isinstance(action, dict)
            or str(action.get("mode")) != "explicit_neighborhood"
            or stored_agents is None
            or list(action.get("agents") or ()) != agents
        ):
            errors.append(f"sequence trial step {position} action differs")
        elif (
            isinstance(action.get("random_seed"), bool)
            or not isinstance(action.get("random_seed"), int)
            or int(action["random_seed"]) < 0
            or action.get("pp_random_seed") != action.get("random_seed")
            or int(action["random_seed"])
            != _paired_seed(
                str(row.get("initial_repair_fingerprint")),
                trial_index,
                position,
            )
        ):
            errors.append(f"sequence trial step {position} paired seed differs")
        low_level = step.get("low_level_delta")
        if not isinstance(low_level, dict):
            errors.append(f"sequence trial step {position} low-level delta is invalid")
        else:
            for name in ("generated", "expanded", "reopened", "runs"):
                value = low_level.get(name)
                if not _strict_nonnegative_int(value):
                    errors.append(
                        f"sequence trial step {position} low-level {name} is invalid"
                    )
                    continue
                low_level_total[name] += value
        if any(
            not isinstance(step.get(name), bool)
            for name in ("terminated", "truncated", "done", "replan_success")
        ):
            errors.append(f"sequence trial step {position} boolean evidence is invalid")
        terminated = step.get("terminated") is True
        truncated = step.get("truncated") is True
        done = step.get("done") is True
        if done != (terminated or truncated) or terminated and truncated:
            errors.append(f"sequence trial step {position} terminal flags differ")
        if step.get("step_applied") is not True:
            errors.append(f"sequence trial step {position} was not applied")
        repair_order = _stored_agent_ids(
            step.get("repair_order"),
            agent_count=int(decision["agent_count"]),
            allow_empty=True,
        )
        if repair_order is None or (
            repair_order and stored_agents is not None and set(repair_order) != set(stored_agents)
        ):
            errors.append(f"sequence trial step {position} repair order differs")
        expected_seed = _paired_seed(
            row.get("initial_repair_fingerprint"), trial_index, position
        )
        if step.get("requested_pp_seed") != expected_seed:
            errors.append(f"sequence trial step {position} requested PP seed differs")
        expected_applied = expected_seed if repair_order else -1
        if step.get("applied_pp_seed") != expected_applied:
            errors.append(f"sequence trial step {position} applied PP seed differs")
        try:
            expected_outcome = classify_repair_outcome(
                before_fingerprint=str(step["before_repair_fingerprint"]),
                after_fingerprint=str(step["after_repair_fingerprint"]),
                replan_success=step.get("replan_success") is True,
                conflicts_before=before_conflicts,
                conflicts_after=after_conflicts,
                feasible=terminated,
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(
                f"sequence trial step {position} outcome is invalid: {error}"
            )
        else:
            if str(step.get("repair_outcome")) != expected_outcome:
                errors.append(
                    f"sequence trial step {position} repair outcome differs"
                )
            if not repair_order and (
                step.get("replan_success") is True
                or expected_outcome != "hard_failure"
            ):
                errors.append(
                    f"sequence trial step {position} empty repair order differs"
                )
        terminal_seen = terminal_seen or done

    if len(conflicts) != executed + 1:
        errors.append("sequence trial trajectory length differs from executed steps")
    if not _strict_nonnegative_int(row.get("executed_steps")) or row.get(
        "executed_steps"
    ) != executed:
        errors.append("sequence trial executed_steps differs")
    if unavailable_seen:
        expected_stop = "template_unavailable"
        expected_truncated = False
    elif steps and isinstance(steps[-1], dict) and steps[-1].get("done") is True:
        expected_stop = (
            "feasible"
            if steps[-1].get("terminated") is True
            else "deadline_or_iteration_truncation"
        )
        expected_truncated = steps[-1].get("truncated") is True
    elif executed == S3_HORIZON:
        expected_stop = "horizon_complete"
        expected_truncated = False
    else:
        expected_stop = ""
        expected_truncated = False
        errors.append("sequence trial stopped before a registered boundary")
    if str(row.get("stop_reason")) != expected_stop:
        errors.append("sequence trial stop_reason differs")
    if row.get("truncated") is not expected_truncated:
        errors.append("sequence trial truncated flag differs")
    if str(row.get("stop_reason")) not in V3_S3_STOP_REASONS:
        errors.append("sequence trial stop_reason is unsupported")
    if _stored_fingerprint(row.get("final_fingerprint")) != previous_full:
        errors.append("sequence trial final fingerprint differs")
    if _stored_fingerprint(row.get("final_repair_fingerprint")) != previous_repair:
        errors.append("sequence trial final repair fingerprint differs")
    if not _float_matches(row.get("total_seconds"), selection_total + repair_total):
        errors.append("sequence trial total_seconds differs")
    if not _float_matches(row.get("pp_replan_seconds"), pp_total):
        errors.append("sequence trial pp_replan_seconds differs")
    for name in ("generated", "expanded", "reopened", "runs"):
        if not _strict_nonnegative_int(row.get(name)) or row.get(name) != int(
            low_level_total[name]
        ):
            errors.append(f"sequence trial {name} total differs")
    if conflicts:
        expected_reduction = max(0, conflicts[0] - conflicts[-1])
        expected_best = max(0, conflicts[0] - min(conflicts))
        if not _strict_nonnegative_int(row.get("conflict_reduction")) or row.get(
            "conflict_reduction"
        ) != expected_reduction:
            errors.append("sequence trial conflict_reduction differs")
        if not _strict_nonnegative_int(row.get("best_conflict_reduction")) or row.get(
            "best_conflict_reduction"
        ) != expected_best:
            errors.append("sequence trial best_conflict_reduction differs")
        if row.get("no_progress") is not (min(conflicts) >= conflicts[0]):
            errors.append("sequence trial no_progress differs")
    if row.get("feasible") is not (expected_stop == "feasible"):
        errors.append("sequence trial feasible flag differs")
    return errors


def _baseline_trial_errors(
    row: Any,
    *,
    decision: dict[str, Any],
    controller: str,
) -> list[str]:
    errors = _artifact_header_errors(
        row, run_fingerprint=None, require_complete=True
    )
    if not isinstance(row, dict):
        return errors
    errors.extend(_row_identity_errors(row, decision, label="baseline trial"))
    if str(row.get("controller")) != controller:
        errors.append("baseline controller differs")
    if _stored_fingerprint(row.get("initial_fingerprint")) != decision[
        "before_fingerprint"
    ]:
        errors.append("baseline initial fingerprint differs")
    if _stored_fingerprint(row.get("initial_repair_fingerprint")) != decision[
        "before_repair_fingerprint"
    ]:
        errors.append("baseline initial repair fingerprint differs")
    trial_index = row.get("trial_index")
    if not _strict_nonnegative_int(trial_index):
        trial_index = -1
        errors.append("baseline trial index is invalid")
    steps = row.get("steps")
    trajectory = row.get("conflict_trajectory")
    if not isinstance(steps, list) or not 0 <= len(steps) <= S3_HORIZON:
        return [*errors, "baseline step coverage is invalid"]
    if not isinstance(trajectory, list):
        return [*errors, "baseline conflict trajectory is invalid"]
    if any(not _strict_nonnegative_int(value) for value in trajectory):
        return [*errors, "baseline conflict trajectory is non-integral"]
    conflicts = list(trajectory)
    if len(conflicts) != len(steps) + 1 or not conflicts:
        errors.append("baseline trajectory length differs")
    total_seconds = 0.0
    pp_total = 0.0
    low_level_total = collections.Counter()
    previous_full = _stored_fingerprint(row.get("initial_fingerprint"))
    previous_repair = _stored_fingerprint(row.get("initial_repair_fingerprint"))
    terminal_seen = False
    for position, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            errors.append(f"baseline step {position} is not an object")
            continue
        if (
            not _strict_nonnegative_int(step.get("step"))
            or step.get("step") != position
            or terminal_seen
        ):
            errors.append(f"baseline step {position} order differs")
        for name in ("selection_seconds", "repair_seconds", "pp_replan_seconds"):
            if not _finite_nonnegative(step.get(name)):
                errors.append(f"baseline step {position} {name} is invalid")
        selection = float(step.get("selection_seconds", 0.0)) if _finite_nonnegative(step.get("selection_seconds")) else 0.0
        repair = float(step.get("repair_seconds", 0.0)) if _finite_nonnegative(step.get("repair_seconds")) else 0.0
        pp_seconds = float(step.get("pp_replan_seconds", 0.0)) if _finite_nonnegative(step.get("pp_replan_seconds")) else 0.0
        total_seconds += selection + repair
        pp_total += pp_seconds
        if not _float_matches(step.get("total_seconds"), selection + repair):
            errors.append(f"baseline step {position} total time differs")
        before_conflicts = 0
        after_conflicts = 0
        try:
            before_conflicts = step["conflicts_before"]
            after_conflicts = step["conflicts_after"]
            if not _strict_nonnegative_int(before_conflicts) or not _strict_nonnegative_int(
                after_conflicts
            ):
                raise TypeError("conflicts are not strict integers")
            if before_conflicts != conflicts[position - 1] or after_conflicts != conflicts[position]:
                errors.append(f"baseline step {position} conflict chain differs")
            if not _strict_nonnegative_int(step.get("conflict_reduction")) or step.get("conflict_reduction") != max(0, before_conflicts - after_conflicts):
                errors.append(f"baseline step {position} reduction differs")
        except (IndexError, KeyError, TypeError, ValueError):
            errors.append(f"baseline step {position} conflicts are invalid")
        if _stored_fingerprint(step.get("before_fingerprint")) != previous_full:
            errors.append(f"baseline step {position} fingerprint chain differs")
        if _stored_fingerprint(step.get("before_repair_fingerprint")) != previous_repair:
            errors.append(f"baseline step {position} repair fingerprint chain differs")
        previous_full = _stored_fingerprint(step.get("after_fingerprint"))
        previous_repair = _stored_fingerprint(step.get("after_repair_fingerprint"))
        if previous_full is None or previous_repair is None:
            errors.append(f"baseline step {position} final fingerprint is empty")
        action = step.get("action")
        stored_agents = _stored_agent_ids(
            step.get("agents"), agent_count=int(decision["agent_count"])
        )
        expected_mode = "official" if controller == "official_adaptive" else "explicit_neighborhood"
        if (
            not isinstance(action, dict)
            or str(action.get("mode")) != expected_mode
            or stored_agents is None
            or (
                controller == "v2-full"
                and list(action.get("agents") or ()) != stored_agents
            )
        ):
            errors.append(f"baseline step {position} action differs")
        elif (
            isinstance(action.get("random_seed"), bool)
            or not isinstance(action.get("random_seed"), int)
            or int(action["random_seed"]) < 0
            or action.get("pp_random_seed") != action.get("random_seed")
            or int(action["random_seed"])
            != _paired_seed(
                str(row.get("initial_repair_fingerprint")),
                trial_index,
                position,
            )
        ):
            errors.append(f"baseline step {position} paired seed differs")
        low_level = step.get("low_level_delta")
        if not isinstance(low_level, dict):
            errors.append(f"baseline step {position} low-level delta is invalid")
        else:
            for name in ("generated", "expanded", "reopened", "runs"):
                value = low_level.get(name)
                if not _strict_nonnegative_int(value):
                    errors.append(f"baseline step {position} low-level {name} is invalid")
                    continue
                low_level_total[name] += value
        if any(
            not isinstance(step.get(name), bool)
            for name in ("terminated", "truncated", "done", "replan_success")
        ):
            errors.append(f"baseline step {position} boolean evidence is invalid")
        terminated = step.get("terminated") is True
        truncated = step.get("truncated") is True
        done = step.get("done") is True
        if done != (terminated or truncated) or terminated and truncated:
            errors.append(f"baseline step {position} terminal flags differ")
        if step.get("step_applied") is not True:
            errors.append(f"baseline step {position} was not applied")
        repair_order = _stored_agent_ids(
            step.get("repair_order"),
            agent_count=int(decision["agent_count"]),
            allow_empty=True,
        )
        if repair_order is None or (
            repair_order and stored_agents is not None and set(repair_order) != set(stored_agents)
        ):
            errors.append(f"baseline step {position} repair order differs")
        expected_seed = _paired_seed(
            row.get("initial_repair_fingerprint"), trial_index, position
        )
        if step.get("requested_pp_seed") != expected_seed:
            errors.append(f"baseline step {position} requested PP seed differs")
        expected_applied = expected_seed if repair_order else -1
        if step.get("applied_pp_seed") != expected_applied:
            errors.append(f"baseline step {position} applied PP seed differs")
        try:
            expected_outcome = classify_repair_outcome(
                before_fingerprint=str(step["before_repair_fingerprint"]),
                after_fingerprint=str(step["after_repair_fingerprint"]),
                replan_success=step.get("replan_success") is True,
                conflicts_before=before_conflicts,
                conflicts_after=after_conflicts,
                feasible=terminated,
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"baseline step {position} outcome is invalid: {error}")
        else:
            if str(step.get("repair_outcome")) != expected_outcome:
                errors.append(f"baseline step {position} repair outcome differs")
            if not repair_order and (
                step.get("replan_success") is True
                or expected_outcome != "hard_failure"
            ):
                errors.append(
                    f"baseline step {position} empty repair order differs"
                )
        terminal_seen = terminal_seen or done
    if not _strict_nonnegative_int(row.get("executed_steps")) or row.get(
        "executed_steps"
    ) != len(steps):
        errors.append("baseline executed_steps differs")
    if steps and isinstance(steps[-1], dict) and steps[-1].get("done") is True:
        expected_stop = (
            "feasible"
            if steps[-1].get("terminated") is True
            else "deadline_or_iteration_truncation"
        )
        expected_truncated = steps[-1].get("truncated") is True
    elif len(steps) == S3_HORIZON:
        expected_stop = "horizon_complete"
        expected_truncated = False
    else:
        expected_stop = ""
        expected_truncated = False
        errors.append("baseline stopped before a registered boundary")
    if str(row.get("stop_reason")) != expected_stop:
        errors.append("baseline stop_reason differs")
    if row.get("truncated") is not expected_truncated:
        errors.append("baseline truncated flag differs")
    if _stored_fingerprint(row.get("final_fingerprint")) != previous_full:
        errors.append("baseline final fingerprint differs")
    if _stored_fingerprint(row.get("final_repair_fingerprint")) != previous_repair:
        errors.append("baseline final repair fingerprint differs")
    if not _float_matches(row.get("total_seconds"), total_seconds):
        errors.append("baseline total_seconds differs")
    if not _float_matches(row.get("pp_replan_seconds"), pp_total):
        errors.append("baseline pp_replan_seconds differs")
    for name in ("generated", "expanded", "reopened", "runs"):
        if not _strict_nonnegative_int(row.get(name)) or row.get(name) != int(low_level_total[name]):
            errors.append(f"baseline {name} total differs")
    if conflicts:
        if not _strict_nonnegative_int(row.get("conflict_reduction")) or row.get(
            "conflict_reduction"
        ) != max(0, conflicts[0] - conflicts[-1]):
            errors.append("baseline conflict_reduction differs")
        if not _strict_nonnegative_int(row.get("best_conflict_reduction")) or row.get(
            "best_conflict_reduction"
        ) != max(0, conflicts[0] - min(conflicts)):
            errors.append("baseline best_conflict_reduction differs")
        if row.get("no_progress") is not (min(conflicts) >= conflicts[0]):
            errors.append("baseline no_progress differs")
    if row.get("feasible") is not (expected_stop == "feasible"):
        errors.append("baseline feasible flag differs")
    return errors


def _state_artifact_errors(
    payload: Any,
    *,
    run_fingerprint: str | None = None,
    selected_row: dict[str, Any],
    qualified: dict[str, Any],
) -> list[str]:
    errors = _artifact_header_errors(
        payload, run_fingerprint=run_fingerprint, require_complete=True
    )
    if not isinstance(payload, dict):
        return errors
    decision = qualified.get("decision")
    if not isinstance(decision, dict):
        return [*errors, "state qualification decision is invalid"]
    required_decision_fields = {
        "state_id",
        "before_fingerprint",
        "before_repair_fingerprint",
        "split",
        "map_id",
        "layout_mode",
        "agent_count",
        "source_stratum",
        "temporal_context",
        "decision_index",
    }
    missing_decision_fields = sorted(required_decision_fields - set(decision))
    if missing_decision_fields:
        return [
            *errors,
            f"state qualification decision is missing {missing_decision_fields}",
        ]
    qualification_errors = _qualification_artifact_errors(
        qualified,
        run_fingerprint=run_fingerprint,
        expected_decision=None,
    )
    errors.extend(f"qualification: {error}" for error in qualification_errors)
    state_id = str(decision.get("state_id", ""))
    for name in (
        "state_id",
        "before_fingerprint",
        "before_repair_fingerprint",
        "split",
        "map_id",
        "layout_mode",
        "agent_count",
    ):
        if selected_row.get(name) != decision.get(name):
            errors.append(f"state selection {name} differs from qualification")
    if str(payload.get("state_id")) != state_id:
        errors.append("state artifact state_id differs")
    if not isinstance(payload.get("ambiguous"), bool):
        errors.append("state artifact ambiguous flag is not boolean")
    features = payload.get("features")
    trials = payload.get("trials")
    baselines = payload.get("external_baselines")
    if not isinstance(features, list) or not isinstance(trials, list) or not isinstance(baselines, list):
        return [*errors, "state artifact row groups are invalid"]

    base_sequences = list(balanced_sequence_templates(state_id))
    registered = {sequence_id(value): value for value in base_sequences}
    if payload.get("ambiguous") is True:
        try:
            additions = _ambiguous_additional_sequences(qualified, state_id)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"state ambiguous sequence reconstruction failed: {error}")
            additions = ()
        registered.update({sequence_id(value): value for value in additions})
    feature_by_sequence: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            errors.append(f"state feature {index} is not an object")
            continue
        key = str(feature.get("sequence_id", ""))
        if key in feature_by_sequence:
            errors.append(f"state duplicate feature sequence {key}")
            continue
        feature_by_sequence[key] = feature
        templates = registered.get(key)
        if templates is None:
            errors.append(f"state feature sequence {key} is unregistered")
            continue
        try:
            expected = _sequence_feature(qualified, templates)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"state feature {key} cannot be reconstructed: {error}")
            continue
        if feature != expected:
            errors.append(f"state feature {key} differs from deterministic projection")
        values = feature.get("feature_values")
        if (
            feature.get("feature_profile") != V3_S3_PROFILE
            or tuple(map(str, feature.get("feature_names", ())))
            != V3_S3_FULL_FEATURE_NAMES
            or not isinstance(values, list)
            or len(values) != len(V3_S3_FULL_FEATURE_NAMES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (values or ())
            )
        ):
            errors.append(f"state feature {key} feature order or values are invalid")
    if set(feature_by_sequence) != set(registered):
        errors.append("state registered feature coverage differs")

    trial_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            errors.append(f"state trial {index} is not an object")
            continue
        try:
            key = (str(trial["sequence_id"]), int(trial["trial_index"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"state trial {index} identity is invalid")
            continue
        if key in trial_by_key:
            errors.append(f"state duplicate trial {key}")
            continue
        trial_by_key[key] = trial
        templates = registered.get(key[0])
        if templates is None:
            errors.append(f"state trial sequence {key[0]} is unregistered")
            continue
        try:
            trial_errors = _sequence_trial_errors(
                trial,
                decision=decision,
                qualified=qualified,
                expected_templates=templates,
            )
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            trial_errors = [f"validator rejected malformed trial: {error}"]
        errors.extend(f"{key}: {error}" for error in trial_errors)
    primary_keys = {
        (key, trial_index) for key in registered for trial_index in (0, 1)
    }
    expected_trial_keys = set(primary_keys)
    if payload.get("ambiguous") is True and primary_keys <= set(trial_by_key):
        primary_rows = {
            key: [trial_by_key[(key, 0)], trial_by_key[(key, 1)]]
            for key in registered
        }
        try:
            top = sorted(
                registered,
                key=lambda key: (-_sequence_efficiency(primary_rows[key]), key),
            )[:6]
            expected_trial_keys.update(
                (key, trial_index) for key in top for trial_index in (2, 3)
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"state ambiguous extra-trial ranking failed: {error}")
    if set(trial_by_key) != expected_trial_keys:
        errors.append("state sequence trial coverage differs")

    baseline_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for index, baseline in enumerate(baselines):
        if not isinstance(baseline, dict):
            errors.append(f"state baseline {index} is not an object")
            continue
        try:
            key = (str(baseline["controller"]), int(baseline["trial_index"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"state baseline {index} identity is invalid")
            continue
        if key in baseline_by_key:
            errors.append(f"state duplicate baseline {key}")
            continue
        baseline_by_key[key] = baseline
        try:
            baseline_errors = _baseline_trial_errors(
                baseline,
                decision=decision,
                controller=key[0],
            )
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            baseline_errors = [f"validator rejected malformed baseline: {error}"]
        errors.extend(f"{key}: {error}" for error in baseline_errors)
    expected_baselines = {
        (controller, trial_index)
        for controller in ("v2-full", "official_adaptive")
        for trial_index in (0, 1)
    }
    if set(baseline_by_key) != expected_baselines:
        errors.append("state external baseline coverage differs")
    return errors


def _state_collection_job(job: dict[str, Any]) -> dict[str, Any]:
    qualified = _read_json(Path(job["qualification_file"]))
    qualification_errors = _qualification_artifact_errors(
        qualified,
        run_fingerprint=str(job["run_fingerprint"]),
    )
    if qualification_errors:
        raise ValueError(
            "v3-S3 qualification artifact is invalid: "
            + "; ".join(qualification_errors[:5])
        )
    decision = dict(qualified["decision"])
    output = Path(job["state_file"])
    if bool(job["resume"]) and output.is_file():
        existing = _resume_completed_artifact(
            output,
            run_fingerprint=str(job["run_fingerprint"]),
            validator=_state_artifact_errors,
            label="v3-S3 state",
            selected_row=decision,
            qualified=qualified,
        )
        if existing is not None:
            return {"state_file": str(output), "status": "resumed"}
    base_sequences = list(balanced_sequence_templates(str(decision["state_id"])))
    features = [_sequence_feature(qualified, templates) for templates in base_sequences]
    trials = [
        _sequence_trial(qualified, templates, trial_index)
        for templates in base_sequences
        for trial_index in (0, 1)
    ]
    ambiguous = _ambiguous_sequences(trials)
    if ambiguous:
        additions = _ambiguous_additional_sequences(
            qualified, str(decision["state_id"])
        )
        features.extend(_sequence_feature(qualified, templates) for templates in additions)
        trials.extend(
            _sequence_trial(qualified, templates, trial_index)
            for templates in additions
            for trial_index in (0, 1)
        )
        grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        template_lookup = {
            sequence_id(templates): templates for templates in (*base_sequences, *additions)
        }
        for row in trials:
            grouped[str(row["sequence_id"])].append(row)
        top = sorted(
            grouped,
            key=lambda key: (-_sequence_efficiency(grouped[key]), key),
        )[:6]
        trials.extend(
            _sequence_trial(qualified, template_lookup[key], trial_index)
            for key in top
            for trial_index in (2, 3)
        )
    baselines = [
        _baseline_trial(qualified, controller, trial_index)
        for controller in ("v2-full", "official_adaptive")
        for trial_index in (0, 1)
    ]
    payload = {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "run_fingerprint": str(job["run_fingerprint"]),
        "complete": True,
        "state_id": str(decision["state_id"]),
        "ambiguous": ambiguous,
        "features": features,
        "trials": trials,
        "external_baselines": baselines,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output)
    return {
        "state_file": str(output),
        "status": "ok",
        "peak_memory_bytes": (
            int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss) * 1024
            if _resource is not None
            else 0
        ),
    }


def _trial_semantics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_id": str(row["state_id"]),
        "sequence_id": str(row["sequence_id"]),
        "trial_index": int(row["trial_index"]),
        "initial_fingerprint": str(row["initial_fingerprint"]),
        "final_fingerprint": str(row["final_fingerprint"]),
        "initial_repair_fingerprint": str(row["initial_repair_fingerprint"]),
        "final_repair_fingerprint": str(row["final_repair_fingerprint"]),
        "conflict_trajectory": list(map(int, row["conflict_trajectory"])),
        "conflict_reduction": int(row["conflict_reduction"]),
        "no_progress": bool(row["no_progress"]),
        "feasible": bool(row["feasible"]),
        "stop_reason": str(row["stop_reason"]),
        "truncated": bool(row["truncated"]),
        "generated": int(row["generated"]),
        "expanded": int(row["expanded"]),
        "reopened": int(row["reopened"]),
        "runs": int(row["runs"]),
        "steps": [
            {
                "step": int(step["step"]),
                "template": dict(step["template"]),
                "template_valid": bool(step["template_valid"]),
                "executed": bool(step["executed"]),
                "candidate_id": step.get("candidate_id"),
                "agents": list(map(int, step.get("agents", ()))),
                "action": dict(step.get("action") or {}),
                "conflicts_before": step.get("conflicts_before"),
                "conflicts_after": step.get("conflicts_after"),
                "conflict_reduction": step.get("conflict_reduction"),
                "repair_outcome": step.get("repair_outcome"),
                "replan_success": bool(step.get("replan_success")),
                "before_fingerprint": step.get("before_fingerprint"),
                "after_fingerprint": step.get("after_fingerprint"),
                "before_repair_fingerprint": step.get(
                    "before_repair_fingerprint"
                ),
                "after_repair_fingerprint": step.get("after_repair_fingerprint"),
                "low_level_delta": dict(step.get("low_level_delta") or {}),
                "terminated": bool(step.get("terminated")),
                "truncated": bool(step.get("truncated")),
                "done": bool(step.get("done")),
            }
            for step in row["steps"]
        ],
    }


def _strict_retest_job(job: dict[str, Any]) -> dict[str, Any]:
    qualified = _read_json(Path(job["qualification_file"]))
    qualification_errors = _qualification_artifact_errors(
        qualified,
        run_fingerprint=str(job["run_fingerprint"]),
    )
    if qualification_errors:
        raise ValueError(
            "strict retest qualification is invalid: "
            + "; ".join(qualification_errors[:5])
        )
    expected = dict(job["expected"])
    templates = tuple(
        S3ActionTemplate.from_payload(value) for value in expected["templates"]
    )
    observed = _sequence_trial(qualified, templates, int(expected["trial_index"]))
    expected_semantics = _trial_semantics(expected)
    observed_semantics = _trial_semantics(observed)
    return {
        "state_id": str(expected["state_id"]),
        "sequence_id": str(expected["sequence_id"]),
        "trial_index": int(expected["trial_index"]),
        "passed": observed_semantics == expected_semantics,
        "expected_fingerprint": _fingerprint(expected_semantics),
        "observed_fingerprint": _fingerprint(observed_semantics),
    }


def _strict_retest(
    *,
    selected: list[dict[str, Any]],
    state_files: list[Path],
    output_root: Path,
    controller_bundle: Path,
    run_fingerprint: str,
    fraction: float = 0.15,
) -> dict[str, Any]:
    report_path = output_root / "strict_retest_report.json"
    file_by_state = {}
    input_files = []
    for path in sorted(map(Path, state_files)):
        payload = _read_json(path)
        state_id = str(payload["state_id"])
        file_by_state[state_id] = (path, payload)
        input_files.append(
            {
                "kind": "state",
                "state_id": state_id,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    for row in sorted(selected, key=lambda value: str(value["state_id"])):
        qualification_path = Path(str(row["qualification_file"])).resolve()
        input_files.append(
            {
                "kind": "qualification",
                "state_id": str(row["state_id"]),
                "path": str(qualification_path),
                "sha256": sha256_file(qualification_path),
            }
        )
    strict_input_identity = {
        "run_fingerprint": str(run_fingerprint),
        "fraction": float(fraction),
        "input_files": input_files,
        "controller_bundle_fingerprint": _directory_content_fingerprint(
            Path(controller_bundle)
        ),
    }
    strict_input_sha256 = _fingerprint(strict_input_identity)
    if report_path.is_file():
        try:
            previous = _read_json(report_path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            previous = None
        if (
            isinstance(previous, dict)
            and str(previous.get("schema")) == V3_S3_COLLECTION_SCHEMA
            and previous.get("schema_version") == V3_S3_COLLECTION_VERSION
            and str(previous.get("run_fingerprint")) == str(run_fingerprint)
            and str(previous.get("input_sha256")) == strict_input_sha256
            and previous.get("input_identity") == strict_input_identity
            and isinstance(previous.get("passed"), bool)
        ):
            return previous
    count = max(1, math.ceil(float(fraction) * len(selected)))
    sample = sorted(
        selected,
        key=lambda row: _stable_key(
            ("strict-retest", row["state_id"], row["before_repair_fingerprint"])
        ),
    )[:count]
    jobs = []
    for row in sample:
        _path, payload = file_by_state[str(row["state_id"])]
        expected = min(
            payload["trials"],
            key=lambda trial: (str(trial["sequence_id"]), int(trial["trial_index"])),
        )
        jobs.append(
            {
                "state_id": str(row["state_id"]),
                "qualification_file": str(row["qualification_file"]),
                "expected": expected,
                "run_fingerprint": str(run_fingerprint),
            }
        )
    completed, errors = _run_jobs(
        jobs,
        workers=1,
        controller_bundle=controller_bundle,
        status_path=output_root / "strict_retest_status.json",
        phase="strict-retest",
        job_function=_strict_retest_job,
    )
    mismatches = [row for row in completed if not bool(row["passed"])]
    report = {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "run_fingerprint": str(run_fingerprint),
        "input_sha256": strict_input_sha256,
        "input_identity": strict_input_identity,
        "fraction": float(fraction),
        "requested_state_count": count,
        "completed_state_count": len(completed),
        "error_count": len(errors),
        "mismatch_count": len(mismatches),
        "errors": errors,
        "mismatches": mismatches,
        "passed": len(completed) == count and not errors and not mismatches,
    }
    _write_json(report_path, report)
    return report


def _job_progress(
    completed: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    total: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    resumed = sum(str(row.get("status")) == "resumed" for row in completed)
    finished = len(completed) + len(errors)
    processed = finished - resumed
    rate = (
        60.0 * processed / elapsed_seconds
        if elapsed_seconds > 0.0 and processed > 0
        else 0.0
    )
    return {
        "finished_states": finished,
        "resumed_states": resumed,
        "processed_states": processed,
        "states_per_minute": rate,
        "estimated_remaining_seconds": (
            60.0 * (int(total) - finished) / rate if rate > 0.0 else None
        ),
    }


def _run_jobs(
    jobs: list[dict[str, Any]],
    *,
    workers: int,
    controller_bundle: Path,
    status_path: Path,
    phase: str,
    job_function: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed = []
    errors = []
    started = time.perf_counter()
    cpu_sets = tuple(isolated_lane_cpu_sets(workers))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_initialize,
        initargs=(cpu_sets, str(controller_bundle)),
    ) as pool:
        futures = {pool.submit(job_function, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "state_id": str(job.get("state_id") or job.get("decision", {}).get("state_id")),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            rejected = sum(
                str(row.get("status")) == "rejected" for row in completed
            )
            elapsed = time.perf_counter() - started
            progress = _job_progress(
                completed,
                errors,
                total=len(jobs),
                elapsed_seconds=elapsed,
            )
            _write_json(
                status_path,
                {
                    "schema": V3_S3_COLLECTION_SCHEMA,
                    "schema_version": V3_S3_COLLECTION_VERSION,
                    "phase": phase,
                    "status": "running",
                    "completed_states": len(completed),
                    "total_states": len(jobs),
                    "error_states": len(errors),
                    "rejected_states": rejected,
                    "resumed_states": progress["resumed_states"],
                    "processed_states": progress["processed_states"],
                    "states_per_minute": progress["states_per_minute"],
                    "estimated_remaining_seconds": progress[
                        "estimated_remaining_seconds"
                    ],
                },
            )
    elapsed = time.perf_counter() - started
    rejected = sum(str(row.get("status")) == "rejected" for row in completed)
    progress = _job_progress(
        completed,
        errors,
        total=len(jobs),
        elapsed_seconds=elapsed,
    )
    _write_json(
        status_path,
        {
            "schema": V3_S3_COLLECTION_SCHEMA,
            "schema_version": V3_S3_COLLECTION_VERSION,
            "phase": phase,
            "status": "error" if errors else "complete",
            "completed_states": len(completed),
            "total_states": len(jobs),
            "error_states": len(errors),
            "rejected_states": rejected,
            "resumed_states": progress["resumed_states"],
            "processed_states": progress["processed_states"],
            "states_per_minute": progress["states_per_minute"],
            "estimated_remaining_seconds": 0.0,
        },
    )
    return completed, errors


def _select_qualified_states(
    qualification_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available = []
    for result in qualification_results:
        if str(result.get("status")) not in {"ok", "resumed"}:
            continue
        payload = _read_json(Path(result["state_file"]))
        available.append(
            {
                **dict(payload["decision"]),
                "qualification_file": str(Path(result["state_file"]).resolve()),
            }
        )
    plan = _adaptive_selection_plan(available)
    if not bool(plan["passed"]):
        raise ValueError(
            f"v3-S3 qualified overall coverage is insufficient: {plan['errors']}"
        )
    selected = []
    for split, split_plan in dict(plan["by_split"]).items():
        rows = [row for row in available if str(row["split"]) == split]
        selected.extend(
            _balanced_state_take(
                rows,
                int(split_plan["target_state_count"]),
                split=split,
            )
        )
    expected = int(plan["target_state_count"])
    if len(selected) != expected:
        raise ValueError(
            "v3-S3 adaptive selection did not produce its registered target: "
            f"expected={expected}, observed={len(selected)}"
        )
    return selected, {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "qualified_state_count": len(available),
        "selected_state_count": len(selected),
        "target_state_cap": S3_TARGET_STATE_CAP,
        "selection_policy": str(plan["policy"]),
        "selection_plan": plan,
        "selected_by_split": dict(
            collections.Counter(str(row["split"]) for row in selected)
        ),
        "selected_by_agent_count": dict(
            collections.Counter(int(row["agent_count"]) for row in selected)
        ),
        "selected_by_layout": dict(
            collections.Counter(str(row["layout_mode"]) for row in selected)
        ),
        "shortages": [],
        "passed": len(selected) == expected,
    }


def _stream_manifests(
    state_files: Iterable[Path], output_root: Path
) -> dict[str, int]:
    paths = {
        "features": output_root / "sequence_features.jsonl",
        "trials": output_root / "sequence_trials.jsonl",
        "baselines": output_root / "external_baselines.jsonl",
    }
    partials = {
        name: path.with_name(path.name + ".partial") for name, path in paths.items()
    }
    streams = {}
    counts = collections.Counter()
    succeeded = False
    try:
        for name, path in partials.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            streams[name] = path.open("w", encoding="utf-8", newline="\n")
        for state_file in sorted(map(Path, state_files)):
            payload = _read_json(state_file)
            for name, key in (
                ("features", "features"),
                ("trials", "trials"),
                ("baselines", "external_baselines"),
            ):
                for row in payload[key]:
                    streams[name].write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    counts[name] += 1
        succeeded = True
    finally:
        for stream in streams.values():
            stream.close()
    if succeeded:
        for name, path in paths.items():
            os.replace(partials[name], path)
    return dict(counts)


def _coverage(
    selected: list[dict[str, Any]],
    state_files: Iterable[Path],
    *,
    run_fingerprint: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    covered: set[str] = set()
    selected_by_id = {str(row.get("state_id")): row for row in selected}
    if len(selected_by_id) != len(selected):
        errors.append("selected states contain duplicate state_id values")
    observed_state_ids: list[str] = []
    trial_count = 0
    feature_count = 0
    baseline_count = 0
    for path in map(Path, state_files):
        try:
            payload = _read_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            errors.append(f"{path}: state artifact is unreadable: {error}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path}: state artifact is not an object")
            continue
        state_id = str(payload.get("state_id", ""))
        observed_state_ids.append(state_id)
        covered.add(state_id)
        selected_row = selected_by_id.get(state_id)
        if selected_row is None:
            errors.append(f"{state_id}: state file is not part of the selected cohort")
            continue
        qualification_file = selected_row.get("qualification_file")
        if not qualification_file:
            errors.append(f"{state_id}: missing qualification file")
            continue
        try:
            qualified = _read_json(Path(str(qualification_file)))
            state_errors = _state_artifact_errors(
                payload,
                run_fingerprint=(
                    str(run_fingerprint)
                    if run_fingerprint is not None
                    else str(payload.get("run_fingerprint", ""))
                ),
                selected_row=selected_row,
                qualified=qualified,
            )
        except (KeyError, OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            errors.append(
                f"{state_id}: semantic validation failed: "
                f"{type(error).__name__}: {error}"
            )
        else:
            errors.extend(f"{state_id}: {error}" for error in state_errors)
        features = payload.get("features")
        trials = payload.get("trials")
        baselines = payload.get("external_baselines")
        feature_count += len(features) if isinstance(features, list) else 0
        trial_count += len(trials) if isinstance(trials, list) else 0
        baseline_count += len(baselines) if isinstance(baselines, list) else 0
    expected = set(selected_by_id)
    if len(observed_state_ids) != len(set(observed_state_ids)):
        errors.append("state files contain duplicate state_id values")
    return {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "state_count": len(selected),
        "covered_state_count": len(covered),
        "feature_count": feature_count,
        "trial_count": trial_count,
        "external_baseline_count": baseline_count,
        "error_count": len(errors),
        "errors": errors,
        "passed": covered == expected and not errors,
    }


def collect_v3_s3_data(
    *,
    source_roots: dict[str, list[Path]],
    output: str | Path,
    controller_bundle: str | Path,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("v3-S3 workers must be positive")
    output_root = Path(output).resolve()
    controller_path = Path(controller_bundle).resolve()
    decisions = source_decisions(source_roots)
    pool, pool_report = qualification_pool(decisions)
    collection_producer_identity = producer_identity(
        project_root=Path(__file__).resolve().parents[1],
        source_files=V3_S3_COLLECTION_PRODUCER_FILES,
        native_required=True,
        package_names=("numpy", "scikit-learn"),
        optional_package_names=("joblib",),
    )
    identity = {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "trace_replay_contract": TRACE_REPLAY_CONTRACT,
        "source_run_fingerprints": sorted(
            {
                str(row["source_run_fingerprint"])
                for row in pool
            }
        ),
        "qualification_pool_fingerprint": _fingerprint(
            [
                (row["state_id"], row["before_repair_fingerprint"])
                for row in pool
            ]
        ),
        "controller_bundle": str(controller_path),
        "controller_bundle_fingerprint": _directory_content_fingerprint(
            controller_path
        ),
        "producer_identity": collection_producer_identity,
        "producer_identity_fingerprint": _fingerprint(
            collection_producer_identity
        ),
        "workers": int(workers),
        "base_sequences_per_state": 36,
        "paired_trials": 2,
        "horizon": 3,
        "runtime_fallback": None,
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("v3-S3 collection output belongs to another run")
        if not resume:
            raise ValueError("v3-S3 collection exists; pass --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    _write_json(output_root / "qualification_pool_report.json", pool_report)
    _write_jsonl(output_root / "qualification_pool.jsonl", pool)
    qualification_jobs = []
    for row in pool:
        key = _fingerprint((row["state_id"], row["before_repair_fingerprint"]))[:20]
        qualification_jobs.append(
            {
                "state_id": row["state_id"],
                "decision": row,
                "state_file": str(output_root / "qualification" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    qualified, qualification_errors = _run_jobs(
        qualification_jobs,
        workers=workers,
        controller_bundle=controller_path,
        status_path=output_root / "status.json",
        phase="qualification",
        job_function=_qualification_job,
    )
    qualification_rejections = [
        row for row in qualified if str(row.get("status")) == "rejected"
    ]
    selected, selection_report = _select_qualified_states(qualified)
    _write_jsonl(output_root / "state_selection.jsonl", selected)
    _write_json(output_root / "state_selection_report.json", selection_report)
    collection_jobs = []
    for row in selected:
        key = _fingerprint((row["state_id"], row["before_repair_fingerprint"]))[:20]
        collection_jobs.append(
            {
                "state_id": row["state_id"],
                "qualification_file": row["qualification_file"],
                "state_file": str(
                    output_root / "states" / str(row["split"]) / f"{key}.json"
                ),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    completed, collection_errors = _run_jobs(
        collection_jobs,
        workers=workers,
        controller_bundle=controller_path,
        status_path=output_root / "status.json",
        phase="sequence-collection",
        job_function=_state_collection_job,
    )
    state_files = [Path(row["state_file"]) for row in completed]
    strict_retest = (
        _strict_retest(
            selected=selected,
            state_files=state_files,
            output_root=output_root,
            controller_bundle=controller_path,
            run_fingerprint=run_fingerprint,
        )
        if not collection_errors and len(completed) == len(selected)
        else {
            "schema": V3_S3_COLLECTION_SCHEMA,
            "schema_version": V3_S3_COLLECTION_VERSION,
            "run_fingerprint": run_fingerprint,
            "passed": False,
            "skipped": "incomplete sequence collection",
        }
    )
    state_collection_complete = (
        not collection_errors and len(completed) == len(selected)
    )
    manifest_counts = (
        _stream_manifests(state_files, output_root)
        if state_collection_complete
        else {}
    )
    coverage = _coverage(
        selected, state_files, run_fingerprint=run_fingerprint
    )
    _write_json(output_root / "coverage_report.json", coverage)
    report = {
        "schema": V3_S3_COLLECTION_SCHEMA,
        "schema_version": V3_S3_COLLECTION_VERSION,
        "run_fingerprint": run_fingerprint,
        "qualification_pool": pool_report,
        "qualification_completed_count": len(qualified),
        "qualification_error_count": len(qualification_errors),
        "qualification_errors": qualification_errors,
        "qualification_rejected_count": len(qualification_rejections),
        "qualification_rejections": qualification_rejections,
        "selection": selection_report,
        "requested_state_count": len(selected),
        "completed_state_count": len(completed),
        "error_state_count": len(collection_errors),
        "errors": collection_errors,
        "manifest_counts": manifest_counts,
        "coverage": coverage,
        "strict_retest": strict_retest,
        "sequence_features_sha256": (
            sha256_file(output_root / "sequence_features.jsonl")
            if state_collection_complete
            else None
        ),
        "sequence_trials_sha256": (
            sha256_file(output_root / "sequence_trials.jsonl")
            if state_collection_complete
            else None
        ),
        "external_baselines_sha256": (
            sha256_file(output_root / "external_baselines.jsonl")
            if state_collection_complete
            else None
        ),
        "parallel_runtime": parallel_runtime_metadata(workers),
        "complete": not qualification_errors
        and not collection_errors
        and len(completed) == len(selected)
        and bool(coverage["passed"])
        and bool(strict_retest["passed"]),
    }
    _write_json(output_root / "collection_report.json", report)
    _write_json(
        output_root / "status.json",
        {
            "schema": V3_S3_COLLECTION_SCHEMA,
            "schema_version": V3_S3_COLLECTION_VERSION,
            "phase": "complete",
            "status": "complete" if report["complete"] else "error",
            "completed_states": len(completed),
            "total_states": len(selected),
            "error_states": len(collection_errors),
        },
    )
    return report


def revalidate_v3_s3_collection(output: str | Path) -> dict[str, Any]:
    output_root = Path(output).resolve()
    report_path = output_root / "collection_report.json"
    selection_path = output_root / "state_selection.jsonl"
    strict_retest_path = output_root / "strict_retest_report.json"
    for required in (report_path, selection_path, strict_retest_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    previous = _read_json(report_path)
    if (
        not isinstance(previous, dict)
        or str(previous.get("schema")) != V3_S3_COLLECTION_SCHEMA
        or previous.get("schema_version") != V3_S3_COLLECTION_VERSION
        or not str(previous.get("run_fingerprint", ""))
    ):
        raise ValueError(
            "legacy or malformed v3-S3 collection is read-only; "
            "collect the current schema into a new output directory"
        )
    selected = _read_jsonl(selection_path)
    state_files = sorted((output_root / "states").rglob("*.json"))
    strict_retest = _read_json(strict_retest_path)
    if (
        not isinstance(strict_retest, dict)
        or str(strict_retest.get("schema")) != V3_S3_COLLECTION_SCHEMA
        or strict_retest.get("schema_version") != V3_S3_COLLECTION_VERSION
        or str(strict_retest.get("run_fingerprint"))
        != str(previous["run_fingerprint"])
        or not str(strict_retest.get("input_sha256", ""))
        or not isinstance(strict_retest.get("passed"), bool)
    ):
        raise ValueError("v3-S3 strict-retest report is invalid")
    coverage = _coverage(
        selected,
        state_files,
        run_fingerprint=str(previous.get("run_fingerprint", "")),
    )
    _write_json(output_root / "coverage_report.json", coverage)

    state_collection_complete = len(state_files) == len(selected)
    state_collection_valid = state_collection_complete and bool(coverage["passed"])
    manifest_counts = (
        _stream_manifests(state_files, output_root)
        if state_collection_valid
        else {}
    )
    qualification_errors = list(previous.get("qualification_errors") or ())
    collection_errors = list(previous.get("errors") or ())
    complete = (
        not qualification_errors
        and not collection_errors
        and state_collection_complete
        and bool(coverage["passed"])
        and bool(strict_retest.get("passed"))
    )
    report = {
        **previous,
        "requested_state_count": len(selected),
        "completed_state_count": len(state_files),
        "error_state_count": len(collection_errors),
        "manifest_counts": manifest_counts,
        "coverage": coverage,
        "strict_retest": strict_retest,
        "sequence_features_sha256": (
            sha256_file(output_root / "sequence_features.jsonl")
            if state_collection_valid
            else None
        ),
        "sequence_trials_sha256": (
            sha256_file(output_root / "sequence_trials.jsonl")
            if state_collection_valid
            else None
        ),
        "external_baselines_sha256": (
            sha256_file(output_root / "external_baselines.jsonl")
            if state_collection_valid
            else None
        ),
        "revalidation": {
            "schema": "lns2.v3_s3_revalidation.v2",
            "schema_version": V3_S3_COLLECTION_VERSION,
            "coverage_contract": "ambiguous-additional-sequences-v2",
            "validator_sha256": sha256_file(Path(__file__).resolve()),
            "previous_complete": bool(previous.get("complete")),
        },
        "complete": complete,
    }
    _write_json(report_path, report)
    _write_json(
        output_root / "status.json",
        {
            "schema": V3_S3_COLLECTION_SCHEMA,
            "schema_version": V3_S3_COLLECTION_VERSION,
            "phase": "complete",
            "status": "complete" if complete else "error",
            "completed_states": len(state_files),
            "total_states": len(selected),
            "error_states": len(collection_errors),
        },
    )
    return report


__all__ = [
    "S3_AGENT_COUNTS",
    "S3_LAYOUTS",
    "S3_SPLIT_CELL_QUOTAS",
    "S3_SPLIT_MINIMUM_COUNTS",
    "S3_SPLIT_TARGET_COUNTS",
    "S3_STRATUM_QUOTAS",
    "S3_TARGET_STATE_CAP",
    "V3_S3_COLLECTION_SCHEMA",
    "V3_S3_COLLECTION_VERSION",
    "audit_v3_s3_parallelism",
    "assign_source_strata",
    "collect_v3_s3_data",
    "qualification_pool",
    "revalidate_v3_s3_collection",
    "source_decisions",
    "temporal_context",
]
