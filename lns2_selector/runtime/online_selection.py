from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Iterable

from experiments.context_audit import _pair_vector
from experiments.neighborhood_candidates import (
    select_representative_neighborhood_groups,
    select_representative_neighborhoods,
)
from experiments.neighborhood_features import (
    _feature_profiles_from_shared,
    candidate_feature_cache,
    state_dynamic_features,
    static_context_features,
)
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    select_seed_agents,
    state_fingerprint,
)
from experiments.state_analysis import (
    StateAnalysis,
    StaticGridAnalysis,
    analyze_state,
    analyze_static_grid,
)
from lns2_selector.runtime.topology_candidates import (
    generate_topology_boundary_candidates,
    merge_topology_anchor_candidates,
)


CONTROLLER_RUNTIMES = ("reference", "optimized", "auto")

_TOPOLOGY_BOUNDARY_LEGACY_CONFIG = {
    "enabled": True,
    "generator_id": "stride-topoboundary-v1",
    "neighborhood_size": 16,
    "core_budget": 4,
    "maximum_added_candidates": 2,
}
_TOPOLOGY_BOUNDARY_STATIC_CACHE_CONFIG = {
    **_TOPOLOGY_BOUNDARY_LEGACY_CONFIG,
    "runtime_id": "stride-boundary-static-cache-v1",
    "static_grid_cache": True,
}
_TOPOLOGY_BOUNDARY_CHEAP_GATE_CONFIG = {
    **_TOPOLOGY_BOUNDARY_LEGACY_CONFIG,
    "runtime_id": "stride-boundary-cheap-gate-v1",
    "static_grid_cache": True,
    "activation_gate": {
        "gate_id": "stride-boundary-map-topology-v1",
        "minimum_low_degree_cell_ratio": 0.06,
    },
}
_TOPOLOGY_BOUNDARY_MAP_GATE_CONFIG = {
    **_TOPOLOGY_BOUNDARY_LEGACY_CONFIG,
    "runtime_id": "stride-boundary-map-gate-v1",
    "static_grid_cache": True,
    "activation_gate": {
        "gate_id": "stride-boundary-map-topology-v1",
        "minimum_low_degree_cell_ratio": 0.06,
    },
}
_TOPOLOGY_BOUNDARY_STALL_GUARD_CONFIG = {
    **_TOPOLOGY_BOUNDARY_MAP_GATE_CONFIG,
    "runtime_id": "stride-boundary-stall-guard-v1",
    "phase_guard": {
        "gate_id": "stride-boundary-stall-guard-v1",
        "maximum_no_progress_streak": 5,
    },
}
_TOPOLOGY_BOUNDARY_PHASE_GUARD_CONFIG = {
    **_TOPOLOGY_BOUNDARY_CHEAP_GATE_CONFIG,
    "runtime_id": "stride-boundary-phase-guard-v2",
    "phase_guard": {
        "gate_id": "stride-boundary-phase-guard-v2",
        "low_conflict_pair_threshold": 2,
        "low_conflict_no_progress_streak": 2,
        "maximum_no_progress_streak": 5,
        "minimum_remaining_wall_seconds": 5.0,
    },
}


def validate_topology_boundary_augmentation(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    result = dict(value)
    if result not in (
        _TOPOLOGY_BOUNDARY_LEGACY_CONFIG,
        _TOPOLOGY_BOUNDARY_STATIC_CACHE_CONFIG,
        _TOPOLOGY_BOUNDARY_CHEAP_GATE_CONFIG,
        _TOPOLOGY_BOUNDARY_MAP_GATE_CONFIG,
        _TOPOLOGY_BOUNDARY_STALL_GUARD_CONFIG,
        _TOPOLOGY_BOUNDARY_PHASE_GUARD_CONFIG,
    ):
        raise ValueError("unsupported topology-boundary runtime augmentation")
    return result


class ClosedLoopExecutionError(RuntimeError):
    def __init__(
        self,
        kind: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.details = dict(details or {})


def proposal_random_seed(
    task_id: str,
    solver_seed: int,
    state_hash: str,
    decision_index: int,
    seed_agent: int,
    heuristic: str,
    size: int,
    trial_index: int,
) -> int:
    return proposal_random_seeds(
        task_id,
        solver_seed,
        state_hash,
        decision_index,
        [(seed_agent, heuristic, size, trial_index)],
    )[0]


def proposal_random_seeds(
    task_id: str,
    solver_seed: int,
    state_hash: str,
    decision_index: int,
    requests: Iterable[tuple[int, str, int, int]],
) -> list[int]:
    """Return the exact v1 proposal seeds without rebuilding generic dicts.

    The payload below is the byte-for-byte ``sort_keys=True`` JSON form used by
    ``_fingerprint``.  Encoding the invariant strings once removes most of the
    old 288-request Python bookkeeping while preserving every historical seed.
    """

    task_json = json.dumps(str(task_id), ensure_ascii=True, separators=(",", ":"))
    state_json = json.dumps(
        str(state_hash), ensure_ascii=True, separators=(",", ":")
    )
    namespace_json = '"closed-loop-proposal-v1"'
    result: list[int] = []
    for seed_agent, heuristic, size, trial_index in requests:
        heuristic_json = json.dumps(
            str(heuristic), ensure_ascii=True, separators=(",", ":")
        )
        payload = (
            '{"decision_index":'
            + str(int(decision_index))
            + ',"heuristic":'
            + heuristic_json
            + ',"namespace":'
            + namespace_json
            + ',"seed_agent":'
            + str(int(seed_agent))
            + ',"size":'
            + str(int(size))
            + ',"solver_seed":'
            + str(int(solver_seed))
            + ',"state_fingerprint":'
            + state_json
            + ',"task_id":'
            + task_json
            + ',"trial_index":'
            + str(int(trial_index))
            + "}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        result.append(int.from_bytes(digest[:8], "big") % (2**31))
    return result


def repair_random_seed(
    task_id: str,
    solver_seed: int,
    state_hash: str,
    decision_index: int,
    candidate_id: str,
    forbidden: Iterable[int],
) -> int:
    value = int(
        _fingerprint(
            {
                "namespace": "closed-loop-explicit-repair-v1",
                "task_id": task_id,
                "solver_seed": solver_seed,
                "state_fingerprint": state_hash,
                "decision_index": decision_index,
                "candidate_id": candidate_id,
            }
        )[:16],
        16,
    ) % (2**31)
    excluded = set(map(int, forbidden))
    while value in excluded:
        value = (value + 1) % (2**31)
    return value


def pp_replay_random_seed(
    task_id: str,
    solver_seed: int,
    state_hash: str,
    decision_index: int,
    route: str,
) -> int:
    """Return a controller-independent PP seed for paired trace replay.

    ``route`` remains in the public signature for backward-compatible callers,
    but deliberately does not enter the seed.  At the same task, solver state,
    and decision index, official Adaptive and learned selectors must therefore
    give PP the same random stream; only the selected neighborhood may differ.
    """

    if not isinstance(route, str) or not route:
        raise ValueError("PP replay route must be a non-empty string")

    return int(
        _fingerprint(
            {
                "namespace": "closed-loop-pp-replay-v1",
                "task_id": str(task_id),
                "solver_seed": int(solver_seed),
                "state_fingerprint": str(state_hash),
                "decision_index": int(decision_index),
            }
        )[:16],
        16,
    ) % (2**31)


def online_candidate_rows(
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    static_grid: StaticGridAnalysis | None = None,
) -> list[dict[str, Any]]:
    analysis = analyze_state(state, static_grid=static_grid)
    dynamic = state_dynamic_features(state, analysis)
    context = static_context_features(state)
    feature_cache = candidate_feature_cache(state, analysis)
    rows = []
    state_hash = state_fingerprint(state)
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        rows.append(
            {
                "state_id": state_hash,
                "candidate_id": candidate_id,
                "candidate_key": candidate_id,
                "features": _feature_profiles_from_shared(
                    state,
                    analysis,
                    candidate,
                    dynamic=dynamic,
                    context=context,
                    feature_cache=feature_cache,
                ),
            }
        )
    return rows


def score_online_candidates(
    rows: list[dict[str, Any]], model: Any
) -> tuple[int, list[float], float]:
    if not rows:
        raise ValueError("cannot score an empty candidate pool")
    direct_scorer = getattr(model, "score_candidates", None)
    direct_scores = direct_scorer(rows) if callable(direct_scorer) else None
    scores = (
        list(map(float, direct_scores))
        if direct_scores is not None
        else [0.0] * len(rows)
    )
    if len(scores) != len(rows):
        raise ValueError("direct candidate scorer returned the wrong number of scores")
    vectors = []
    reverse_vectors = []
    pairs = []
    if direct_scores is None:
        pair_vectors = getattr(model, "pair_vectors", None)
        if callable(pair_vectors):
            vectors, reverse_vectors, pairs = pair_vectors(rows)
        else:
            pair_vector = getattr(model, "pair_vector", None)
            for left in range(len(rows)):
                for right in range(left + 1, len(rows)):
                    if callable(pair_vector):
                        vectors.append(pair_vector(rows[left], rows[right]))
                        reverse_vectors.append(pair_vector(rows[right], rows[left]))
                    else:
                        vectors.append(
                            _pair_vector(
                                rows[left], rows[right], model.profile, model.feature_names
                            )
                        )
                        reverse_vectors.append(
                            _pair_vector(
                                rows[right], rows[left], model.profile, model.feature_names
                            )
                        )
                    pairs.append((left, right))
    if vectors:
        predict_positive = getattr(model, "predict_positive", None)
        if callable(predict_positive):
            forward = predict_positive(vectors)
            reverse = predict_positive(reverse_vectors)
        else:
            import numpy as np

            forward = model.estimator.predict_proba(np.asarray(vectors, dtype=float))[:, 1]
            reverse = model.estimator.predict_proba(
                np.asarray(reverse_vectors, dtype=float)
            )[:, 1]
        probabilities = [
            (float(first) + (1.0 - float(second))) / 2.0
            for first, second in zip(forward, reverse)
        ]
        for probability, (left, right) in zip(probabilities, pairs):
            scores[left] += float(probability)
            scores[right] += 1.0 - float(probability)
    # Native sklearn, Python and the C++ portable evaluator can differ by a
    # handful of floating-point ulps. Treat those numerically identical scores
    # as ties so the candidate hash remains the cross-platform decision rule.
    stable_scores = [round(score, 12) for score in scores]
    order = sorted(
        range(len(rows)),
        key=lambda index: (-stable_scores[index], str(rows[index]["candidate_key"])),
    )
    margin = (
        stable_scores[order[0]] - stable_scores[order[1]]
        if len(order) > 1
        else stable_scores[order[0]]
    )
    return order[0], scores, margin


def pairwise_win_probability(
    rows: list[dict[str, Any]], model: Any, left: int, right: int
) -> float:
    """Return the symmetric probability that ``left`` beats ``right``.

    This uses the same forward/reverse averaging rule as
    :func:`score_online_candidates`, but evaluates only the requested pair.
    Keeping the helper on the runtime path makes guarded selectors portable
    across sklearn, Python compact-tree, and native compact-tree backends.
    """

    if not rows or left == right:
        raise ValueError("pairwise evidence requires two distinct candidates")
    if min(left, right) < 0 or max(left, right) >= len(rows):
        raise IndexError("pairwise evidence candidate index is out of range")
    pair_vector = getattr(model, "pair_vector", None)
    if callable(pair_vector):
        forward_vector = pair_vector(rows[left], rows[right])
        reverse_vector = pair_vector(rows[right], rows[left])
    else:
        forward_vector = _pair_vector(
            rows[left], rows[right], model.profile, model.feature_names
        )
        reverse_vector = _pair_vector(
            rows[right], rows[left], model.profile, model.feature_names
        )
    predict_positive = getattr(model, "predict_positive", None)
    if callable(predict_positive):
        forward = float(predict_positive([forward_vector])[0])
        reverse = float(predict_positive([reverse_vector])[0])
    else:
        import numpy as np

        forward = float(
            model.estimator.predict_proba(
                np.asarray([forward_vector], dtype=float)
            )[0, 1]
        )
        reverse = float(
            model.estimator.predict_proba(
                np.asarray([reverse_vector], dtype=float)
            )[0, 1]
        )
    probability = (forward + (1.0 - reverse)) / 2.0
    if not math.isfinite(probability):
        raise ValueError("pairwise evidence is non-finite")
    return min(1.0, max(0.0, probability))


def feature_range_diagnostic(
    row: dict[str, Any], profile: str, ranges: dict[str, tuple[float, float]]
) -> dict[str, Any]:
    if "feature_values" in row:
        if str(row.get("feature_profile")) != profile:
            raise ValueError("dense feature row has the wrong profile")
        names = tuple(map(str, row.get("feature_names", ())))
        values = tuple(map(float, row["feature_values"]))
        if len(names) != len(values) or len(names) != len(set(names)):
            raise ValueError("dense feature row is invalid")
        features = dict(zip(names, values))
    else:
        features = dict(row["features"][profile])
    outside = []
    for name, (minimum, maximum) in ranges.items():
        value = float(features.get(name, 0.0))
        if value < minimum or value > maximum:
            outside.append(name)
    return {
        "feature_count": len(ranges),
        "outside_count": len(outside),
        "outside_fraction": len(outside) / len(ranges) if ranges else 0.0,
        "outside_features": sorted(outside),
    }


def generate_online_candidates(
    environment: Any,
    state: dict[str, Any],
    *,
    task_id: str,
    solver_seed: int,
    decision_index: int,
    proposal_config: dict[str, Any],
    state_hash: str | None = None,
    verify_full_state: bool = True,
    proposal_backend: str = "reference",
    shadow_validation: bool = False,
    seed_agents_override: Iterable[int] | None = None,
    topology_static_grid: StaticGridAnalysis | None = None,
    topology_state_analysis: StateAnalysis | None = None,
    topology_state_analysis_seconds: float = 0.0,
    topology_no_progress_streak: int = 0,
    topology_remaining_wall_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if proposal_backend not in CONTROLLER_RUNTIMES:
        raise ValueError(f"unsupported proposal backend: {proposal_backend}")
    state_hash = state_fingerprint(state) if state_hash is None else str(state_hash)
    get_revision = getattr(environment, "get_state_revision", None)
    revision_before = int(get_revision()) if callable(get_revision) else None
    request_generation_started = time.perf_counter()
    if seed_agents_override is None:
        seed_agents = select_seed_agents(
            state,
            int(proposal_config["max_seed_agents"]),
            state_hash=state_hash,
        )
    else:
        seed_agents = list(map(int, seed_agents_override))
        conflicting_agents = {
            int(agent["id"])
            for agent in state.get("agents", [])
            if int(agent.get("conflict_degree", 0)) > 0
        }
        if (
            not seed_agents
            or len(seed_agents) != len(set(seed_agents))
            or len(seed_agents) > int(proposal_config["max_seed_agents"])
            or not set(seed_agents) <= conflicting_agents
        ):
            raise ValueError("proposal seed override is not a valid conflicting subset")
    heuristics = list(map(str, proposal_config["heuristics"]))
    sizes = list(map(int, proposal_config["neighborhood_sizes"]))
    trials = int(proposal_config["trials"])
    request_specs: list[tuple[int, str, int, int]] = []
    for seed_agent in seed_agents:
        for heuristic in heuristics:
            for size in sizes:
                for trial_index in range(trials):
                    request_specs.append(
                        (seed_agent, heuristic, size, trial_index)
                    )
    random_seeds = proposal_random_seeds(
        task_id,
        solver_seed,
        state_hash,
        decision_index,
        request_specs,
    )
    requests: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for (seed_agent, heuristic, size, _trial_index), random_seed in zip(
        request_specs, random_seeds
    ):
        action = {
            "mode": "seed",
            "heuristic": heuristic,
            "seed_agent": seed_agent,
            "neighborhood_size": size,
            "random_seed": random_seed,
        }
        requests.append(
            (
                action,
                {
                    "family": f"{heuristic}:{size}",
                    "seed_agent": seed_agent,
                    "proposal_seed": random_seed,
                    "requested_size": size,
                },
            )
        )
    request_generation_seconds = time.perf_counter() - request_generation_started
    actions = [action for action, _ in requests]
    started = time.perf_counter()
    propose_compact = getattr(environment, "propose_batch_compact", None)
    propose_seed_grid = getattr(environment, "propose_seed_grid_compact", None)
    propose_seed_grid_grouped = getattr(
        environment, "propose_seed_grid_grouped", None
    )
    propose_batch = getattr(environment, "propose_batch", None)
    use_seed_grid_grouped = proposal_backend in {
        "optimized",
        "auto",
    } and callable(propose_seed_grid_grouped)
    use_seed_grid = proposal_backend in {"optimized", "auto"} and callable(
        propose_seed_grid
    )
    use_compact = proposal_backend in {"optimized", "auto"} and callable(
        propose_compact
    )
    if proposal_backend == "optimized" and not (
        callable(propose_seed_grid_grouped)
        or callable(propose_seed_grid)
        or callable(propose_compact)
    ):
        raise RuntimeError(
            "optimized controller runtime requires propose_batch_compact"
        )
    proposal_groups: list[dict[str, Any]] | None = None
    results: list[dict[str, Any]] | None = None
    if use_seed_grid_grouped:
        grouped_payload = _plain(
            propose_seed_grid_grouped(
                seed_agents,
                heuristics,
                sizes,
                random_seeds,
                trials,
            )
        )
        if not isinstance(grouped_payload, dict):
            raise RuntimeError("grouped proposal grid returned an invalid payload")
        if int(grouped_payload.get("proposal_count", -1)) != len(requests):
            raise RuntimeError("grouped proposal grid returned the wrong request count")
        invalid_indices = list(map(int, grouped_payload.get("invalid_indices", [])))
        if invalid_indices:
            raise RuntimeError("valid online proposal was rejected")
        proposal_groups = []
        for value in list(grouped_payload.get("rows", [])):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise RuntimeError("grouped proposal grid returned an invalid row")
            agents = list(map(int, value[0]))
            source_indices = list(map(int, value[1]))
            if not agents or not source_indices:
                raise RuntimeError("grouped proposal grid returned an empty group")
            if any(index < 0 or index >= len(requests) for index in source_indices):
                raise RuntimeError("grouped proposal grid returned an invalid source")
            proposal_groups.append(
                {
                    "agents": agents,
                    "sources": [requests[index][1] for index in source_indices],
                    "source_indices": source_indices,
                }
            )
        if len(proposal_groups) != int(
            grouped_payload.get("unique_neighborhood_count", -1)
        ):
            raise RuntimeError("grouped proposal grid returned the wrong group count")
        backend = "grouped_seed_grid"
    elif use_seed_grid:
        compact_results = [
            _plain(value)
            for value in propose_seed_grid(
                seed_agents,
                heuristics,
                sizes,
                random_seeds,
                trials,
            )
        ]
        results = []
        for value in compact_results:
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise RuntimeError("compact proposal grid returned an invalid row")
            results.append(
                {
                    "action_valid": bool(value[0]),
                    "generated": bool(value[1]),
                    "neighborhood": list(map(int, value[2])),
                }
            )
        backend = "compact_seed_grid"
    elif use_compact:
        compact_results = [_plain(value) for value in propose_compact(actions)]
        results = []
        for value in compact_results:
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise RuntimeError("compact proposal batch returned an invalid row")
            results.append(
                {
                    "action_valid": bool(value[0]),
                    "generated": bool(value[1]),
                    "neighborhood": list(map(int, value[2])),
                }
            )
        backend = "compact"
    elif callable(propose_batch):
        results = [_plain(value) for value in propose_batch(actions)]
        backend = "batch"
    else:
        results = [_plain(environment.propose(action)) for action in actions]
        backend = "single_fallback"
    proposal_seconds = time.perf_counter() - started
    proposal_shadow_seconds = 0.0
    if shadow_validation:
        shadow_started = time.perf_counter()
        if use_seed_grid_grouped or use_seed_grid or use_compact:
            if not callable(propose_batch):
                raise RuntimeError("proposal shadow validation requires propose_batch")
            shadow_results = [_plain(value) for value in propose_batch(actions)]
        else:
            if not callable(propose_compact):
                raise RuntimeError(
                    "proposal shadow validation requires propose_batch_compact"
                )
            shadow_results = []
            for value in [_plain(item) for item in propose_compact(actions)]:
                if not isinstance(value, (list, tuple)) or len(value) != 3:
                    raise RuntimeError("compact proposal shadow returned an invalid row")
                shadow_results.append(
                    {
                        "action_valid": bool(value[0]),
                        "generated": bool(value[1]),
                        "neighborhood": list(map(int, value[2])),
                    }
                )
        proposal_shadow_seconds = time.perf_counter() - shadow_started
        if proposal_groups is not None:
            primary_signature: list[tuple[bool, bool, tuple[int, ...]] | None] = [
                None
            ] * len(requests)
            for group in proposal_groups:
                signature = (True, True, tuple(map(int, group["agents"])))
                for index in group["source_indices"]:
                    primary_signature[index] = signature
            if any(value is None for value in primary_signature):
                raise RuntimeError("grouped proposal grid omitted a request")
        else:
            assert results is not None
            primary_signature = [
                (
                    bool(value.get("action_valid")),
                    bool(value.get("generated")),
                    tuple(sorted(map(int, value.get("neighborhood", [])))),
                )
                for value in results
            ]
        shadow_signature = [
            (
                bool(value.get("action_valid")),
                bool(value.get("generated")),
                tuple(sorted(map(int, value.get("neighborhood", [])))),
            )
            for value in shadow_results
        ]
        if primary_signature != shadow_signature:
            raise ClosedLoopExecutionError(
                "proposal_shadow_mismatch",
                "reference and compact proposal batches differ",
            )
    if results is not None and len(results) != len(requests):
        raise RuntimeError("online proposal batch returned an unexpected result count")
    state_check_started = time.perf_counter()
    revision_after = int(get_revision()) if callable(get_revision) else None
    if revision_before is not None and revision_after != revision_before:
        raise ClosedLoopExecutionError(
            "revision_mismatch", "proposal changed the structural state revision"
        )
    full_state_verified = bool(verify_full_state or revision_before is None)
    state_check_fingerprint_seconds = 0.0
    if full_state_verified:
        after = _plain(environment.get_state())
        state_check_fingerprint_started = time.perf_counter()
        after_fingerprint = state_fingerprint(after)
        state_check_fingerprint_seconds = (
            time.perf_counter() - state_check_fingerprint_started
        )
        # ``runtime`` is a live wall-clock field and legitimately advances
        # while a read-only proposal batch is being generated.  The
        # deterministic fingerprint deliberately excludes wall-clock/context
        # fields, while the native revision above protects structural state.
        # Comparing the complete dictionaries therefore creates false
        # positives without adding a repair-state invariant.
        if after_fingerprint != state_hash:
            raise ClosedLoopExecutionError(
                "fingerprint_mismatch", "proposal changed the closed-loop repair state"
            )
    state_check_seconds = time.perf_counter() - state_check_started
    candidate_postprocess_started = time.perf_counter()
    proposals: list[dict[str, Any]] = []
    if proposal_groups is not None:
        candidates = select_representative_neighborhood_groups(
            proposal_groups, int(proposal_config["candidates_per_family"])
        )
        proposal_count = len(requests)
        unique_neighborhood_count = len(proposal_groups)
    else:
        assert results is not None
        for (_, metadata), result in zip(requests, results):
            if not bool(result.get("action_valid")) or not bool(
                result.get("generated")
            ):
                raise RuntimeError("valid online proposal was rejected")
            agents = sorted(map(int, result.get("neighborhood", [])))
            if not agents or len(agents) != len(set(agents)):
                raise RuntimeError("online proposal returned an invalid neighborhood")
            proposals.append({**metadata, "agents": agents})
        candidates = select_representative_neighborhoods(
            proposals, int(proposal_config["candidates_per_family"])
        )
        proposal_count = len(proposals)
        unique_neighborhood_count = len(
            {tuple(row["agents"]) for row in proposals}
        )
    base_candidate_count = len(candidates)
    topology_boundary_analysis_seconds = 0.0
    topology_boundary_static_seconds = 0.0
    topology_boundary_dynamic_seconds = 0.0
    topology_boundary_candidate_seconds = 0.0
    topology_boundary_merge_seconds = 0.0
    topology_boundary_static_cache_hit = False
    topology_boundary_gate_seconds = 0.0
    topology_boundary_gate_evaluated = False
    topology_boundary_gate_passed = True
    topology_boundary_gate_reason = "legacy_unconditional"
    topology_boundary_low_degree_cell_ratio: float | None = None
    topology_boundary_no_progress_streak = int(topology_no_progress_streak)
    topology_boundary_remaining_wall_seconds = (
        None
        if topology_remaining_wall_seconds is None
        else float(topology_remaining_wall_seconds)
    )
    if topology_boundary_no_progress_streak < 0:
        raise ValueError("topology no-progress streak must be non-negative")
    if topology_state_analysis_seconds < 0.0:
        raise ValueError("topology state-analysis time must be non-negative")
    if (
        topology_boundary_remaining_wall_seconds is not None
        and topology_boundary_remaining_wall_seconds < 0.0
    ):
        raise ValueError("topology remaining wall time must be non-negative")
    topology_boundary_generated_count = 0
    topology_boundary_added_candidate_count = 0
    topology_boundary = validate_topology_boundary_augmentation(
        proposal_config.get("topology_boundary")
    )
    if topology_boundary is not None:
        topology_started = time.perf_counter()
        topology_static_started = time.perf_counter()
        static_grid = topology_static_grid
        if static_grid is None:
            static_grid = analyze_static_grid(state)
        else:
            topology_boundary_static_cache_hit = True
        topology_boundary_static_seconds = (
            time.perf_counter() - topology_static_started
        )
        maximum = int(topology_boundary["maximum_added_candidates"])
        activation_gate = dict(topology_boundary.get("activation_gate") or {})
        if activation_gate:
            topology_boundary_gate_evaluated = True
            topology_gate_started = time.perf_counter()
            topology_boundary_low_degree_cell_ratio = (
                sum(
                    int(static_grid.degrees.get(cell, 0)) <= 2
                    for cell in static_grid.free_cells
                )
                / len(static_grid.free_cells)
                if static_grid.free_cells
                else 0.0
            )
            minimum_ratio = float(
                activation_gate["minimum_low_degree_cell_ratio"]
            )
            topology_boundary_gate_passed = (
                topology_boundary_low_degree_cell_ratio >= minimum_ratio
            )
            topology_boundary_gate_reason = (
                "map_topology_passed"
                if topology_boundary_gate_passed
                else "low_degree_cell_ratio_below_threshold"
            )
            topology_boundary_gate_seconds = time.perf_counter() - topology_gate_started
        phase_guard = dict(topology_boundary.get("phase_guard") or {})
        if topology_boundary_gate_passed and phase_guard:
            topology_boundary_gate_evaluated = True
            topology_phase_started = time.perf_counter()
            current_conflicts = int(state.get("num_of_colliding_pairs", 0))
            if (
                "low_conflict_pair_threshold" in phase_guard
                and current_conflicts
                < int(phase_guard["low_conflict_pair_threshold"])
                and topology_boundary_no_progress_streak
                >= int(phase_guard["low_conflict_no_progress_streak"])
            ):
                topology_boundary_gate_passed = False
                topology_boundary_gate_reason = "low_conflict_no_progress"
            elif (
                "maximum_no_progress_streak" in phase_guard
                and topology_boundary_no_progress_streak
                >= int(phase_guard["maximum_no_progress_streak"])
            ):
                topology_boundary_gate_passed = False
                topology_boundary_gate_reason = "no_progress_streak"
            elif (
                "minimum_remaining_wall_seconds" in phase_guard
                and topology_boundary_remaining_wall_seconds is not None
                and topology_boundary_remaining_wall_seconds
                < float(phase_guard["minimum_remaining_wall_seconds"])
            ):
                topology_boundary_gate_passed = False
                topology_boundary_gate_reason = "insufficient_remaining_wall_time"
            else:
                topology_boundary_gate_reason = "map_and_phase_passed"
            topology_boundary_gate_seconds += (
                time.perf_counter() - topology_phase_started
            )
        if topology_boundary_gate_passed:
            if topology_state_analysis is None:
                topology_dynamic_started = time.perf_counter()
                analysis = analyze_state(state, static_grid=static_grid)
                topology_boundary_dynamic_seconds = (
                    time.perf_counter() - topology_dynamic_started
                )
            else:
                analysis = topology_state_analysis
                topology_boundary_dynamic_seconds = float(
                    topology_state_analysis_seconds
                )
            topology_candidate_started = time.perf_counter()
            topology_candidates = generate_topology_boundary_candidates(
                state,
                analysis,
                neighborhood_size=int(topology_boundary["neighborhood_size"]),
                core_budget=int(topology_boundary["core_budget"]),
            )
            topology_boundary_candidate_seconds = (
                time.perf_counter() - topology_candidate_started
            )
            if len(topology_candidates) > maximum:
                raise RuntimeError("topology-boundary runtime candidate cap exceeded")
            topology_boundary_generated_count = len(topology_candidates)
            topology_merge_started = time.perf_counter()
            candidates = merge_topology_anchor_candidates(candidates, topology_candidates)
            topology_boundary_merge_seconds = (
                time.perf_counter() - topology_merge_started
            )
        topology_boundary_analysis_seconds = (
            time.perf_counter() - topology_started
            + (
                topology_boundary_dynamic_seconds
                if topology_state_analysis is not None
                and topology_boundary_gate_passed
                else 0.0
            )
        )
        topology_boundary_added_candidate_count = len(candidates) - base_candidate_count
        if not 0 <= topology_boundary_added_candidate_count <= maximum:
            raise RuntimeError("topology-boundary runtime merge changed the candidate cap")
    if not candidates:
        raise RuntimeError("online proposal stage produced no explicit candidates")
    candidate_postprocess_seconds = (
        time.perf_counter() - candidate_postprocess_started
    )
    candidate_generation_seconds = (
        request_generation_seconds
        + proposal_seconds
        + candidate_postprocess_seconds
        + (
            topology_boundary_dynamic_seconds
            if topology_state_analysis is not None
            and topology_boundary is not None
            and topology_boundary_gate_passed
            else 0.0
        )
    )
    return candidates, {
        "proposal_count": proposal_count,
        "unique_neighborhood_count": unique_neighborhood_count,
        "candidate_count": len(candidates),
        "base_candidate_count": base_candidate_count,
        "topology_boundary_enabled": topology_boundary is not None,
        "topology_boundary_generated_count": topology_boundary_generated_count,
        "topology_boundary_added_candidate_count": topology_boundary_added_candidate_count,
        "topology_boundary_analysis_seconds": topology_boundary_analysis_seconds,
        "topology_boundary_static_seconds": topology_boundary_static_seconds,
        "topology_boundary_dynamic_seconds": topology_boundary_dynamic_seconds,
        "topology_boundary_candidate_seconds": topology_boundary_candidate_seconds,
        "topology_boundary_merge_seconds": topology_boundary_merge_seconds,
        "topology_boundary_static_cache_hit": topology_boundary_static_cache_hit,
        "topology_boundary_gate_seconds": topology_boundary_gate_seconds,
        "topology_boundary_gate_evaluated": topology_boundary_gate_evaluated,
        "topology_boundary_gate_passed": topology_boundary_gate_passed,
        "topology_boundary_gate_reason": topology_boundary_gate_reason,
        "topology_boundary_low_degree_cell_ratio": (
            topology_boundary_low_degree_cell_ratio
        ),
        "topology_boundary_no_progress_streak": topology_boundary_no_progress_streak,
        "topology_boundary_remaining_wall_seconds": (
            topology_boundary_remaining_wall_seconds
        ),
        "seed_agents": list(seed_agents),
        "seed_agent_count": len(seed_agents),
        "seed_agents_overridden": seed_agents_override is not None,
        "proposal_seconds": proposal_seconds,
        "proposal_shadow_seconds": proposal_shadow_seconds,
        "request_generation_seconds": request_generation_seconds,
        "candidate_postprocess_seconds": candidate_postprocess_seconds,
        "candidate_generation_seconds": candidate_generation_seconds,
        "state_check_seconds": state_check_seconds,
        "state_check_fingerprint_seconds": state_check_fingerprint_seconds,
        "state_check_backend": (
            "revision_and_full" if revision_before is not None and full_state_verified
            else "revision" if revision_before is not None
            else "full_state"
        ),
        "full_state_verified": full_state_verified,
        "state_revision": revision_after,
        "backend": backend,
        "shadow_validation": bool(shadow_validation),
        "shadow_validation_passed": bool(shadow_validation),
    }

__all__ = [
    "ClosedLoopExecutionError",
    "feature_range_diagnostic",
    "generate_online_candidates",
    "online_candidate_rows",
    "pp_replay_random_seed",
    "proposal_random_seed",
    "proposal_random_seeds",
    "repair_random_seed",
    "score_online_candidates",
    "validate_topology_boundary_augmentation",
]
