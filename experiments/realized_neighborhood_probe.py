from __future__ import annotations

import collections
import itertools
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import action_family as _family
from experiments.independent_layout_probe import _map_bootstrap
from experiments.movingai_mechanism_probe import _mean
from experiments.movingai_probe_quality import (
    _aggregate_candidate,
    _candidate_trial_stability,
    _jaccard,
    _pareto_keys,
    _pareto_sensitivity,
    _statewise_eta_squared,
)
from experiments.repair_collection import (
    SCHEMA_VERSION,
    CollectionLockError,
    _CollectionRunLock,
    _dataset_fingerprint,
    _fingerprint,
    _horizon_outcomes,
    _load_dataset_rows,
    _make_environment,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)


PROBE_SCHEMA = "lns2.realized_neighborhood_probe.v1"


def _actual_neighborhood(outcome: dict[str, Any]) -> tuple[int, ...]:
    steps = [
        row for row in outcome.get("steps", []) if int(row.get("step", -1)) == 1
    ]
    if len(steps) != 1:
        raise ValueError("proposal outcome must contain exactly one first step")
    neighborhood = steps[0].get("metrics", {}).get("neighborhood")
    if not isinstance(neighborhood, list) or not neighborhood:
        raise ValueError("proposal outcome is missing its actual neighborhood")
    values = tuple(sorted(int(value) for value in neighborhood))
    if len(values) != len(set(values)):
        raise ValueError("proposal neighborhood contains duplicate agents")
    return values


def _candidate_id(agents: Iterable[int]) -> str:
    return f"neighborhood-{_fingerprint(sorted(map(int, agents)))[:16]}"


def _distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    return 1.0 - _jaccard(left, right)


def select_representative_neighborhoods(
    proposals: list[dict[str, Any]], candidates_per_family: int
) -> list[dict[str, Any]]:
    if candidates_per_family <= 0:
        raise ValueError("candidates_per_family must be positive")
    by_agents: dict[tuple[int, ...], dict[str, Any]] = {}
    by_family: dict[str, collections.Counter[tuple[int, ...]]] = (
        collections.defaultdict(collections.Counter)
    )
    for proposal in proposals:
        agents = tuple(sorted(int(value) for value in proposal["agents"]))
        if not agents or len(agents) != len(set(agents)):
            raise ValueError("proposal agents must be a non-empty unique set")
        family = str(proposal["family"])
        by_family[family][agents] += 1
        record = by_agents.setdefault(
            agents,
            {
                "candidate_id": _candidate_id(agents),
                "agents": list(agents),
                "proposal_count_by_family": collections.Counter(),
                "proposal_seeds": set(),
                "seed_agents": set(),
                "selection_families": set(),
            },
        )
        record["proposal_count_by_family"][family] += 1
        record["proposal_seeds"].add(int(proposal["proposal_seed"]))
        record["seed_agents"].add(int(proposal["seed_agent"]))

    for family, counts in sorted(by_family.items()):
        remaining = set(counts)
        chosen: list[tuple[int, ...]] = []
        while remaining and len(chosen) < candidates_per_family:
            if not chosen:
                selected = min(
                    remaining,
                    key=lambda agents: (-counts[agents], _candidate_id(agents)),
                )
            else:
                selected = min(
                    remaining,
                    key=lambda agents: (
                        -min(_distance(agents, previous) for previous in chosen),
                        -counts[agents],
                        _candidate_id(agents),
                    ),
                )
            chosen.append(selected)
            remaining.remove(selected)
            by_agents[selected]["selection_families"].add(family)

    selected_rows = []
    for agents, record in by_agents.items():
        if not record["selection_families"]:
            continue
        selected_rows.append(
            {
                "candidate_id": record["candidate_id"],
                "agents": list(agents),
                "actual_size": len(agents),
                "selection_families": sorted(record["selection_families"]),
                "proposal_count_by_family": dict(
                    sorted(record["proposal_count_by_family"].items())
                ),
                "proposal_seeds": sorted(record["proposal_seeds"]),
                "seed_agents": sorted(record["seed_agents"]),
            }
        )
    return sorted(selected_rows, key=lambda row: str(row["candidate_id"]))


def evaluation_seed(
    state_id: str,
    candidate_id: str,
    trial_index: int,
    proposal_seeds: Iterable[int],
) -> int:
    if trial_index < 0:
        raise ValueError("trial_index must be non-negative")
    value = int(
        _fingerprint(
            {
                "namespace": "explicit-neighborhood-evaluation-v1",
                "state_id": state_id,
                "candidate_id": candidate_id,
                "trial_index": trial_index,
            }
        )[:16],
        16,
    ) % (2**31)
    forbidden = set(map(int, proposal_seeds))
    while value in forbidden:
        value = (value + 1) % (2**31)
    return value


def build_candidate_rows(
    source_collection: str | Path, candidates_per_family: int
) -> list[dict[str, Any]]:
    root = Path(source_collection).resolve()
    baseline = {
        str(row["episode_id"]): row
        for row in _read_jsonl(root / "collection_manifest.jsonl")
        if str(row.get("policy")) == "official_adaptive"
    }
    rows = []
    seen_states: set[str] = set()
    for manifest in _read_jsonl(root / "counterfactual_manifest.jsonl"):
        if str(manifest.get("status")) not in {"ok", "resumed"}:
            raise ValueError("source counterfactual manifest contains an error")
        states = _read_jsonl(root / str(manifest["states_file"]))
        outcomes = _read_jsonl(root / str(manifest["outcomes_file"]))
        outcomes_by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for outcome in outcomes:
            outcomes_by_state[str(outcome["state_id"])].append(outcome)
        for state in states:
            state_id = str(state["state_id"])
            if state_id in seen_states:
                raise ValueError(f"duplicate source state: {state_id}")
            seen_states.add(state_id)
            episode_id = str(state["episode_id"])
            source_episode = baseline.get(episode_id)
            if source_episode is None:
                raise ValueError(f"missing Adaptive source episode: {episode_id}")
            proposals = []
            for outcome in outcomes_by_state.get(state_id, []):
                if not bool(outcome.get("action_valid")):
                    raise ValueError(f"source state has an invalid proposal: {state_id}")
                action = outcome["candidate_action"]
                proposals.append(
                    {
                        "agents": list(_actual_neighborhood(outcome)),
                        "family": _family(action),
                        "seed_agent": int(action["seed_agent"]),
                        "proposal_seed": int(outcome["trial_seed"]),
                    }
                )
            candidates = select_representative_neighborhoods(
                proposals, candidates_per_family
            )
            if not candidates:
                raise ValueError(f"source state has no realized candidates: {state_id}")
            context = state["state"].get("context", {})
            rows.append(
                {
                    "schema": PROBE_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "state_id": state_id,
                    "state_fingerprint": str(state["state_fingerprint"]),
                    "episode_id": episode_id,
                    "decision_index": int(state["decision_index"]),
                    "prefix_actions": state["prefix_actions"],
                    "state": state["state"],
                    "split": str(context.get("split", source_episode["split"])),
                    "map_id": str(context.get("map_id", source_episode["map_id"])),
                    "task_id": str(context.get("task_id", source_episode["task_id"])),
                    "layout_mode": str(context.get("layout_mode", "unknown")),
                    "task_variant": str(context.get("task_variant", "unknown")),
                    "agent_count": int(context.get("agent_count", source_episode["agent_count"])),
                    "solver_seed": int(source_episode["solver_seed"]),
                    "candidate_count": len(candidates),
                    "proposal_outcome_count": len(proposals),
                    "unique_proposal_neighborhood_count": len(
                        {tuple(proposal["agents"]) for proposal in proposals}
                    ),
                    "candidates": candidates,
                }
            )
    return sorted(rows, key=lambda row: str(row["state_id"]))


def _explicit_worker(job: dict[str, Any]) -> dict[str, Any]:
    state_row = job["state_row"]
    state_id = str(state_row["state_id"])
    output_root = Path(job["output_root"])
    episode_root = output_root / "explicit" / str(state_row["split"]) / state_id
    metadata_path = episode_root / "metadata.json"
    if job["resume"] and metadata_path.is_file():
        metadata = _read_json(metadata_path)
        if (
            metadata.get("run_fingerprint") == job["run_fingerprint"]
            and metadata.get("complete") is True
        ):
            return {**metadata, "state_count": 1, "status": "resumed"}
    outcomes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    expected_fingerprint = str(state_row["state_fingerprint"])
    trials = int(job["evaluation_trials"])
    for candidate in state_row["candidates"]:
        agents = sorted(map(int, candidate["agents"]))
        proposal_seeds = list(map(int, candidate["proposal_seeds"]))
        for trial_index in range(trials):
            seed = evaluation_seed(
                state_id,
                str(candidate["candidate_id"]),
                trial_index,
                proposal_seeds,
            )
            try:
                environment = _make_environment(
                    job["dataset_root"], job["row"], job["environment"], "Adaptive"
                )
                replayed = _plain(environment.reset(seed=int(state_row["solver_seed"])))
                for prefix_action in state_row["prefix_actions"]:
                    if replayed["done"]:
                        raise RuntimeError("prefix replay terminated before source state")
                    replayed = _plain(environment.step(prefix_action))["observation"]
                actual_fingerprint = state_fingerprint(replayed)
                if actual_fingerprint != expected_fingerprint:
                    raise RuntimeError(
                        "replay fingerprint mismatch: "
                        f"expected {expected_fingerprint}, got {actual_fingerprint}"
                    )
                action = {
                    "mode": "explicit_neighborhood",
                    "agents": agents,
                    "random_seed": seed,
                }
                result = _plain(environment.step(action))
                metrics = result["metrics"]
                actual = sorted(map(int, metrics.get("neighborhood", [])))
                if not bool(metrics.get("action_valid")):
                    raise RuntimeError("explicit neighborhood was rejected")
                if actual != agents:
                    raise RuntimeError(
                        f"explicit neighborhood changed: expected {agents}, got {actual}"
                    )
                current = result["observation"]
                points = [
                    {
                        "step": 0,
                        "state": replayed,
                        "action": None,
                        "metrics": None,
                        "step_runtime": 0.0,
                    },
                    {
                        "step": 1,
                        "state": current,
                        "action": action,
                        "metrics": metrics,
                        "step_runtime": float(metrics["step_runtime"]),
                    },
                ]
                horizon = _horizon_outcomes(replayed, points, [1])[0]
                outcomes.append(
                    {
                        "schema": PROBE_SCHEMA,
                        "schema_version": SCHEMA_VERSION,
                        "run_fingerprint": job["run_fingerprint"],
                        "state_id": state_id,
                        "state_fingerprint": expected_fingerprint,
                        "candidate_id": candidate["candidate_id"],
                        "agents": agents,
                        "selection_families": candidate["selection_families"],
                        "proposal_seeds": proposal_seeds,
                        "evaluation_trial_index": trial_index,
                        "evaluation_seed": seed,
                        "evaluation_seed_disjoint": seed not in set(proposal_seeds),
                        "actual_neighborhood": actual,
                        "action_valid": True,
                        "solved": bool(horizon["solved"]),
                        "conflicts_before": int(replayed["num_of_colliding_pairs"]),
                        "conflicts_after": int(horizon["conflicts_after"]),
                        "conflict_auc": float(horizon["conflict_auc"]),
                        "sum_of_costs_after": int(horizon["sum_of_costs_after"]),
                        "generated": int(horizon["low_level_delta"]["generated"]),
                        "runtime": float(horizon["branch_runtime"]),
                    }
                )
            except Exception as error:
                errors.append(
                    {
                        "schema": PROBE_SCHEMA,
                        "schema_version": SCHEMA_VERSION,
                        "state_id": state_id,
                        "candidate_id": candidate["candidate_id"],
                        "evaluation_trial_index": trial_index,
                        "evaluation_seed": seed,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    outcomes_path = episode_root / "outcomes.jsonl"
    errors_path = episode_root / "errors.jsonl"
    _write_jsonl(outcomes_path, outcomes)
    _write_jsonl(errors_path, errors)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": job["run_fingerprint"],
        "state_id": state_id,
        "episode_id": state_row["episode_id"],
        "split": state_row["split"],
        "map_id": state_row["map_id"],
        "task_id": state_row["task_id"],
        "state_count": 1,
        "candidate_count": len(state_row["candidates"]),
        "outcome_count": len(outcomes),
        "error_count": len(errors),
        "outcomes_file": outcomes_path.relative_to(output_root).as_posix(),
        "errors_file": errors_path.relative_to(output_root).as_posix(),
        "metadata_file": metadata_path.relative_to(output_root).as_posix(),
        "complete": not errors,
        "status": "ok" if not errors else "error",
    }
    _write_json(metadata_path, metadata)
    return metadata


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported realized-neighborhood probe config")
    for key in ("candidates_per_family", "evaluation_trials", "workers"):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(config.get("episode_wall_time_limit_seconds", 0.0)) <= 0:
        raise ValueError("episode_wall_time_limit_seconds must be positive")


def run_realized_collection(
    dataset: str | Path,
    source_collection: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    workers: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset).resolve()
    source_root = Path(source_collection).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    _validate_config(config)
    effective_workers = int(workers or config["workers"])
    source_run = _read_json(source_root / "run_config.json")
    actual_dataset_fingerprint = _dataset_fingerprint(dataset_root)
    if actual_dataset_fingerprint != str(source_run["dataset_fingerprint"]):
        raise ValueError("dataset fingerprint differs from the source collection")
    candidates = build_candidate_rows(
        source_root, int(config["candidates_per_family"])
    )
    expected_split = str(config["source_split"])
    if any(str(row["split"]) != expected_split for row in candidates):
        raise ValueError("candidate source contains a non-probe split")
    candidate_fingerprint = _fingerprint(candidates)
    config_fingerprint = _fingerprint(config)
    run_fingerprint = _fingerprint(
        {
            "dataset_fingerprint": actual_dataset_fingerprint,
            "source_run_fingerprint": source_run["run_fingerprint"],
            "candidate_fingerprint": candidate_fingerprint,
            "configuration_fingerprint": config_fingerprint,
        }
    )
    total_candidates = sum(int(row["candidate_count"]) for row in candidates)
    estimate = {
        "state_count": len(candidates),
        "candidate_count": total_candidates,
        "evaluation_trials": int(config["evaluation_trials"]),
        "outcome_count": total_candidates * int(config["evaluation_trials"]),
        "environment_reset_count": total_candidates * int(config["evaluation_trials"]),
        "workers": effective_workers,
    }
    if dry_run:
        return {
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "run_fingerprint": run_fingerprint,
            "estimate": estimate,
        }

    run_config = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset_root),
        "dataset_fingerprint": actual_dataset_fingerprint,
        "source_collection": str(source_root),
        "source_run_fingerprint": source_run["run_fingerprint"],
        "candidate_fingerprint": candidate_fingerprint,
        "configuration": config,
        "configuration_fingerprint": config_fingerprint,
        "run_fingerprint": run_fingerprint,
    }
    existing_run_path = output_root / "run_config.json"
    if existing_run_path.is_file():
        existing = _read_json(existing_run_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("output contains a different realized-probe configuration")
        if not resume:
            raise ValueError("output already exists; pass resume to reuse it")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(existing_run_path, run_config)
    _write_jsonl(output_root / "candidates.jsonl", candidates)

    split_rows = _load_dataset_rows(dataset_root, [expected_split])
    dataset_index = {str(row["task_id"]): row for row in split_rows}
    jobs = []
    for state_row in candidates:
        task_id = str(state_row["task_id"])
        if task_id not in dataset_index:
            raise ValueError(f"candidate references unknown task: {task_id}")
        jobs.append(
            {
                "row": dataset_index[task_id],
                "state_row": state_row,
                "solver_seed": int(state_row["solver_seed"]),
                "dataset_root": str(dataset_root),
                "environment": source_run["configuration"]["environment"],
                "evaluation_trials": int(config["evaluation_trials"]),
                "output_root": str(output_root),
                "run_fingerprint": run_fingerprint,
                "resume": resume,
            }
        )

    manifest_path = output_root / "collection_manifest.jsonl"
    existing_manifest = (
        _read_jsonl(manifest_path) if resume and manifest_path.is_file() else []
    )
    manifest_by_key = {
        str(row.get("state_id", row.get("task_id"))): row
        for row in existing_manifest
    }

    def record(result: dict[str, Any]) -> None:
        key = str(result.get("state_id", result.get("task_id")))
        manifest_by_key[key] = result
        _write_jsonl(
            manifest_path,
            [manifest_by_key[name] for name in sorted(manifest_by_key)],
        )

    with _CollectionRunLock(output_root, run_fingerprint, "realized-neighborhood"):
        results = _run_jobs(
            _explicit_worker,
            jobs,
            effective_workers,
            phase="realized-neighborhood",
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            timeout_seconds=float(config["episode_wall_time_limit_seconds"]),
            on_result=record,
        )
    errors = sum(
        int(row.get("error_count", 0))
        or int(str(row.get("status")) not in {"ok", "resumed"})
        for row in results
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "state_count": len(candidates),
        "candidate_count": total_candidates,
        "outcome_count": sum(int(row.get("outcome_count", 0)) for row in results),
        "error_count": errors,
        "estimate": estimate,
    }


def _candidate_pool_diversity(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    counts = []
    for row in candidate_rows:
        neighborhoods = [set(map(int, candidate["agents"])) for candidate in row["candidates"]]
        pairs = [
            1.0 - _jaccard(left, right)
            for left, right in itertools.combinations(neighborhoods, 2)
        ]
        values.extend(pairs)
        counts.append(len(neighborhoods))
    return {
        "candidate_count_min": min(counts, default=0),
        "candidate_count_mean": _mean(counts),
        "candidate_count_max": max(counts, default=0),
        "mean_pairwise_jaccard_distance": _mean(values),
    }


def analyze_realized_records(
    candidate_rows: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    config: dict[str, Any],
    nominal_report: dict[str, Any],
) -> dict[str, Any]:
    candidate_index = {
        (str(row["state_id"]), str(candidate["candidate_id"])): (row, candidate)
        for row in candidate_rows
        for candidate in row["candidates"]
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    orphan_outcomes = 0
    neighborhood_mismatches = 0
    seed_overlaps = 0
    for outcome in outcomes:
        key = (str(outcome["state_id"]), str(outcome["candidate_id"]))
        source = candidate_index.get(key)
        if source is None:
            orphan_outcomes += 1
            continue
        expected = sorted(map(int, source[1]["agents"]))
        neighborhood_mismatches += sorted(map(int, outcome["actual_neighborhood"])) != expected
        seed_overlaps += not bool(outcome.get("evaluation_seed_disjoint"))
        grouped[key].append(outcome)

    expected_trials = int(config["evaluation_trials"])
    trial_mismatches = 0
    state_rows = []
    for source_state in candidate_rows:
        candidates = []
        before = float(source_state["state"]["num_of_colliding_pairs"])
        for source_candidate in source_state["candidates"]:
            key = (str(source_state["state_id"]), str(source_candidate["candidate_id"]))
            trials = sorted(
                grouped.get(key, []),
                key=lambda row: int(row["evaluation_trial_index"]),
            )
            if (
                len(trials) != expected_trials
                or {int(row["evaluation_trial_index"]) for row in trials}
                != set(range(expected_trials))
            ):
                trial_mismatches += 1
            normalized_trials = [
                {
                    "episode_id": str(source_state["episode_id"]),
                    "trial_index": int(row["evaluation_trial_index"]),
                    "solved": bool(row["solved"]),
                    "conflicts_after": float(row["conflicts_after"]),
                    "generated": float(row["generated"]),
                    "runtime": float(row["runtime"]),
                    "neighborhood": tuple(map(int, row["actual_neighborhood"])),
                }
                for row in trials
            ]
            candidates.append(
                {
                    "candidate_key": source_candidate["candidate_id"],
                    "family": source_candidate["candidate_id"],
                    "heuristic": "explicit",
                    "size": int(source_candidate["actual_size"]),
                    "seed_agent": -1,
                    "trial_count": len(normalized_trials),
                    "trial_conflicts": [row["conflicts_after"] for row in normalized_trials],
                    "trials": normalized_trials,
                    "selection_families": source_candidate["selection_families"],
                }
            )
        state_rows.append(
            {
                "state_fingerprint": source_state["state_fingerprint"],
                "map_id": source_state["map_id"],
                "task_id": source_state["task_id"],
                "layout_mode": source_state["layout_mode"],
                "task_variant": source_state["task_variant"],
                "conflicts_before": before,
                "action_effect_eta_squared": _statewise_eta_squared(candidates, before),
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )

    collection_errors = sum(
        str(row.get("status")) not in {"ok", "resumed"}
        or int(row.get("error_count", 0)) > 0
        for row in manifest
    )
    expected_outcomes = len(candidate_index) * expected_trials
    integrity_errors = {
        "collection_errors": collection_errors,
        "orphan_outcomes": orphan_outcomes,
        "neighborhood_mismatches": neighborhood_mismatches,
        "proposal_evaluation_seed_overlaps": seed_overlaps,
        "trial_mismatches": trial_mismatches,
        "outcome_count_mismatch": int(len(outcomes) != expected_outcomes),
        "duplicate_state_ids": len(candidate_rows)
        - len({str(row["state_id"]) for row in candidate_rows}),
        "non_probe_states": sum(str(row["split"]) != "probe" for row in candidate_rows),
    }
    integrity_passed = all(value == 0 for value in integrity_errors.values())
    stability = _candidate_trial_stability(
        state_rows, bootstrap_samples=500
    )
    pareto = _pareto_sensitivity(state_rows)
    etas = [float(row["action_effect_eta_squared"]) for row in state_rows]
    nominal_eta = float(nominal_report["action_effect"]["mean_eta_squared"])
    eta_mean = _mean(etas)
    eta_improvement = eta_mean - nominal_eta
    distinct_state_rate = _mean(
        len(
            {
                (
                    round(_mean(trial["solved"] for trial in candidate["trials"]), 8),
                    round(_mean(trial["conflicts_after"] for trial in candidate["trials"]), 8),
                )
                for candidate in row["candidates"]
            }
        )
        > 1
        for row in state_rows
    )

    proposal_family_winners: collections.Counter[str] = collections.Counter()
    unique_family_states = 0
    for source_state, row in zip(candidate_rows, state_rows):
        aggregated = [
            _aggregate_candidate(candidate, candidate["trials"])
            for candidate in row["candidates"]
        ]
        pareto_keys = _pareto_keys(aggregated)
        source_by_id = {
            str(candidate["candidate_id"]): candidate
            for candidate in source_state["candidates"]
        }
        families = {
            family
            for key in pareto_keys
            for family in source_by_id[key]["selection_families"]
        }
        if len(families) == 1:
            proposal_family_winners[next(iter(families))] += 1
            unique_family_states += 1
    maximum_family_share = max(proposal_family_winners.values(), default=0) / max(
        1, len(state_rows)
    )
    thresholds = config["thresholds"]
    gates = {
        "integrity": integrity_passed,
        "state_coverage": len(state_rows) >= int(thresholds["minimum_state_count"]),
        "candidate_coverage": min(
            (int(row["candidate_count"]) for row in state_rows), default=0
        )
        >= int(thresholds["minimum_candidates_per_state"]),
        "realized_action_eta": eta_mean
        >= float(thresholds["minimum_action_eta_squared"]),
        "eta_improvement_over_nominal": eta_improvement
        >= float(thresholds["minimum_eta_improvement"]),
        "trial_split_spearman": stability["mean_rank_spearman"]
        >= float(thresholds["minimum_trial_split_spearman"]),
        "pareto_candidate_jaccard": stability["mean_pareto_family_jaccard"]
        >= float(thresholds["minimum_pareto_candidate_jaccard"]),
        "best_candidate_jaccard": stability["mean_best_candidate_jaccard"]
        >= float(thresholds["minimum_best_candidate_jaccard"]),
        "distinct_outcome_states": distinct_state_rate
        >= float(thresholds["minimum_distinct_outcome_state_rate"]),
        "no_fixed_proposal_family": maximum_family_share
        <= float(thresholds["maximum_fixed_proposal_family_share"]),
    }
    if not gates["integrity"] or not gates["state_coverage"]:
        decision = "stop_invalid_realized_probe"
    elif all(gates.values()):
        decision = "proceed_to_realized_neighborhood_ranking_audit"
    elif gates["realized_action_eta"] and gates["eta_improvement_over_nominal"]:
        decision = "model_pp_order_or_use_robust_realized_objective"
    else:
        decision = "include_replan_order_in_action_and_redesign_repair_control"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "passed": all(gates.values()),
        "integrity": {
            "passed": integrity_passed,
            "errors": integrity_errors,
            "state_count": len(state_rows),
            "candidate_count": len(candidate_index),
            "outcome_count": len(outcomes),
            "expected_outcome_count": expected_outcomes,
        },
        "candidate_pool": _candidate_pool_diversity(candidate_rows),
        "action_effect": {
            "nominal_eta_squared": nominal_eta,
            "realized_eta_squared": eta_mean,
            "eta_improvement": eta_improvement,
            "median_eta_squared": statistics.median(etas) if etas else 0.0,
            "map_bootstrap": _map_bootstrap(
                state_rows, "action_effect_eta_squared", 2000
            ),
        },
        "trial_stability": stability,
        "distinct_outcome_state_rate": distinct_state_rate,
        "proposal_family": {
            "unique_pareto_counts": dict(sorted(proposal_family_winners.items())),
            "unique_family_state_count": unique_family_states,
            "maximum_fixed_share": maximum_family_share,
        },
        "pareto_sensitivity": pareto,
        "gates": gates,
        "limitations": [
            "Candidate sets are proposed by the existing Target/Collision/Random generators.",
            "All states still come from official Adaptive trajectories.",
            "Only Horizon-1 repair is evaluated; no learned ranker is trained here.",
            "Evaluation seeds change PP repair order while the explicit agent set remains fixed.",
        ],
    }


def render_report(report: dict[str, Any]) -> str:
    effect = report["action_effect"]
    stability = report["trial_stability"]
    lines = [
        "# Realized-neighborhood stability probe",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Coverage",
        "",
        f"- States: {report['integrity']['state_count']}",
        f"- Explicit candidates: {report['integrity']['candidate_count']}",
        f"- Evaluation outcomes: {report['integrity']['outcome_count']}",
        "",
        "## Stability",
        "",
        f"- Nominal-action eta-squared: {effect['nominal_eta_squared']:.3f}",
        f"- Realized-neighborhood eta-squared: {effect['realized_eta_squared']:.3f}",
        f"- Eta improvement: {effect['eta_improvement']:+.3f}",
        f"- Trial-split Spearman: {stability['mean_rank_spearman']:.3f}",
        f"- Pareto-candidate Jaccard: {stability['mean_pareto_family_jaccard']:.3f}",
        f"- Best-candidate Jaccard: {stability['mean_best_candidate_jaccard']:.3f}",
        f"- Distinct-outcome state rate: {report['distinct_outcome_state_rate']:.1%}",
        "",
        "## Gates",
        "",
    ]
    for name, passed in report["gates"].items():
        lines.append(f"- {name}: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "This probe fixes the actual agent set and varies only the explicit-action "
            "random seed, which controls PP repair order. It does not train a model "
            "or use Test/OOD labels.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_realized_collection(
    collection: str | Path,
    config_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    root = Path(collection).resolve()
    config = _read_json(Path(config_path).resolve())
    run_config = _read_json(root / "run_config.json")
    candidate_rows = _read_jsonl(root / "candidates.jsonl")
    manifest = _read_jsonl(root / "collection_manifest.jsonl")
    outcomes = []
    for row in manifest:
        if str(row.get("status")) not in {"ok", "resumed"}:
            continue
        outcomes.extend(_read_jsonl(root / str(row["outcomes_file"])))
    nominal_path = Path(config["nominal_report"])
    if not nominal_path.is_absolute():
        nominal_path = Path.cwd() / nominal_path
    nominal_report = _read_json(nominal_path.resolve())
    report = analyze_realized_records(
        candidate_rows, outcomes, manifest, config, nominal_report
    )
    report["source"] = {
        "run_fingerprint": run_config["run_fingerprint"],
        "source_run_fingerprint": run_config["source_run_fingerprint"],
        "candidate_fingerprint": run_config["candidate_fingerprint"],
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "realized_neighborhood_probe.json", report)
    (output_root / "realized_neighborhood_probe.md").write_text(
        render_report(report), encoding="utf-8"
    )
    return report


__all__ = [
    "CollectionLockError",
    "analyze_realized_collection",
    "analyze_realized_records",
    "build_candidate_rows",
    "evaluation_seed",
    "run_realized_collection",
    "select_representative_neighborhoods",
]
