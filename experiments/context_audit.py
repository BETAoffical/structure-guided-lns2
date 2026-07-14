from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import pickle
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.repair_quality import pareto_indices


AUDIT_SCHEMA_VERSION = 1
MODEL_SEED = 20260714
FEATURE_PROFILES = ("action_seed", "dynamic", "full_context")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing collection file: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"collection path escapes its root: {relative}") from error
    return path


def _portable_path(value: str) -> Path:
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        drive = value[5].upper()
        return Path(f"{drive}:/{value[7:]}")
    return Path(value)


def _dataset_root(collection: Path, explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        root = Path(explicit).resolve()
        if not root.is_dir():
            raise ValueError(f"dataset directory does not exist: {root}")
        return root
    config_path = collection / "run_config.json"
    if not config_path.is_file():
        return None
    configured = _portable_path(
        str(json.loads(config_path.read_text(encoding="utf-8")).get("dataset", ""))
    )
    return configured.resolve() if configured.is_dir() else None


def _dataset_contexts(root: Path | None) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}
    contexts: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(root.glob("*/manifest.jsonl")):
        split_root = manifest_path.parent
        for row in _read_jsonl(manifest_path):
            task_path = _resolve(split_root, str(row["task_file"]))
            task = json.loads(task_path.read_text(encoding="utf-8"))
            metadata = dict(task.get("metadata", {}))
            contexts[str(row["task_id"])] = {
                "flow_type": row.get("flow_type", metadata.get("flow_type")),
                "opposing_flow_ratio": metadata.get("opposing_flow_ratio", 0),
                "shared_corridor_ratio": metadata.get("shared_corridor_ratio", 0),
                "realized_intersection_crossing_ratio": metadata.get(
                    "realized_intersection_crossing_ratio", 0
                ),
                "agent_density_free_cells": metadata.get(
                    "agent_density_free_cells", 0
                ),
                "agent_density_service_cells": metadata.get(
                    "agent_density_service_cells", 0
                ),
                "minimum_shortest_distance": metadata.get(
                    "minimum_shortest_distance", 0
                ),
                "maximum_shortest_distance": metadata.get(
                    "maximum_shortest_distance", 0
                ),
                "realized_flow_counts": metadata.get("realized_flow_counts", {}),
                "od_quota_counts": metadata.get("od_quota_counts", {}),
                "candidate_intersection_count": len(
                    metadata.get("candidate_intersection_components", [])
                ),
                "selected_intersection_count": len(
                    metadata.get("selected_intersection_components", [])
                ),
            }
    return contexts


def _mean(values: Iterable[float | int]) -> float:
    numbers = [float(value) for value in values]
    return statistics.fmean(numbers) if numbers else 0.0


def _std(values: Iterable[float | int]) -> float:
    numbers = [float(value) for value in values]
    return statistics.pstdev(numbers) if len(numbers) > 1 else 0.0


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _horizon(row: dict[str, Any], horizon: int) -> dict[str, Any] | None:
    for value in row.get("horizon_outcomes", []):
        if int(value["horizon"]) == horizon and bool(value.get("available", True)):
            return value
    return None


def _candidate_key(row: dict[str, Any]) -> tuple[int, str, int]:
    action = row["candidate_action"]
    return (
        int(action["seed_agent"]),
        str(action["heuristic"]),
        int(action["neighborhood_size"]),
    )


def _average_outcomes(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    values = [_horizon(row, horizon) for row in rows]
    available = [value for value in values if value is not None]
    if not available:
        raise ValueError(f"candidate has no Horizon {horizon} outcome")
    generated = [
        float(value.get("low_level_delta", {}).get("generated", 0))
        for value in available
    ]
    solved_rate = _mean(float(bool(value["solved"])) for value in available)
    return {
        "solved": solved_rate >= 1.0,
        "solved_rate": solved_rate,
        "conflicts_after": _mean(value["conflicts_after"] for value in available),
        "conflict_auc": _mean(value["conflict_auc"] for value in available),
        "generated": _mean(generated),
        "branch_runtime": _mean(value["branch_runtime"] for value in available),
        "trial_count": len(available),
        "low_level_delta": {"generated": _mean(generated)},
    }


def _connected_components(
    agent_count: int, edges: list[list[int]]
) -> tuple[int, int, dict[int, int]]:
    adjacency: list[list[int]] = [[] for _ in range(agent_count)]
    active: set[int] = set()
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
        active.update((int(left), int(right)))
    component_sizes: dict[int, int] = {}
    component_count = 0
    largest = 0
    visited: set[int] = set()
    for start in sorted(active):
        if start in visited:
            continue
        stack = [start]
        members: list[int] = []
        visited.add(start)
        while stack:
            vertex = stack.pop()
            members.append(vertex)
            for neighbor in adjacency[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        size = len(members)
        component_count += 1
        largest = max(largest, size)
        for member in members:
            component_sizes[member] = size
    return component_count, largest, component_sizes


def _categorical(features: dict[str, float], prefix: str, value: Any) -> None:
    normalized = str(value or "unknown").strip().lower().replace(" ", "_")
    features[f"{prefix}={normalized}"] = 1.0


def candidate_features(
    state_row: dict[str, Any],
    outcome_row: dict[str, Any],
    stage: str,
    dataset_context: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    state = state_row["state"]
    context = dict(state.get("context", {}))
    action = outcome_row["candidate_action"]
    agents = list(state.get("agents", []))
    by_id = {int(agent["id"]): agent for agent in agents}
    seed_id = int(action["seed_agent"])
    if seed_id not in by_id:
        raise ValueError(f"seed agent {seed_id} is absent from state")
    seed = by_id[seed_id]
    agent_count = max(len(agents), int(context.get("agent_count", len(agents))))
    edges = list(state.get("conflict_edges", []))
    component_count, largest_component, component_sizes = _connected_components(
        agent_count, edges
    )
    conflict_degrees = [float(agent.get("conflict_degree", 0)) for agent in agents]
    delays = [float(agent.get("delay", 0)) for agent in agents]
    path_costs = [float(agent.get("path_cost", 0)) for agent in agents]
    shortest_costs = [float(agent.get("shortest_path_cost", 0)) for agent in agents]
    rows = int(state.get("rows", 0))
    cols = int(state.get("cols", 0))
    obstacles = list(state.get("obstacles", []))
    free_cells = max(0, len(obstacles) - sum(int(value) for value in obstacles))

    action_seed: dict[str, float] = {
        "action.neighborhood_size": float(action["neighborhood_size"]),
        "seed.conflict_degree": float(seed.get("conflict_degree", 0)),
        "seed.delay": float(seed.get("delay", 0)),
        "seed.path_cost": float(seed.get("path_cost", 0)),
        "seed.shortest_path_cost": float(seed.get("shortest_path_cost", 0)),
        "seed.path_stretch": _safe_ratio(
            seed.get("path_cost", 0), max(1, seed.get("shortest_path_cost", 0))
        ),
        "seed.component_size": float(component_sizes.get(seed_id, 0)),
    }
    _categorical(action_seed, "action.heuristic", action["heuristic"])

    dynamic = dict(action_seed)
    dynamic.update(
        {
            "state.iteration": float(state.get("iteration", 0)),
            "state.colliding_pairs": float(state.get("num_of_colliding_pairs", 0)),
            "state.conflict_edge_density": _safe_ratio(len(edges), agent_count),
            "state.conflicting_agent_ratio": _safe_ratio(
                sum(value > 0 for value in conflict_degrees), agent_count
            ),
            "state.component_count": float(component_count),
            "state.largest_component": float(largest_component),
            "state.largest_component_ratio": _safe_ratio(largest_component, agent_count),
            "state.degree_mean": _mean(conflict_degrees),
            "state.degree_std": _std(conflict_degrees),
            "state.degree_max": max(conflict_degrees, default=0.0),
            "state.delay_mean": _mean(delays),
            "state.delay_std": _std(delays),
            "state.delay_max": max(delays, default=0.0),
            "state.path_cost_mean": _mean(path_costs),
            "state.path_stretch_mean": _safe_ratio(
                sum(path_costs), max(1.0, sum(shortest_costs))
            ),
            "state.sum_of_costs_per_agent": _safe_ratio(
                state.get("sum_of_costs", 0), agent_count
            ),
            "state.low_level_generated_per_agent": _safe_ratio(
                state.get("low_level", {}).get("generated", 0), agent_count
            ),
            "state.low_level_runs_per_agent": _safe_ratio(
                state.get("low_level", {}).get("runs", 0), agent_count
            ),
        }
    )
    _categorical(dynamic, "state.stage", stage)

    full = dict(dynamic)
    topology = dict(context.get("topology_metrics", {}))
    dataset_context = dataset_context or {}
    full.update(
        {
            "context.agent_count": float(agent_count),
            "context.rows": float(rows),
            "context.cols": float(cols),
            "context.free_cells": float(free_cells),
            "context.free_cell_ratio": _safe_ratio(free_cells, max(1, rows * cols)),
            "context.agent_density": _safe_ratio(agent_count, max(1, free_cells)),
            "context.mean_shortest_distance": float(
                context.get("mean_shortest_distance", 0)
            ),
            "context.dominant_flow_ratio": float(
                context.get("dominant_flow_ratio", 0)
            ),
            "context.hotspot_skew": float(context.get("hotspot_skew", 0)),
            "context.required_bottleneck_crossing_ratio": float(
                context.get("required_bottleneck_crossing_ratio", 0)
            ),
            "context.required_intersection_crossing_ratio": float(
                context.get("required_intersection_crossing_ratio", 0)
            ),
            "context.swap_pair_ratio": float(context.get("swap_pair_ratio", 0)),
            "context.articulation_count": float(
                topology.get("articulation_count", 0)
            ),
            "context.average_free_degree": float(
                topology.get("average_free_degree", 0)
            ),
            "context.dead_end_cell_count": float(
                topology.get("dead_end_cell_count", 0)
            ),
            "context.route_redundancy_proxy": float(
                topology.get("route_redundancy_proxy", 0)
            ),
            "context.opposing_flow_ratio": float(
                dataset_context.get("opposing_flow_ratio", 0)
            ),
            "context.shared_corridor_ratio": float(
                dataset_context.get("shared_corridor_ratio", 0)
            ),
            "context.realized_intersection_crossing_ratio": float(
                dataset_context.get("realized_intersection_crossing_ratio", 0)
            ),
            "context.agent_density_free_cells": float(
                dataset_context.get("agent_density_free_cells", 0)
            ),
            "context.agent_density_service_cells": float(
                dataset_context.get("agent_density_service_cells", 0)
            ),
            "context.minimum_shortest_distance": float(
                dataset_context.get("minimum_shortest_distance", 0)
            ),
            "context.maximum_shortest_distance": float(
                dataset_context.get("maximum_shortest_distance", 0)
            ),
            "context.candidate_intersection_count": float(
                dataset_context.get("candidate_intersection_count", 0)
            ),
            "context.selected_intersection_count": float(
                dataset_context.get("selected_intersection_count", 0)
            ),
        }
    )
    for group in ("realized_flow_counts", "od_quota_counts"):
        for name, count in dict(dataset_context.get(group, {}) or {}).items():
            full[f"context.{group}.{name}_ratio"] = _safe_ratio(count, agent_count)
    _categorical(full, "context.layout_mode", context.get("layout_mode"))
    _categorical(full, "context.layout_variant", context.get("layout_variant"))
    _categorical(full, "context.scenario_type", context.get("scenario_type"))
    _categorical(full, "context.task_variant", context.get("task_variant"))
    _categorical(full, "context.flow_type", dataset_context.get("flow_type"))
    return {
        "action_seed": action_seed,
        "dynamic": dynamic,
        "full_context": full,
    }


def _stage_labels(states: list[dict[str, Any]]) -> dict[str, str]:
    by_episode: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for state in states:
        by_episode[str(state["episode_id"])].append(state)
    labels: dict[str, str] = {}
    for values in by_episode.values():
        values.sort(key=lambda row: int(row["decision_index"]))
        if len(values) == 1:
            labels[str(values[0]["state_id"])] = "only"
            continue
        for index, value in enumerate(values):
            label = "early" if index == 0 else "late" if index == len(values) - 1 else "middle"
            labels[str(value["state_id"])] = label
    return labels


def build_index(
    collection: str | Path,
    horizon: int = 4,
    dataset: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(collection).resolve()
    dataset_contexts = _dataset_contexts(_dataset_root(root, dataset))
    manifests = _read_jsonl(root / "counterfactual_manifest.jsonl")
    states: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for manifest in manifests:
        if str(manifest.get("status")) not in {"ok", "resumed"}:
            continue
        states.extend(_read_jsonl(_resolve(root, str(manifest["states_file"]))))
        outcomes.extend(_read_jsonl(_resolve(root, str(manifest["outcomes_file"]))))
    stage_labels = _stage_labels(states)
    state_index = {str(row["state_id"]): row for row in states}
    grouped: dict[tuple[str, tuple[int, str, int]], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in outcomes:
        if bool(row.get("action_valid", False)) and _horizon(row, horizon) is not None:
            grouped[(str(row["state_id"]), _candidate_key(row))].append(row)

    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for (state_id, key), rows in sorted(grouped.items()):
        state_row = state_index[state_id]
        averaged = _average_outcomes(rows, horizon)
        representative = rows[0]
        context = state_row["state"].get("context", {})
        features = candidate_features(
            state_row,
            representative,
            stage_labels.get(state_id, "unknown"),
            dataset_contexts.get(str(context.get("task_id", ""))),
        )
        by_state[state_id].append(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "state_id": state_id,
                "episode_id": str(state_row["episode_id"]),
                "split": str(context.get("split", "unknown")),
                "map_id": str(context.get("map_id", "unknown")),
                "task_id": str(context.get("task_id", "unknown")),
                "decision_index": int(state_row["decision_index"]),
                "stage": stage_labels.get(state_id, "unknown"),
                "candidate_key": f"{key[0]}:{key[1]}:{key[2]}",
                "candidate_action": {
                    "seed_agent": key[0],
                    "heuristic": key[1],
                    "neighborhood_size": key[2],
                },
                "features": features,
                "outcome": averaged,
            }
        )

    result: list[dict[str, Any]] = []
    for state_id, candidates in sorted(by_state.items()):
        values = [candidate["outcome"] for candidate in candidates]
        pareto = set(pareto_indices(values))
        for index, candidate in enumerate(candidates):
            candidate["pareto"] = index in pareto
            candidate["candidate_index"] = index
            result.append(candidate)
    return result


def _validate_split_isolation(index: list[dict[str, Any]]) -> dict[str, Any]:
    unexpected = sorted(
        {str(row["split"]) for row in index} - {"train", "validation"}
    )
    if unexpected:
        raise ValueError(
            f"context audit contains forbidden Test/OOD label splits: {unexpected}"
        )
    train = [row for row in index if row["split"] == "train"]
    validation = [row for row in index if row["split"] == "validation"]
    train_maps = {str(row["map_id"]) for row in train}
    validation_maps = {str(row["map_id"]) for row in validation}
    train_tasks = {str(row["task_id"]) for row in train}
    validation_tasks = {str(row["task_id"]) for row in validation}
    map_overlap = sorted(train_maps & validation_maps)
    task_overlap = sorted(train_tasks & validation_tasks)
    if map_overlap or task_overlap:
        raise ValueError(
            "Train/Validation isolation failed: "
            f"map overlap={map_overlap}, task overlap={task_overlap}"
        )
    return {
        "train_map_count": len(train_maps),
        "validation_map_count": len(validation_maps),
        "train_task_count": len(train_tasks),
        "validation_task_count": len(validation_tasks),
        "map_overlap": map_overlap,
        "task_overlap": task_overlap,
    }


def _feature_names(rows: list[dict[str, Any]], profile: str) -> list[str]:
    return sorted(
        {
            name
            for row in rows
            for name in row["features"][profile]
        }
    )


def _vector(row: dict[str, Any], profile: str, names: list[str]) -> list[float]:
    values = row["features"][profile]
    return [float(values.get(name, 0.0)) for name in names]


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [first - second for first, second in zip(left, right)]


def _pair_vector(
    left: dict[str, Any],
    right: dict[str, Any],
    profile: str,
    names: list[str],
) -> list[float]:
    left_vector = _vector(left, profile, names)
    right_vector = _vector(right, profile, names)
    shared = [
        (first + second) / 2.0
        for name, first, second in zip(names, left_vector, right_vector)
        if name.startswith(("state.", "context."))
    ]
    return _subtract(left_vector, right_vector) + shared


def _objective_values(value: dict[str, Any]) -> tuple[float, ...]:
    return (
        -float(bool(value["solved"])),
        float(value["conflicts_after"]),
        float(value["conflict_auc"]),
        float(value["generated"]),
        float(value["branch_runtime"]),
    )


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = _objective_values(left)
    right_values = _objective_values(right)
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _pairwise_examples(
    rows: list[dict[str, Any]], profile: str, names: list[str]
) -> tuple[list[list[float]], list[int]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    examples: list[list[float]] = []
    labels: list[int] = []
    for candidates in grouped.values():
        candidates.sort(key=lambda row: str(row["candidate_key"]))
        for left_index, left in enumerate(candidates):
            for right_index in range(left_index + 1, len(candidates)):
                right = candidates[right_index]
                if _dominates(left["outcome"], right["outcome"]):
                    examples.append(_pair_vector(left, right, profile, names))
                    labels.append(1)
                    examples.append(_pair_vector(right, left, profile, names))
                    labels.append(0)
                elif _dominates(right["outcome"], left["outcome"]):
                    examples.append(_pair_vector(left, right, profile, names))
                    labels.append(0)
                    examples.append(_pair_vector(right, left, profile, names))
                    labels.append(1)
    if not examples:
        raise ValueError("no Pareto-dominance pairs are available for training")
    return examples, labels


@dataclass
class PairwiseModel:
    profile: str
    feature_names: list[str]
    estimator: Any

    def select(self, candidates: list[dict[str, Any]]) -> int:
        import numpy as np

        comparisons: list[list[float]] = []
        reverse_comparisons: list[list[float]] = []
        pairs: list[tuple[int, int]] = []
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                comparisons.append(
                    _pair_vector(
                        candidates[left],
                        candidates[right],
                        self.profile,
                        self.feature_names,
                    )
                )
                reverse_comparisons.append(
                    _pair_vector(
                        candidates[right],
                        candidates[left],
                        self.profile,
                        self.feature_names,
                    )
                )
                pairs.append((left, right))
        if not comparisons:
            return 0
        forward = self.estimator.predict_proba(
            np.asarray(comparisons, dtype=float)
        )[:, 1]
        reverse = self.estimator.predict_proba(
            np.asarray(reverse_comparisons, dtype=float)
        )[:, 1]
        probabilities = (forward + (1.0 - reverse)) / 2.0
        scores = [0.0] * len(candidates)
        for probability, (left, right) in zip(probabilities, pairs):
            scores[left] += float(probability)
            scores[right] += 1.0 - float(probability)
        return min(
            range(len(candidates)),
            key=lambda index: (-scores[index], str(candidates[index]["candidate_key"])),
        )


def train_models(
    index: list[dict[str, Any]], output: str | Path
) -> dict[str, PairwiseModel]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    output_root = Path(output).resolve()
    train_rows = [row for row in index if row["split"] == "train"]
    if not train_rows:
        raise ValueError("context audit index has no Train rows")
    models: dict[str, PairwiseModel] = {}
    for profile in FEATURE_PROFILES:
        names = _feature_names(train_rows, profile)
        examples, labels = _pairwise_examples(train_rows, profile, names)
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=MODEL_SEED,
        )
        estimator.fit(
            np.asarray(examples, dtype=float), np.asarray(labels, dtype=int)
        )
        model = PairwiseModel(profile, names, estimator)
        models[profile] = model
        path = output_root / "models" / f"{profile}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(model, stream)
    return models


def _bootstrap_interval(values: list[float], samples: int = 2000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(MODEL_SEED)
    estimates = []
    for _ in range(samples):
        estimates.append(_mean(rng.choice(values) for _ in values))
    estimates.sort()
    return [
        estimates[int(0.025 * (len(estimates) - 1))],
        estimates[int(0.975 * (len(estimates) - 1))],
    ]


def _profile_evaluation(
    rows: list[dict[str, Any]], model: PairwiseModel
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    records: dict[str, dict[str, float]] = {}
    families: collections.Counter[str] = collections.Counter()
    for state_id, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda row: str(row["candidate_key"]))
        selected = candidates[model.select(candidates)]
        minimum_auc = min(float(row["outcome"]["conflict_auc"]) for row in candidates)
        minimum_conflicts = min(
            float(row["outcome"]["conflicts_after"]) for row in candidates
        )
        selected_auc = float(selected["outcome"]["conflict_auc"])
        selected_conflicts = float(selected["outcome"]["conflicts_after"])
        action = selected["candidate_action"]
        family = f"{action['heuristic']}:{int(action['neighborhood_size'])}"
        families[family] += 1
        records[state_id] = {
            "pareto_hit": float(bool(selected["pareto"])),
            "auc_regret": _safe_ratio(selected_auc - minimum_auc, max(1.0, abs(minimum_auc))),
            "conflict_regret": _safe_ratio(
                selected_conflicts - minimum_conflicts, max(1.0, abs(minimum_conflicts))
            ),
        }
    return (
        {
            "state_count": len(records),
            "pareto_top1_hit_rate": _mean(row["pareto_hit"] for row in records.values()),
            "mean_auc_regret": _mean(row["auc_regret"] for row in records.values()),
            "mean_conflict_regret": _mean(
                row["conflict_regret"] for row in records.values()
            ),
            "selected_action_families": dict(sorted(families.items())),
        },
        records,
    )


def evaluate_models(
    index: list[dict[str, Any]], models: dict[str, PairwiseModel]
) -> dict[str, Any]:
    validation = [row for row in index if row["split"] == "validation"]
    if not validation:
        raise ValueError("context audit index has no Validation rows")
    evaluations: dict[str, Any] = {}
    records: dict[str, dict[str, dict[str, float]]] = {}
    for profile, model in models.items():
        evaluations[profile], records[profile] = _profile_evaluation(validation, model)
    dynamic = records["dynamic"]
    full = records["full_context"]
    state_ids = sorted(set(dynamic) & set(full))
    hit_differences = [
        full[state_id]["pareto_hit"] - dynamic[state_id]["pareto_hit"]
        for state_id in state_ids
    ]
    auc_improvements = [
        dynamic[state_id]["auc_regret"] - full[state_id]["auc_regret"]
        for state_id in state_ids
    ]
    dynamic_auc = evaluations["dynamic"]["mean_auc_regret"]
    full_auc = evaluations["full_context"]["mean_auc_regret"]
    relative_auc_reduction = _safe_ratio(dynamic_auc - full_auc, max(1e-12, dynamic_auc))
    hit_gain = (
        evaluations["full_context"]["pareto_top1_hit_rate"]
        - evaluations["dynamic"]["pareto_top1_hit_rate"]
    )
    hit_interval = _bootstrap_interval(hit_differences)
    auc_interval = _bootstrap_interval(auc_improvements)
    gates = [
        {
            "name": "pareto_hit_gain",
            "actual": hit_gain,
            "requirement": ">= 0.05",
            "passed": hit_gain >= 0.05,
        },
        {
            "name": "relative_auc_regret_reduction",
            "actual": relative_auc_reduction,
            "requirement": ">= 0.05",
            "passed": relative_auc_reduction >= 0.05,
        },
        {
            "name": "paired_bootstrap_no_significant_degradation",
            "actual": {
                "hit_gain_95_ci": hit_interval,
                "auc_improvement_95_ci": auc_interval,
            },
            "requirement": "both upper bounds >= 0",
            "passed": hit_interval[1] >= 0.0 and auc_interval[1] >= 0.0,
        },
    ]
    return {
        "profiles": evaluations,
        "comparison": {
            "full_context_minus_dynamic_hit_rate": hit_gain,
            "relative_auc_regret_reduction": relative_auc_reduction,
            "hit_gain_95_ci": hit_interval,
            "auc_improvement_95_ci": auc_interval,
        },
        "acceptance": {
            "passed": all(bool(gate["passed"]) for gate in gates),
            "gates": gates,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# InitLNS Context Learnability Audit",
        "",
        f"Offline acceptance: **{'PASS' if report['acceptance']['passed'] else 'FAIL'}**",
        "",
        "## Dataset",
        "",
        f"- Candidate rows: {report['counts']['candidate_rows']}",
        f"- Train states: {report['counts']['train_states']}",
        f"- Validation states: {report['counts']['validation_states']}",
        "",
        "## Ablations",
        "",
        "| Profile | States | Pareto top-1 | Mean AUC regret | Mean conflict regret |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for profile in FEATURE_PROFILES:
        value = report["profiles"][profile]
        lines.append(
            f"| {profile} | {value['state_count']} | "
            f"{value['pareto_top1_hit_rate']:.4f} | "
            f"{value['mean_auc_regret']:.4f} | "
            f"{value['mean_conflict_regret']:.4f} |"
        )
    comparison = report["comparison"]
    lines.extend(
        [
            "",
            "## Full Context vs Dynamic",
            "",
            f"- Pareto hit-rate gain: {comparison['full_context_minus_dynamic_hit_rate']:.4f}",
            f"- Relative AUC-regret reduction: {comparison['relative_auc_regret_reduction']:.4f}",
            f"- Hit gain 95% CI: {comparison['hit_gain_95_ci']}",
            f"- AUC improvement 95% CI: {comparison['auc_improvement_95_ci']}",
            "",
            "## Gates",
            "",
            "| Gate | Actual | Requirement | Result |",
            "| --- | --- | --- | --- |",
        ]
    )
    for gate in report["acceptance"]["gates"]:
        actual = json.dumps(gate["actual"], sort_keys=True)
        lines.append(
            f"| {gate['name']} | `{actual}` | {gate['requirement']} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(["", "## Selected Action Families", ""])
    for profile in FEATURE_PROFILES:
        families = report["profiles"][profile]["selected_action_families"]
        rendered = ", ".join(f"{name}={count}" for name, count in families.items())
        lines.append(f"- `{profile}`: {rendered}")
    return "\n".join(lines) + "\n"


def run_audit(
    collection: str | Path,
    output: str | Path,
    dataset: str | Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output).resolve()
    collection_root = Path(collection).resolve()
    resolved_dataset = _dataset_root(collection_root, dataset)
    index = build_index(collection_root, dataset=resolved_dataset)
    isolation = _validate_split_isolation(index)
    _write_jsonl(output_root / "candidate_index.jsonl", index)
    models = train_models(index, output_root)
    evaluation = evaluate_models(index, models)
    state_counts = {
        split: len({row["state_id"] for row in index if row["split"] == split})
        for split in ("train", "validation")
    }
    digest = hashlib.sha256(
        "\n".join(
            f"{row['state_id']}|{row['candidate_key']}|{row['pareto']}"
            for row in index
        ).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "model_seed": MODEL_SEED,
        "collection": str(collection_root),
        "dataset": str(resolved_dataset) if resolved_dataset else None,
        "index_sha256": digest,
        "counts": {
            "candidate_rows": len(index),
            "train_states": state_counts["train"],
            "validation_states": state_counts["validation"],
        },
        "split_isolation": isolation,
        **evaluation,
    }
    _write_json(output_root / "context_audit.json", report)
    (output_root / "context_audit.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    closed_loop = {
        "schema_version": 1,
        "executed": False,
        "status": (
            "ready" if report["acceptance"]["passed"] else "blocked_by_offline_gate"
        ),
        "offline_acceptance": bool(report["acceptance"]["passed"]),
        "required_validation_instance_seeds": 24,
        "requirements": {
            "successes": ">= official Adaptive",
            "conflict_auc_or_time_to_feasible": ">= 5% improvement",
        },
        "reason": (
            "The offline context gate passed; run the native closed-loop evaluator."
            if report["acceptance"]["passed"]
            else "The offline context gate failed, so closed-loop evaluation is intentionally not run."
        ),
    }
    _write_json(output_root / "closed_loop_gate.json", closed_loop)
    (output_root / "closed_loop_gate.md").write_text(
        "# InitLNS Closed-Loop Gate\n\n"
        f"Status: **{closed_loop['status']}**\n\n"
        f"{closed_loop['reason']}\n",
        encoding="utf-8",
    )
    return report
