from __future__ import annotations

import collections
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable


K_OPTIONS = (3, 5, 7, 11)
RUN_WEIGHT_OPTIONS = (
    {"map": 1.0, "task": 1.0},
    {"map": 2.0, "task": 1.0},
    {"map": 1.0, "task": 2.0},
)
REPAIR_WEIGHT_OPTIONS = tuple(
    {"map": float(map_weight), "task": float(task_weight), "conflict": float(conflict_weight)}
    for map_weight, task_weight, conflict_weight in itertools.product(
        (0.5, 1.0, 2.0), repeat=3
    )
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _group_for_key(key: str) -> str:
    return key.split(".", 1)[0]


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _base_raw_features(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    numeric: dict[str, float] = {}
    categorical: dict[str, str] = {}
    for name, value in case["map_features"].items():
        if _numeric(value):
            numeric[f"map.{name}"] = float(value)
    task_features = case["task_features"]
    for name, value in task_features.items():
        if name == "realized_flow_counts":
            total = max(1.0, float(sum(value.values())))
            for flow, count in sorted(value.items()):
                numeric[f"task.flow_ratio.{flow}"] = float(count) / total
        elif _numeric(value):
            numeric[f"task.{name}"] = float(value)
    return {"numeric": numeric, "categorical": categorical}


def run_raw_features(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _base_raw_features(case)


def _path_overlap(path: list[list[int]], cells: set[tuple[int, int]]) -> float:
    if not path:
        return 0.0
    return sum(tuple(cell) in cells for cell in path) / len(path)


def repair_raw_features(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = _base_raw_features(case)
    numeric = raw["numeric"]
    categorical = raw["categorical"]
    events = case["conflict_events_before"]
    heatmap = case["conflict_heatmap_before"]
    rows = max(1.0, float(case["map_features"]["rows"] - 1))
    cols = max(1.0, float(case["map_features"]["cols"] - 1))
    total_weight = sum(float(item["weight"]) for item in heatmap)
    if total_weight:
        centroid_row = sum(
            item["cell"][0] * float(item["weight"]) for item in heatmap
        ) / total_weight
        centroid_col = sum(
            item["cell"][1] * float(item["weight"]) for item in heatmap
        ) / total_weight
        row_variance = sum(
            ((item["cell"][0] - centroid_row) / rows) ** 2
            * float(item["weight"])
            for item in heatmap
        ) / total_weight
        col_variance = sum(
            ((item["cell"][1] - centroid_col) / cols) ** 2
            * float(item["weight"])
            for item in heatmap
        ) / total_weight
    else:
        centroid_row = centroid_col = row_variance = col_variance = 0.0
    numeric.update(
        {
            "conflict.event_count": float(len(events)),
            "conflict.vertex_ratio": (
                sum(event["type"] == "vertex" for event in events)
                / max(1, len(events))
            ),
            "conflict.edge_swap_ratio": (
                sum(event["type"] == "edge_swap" for event in events)
                / max(1, len(events))
            ),
            "conflict.centroid_row": centroid_row / rows,
            "conflict.centroid_col": centroid_col / cols,
            "conflict.row_spread": math.sqrt(row_variance),
            "conflict.col_spread": math.sqrt(col_variance),
        }
    )
    seed_pair = sorted(int(value) for value in case["seed_conflict"])
    seed_event = next(
        (
            event
            for event in events
            if sorted(int(value) for value in event["agents"]) == seed_pair
        ),
        None,
    )
    categorical["conflict.seed_type"] = (
        str(seed_event["type"]) if seed_event is not None else "unknown"
    )
    if seed_event and seed_event["cells"]:
        numeric["conflict.seed_row"] = (
            sum(cell[0] for cell in seed_event["cells"])
            / len(seed_event["cells"])
            / rows
        )
        numeric["conflict.seed_col"] = (
            sum(cell[1] for cell in seed_event["cells"])
            / len(seed_event["cells"])
            / cols
        )
    else:
        numeric["conflict.seed_row"] = 0.0
        numeric["conflict.seed_col"] = 0.0

    agents = {int(agent["agent"]): agent for agent in case["agents"]}
    conflict_cells = {tuple(item["cell"]) for item in heatmap}
    for slot, agent_id in enumerate(seed_pair):
        agent = agents.get(agent_id)
        prefix = f"conflict.seed_agent_{slot}"
        if agent is None:
            categorical[f"{prefix}.start_zone"] = "unknown"
            categorical[f"{prefix}.goal_zone"] = "unknown"
            categorical[f"{prefix}.flow"] = "unknown"
            numeric[f"{prefix}.shortest_distance"] = 0.0
            numeric[f"{prefix}.path_stretch"] = 0.0
            numeric[f"{prefix}.path_conflict_overlap"] = 0.0
            continue
        categorical[f"{prefix}.start_zone"] = str(agent["start_zone"])
        categorical[f"{prefix}.goal_zone"] = str(agent["goal_zone"])
        categorical[f"{prefix}.flow"] = str(agent["flow_assignment"])
        shortest = max(1.0, float(agent["shortest_distance"]))
        path = agent["path_before"]
        numeric[f"{prefix}.shortest_distance"] = shortest
        numeric[f"{prefix}.path_stretch"] = (len(path) - 1) / shortest
        numeric[f"{prefix}.path_conflict_overlap"] = _path_overlap(
            path, conflict_cells
        )
    return raw


def fit_feature_schema(
    raw_rows: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not raw_rows:
        raise ValueError("cannot fit a feature schema without rows")
    numeric_keys = sorted(
        {key for row in raw_rows for key in row["numeric"]}
    )
    categorical_keys = sorted(
        {key for row in raw_rows for key in row["categorical"]}
    )
    fields: list[dict[str, Any]] = []
    dropped: list[str] = []
    for key in numeric_keys:
        values = [float(row["numeric"].get(key, 0.0)) for row in raw_rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        standard_deviation = math.sqrt(variance)
        if standard_deviation <= 1e-12:
            dropped.append(key)
            continue
        fields.append(
            {
                "name": key,
                "kind": "numeric",
                "group": _group_for_key(key),
                "mean": mean,
                "standard_deviation": standard_deviation,
            }
        )
    for key in categorical_keys:
        values = sorted(
            {str(row["categorical"].get(key, "__unknown__")) for row in raw_rows}
        )
        if len(values) <= 1:
            dropped.append(key)
            continue
        for value in [*values, "__unknown__"]:
            fields.append(
                {
                    "name": f"{key}={value}",
                    "source": key,
                    "value": value,
                    "kind": "categorical",
                    "group": _group_for_key(key),
                }
            )
    return {
        "schema_version": 1,
        "fields": fields,
        "dropped_zero_variance": dropped,
        "feature_count": len(fields),
    }


def vectorize(
    raw: dict[str, dict[str, Any]], schema: dict[str, Any]
) -> list[float]:
    result = []
    known_values: dict[str, set[str]] = collections.defaultdict(set)
    for field in schema["fields"]:
        if field["kind"] == "categorical":
            known_values[field["source"]].add(field["value"])
    for field in schema["fields"]:
        if field["kind"] == "numeric":
            value = float(raw["numeric"].get(field["name"], field["mean"]))
            result.append(
                (value - field["mean"]) / field["standard_deviation"]
            )
        else:
            source = field["source"]
            value = str(raw["categorical"].get(source, "__unknown__"))
            if value not in known_values[source]:
                value = "__unknown__"
            result.append(1.0 if value == field["value"] else 0.0)
    return result


def grouped_distance(
    first: list[float],
    second: list[float],
    schema: dict[str, Any],
    weights: dict[str, float],
) -> float:
    if len(first) != len(second) or len(first) != len(schema["fields"]):
        raise ValueError("feature vector length mismatch")
    totals: collections.defaultdict[str, float] = collections.defaultdict(float)
    counts: collections.Counter[str] = collections.Counter()
    for left, right, field in zip(first, second, schema["fields"]):
        group = field["group"]
        totals[group] += (left - right) ** 2
        counts[group] += 1
    weighted_total = 0.0
    weight_total = 0.0
    for group, count in counts.items():
        weight = float(weights.get(group, 1.0))
        weighted_total += weight * totals[group] / count
        weight_total += weight
    return math.sqrt(weighted_total / max(weight_total, 1e-12))


def _merge_average_heatmaps(
    rows: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    weights: collections.defaultdict[tuple[int, int], float] = (
        collections.defaultdict(float)
    )
    for row in rows:
        for item in row[field]:
            weights[tuple(item["cell"])] += float(item["weight"]) / len(rows)
    return [
        {"cell": [row, col], "weight": round(weight, 6)}
        for (row, col), weight in sorted(weights.items())
        if weight > 0.0
    ]


def aggregate_run_cases(
    run_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_task: collections.defaultdict[str, list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for run in run_cases:
        by_task[str(run["task_id"])].append(run)
    prototypes = []
    for task_id, rows in sorted(by_task.items()):
        rows.sort(key=lambda item: (int(item["solver_seed"]), item["run_id"]))
        first = rows[0]
        prototypes.append(
            {
                "prototype_id": task_id,
                "split": first["split"],
                "map_id": first["map_id"],
                "task_id": task_id,
                "source_run_ids": [row["run_id"] for row in rows],
                "solver_seeds": [row["solver_seed"] for row in rows],
                "map_features": first["map_features"],
                "task_features": first["task_features"],
                "conflict_probability": round(
                    sum(bool(row["conflict_heatmap"]) for row in rows)
                    / len(rows),
                    6,
                ),
                "conflict_heatmap": _merge_average_heatmaps(
                    rows, "conflict_heatmap"
                ),
            }
        )
    return prototypes


def _role_template(case: dict[str, Any]) -> dict[str, Any]:
    seed_agents = {int(value) for value in case["seed_conflict"]}
    conflict_cells = {
        tuple(item["cell"]) for item in case["conflict_heatmap_before"]
    }
    roles = []
    for agent in case["agents"]:
        shortest = max(1.0, float(agent["shortest_distance"]))
        roles.append(
            {
                "is_seed": int(agent["agent"]) in seed_agents,
                "start_zone": str(agent["start_zone"]),
                "goal_zone": str(agent["goal_zone"]),
                "flow_assignment": str(agent["flow_assignment"]),
                "shortest_distance": float(agent["shortest_distance"]),
                "path_stretch": (len(agent["path_before"]) - 1) / shortest,
                "path_conflict_overlap": _path_overlap(
                    agent["path_before"], conflict_cells
                ),
            }
        )
    return {"neighborhood_size": len(roles), "roles": roles}


def _public_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": schema["schema_version"],
        "feature_count": schema["feature_count"],
        "feature_names": [field["name"] for field in schema["fields"]],
        "feature_groups": [field["group"] for field in schema["fields"]],
        "dropped_zero_variance": schema["dropped_zero_variance"],
    }


def build_retrieval_index(
    memory: str | Path, output: str | Path
) -> dict[str, Any]:
    memory_root = Path(memory).resolve()
    output_root = Path(output).resolve()
    summary = _read_json(memory_root / "experience_summary.json")
    if summary.get("split") != "train" or summary.get("usage", "memory") != "memory":
        raise ValueError("retrieval memory must be Train-only")
    run_cases = _read_jsonl(memory_root / "run_cases.jsonl")
    repair_cases = _read_jsonl(memory_root / "repair_cases.jsonl")
    if any(case["split"] != "train" for case in [*run_cases, *repair_cases]):
        raise ValueError("retrieval memory contains a non-Train case")
    if any(case.get("usage", "memory") != "memory" for case in [*run_cases, *repair_cases]):
        raise ValueError("retrieval memory contains an evaluation case")

    prototypes = aggregate_run_cases(run_cases)
    run_raw = [run_raw_features(row) for row in prototypes]
    repair_raw = [repair_raw_features(row) for row in repair_cases]
    run_schema = fit_feature_schema(run_raw)
    repair_schema = fit_feature_schema(repair_raw)

    run_index = []
    for prototype, raw in zip(prototypes, run_raw):
        run_index.append(
            {
                **prototype,
                "vector": vectorize(raw, run_schema),
            }
        )
    repair_index = []
    for case, raw in zip(repair_cases, repair_raw):
        repair_index.append(
            {
                "case_id": case["case_id"],
                "run_id": case["run_id"],
                "map_id": case["map_id"],
                "task_id": case["task_id"],
                "layout_mode": case["map_features"]["layout_mode"],
                "task_variant": case["task_features"]["task_variant"],
                "label": case["outcome"]["label"],
                "effective": bool(case["outcome"]["effective"]),
                "vector": vectorize(raw, repair_schema),
                "role_template": _role_template(case),
            }
        )

    normalizer = {
        "schema_version": 1,
        "fit_split": "train",
        "run": run_schema,
        "repair": repair_schema,
    }
    feature_schema = {
        "schema_version": 1,
        "run": _public_schema(run_schema),
        "repair": _public_schema(repair_schema),
        "excluded_identifiers": [
            "layout_mode",
            "layout_variant",
            "scenario_type",
            "task_variant",
            "map_id",
            "task_id",
            "solver_seed",
            "agent",
        ],
        "excluded_post_repair_fields": [
            "conflict_events_after",
            "conflict_heatmap_after",
            "path_after",
            "outcome",
            "runtime",
        ],
    }
    _write_json(output_root / "normalizer.json", normalizer)
    _write_json(output_root / "feature_schema.json", feature_schema)
    _write_jsonl(output_root / "run_index.jsonl", run_index)
    _write_jsonl(output_root / "repair_index.jsonl", repair_index)
    index_summary = {
        "schema_version": 1,
        "source_split": "train",
        "source_run_count": len(run_cases),
        "run_prototype_count": len(run_index),
        "repair_case_count": len(repair_index),
        "map_count": len({row["map_id"] for row in run_index}),
        "task_count": len({row["task_id"] for row in run_index}),
        "run_feature_count": run_schema["feature_count"],
        "repair_feature_count": repair_schema["feature_count"],
    }
    _write_json(output_root / "index_summary.json", index_summary)
    return index_summary


def _rank_neighbors(
    query_vector: list[float],
    entries: list[dict[str, Any]],
    schema: dict[str, Any],
    weights: dict[str, float],
    k: int,
    kind: str,
    exclude_id: str | None = None,
    exclude_run: str | None = None,
) -> list[tuple[dict[str, Any], float]]:
    id_field = "prototype_id" if kind == "run" else "case_id"
    ranked = []
    for entry in entries:
        if exclude_id is not None and entry[id_field] == exclude_id:
            continue
        if exclude_run is not None and entry.get("run_id") == exclude_run:
            continue
        ranked.append(
            (
                entry,
                grouped_distance(
                    query_vector, entry["vector"], schema, weights
                ),
            )
        )
    ranked.sort(key=lambda item: (item[1], str(item[0][id_field])))
    if kind == "run":
        return ranked[:k]
    selected = []
    runs: set[str] = set()
    task_counts: collections.Counter[str] = collections.Counter()
    for entry, distance in ranked:
        if entry["run_id"] in runs or task_counts[entry["task_id"]] >= 2:
            continue
        selected.append((entry, distance))
        runs.add(entry["run_id"])
        task_counts[entry["task_id"]] += 1
        if len(selected) == k:
            break
    return selected


def _neighbor_weights(
    neighbors: list[tuple[dict[str, Any], float]],
) -> list[float]:
    return [1.0 / (distance + 1e-6) for _, distance in neighbors]


def _predict_heatmap(
    neighbors: list[tuple[dict[str, Any], float]],
) -> list[dict[str, Any]]:
    if not neighbors:
        return []
    neighbor_weights = _neighbor_weights(neighbors)
    denominator = sum(neighbor_weights)
    result: collections.defaultdict[tuple[int, int], float] = (
        collections.defaultdict(float)
    )
    for (entry, _), weight in zip(neighbors, neighbor_weights):
        for item in entry["conflict_heatmap"]:
            result[tuple(item["cell"])] += (
                weight * float(item["weight"]) / denominator
            )
    return [
        {"cell": [row, col], "weight": round(weight, 6)}
        for (row, col), weight in sorted(result.items())
        if weight > 0.0
    ]


def _distribution(
    values: collections.defaultdict[str, float],
) -> list[dict[str, Any]]:
    total = sum(values.values())
    return [
        {"role": role, "probability": round(weight / total, 6)}
        for role, weight in sorted(values.items())
    ] if total else []


def _aggregate_role_template(
    neighbors: list[tuple[dict[str, Any], float]],
) -> dict[str, Any]:
    effective = [
        (entry, distance)
        for entry, distance in neighbors
        if entry["effective"]
    ]
    seed_roles: collections.defaultdict[str, float] = collections.defaultdict(float)
    other_roles: collections.defaultdict[str, float] = collections.defaultdict(float)
    numeric_totals: collections.defaultdict[str, float] = collections.defaultdict(float)
    numeric_weight = 0.0
    neighborhood_size = 0.0
    case_weight = 0.0
    for (entry, _), weight in zip(effective, _neighbor_weights(effective)):
        template = entry["role_template"]
        neighborhood_size += weight * template["neighborhood_size"]
        case_weight += weight
        for role in template["roles"]:
            signature = (
                f"{role['start_zone']}->{role['goal_zone']}"
                f"|{role['flow_assignment']}"
            )
            target = seed_roles if role["is_seed"] else other_roles
            target[signature] += weight
            numeric_totals["shortest_distance"] += (
                weight * float(role["shortest_distance"])
            )
            numeric_totals["path_stretch"] += (
                weight * float(role["path_stretch"])
            )
            numeric_totals["path_conflict_overlap"] += (
                weight * float(role["path_conflict_overlap"])
            )
            numeric_weight += weight
    return {
        "source_effective_neighbor_count": len(effective),
        "expected_neighborhood_size": (
            round(neighborhood_size / case_weight, 6)
            if case_weight
            else None
        ),
        "seed_role_distribution": _distribution(seed_roles),
        "additional_role_distribution": _distribution(other_roles),
        "mean_shortest_distance": (
            round(numeric_totals["shortest_distance"] / numeric_weight, 6)
            if numeric_weight
            else None
        ),
        "mean_path_stretch": (
            round(numeric_totals["path_stretch"] / numeric_weight, 6)
            if numeric_weight
            else None
        ),
        "mean_path_conflict_overlap": (
            round(
                numeric_totals["path_conflict_overlap"] / numeric_weight, 6
            )
            if numeric_weight
            else None
        ),
    }


def _effective_probability(
    neighbors: list[tuple[dict[str, Any], float]],
) -> float:
    if not neighbors:
        return 0.0
    weights = _neighbor_weights(neighbors)
    return sum(
        weight * float(entry["effective"])
        for (entry, _), weight in zip(neighbors, weights)
    ) / sum(weights)


def _neighbor_records(
    neighbors: list[tuple[dict[str, Any], float]], kind: str
) -> list[dict[str, Any]]:
    id_field = "prototype_id" if kind == "run" else "case_id"
    result = []
    for entry, distance in neighbors:
        item = {
            "id": entry[id_field],
            "task_id": entry["task_id"],
            "distance": round(distance, 9),
        }
        if kind == "repair":
            item.update(
                {
                    "run_id": entry["run_id"],
                    "label": entry["label"],
                    "effective": entry["effective"],
                }
            )
        result.append(item)
    return result


def _heatmap_dict(
    heatmap: list[dict[str, Any]],
) -> dict[tuple[int, int], float]:
    return {
        tuple(item["cell"]): float(item["weight"]) for item in heatmap
    }


def _heatmap_metrics(
    predicted: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> dict[str, float]:
    left = _heatmap_dict(predicted)
    right = _heatmap_dict(actual)
    dot = sum(value * right.get(cell, 0.0) for cell, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 and right_norm == 0.0:
        cosine = 1.0
    elif left_norm == 0.0 or right_norm == 0.0:
        cosine = 0.0
    else:
        cosine = dot / (left_norm * right_norm)
    actual_top = [
        cell for cell, _ in sorted(right.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    predicted_top = {
        cell for cell, _ in sorted(left.items(), key=lambda item: (-item[1], item[0]))[:10]
    }
    if not actual_top:
        top_recall = 1.0 if not predicted_top else 0.0
    else:
        top_recall = sum(cell in predicted_top for cell in actual_top) / len(actual_top)
    return {
        "cosine_similarity": round(cosine, 9),
        "top10_recall": round(top_recall, 9),
    }


def _binary_metrics(values: list[tuple[bool, bool]]) -> dict[str, float | int]:
    true_positive = sum(predicted and actual for predicted, actual in values)
    false_positive = sum(predicted and not actual for predicted, actual in values)
    false_negative = sum(not predicted and actual for predicted, actual in values)
    true_negative = sum(not predicted and not actual for predicted, actual in values)
    total = len(values)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "count": total,
        "accuracy": round((true_positive + true_negative) / max(1, total), 9),
        "precision": round(precision, 9),
        "recall": round(recall, 9),
        "f1": round(f1, 9),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _ood_threshold(
    entries: list[dict[str, Any]],
    schema: dict[str, Any],
    weights: dict[str, float],
    kind: str,
) -> float:
    distances = []
    id_field = "prototype_id" if kind == "run" else "case_id"
    for entry in entries:
        neighbors = _rank_neighbors(
            entry["vector"],
            entries,
            schema,
            weights,
            1,
            kind,
            exclude_id=entry[id_field],
            exclude_run=entry.get("run_id") if kind == "repair" else None,
        )
        if neighbors:
            distances.append(neighbors[0][1])
    return _percentile(distances, 0.95)


def _role_overlap(predicted: dict[str, Any], actual: dict[str, Any]) -> float:
    def actual_distribution(seed: bool) -> dict[str, float]:
        roles = [role for role in actual["roles"] if role["is_seed"] == seed]
        counts: collections.Counter[str] = collections.Counter(
            f"{role['start_zone']}->{role['goal_zone']}|{role['flow_assignment']}"
            for role in roles
        )
        return {
            role: count / max(1, len(roles)) for role, count in counts.items()
        }

    scores = []
    for seed, field in (
        (True, "seed_role_distribution"),
        (False, "additional_role_distribution"),
    ):
        expected = {
            item["role"]: float(item["probability"])
            for item in predicted[field]
        }
        observed = actual_distribution(seed)
        keys = set(expected) | set(observed)
        scores.append(sum(min(expected.get(key, 0.0), observed.get(key, 0.0)) for key in keys))
    return sum(scores) / len(scores)


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _load_index(
    index_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    normalizer = _read_json(index_root / "normalizer.json")
    if normalizer.get("fit_split") != "train":
        raise ValueError("index normalizer was not fit on Train")
    return (
        normalizer,
        _read_jsonl(index_root / "run_index.jsonl"),
        _read_jsonl(index_root / "repair_index.jsonl"),
    )


def _tune_run(
    queries: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    schema: dict[str, Any],
) -> tuple[int, dict[str, float]]:
    candidates = []
    for weights in RUN_WEIGHT_OPTIONS:
        for k in K_OPTIONS:
            metrics = []
            for query in queries:
                neighbors = _rank_neighbors(
                    query["vector"], entries, schema, weights, k, "run"
                )
                metrics.append(
                    _heatmap_metrics(
                        _predict_heatmap(neighbors), query["conflict_heatmap"]
                    )
                )
            candidates.append(
                (
                    _mean([item["cosine_similarity"] for item in metrics]),
                    _mean([item["top10_recall"] for item in metrics]),
                    -k,
                    json.dumps(weights, sort_keys=True),
                    k,
                    weights,
                )
            )
    return max(candidates)[4:6]


def _tune_repair(
    queries: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    schema: dict[str, Any],
) -> tuple[int, dict[str, float]]:
    candidates = []
    for weights in REPAIR_WEIGHT_OPTIONS:
        for k in K_OPTIONS:
            predictions = []
            for query in queries:
                neighbors = _rank_neighbors(
                    query["vector"], entries, schema, weights, k, "repair"
                )
                predictions.append(
                    (
                        _effective_probability(neighbors) >= 0.5,
                        bool(query["effective"]),
                    )
                )
            metrics = _binary_metrics(predictions)
            candidates.append(
                (
                    float(metrics["f1"]),
                    float(metrics["accuracy"]),
                    -k,
                    json.dumps(weights, sort_keys=True),
                    k,
                    weights,
                )
            )
    return max(candidates)[4:6]


def _grouped_repair_metrics(
    guidance: list[dict[str, Any]], field: str
) -> dict[str, dict[str, float | int]]:
    result = {}
    for value in sorted({str(row[field]) for row in guidance}):
        rows = [row for row in guidance if str(row[field]) == value]
        result[value] = _binary_metrics(
            [
                (bool(row["predicted_effective"]), bool(row["actual_effective"]))
                for row in rows
            ]
        )
    return result


def _grouped_heatmap_metrics(
    guidance: list[dict[str, Any]], field: str
) -> dict[str, dict[str, float | int]]:
    result = {}
    for value in sorted({str(row[field]) for row in guidance}):
        rows = [row for row in guidance if str(row[field]) == value]
        result[value] = {
            "count": len(rows),
            "mean_cosine_similarity": round(
                _mean([row["metrics"]["cosine_similarity"] for row in rows]), 9
            ),
            "mean_top10_recall": round(
                _mean([row["metrics"]["top10_recall"] for row in rows]), 9
            ),
        }
    return result


def evaluate_retrieval(
    index: str | Path, queries: str | Path, output: str | Path
) -> dict[str, Any]:
    index_root = Path(index).resolve()
    query_root = Path(queries).resolve()
    output_root = Path(output).resolve()
    query_summary = _read_json(query_root / "experience_summary.json")
    if (
        query_summary.get("split") != "validation"
        or query_summary.get("usage") != "evaluation"
    ):
        raise ValueError("Stage 4 queries must be Validation evaluation data")
    normalizer, run_index, repair_index = _load_index(index_root)
    run_schema = normalizer["run"]
    repair_schema = normalizer["repair"]
    query_runs = _read_jsonl(query_root / "run_cases.jsonl")
    query_repairs = _read_jsonl(query_root / "repair_cases.jsonl")
    if any(row["split"] != "validation" for row in [*query_runs, *query_repairs]):
        raise ValueError("query data contains a non-Validation case")

    run_queries = aggregate_run_cases(query_runs)
    for query in run_queries:
        query["vector"] = vectorize(run_raw_features(query), run_schema)
    repair_queries = []
    for case in query_repairs:
        repair_queries.append(
            {
                "case_id": case["case_id"],
                "run_id": case["run_id"],
                "map_id": case["map_id"],
                "task_id": case["task_id"],
                "layout_mode": case["map_features"]["layout_mode"],
                "task_variant": case["task_features"]["task_variant"],
                "effective": bool(case["outcome"]["effective"]),
                "label": case["outcome"]["label"],
                "vector": vectorize(repair_raw_features(case), repair_schema),
                "role_template": _role_template(case),
            }
        )

    run_k, run_weights = _tune_run(
        run_queries, run_index, run_schema
    )
    repair_k, repair_weights = _tune_repair(
        repair_queries, repair_index, repair_schema
    )
    run_threshold = _ood_threshold(
        run_index, run_schema, run_weights, "run"
    )
    repair_threshold = _ood_threshold(
        repair_index, repair_schema, repair_weights, "repair"
    )

    run_guidance = []
    for query in run_queries:
        neighbors = _rank_neighbors(
            query["vector"],
            run_index,
            run_schema,
            run_weights,
            run_k,
            "run",
        )
        predicted = _predict_heatmap(neighbors)
        nearest = neighbors[0][1] if neighbors else math.inf
        run_guidance.append(
            {
                "schema_version": 1,
                "query_id": query["prototype_id"],
                "map_id": query["map_id"],
                "task_id": query["task_id"],
                "layout_mode": query["map_features"]["layout_mode"],
                "task_variant": query["task_features"]["task_variant"],
                "k": run_k,
                "neighbors": _neighbor_records(neighbors, "run"),
                "nearest_distance": round(nearest, 9),
                "out_of_distribution": nearest > run_threshold,
                "predicted_conflict_heatmap": predicted,
                "actual_has_conflicts": bool(query["conflict_heatmap"]),
                "metrics": _heatmap_metrics(
                    predicted, query["conflict_heatmap"]
                ),
            }
        )

    repair_guidance = []
    for query in repair_queries:
        neighbors = _rank_neighbors(
            query["vector"],
            repair_index,
            repair_schema,
            repair_weights,
            repair_k,
            "repair",
        )
        probability = _effective_probability(neighbors)
        role_template = _aggregate_role_template(neighbors)
        nearest = neighbors[0][1] if neighbors else math.inf
        repair_guidance.append(
            {
                "schema_version": 1,
                "query_id": query["case_id"],
                "map_id": query["map_id"],
                "task_id": query["task_id"],
                "layout_mode": query["layout_mode"],
                "task_variant": query["task_variant"],
                "k": repair_k,
                "neighbors": _neighbor_records(neighbors, "repair"),
                "nearest_distance": round(nearest, 9),
                "out_of_distribution": nearest > repair_threshold,
                "effective_probability": round(probability, 9),
                "predicted_effective": probability >= 0.5,
                "neighborhood_role_template": role_template,
                "actual_label": query["label"],
                "actual_effective": query["effective"],
                "role_overlap": (
                    round(_role_overlap(role_template, query["role_template"]), 9)
                    if query["effective"]
                    and role_template["source_effective_neighbor_count"] > 0
                    else None
                ),
            }
        )

    _write_jsonl(output_root / "run_guidance.jsonl", run_guidance)
    _write_jsonl(output_root / "repair_guidance.jsonl", repair_guidance)
    binary = _binary_metrics(
        [
            (row["predicted_effective"], row["actual_effective"])
            for row in repair_guidance
        ]
    )
    active_run_guidance = [
        row for row in run_guidance if row["actual_has_conflicts"]
    ]
    positive_count = sum(
        row["actual_effective"] for row in repair_guidance
    )
    majority_prediction = positive_count >= (
        len(repair_guidance) - positive_count
    )
    majority_baseline = _binary_metrics(
        [
            (majority_prediction, row["actual_effective"])
            for row in repair_guidance
        ]
    )
    evaluation_summary = {
        "schema_version": 1,
        "index_split": "train",
        "query_split": "validation",
        "test_data_read": False,
        "selected_parameters": {
            "run": {
                "k": run_k,
                "group_weights": run_weights,
                "ood_distance_threshold": round(run_threshold, 9),
            },
            "repair": {
                "k": repair_k,
                "group_weights": repair_weights,
                "ood_distance_threshold": round(repair_threshold, 9),
                "effective_threshold": 0.5,
            },
        },
        "run_evaluation": {
            "query_count": len(run_guidance),
            "mean_cosine_similarity": round(
                _mean(
                    [
                        row["metrics"]["cosine_similarity"]
                        for row in run_guidance
                    ]
                ),
                9,
            ),
            "mean_top10_recall": round(
                _mean(
                    [row["metrics"]["top10_recall"] for row in run_guidance]
                ),
                9,
            ),
            "active_query_count": len(active_run_guidance),
            "active_mean_cosine_similarity": round(
                _mean(
                    [
                        row["metrics"]["cosine_similarity"]
                        for row in active_run_guidance
                    ]
                ),
                9,
            ),
            "active_mean_top10_recall": round(
                _mean(
                    [
                        row["metrics"]["top10_recall"]
                        for row in active_run_guidance
                    ]
                ),
                9,
            ),
            "zero_heatmap_baseline_mean_cosine": round(
                (
                    len(run_guidance) - len(active_run_guidance)
                ) / max(1, len(run_guidance)),
                9,
            ),
            "out_of_distribution_count": sum(
                row["out_of_distribution"] for row in run_guidance
            ),
            "by_layout": _grouped_heatmap_metrics(
                run_guidance, "layout_mode"
            ),
            "by_task_variant": _grouped_heatmap_metrics(
                run_guidance, "task_variant"
            ),
        },
        "repair_evaluation": {
            **binary,
            "majority_baseline": majority_baseline,
            "out_of_distribution_count": sum(
                row["out_of_distribution"] for row in repair_guidance
            ),
            "mean_effective_role_overlap": round(
                _mean(
                    [
                        row["role_overlap"]
                        for row in repair_guidance
                        if row["role_overlap"] is not None
                    ]
                ),
                9,
            ),
            "by_layout": _grouped_repair_metrics(
                repair_guidance, "layout_mode"
            ),
            "by_task_variant": _grouped_repair_metrics(
                repair_guidance, "task_variant"
            ),
        },
        "interpretation": (
            "Offline retrieval metrics do not demonstrate planner improvement; "
            "concrete Agent mapping and solver comparison belong to Stage 5."
        ),
    }
    _write_json(output_root / "evaluation_summary.json", evaluation_summary)
    return evaluation_summary
