from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import shutil
import statistics
from pathlib import Path
from typing import Any, Iterable


PROBE_SCHEMA_VERSION = 1
PROBE_SPLIT = "probe"
MODEL_SEED = 20260714


def _scenario_index_from_path(path: str) -> int:
    match = re.search(r"-random-(\d+)\.scen$", path)
    return int(match.group(1)) if match is not None else 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mean(values: Iterable[float | int]) -> float:
    numbers = [float(value) for value in values]
    return statistics.fmean(numbers) if numbers else 0.0


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _movingai_map_metrics(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers: dict[str, str] = {}
    marker = None
    for index, line in enumerate(lines):
        if line.strip().lower() == "map":
            marker = index
            break
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            headers[parts[0].lower()] = parts[1]
    if marker is None or "height" not in headers or "width" not in headers:
        raise ValueError(f"invalid MovingAI map header: {path}")
    rows = int(headers["height"])
    cols = int(headers["width"])
    grid = lines[marker + 1 : marker + 1 + rows]
    if len(grid) != rows or any(len(row) != cols for row in grid):
        raise ValueError(f"MovingAI map dimensions do not match its header: {path}")
    free = {
        row * cols + col
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value in {".", "G", "S"}
    }
    if not free:
        raise ValueError(f"MovingAI map has no traversable cells: {path}")

    def degree(cell: int) -> int:
        row, col = divmod(cell, cols)
        return sum(
            neighbor in free
            for neighbor in (
                (row - 1) * cols + col if row > 0 else -1,
                (row + 1) * cols + col if row + 1 < rows else -1,
                row * cols + col - 1 if col > 0 else -1,
                row * cols + col + 1 if col + 1 < cols else -1,
            )
        )

    degrees = [degree(cell) for cell in free]
    return {
        "rows": rows,
        "cols": cols,
        "free_cell_count": len(free),
        "obstacle_count": rows * cols - len(free),
        "obstacle_ratio": _ratio(rows * cols - len(free), rows * cols),
        "average_free_degree": _mean(degrees),
        "minimum_free_degree": min(degrees),
        "maximum_free_degree": max(degrees),
        "dead_end_cell_count": sum(value <= 1 for value in degrees),
        "low_degree_cell_ratio": _ratio(sum(value <= 2 for value in degrees), len(degrees)),
    }


def _scenario_prefix_metrics(path: Path, agent_count: int) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = lines[1:] if lines and lines[0].lower().startswith("version") else lines
    if len(rows) < agent_count:
        raise ValueError(
            f"scenario has {len(rows)} agents, fewer than requested {agent_count}: {path}"
        )
    selected = []
    for line in rows[:agent_count]:
        parts = line.split()
        if len(parts) < 9:
            raise ValueError(f"invalid MovingAI scenario row: {line}")
        selected.append(parts)
    starts = [(int(row[4]), int(row[5])) for row in selected]
    goals = [(int(row[6]), int(row[7])) for row in selected]
    if len(set(starts)) != len(starts):
        raise ValueError(f"scenario prefix contains duplicate starts: {path}")
    if len(set(goals)) != len(goals):
        raise ValueError(f"scenario prefix contains duplicate goals: {path}")
    distances = [float(row[8]) for row in selected]
    return {
        "agent_count": agent_count,
        "mean_shortest_distance": _mean(distances),
        "minimum_shortest_distance": min(distances),
        "maximum_shortest_distance": max(distances),
        "unique_start_count": len(set(starts)),
        "unique_goal_count": len(set(goals)),
    }


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file():
        raise ValueError(f"MovingAI source file is missing: {source}")
    actual = _sha256(source)
    if actual != expected_sha256:
        raise ValueError(
            f"MovingAI source checksum mismatch for {source.name}: "
            f"expected {expected_sha256}, got {actual}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or _sha256(destination) != actual:
        shutil.copy2(source, destination)


def prepare_probe_dataset(
    movingai_dataset: str | Path,
    config: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    source_root = Path(movingai_dataset).resolve()
    config_path = Path(config).resolve()
    output_root = Path(output).resolve()
    specification = _read_json(config_path)
    if int(specification.get("schema_version", -1)) != PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported MovingAI mechanism probe config schema")
    split = str(specification.get("split", PROBE_SPLIT))
    if not split or "/" in split or "\\" in split:
        raise ValueError("MovingAI output split must be one path-safe component")
    source_manifest_path = source_root / "manifest.jsonl"
    source_rows = _read_jsonl(source_manifest_path)
    source_index = {str(row["id"]): row for row in source_rows}
    if len(source_index) != len(source_rows) or not source_index:
        raise ValueError("MovingAI source manifest is empty or contains duplicate ids")

    requested: list[tuple[str, int, int]] = []
    case_by_benchmark: dict[str, dict[str, Any]] = {}
    for case in specification.get("cases", []):
        benchmark_id = str(case["benchmark_id"])
        if benchmark_id in case_by_benchmark:
            raise ValueError(f"duplicate MovingAI benchmark case: {benchmark_id}")
        case_by_benchmark[benchmark_id] = dict(case)
        if benchmark_id not in source_index:
            raise ValueError(f"unknown MovingAI benchmark in probe config: {benchmark_id}")
        available = {int(value) for value in source_index[benchmark_id]["agent_counts"]}
        counts = [int(value) for value in case["agent_counts"]]
        if not counts or len(counts) != len(set(counts)):
            raise ValueError(f"agent counts must be non-empty and unique: {benchmark_id}")
        if any(count not in available for count in counts):
            raise ValueError(f"probe requests an unavailable agent count: {benchmark_id}")
        source_scenarios = source_index[benchmark_id].get("scenarios")
        if source_scenarios:
            available_scenarios = {
                int(row["index"]): row for row in source_scenarios
            }
        else:
            fallback_index = _scenario_index_from_path(
                str(source_index[benchmark_id]["scenario_file"])
            )
            available_scenarios = {
                fallback_index: {
                    "index": fallback_index,
                    "file": source_index[benchmark_id]["scenario_file"],
                    "sha256": source_index[benchmark_id]["scenario_sha256"],
                }
            }
        scenario_indices = [
            int(value)
            for value in case.get(
                "scenario_indices", specification.get("scenario_indices", [1])
            )
        ]
        if (
            not scenario_indices
            or len(scenario_indices) != len(set(scenario_indices))
            or any(value not in available_scenarios for value in scenario_indices)
        ):
            raise ValueError(
                f"probe requests unavailable or duplicate scenarios: {benchmark_id}"
            )
        requested.extend(
            (benchmark_id, scenario_index, count)
            for scenario_index in scenario_indices
            for count in counts
        )
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("probe cases must be non-empty and unique")

    source_fingerprint = _fingerprint(
        {
            "manifest_sha256": _sha256(source_manifest_path),
            "config_sha256": _sha256(config_path),
            "requested": requested,
        }
    )
    summary_path = output_root / "dataset_summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if existing.get("configuration_fingerprint") != source_fingerprint:
            raise ValueError("probe output contains a different dataset configuration")

    split_root = output_root / split
    manifest: list[dict[str, Any]] = []
    map_metrics: dict[str, dict[str, Any]] = {}
    for benchmark_id, scenario_index, agent_count in requested:
        source = source_index[benchmark_id]
        source_map = source_root / str(source["map_file"])
        fallback_index = _scenario_index_from_path(str(source["scenario_file"]))
        source_scenarios = source.get("scenarios") or [
            {
                "index": fallback_index,
                "file": source["scenario_file"],
                "sha256": source["scenario_sha256"],
            }
        ]
        scenario = {
            int(row["index"]): row for row in source_scenarios
        }[scenario_index]
        source_scenario = source_root / str(scenario["file"])
        map_file = Path("maps") / source_map.name
        scenario_file = Path("scenarios") / source_scenario.name
        destination_map = split_root / map_file
        destination_scenario = split_root / scenario_file
        _copy_verified(source_map, destination_map, str(source["map_sha256"]))
        _copy_verified(
            source_scenario, destination_scenario, str(scenario["sha256"])
        )
        if benchmark_id not in map_metrics:
            map_metrics[benchmark_id] = _movingai_map_metrics(destination_map)
            _write_json(
                split_root / "maps" / f"{benchmark_id}.json",
                {
                    "schema_version": PROBE_SCHEMA_VERSION,
                    "benchmark_id": benchmark_id,
                    "source": "MovingAI MAPF benchmark",
                    "map_sha256": str(source["map_sha256"]),
                    "topology_metrics": map_metrics[benchmark_id],
                },
            )
        task_id = (
            f"{benchmark_id}__random_{scenario_index:02d}__agents_{agent_count:04d}"
        )
        task_metrics = _scenario_prefix_metrics(destination_scenario, agent_count)
        task_file = Path("tasks") / f"{task_id}.json"
        _write_json(
            split_root / task_file,
            {
                "schema_version": PROBE_SCHEMA_VERSION,
                "task_semantics": (
                    f"static MovingAI random-{scenario_index} scenario prefix"
                ),
                "benchmark_id": benchmark_id,
                "scenario_index": scenario_index,
                "scenario_sha256": str(scenario["sha256"]),
                **task_metrics,
            },
        )
        manifest.append(
            {
                "split": split,
                "map_id": benchmark_id,
                "task_id": task_id,
                "map_file": map_file.as_posix(),
                "scenario_file": scenario_file.as_posix(),
                "map_metadata_file": f"maps/{benchmark_id}.json",
                "task_file": task_file.as_posix(),
                "layout_mode": str(
                    case_by_benchmark[benchmark_id].get(
                        "layout_family", "movingai_standard"
                    )
                ),
                "layout_variant": benchmark_id,
                "scenario_type": f"movingai_random_{scenario_index}",
                "task_variant": f"random_{scenario_index}_agents_{agent_count}",
                "agent_count": agent_count,
                "topology_metrics": map_metrics[benchmark_id],
                "dominant_flow_ratio": 0.0,
                "hotspot_skew": 0.0,
                "required_bottleneck_crossing_ratio": 0.0,
                "mean_shortest_distance": task_metrics["mean_shortest_distance"],
            }
        )
    manifest.sort(
        key=lambda row: (
            str(row["map_id"]),
            str(row["scenario_type"]),
            int(row["agent_count"]),
        )
    )
    _write_jsonl(split_root / "manifest.jsonl", manifest)
    summary = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "configuration_fingerprint": source_fingerprint,
        "source": "MovingAI MAPF benchmark random scenarios",
        "task_semantics": "static scenario prefixes; no release times or task queues",
        "splits": {
            split: {
                "map_count": len(map_metrics),
                "instance_count": len(manifest),
                "scenario_count": len(
                    {str(row["scenario_type"]) for row in manifest}
                ),
                "agent_count_min": min(int(row["agent_count"]) for row in manifest),
                "agent_count_max": max(int(row["agent_count"]) for row in manifest),
            }
        },
    }
    _write_json(summary_path, summary)
    return summary


def _candidate_key(action: dict[str, Any]) -> str:
    return (
        f"{int(action['seed_agent'])}:{str(action['heuristic'])}:"
        f"{int(action['neighborhood_size'])}"
    )


def _family(action: dict[str, Any]) -> str:
    return f"{str(action['heuristic'])}:{int(action['neighborhood_size'])}"


def _horizon_one(outcome: dict[str, Any]) -> dict[str, Any]:
    selected = [
        row for row in outcome.get("horizon_outcomes", []) if int(row["horizon"]) == 1
    ]
    if len(selected) != 1 or not bool(selected[0].get("available")):
        raise ValueError("probe outcome must contain one available Horizon 1 result")
    return selected[0]


def _actual_neighborhood(outcome: dict[str, Any]) -> tuple[int, ...]:
    steps = [row for row in outcome.get("steps", []) if int(row.get("step", -1)) == 1]
    if len(steps) != 1:
        raise ValueError("probe outcome must contain exactly one first repair step")
    neighborhood = steps[0].get("metrics", {}).get("neighborhood")
    if not isinstance(neighborhood, list) or not neighborhood:
        raise ValueError("probe outcome is missing its realized neighborhood")
    values = tuple(sorted(int(value) for value in neighborhood))
    if len(values) != len(set(values)):
        raise ValueError("probe outcome contains duplicate neighborhood members")
    return values


def _dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    left_values = (-left["solved_rate"], left["conflicts_after"], left["conflict_auc"])
    right_values = (
        -right["solved_rate"],
        right["conflicts_after"],
        right["conflict_auc"],
    )
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _group_statistic(
    units: list[dict[str, Any]], labels: list[str], families: list[str]
) -> float:
    if len(set(labels)) < 2 or not units or not families:
        return 0.0
    global_values = {
        family: _mean(unit["support"][family] for unit in units) for family in families
    }
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    values = []
    for indices in grouped.values():
        for family in families:
            group_mean = _mean(units[index]["support"][family] for index in indices)
            values.append((group_mean - global_values[family]) ** 2)
    return _mean(values)


def _density_statistic(
    units: list[dict[str, Any]], counts: list[int], families: list[str]
) -> float:
    values = []
    by_map: dict[str, list[int]] = collections.defaultdict(list)
    for index, unit in enumerate(units):
        by_map[str(unit["map_id"])].append(index)
    for indices in by_map.values():
        if len({counts[index] for index in indices}) < 2:
            continue
        for family in families:
            map_mean = _mean(units[index]["support"][family] for index in indices)
            by_count: dict[int, list[int]] = collections.defaultdict(list)
            for index in indices:
                by_count[counts[index]].append(index)
            for selected in by_count.values():
                count_mean = _mean(
                    units[index]["support"][family] for index in selected
                )
                values.append((count_mean - map_mean) ** 2)
    return _mean(values)


def _permutation_signals(
    units: list[dict[str, Any]], families: list[str], permutations: int
) -> dict[str, Any]:
    if permutations <= 0:
        raise ValueError("probe permutation count must be positive")
    map_labels = [str(unit["map_id"]) for unit in units]
    counts = [int(unit["agent_count"]) for unit in units]
    real_map = _group_statistic(units, map_labels, families)
    real_density = _density_statistic(units, counts, families)
    null_map = []
    null_density = []
    by_map: dict[str, list[int]] = collections.defaultdict(list)
    for index, label in enumerate(map_labels):
        by_map[label].append(index)
    for permutation in range(permutations):
        rng = random.Random(MODEL_SEED + permutation * 7919)
        shuffled_maps = list(map_labels)
        rng.shuffle(shuffled_maps)
        null_map.append(_group_statistic(units, shuffled_maps, families))
        shuffled_counts = list(counts)
        for indices in by_map.values():
            values = [shuffled_counts[index] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values):
                shuffled_counts[index] = value
        null_density.append(_density_statistic(units, shuffled_counts, families))
    return {
        "permutations": permutations,
        "unit": "task instance after duplicate-state aggregation",
        "map_statistic": real_map,
        "map_percentile": _ratio(sum(real_map > value for value in null_map), permutations),
        "map_null_range": [min(null_map, default=0.0), max(null_map, default=0.0)],
        "density_statistic": real_density,
        "density_percentile": _ratio(
            sum(real_density > value for value in null_density), permutations
        ),
        "density_null_range": [
            min(null_density, default=0.0),
            max(null_density, default=0.0),
        ],
    }


def summarize_probe_records(
    qualification: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    states: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    settings: dict[str, Any],
    collection_config: dict[str, Any],
    *,
    require_complete: bool,
    counterfactual_error_count: int = 0,
) -> dict[str, Any]:
    expected_trials = int(collection_config["counterfactual"]["trials"])
    expected_tasks = sum(
        len(case["agent_counts"])
        * len(case.get("scenario_indices", settings.get("scenario_indices", [1])))
        for case in settings["cases"]
    )
    expected_maps = len(settings["cases"])
    solver_seeds = [int(value) for value in collection_config["solver_seeds"]]
    expected_qualification = expected_tasks * len(solver_seeds)
    expected_baseline = expected_qualification * len(collection_config["policies"])
    state_index = {str(row["state_id"]): row for row in states}
    duplicate_state_id_count = len(states) - len(state_index)
    canonical_by_state = {
        state_id: str(row.get("state_fingerprint", state_id))
        for state_id, row in state_index.items()
    }
    canonical_sources: dict[str, dict[str, Any]] = {}
    canonical_state_ids: dict[str, list[str]] = collections.defaultdict(list)
    fingerprint_context_mismatch = 0
    for state_id, row in state_index.items():
        canonical = canonical_by_state[state_id]
        existing = canonical_sources.setdefault(canonical, row)
        existing_context = existing["state"].get("context", {})
        row_context = row["state"].get("context", {})
        if (
            str(existing_context.get("task_id")) != str(row_context.get("task_id"))
            or int(existing.get("decision_index", -1))
            != int(row.get("decision_index", -1))
        ):
            fingerprint_context_mismatch += 1
        canonical_state_ids[canonical].append(state_id)
    duplicate_fingerprint_groups = sum(
        len(values) > 1 for values in canonical_state_ids.values()
    )
    raw_grouped_trials: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    invalid_action_count = 0
    orphan_outcome_count = 0
    for outcome in outcomes:
        state_id = str(outcome["state_id"])
        if state_id not in state_index:
            orphan_outcome_count += 1
            continue
        if not bool(outcome.get("action_valid")):
            invalid_action_count += 1
        raw_grouped_trials[
            (state_id, _candidate_key(outcome["candidate_action"]))
        ].append(outcome)

    trial_mismatch_count = 0
    candidate_count_mismatch = 0
    grouped_trials: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    raw_candidates_by_state: dict[str, int] = collections.Counter(
        state_id for state_id, _ in raw_grouped_trials
    )
    for state_id, row in state_index.items():
        if raw_candidates_by_state.get(state_id, 0) != int(
            row.get("candidate_count", 0)
        ):
            candidate_count_mismatch += 1
    for (state_id, candidate_key), trials in raw_grouped_trials.items():
        trial_indices = [int(row.get("trial_index", -1)) for row in trials]
        if (
            len(trials) != expected_trials
            or len(set(trial_indices)) != expected_trials
        ):
            trial_mismatch_count += 1
        grouped_trials[(canonical_by_state[state_id], candidate_key)].extend(trials)

    candidate_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    between_sum = 0.0
    within_sum = 0.0
    for (state_id, candidate_key), trials in sorted(grouped_trials.items()):
        horizon_rows = [_horizon_one(row) for row in trials]
        conflicts = [float(row["conflicts_after"]) for row in horizon_rows]
        neighborhoods = [_actual_neighborhood(row) for row in trials]
        action = trials[0]["candidate_action"]
        candidate_rows[state_id].append(
            {
                "candidate_key": candidate_key,
                "family": _family(action),
                "trial_count": len(trials),
                "solved_rate": _mean(bool(row["solved"]) for row in horizon_rows),
                "conflicts_after": _mean(conflicts),
                "conflict_auc": _mean(float(row["conflict_auc"]) for row in horizon_rows),
                "generated": _mean(
                    int(row["low_level_delta"]["generated"]) for row in horizon_rows
                ),
                "trial_conflicts": conflicts,
                "neighborhoods": neighborhoods,
                "neighborhood_stable": len(set(neighborhoods)) == 1,
            }
        )

    state_summaries = []
    unique_family_counts: collections.Counter[str] = collections.Counter()
    all_families = sorted(
        {
            f"{heuristic}:{int(size)}"
            for heuristic in collection_config["counterfactual"]["heuristics"]
            for size in collection_config["counterfactual"]["neighborhood_sizes"]
        }
    )
    for state_id, source in sorted(canonical_sources.items()):
        candidates = candidate_rows.get(state_id, [])
        if not candidates:
            continue
        for candidate in candidates:
            candidate["pareto"] = not any(
                other is not candidate and _dominates(other, candidate)
                for other in candidates
            )
        supported = sorted(
            {candidate["family"] for candidate in candidates if candidate["pareto"]}
        )
        if len(supported) == 1:
            unique_family_counts[supported[0]] += 1
        outcome_signatures = {
            (
                round(candidate["solved_rate"], 9),
                round(candidate["conflicts_after"], 9),
                round(candidate["conflict_auc"], 9),
            )
            for candidate in candidates
        }
        neighborhoods = {
            neighborhood
            for candidate in candidates
            for neighborhood in candidate["neighborhoods"]
        }
        all_trial_conflicts = [
            value for candidate in candidates for value in candidate["trial_conflicts"]
        ]
        grand_mean = _mean(all_trial_conflicts)
        for candidate in candidates:
            candidate_mean = _mean(candidate["trial_conflicts"])
            between_sum += len(candidate["trial_conflicts"]) * (
                candidate_mean - grand_mean
            ) ** 2
            within_sum += sum(
                (value - candidate_mean) ** 2
                for value in candidate["trial_conflicts"]
            )
        context = source["state"].get("context", {})
        state_summaries.append(
            {
                "state_id": state_id,
                "source_state_ids": sorted(canonical_state_ids[state_id]),
                "episode_ids": sorted(
                    {
                        str(state_index[value]["episode_id"])
                        for value in canonical_state_ids[state_id]
                    }
                ),
                "map_id": str(context.get("map_id", "unknown")),
                "task_id": str(context.get("task_id", "unknown")),
                "agent_count": int(context.get("agent_count", 0)),
                "candidate_count": len(candidates),
                "recorded_candidate_count": int(source.get("candidate_count", 0)),
                "distinct_outcome_count": len(outcome_signatures),
                "unique_neighborhood_count": len(neighborhoods),
                "conflict_range": max(
                    candidate["conflicts_after"] for candidate in candidates
                )
                - min(candidate["conflicts_after"] for candidate in candidates),
                "auc_range": max(candidate["conflict_auc"] for candidate in candidates)
                - min(candidate["conflict_auc"] for candidate in candidates),
                "trial_conflict_disagreement_rate": _mean(
                    len(set(candidate["trial_conflicts"])) > 1
                    for candidate in candidates
                ),
                "trial_neighborhood_stability_rate": _mean(
                    candidate["neighborhood_stable"] for candidate in candidates
                ),
                "pareto_families": supported,
                "family_support": {
                    family: float(family in supported) for family in all_families
                },
            }
        )

    task_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in state_summaries:
        task_groups[row["task_id"]].append(row)
    units = []
    for task_id, selected in sorted(task_groups.items()):
        units.append(
            {
                "task_id": task_id,
                "map_id": selected[0]["map_id"],
                "agent_count": selected[0]["agent_count"],
                "support": {
                    family: _mean(row["family_support"][family] for row in selected)
                    for family in all_families
                },
            }
        )
    permutations = int(settings["analysis"]["permutations"])
    context_signal = _permutation_signals(units, all_families, permutations)

    group_summary = {}
    for key in sorted({(row["map_id"], row["agent_count"]) for row in state_summaries}):
        selected = [
            row
            for row in state_summaries
            if (row["map_id"], row["agent_count"]) == key
        ]
        group_summary[f"{key[0]}:agents_{key[1]}"] = {
            "state_count": len(selected),
            "mean_conflict_range": _mean(row["conflict_range"] for row in selected),
            "mean_auc_range": _mean(row["auc_range"] for row in selected),
            "family_pareto_support_rate": {
                family: _mean(row["family_support"][family] for row in selected)
                for family in all_families
            },
        }

    qualification_errors = sum(str(row.get("status")) == "error" for row in qualification)
    baseline_errors = sum(str(row.get("status")) == "error" for row in baseline)
    repairable = sum(
        bool(row.get("repairable"))
        for row in qualification
        if str(row.get("status")) == "ok"
    )
    valid_qualification = len(qualification) - qualification_errors
    expected_counterfactual = sum(
        str(row.get("policy")) == collection_config["counterfactual"]["source_policy"]
        and str(row.get("status")) != "error"
        and bool(row.get("summary", {}).get("repairable"))
        for row in baseline
    )
    qualification_maps = {str(row["map_id"]) for row in qualification}
    qualification_tasks = {str(row["task_id"]) for row in qualification}
    counterfactual_episodes = {
        str(row["episode_id"]) for row in states if row.get("episode_id") is not None
    }
    volume_checks = {
        "qualification": len(qualification) == expected_qualification,
        "baseline": len(baseline) == expected_baseline,
        "counterfactual_episodes": len(counterfactual_episodes)
        == expected_counterfactual,
        "map_coverage": len(qualification_maps) == expected_maps,
        "task_coverage": len(qualification_tasks) == expected_tasks,
    }
    complete_volume = all(volume_checks.values())
    integrity_passed = (
        duplicate_state_id_count == 0
        and fingerprint_context_mismatch == 0
        and orphan_outcome_count == 0
        and invalid_action_count == 0
        and trial_mismatch_count == 0
        and candidate_count_mismatch == 0
        and qualification_errors == 0
        and baseline_errors == 0
        and counterfactual_error_count == 0
        and (complete_volume or not require_complete)
    )
    action_effect_ratio = _ratio(between_sum, between_sum + within_sum)
    state_count = len(state_summaries)
    action_diversity_rate = _ratio(
        sum(row["distinct_outcome_count"] > 1 for row in state_summaries), state_count
    )
    neighborhood_diversity_rate = _ratio(
        sum(row["unique_neighborhood_count"] > 1 for row in state_summaries), state_count
    )
    maximum_fixed_share = _ratio(
        max(unique_family_counts.values(), default=0), state_count
    )
    thresholds = settings["analysis"]
    gates = {
        "integrity": {
            "passed": integrity_passed,
            "actual": {
                "volume_checks": volume_checks,
                "invalid_actions": invalid_action_count,
                "trial_mismatches": trial_mismatch_count,
                "candidate_count_mismatches": candidate_count_mismatch,
                "fingerprint_context_mismatches": fingerprint_context_mismatch,
                "collection_errors": qualification_errors
                + baseline_errors
                + counterfactual_error_count,
            },
        },
        "repair_coverage": {
            "passed": _ratio(repairable, valid_qualification)
            >= float(thresholds["minimum_repairable_rate"]),
            "actual": _ratio(repairable, valid_qualification),
            "requirement": float(thresholds["minimum_repairable_rate"]),
        },
        "state_coverage": {
            "passed": state_count >= int(thresholds["minimum_state_count"]),
            "actual": state_count,
            "requirement": int(thresholds["minimum_state_count"]),
        },
        "action_diversity": {
            "passed": action_diversity_rate
            >= float(thresholds["minimum_action_diversity_rate"]),
            "actual": action_diversity_rate,
            "requirement": float(thresholds["minimum_action_diversity_rate"]),
        },
        "neighborhood_diversity": {
            "passed": neighborhood_diversity_rate
            >= float(thresholds["minimum_neighborhood_diversity_rate"]),
            "actual": neighborhood_diversity_rate,
            "requirement": float(thresholds["minimum_neighborhood_diversity_rate"]),
        },
        "action_signal_over_trial_noise": {
            "passed": action_effect_ratio
            >= float(thresholds["minimum_action_effect_ratio"]),
            "actual": action_effect_ratio,
            "requirement": float(thresholds["minimum_action_effect_ratio"]),
        },
        "no_fixed_family_dominance": {
            "passed": maximum_fixed_share
            <= float(thresholds["maximum_fixed_unique_pareto_share"]),
            "actual": maximum_fixed_share,
            "requirement": float(thresholds["maximum_fixed_unique_pareto_share"]),
        },
        "static_context_heterogeneity": {
            "passed": max(
                context_signal["map_percentile"],
                context_signal["density_percentile"],
            )
            >= float(thresholds["context_permutation_percentile"]),
            "actual": {
                "map_percentile": context_signal["map_percentile"],
                "density_percentile": context_signal["density_percentile"],
            },
            "requirement": float(thresholds["context_permutation_percentile"]),
        },
    }
    if not gates["integrity"]["passed"]:
        decision = "incomplete_or_invalid_probe"
    elif not gates["repair_coverage"]["passed"]:
        decision = "increase_probe_density_before_counterfactual_collection"
    elif not gates["state_coverage"]["passed"]:
        decision = "collect_more_repair_states_before_interpretation"
    elif not (
        gates["action_diversity"]["passed"]
        and gates["neighborhood_diversity"]["passed"]
    ):
        decision = "revise_high_level_action_space"
    elif not gates["action_signal_over_trial_noise"]["passed"]:
        decision = "increase_trials_before_modeling"
    elif gates["static_context_heterogeneity"]["passed"]:
        decision = "test_contextual_candidate_ranking_on_independent_maps"
    else:
        decision = "retain_dynamic_policy_and_narrow_static_transfer_claim"
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "pre_registration": {
            "maps": expected_maps,
            "tasks": expected_tasks,
            "solver_seeds": solver_seeds,
            "trials": expected_trials,
            "horizon": 1,
            "families": all_families,
            "require_complete": require_complete,
        },
        "integrity": {
            "passed": integrity_passed,
            "qualification_rows": len(qualification),
            "baseline_rows": len(baseline),
            "state_rows": len(states),
            "unique_state_rows": len(canonical_sources),
            "outcome_rows": len(outcomes),
            "candidate_rows": sum(len(value) for value in candidate_rows.values()),
            "duplicate_state_ids": duplicate_state_id_count,
            "duplicate_fingerprint_groups": duplicate_fingerprint_groups,
            "fingerprint_context_mismatches": fingerprint_context_mismatch,
            "orphan_outcomes": orphan_outcome_count,
            "invalid_actions": invalid_action_count,
            "trial_mismatches": trial_mismatch_count,
            "candidate_count_mismatches": candidate_count_mismatch,
            "counterfactual_errors": counterfactual_error_count,
            "volume_checks": volume_checks,
        },
        "qualification": {
            "valid_count": valid_qualification,
            "repairable_count": repairable,
            "repairable_rate": _ratio(repairable, valid_qualification),
            "map_count": len(qualification_maps),
            "task_count": len(qualification_tasks),
        },
        "mechanism": {
            "state_count": state_count,
            "action_diversity_rate": action_diversity_rate,
            "neighborhood_diversity_rate": neighborhood_diversity_rate,
            "action_effect_ratio": action_effect_ratio,
            "mean_conflict_range": _mean(
                row["conflict_range"] for row in state_summaries
            ),
            "mean_auc_range": _mean(row["auc_range"] for row in state_summaries),
            "mean_trial_conflict_disagreement_rate": _mean(
                row["trial_conflict_disagreement_rate"] for row in state_summaries
            ),
            "mean_trial_neighborhood_stability_rate": _mean(
                row["trial_neighborhood_stability_rate"] for row in state_summaries
            ),
            "unique_pareto_family_counts": dict(sorted(unique_family_counts.items())),
            "maximum_fixed_unique_pareto_share": maximum_fixed_share,
        },
        "context_signal": context_signal,
        "by_map_density": group_summary,
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates.values()),
        "decision": decision,
    }


def render_probe_markdown(report: dict[str, Any]) -> str:
    mechanism = report["mechanism"]
    lines = [
        "# MovingAI InitLNS mechanism probe",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Coverage",
        "",
        f"- Qualification: {report['qualification']['valid_count']} valid, "
        f"{report['qualification']['repairable_count']} repairable "
        f"({report['qualification']['repairable_rate']:.1%})",
        f"- Counterfactual states: {mechanism['state_count']}",
        f"- Counterfactual outcomes: {report['integrity']['outcome_rows']}",
        "",
        "## Mechanism",
        "",
        f"- States with distinct action outcomes: {mechanism['action_diversity_rate']:.1%}",
        f"- States with distinct realized neighborhoods: "
        f"{mechanism['neighborhood_diversity_rate']:.1%}",
        f"- Action effect ratio over trial noise: {mechanism['action_effect_ratio']:.3f}",
        f"- Maximum fixed family unique-Pareto share: "
        f"{mechanism['maximum_fixed_unique_pareto_share']:.1%}",
        "",
        "## Static conditioning",
        "",
        f"- Map permutation percentile: {report['context_signal']['map_percentile']:.3f}",
        f"- Within-map density percentile: "
        f"{report['context_signal']['density_percentile']:.3f}",
        "",
        "## Gates",
        "",
    ]
    for name, gate in report["gates"].items():
        lines.append(f"- {name}: **{'PASS' if gate['passed'] else 'FAIL'}**")
    lines.extend(
        [
            "",
            "This probe diagnoses whether the high-level action space has an immediate, "
            "repeatable mechanism signal. It is not a transfer-learning or RL result.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_probe(
    collection: str | Path,
    config: str | Path,
    output: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    collection_root = Path(collection).resolve()
    output_root = Path(output).resolve()
    settings = _read_json(Path(config).resolve())
    run_config = _read_json(collection_root / "run_config.json")
    collection_config = run_config["configuration"]
    qualification = _read_jsonl(collection_root / "qualification_manifest.jsonl")
    baseline = _read_jsonl(collection_root / "collection_manifest.jsonl")
    counterfactual_manifest = _read_jsonl(
        collection_root / "counterfactual_manifest.jsonl"
    )
    states = []
    outcomes = []
    counterfactual_errors = 0
    for row in counterfactual_manifest:
        counterfactual_errors += int(row.get("error_count", 0))
        if str(row.get("status")) not in {"ok", "resumed"}:
            continue
        states.extend(_read_jsonl(collection_root / str(row["states_file"])))
        outcomes.extend(_read_jsonl(collection_root / str(row["outcomes_file"])))
    report = summarize_probe_records(
        qualification,
        baseline,
        states,
        outcomes,
        settings,
        collection_config,
        require_complete=require_complete,
        counterfactual_error_count=counterfactual_errors,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "movingai_mechanism_probe.json", report)
    (output_root / "movingai_mechanism_probe.md").write_text(
        render_probe_markdown(report), encoding="utf-8"
    )
    return report
