from __future__ import annotations

import collections
import json
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .repair_experience import (
    _cell_zone,
    _map_features,
    _sparse_heatmap,
    _task_features,
)
from .retrieval import (
    _aggregate_role_template,
    _effective_probability,
    _rank_neighbors,
    repair_raw_features,
    vectorize,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _manifest_row(
    dataset_root: Path, split: str, task_id: str
) -> dict[str, Any]:
    for row in _read_jsonl(dataset_root / split / "manifest.jsonl"):
        if row["task_id"] == task_id:
            return row
    raise ValueError(f"unknown task in {split}: {task_id}")


class RepairGuide:
    def __init__(
        self,
        dataset: str | Path,
        split: str,
        task_id: str,
        index: str | Path,
        evaluation: str | Path,
        effective_threshold: float = 0.67,
        role_weight: float = 0.5,
        conflict_weight: float = 0.35,
        baseline_weight: float = 0.15,
    ) -> None:
        if split not in {"validation", "test"}:
            raise ValueError("guided solving is limited to validation or test")
        if not 0.0 <= effective_threshold <= 1.0:
            raise ValueError("effective threshold must be within [0, 1]")
        score_weight = role_weight + conflict_weight + baseline_weight
        if abs(score_weight - 1.0) > 1e-9:
            raise ValueError("Agent mapping weights must sum to one")
        self.dataset_root = Path(dataset).resolve()
        self.split = split
        self.task_id = task_id
        self.index_root = Path(index).resolve()
        self.evaluation_root = Path(evaluation).resolve()
        self.effective_threshold = effective_threshold
        self.role_weight = role_weight
        self.conflict_weight = conflict_weight
        self.baseline_weight = baseline_weight

        self.manifest = _manifest_row(
            self.dataset_root, split, task_id
        )
        self.map_document = _read_json(
            self.dataset_root / split / self.manifest["map_file"]
        )
        self.task_document = _read_json(
            self.dataset_root / split / self.manifest["task_file"]
        )
        self.map_features = _map_features(
            self.map_document, self.manifest
        )
        self.task_features = _task_features(
            self.task_document, self.manifest
        )
        self.zone_sets = {
            name: {tuple(cell) for cell in cells}
            for name, cells in self.map_document["metadata"]["zones"].items()
        }
        normalizer = _read_json(self.index_root / "normalizer.json")
        if normalizer.get("fit_split") != "train":
            raise ValueError("guidance index was not fit on Train")
        self.schema = normalizer["repair"]
        self.entries = _read_jsonl(
            self.index_root / "repair_index.jsonl"
        )
        if any(
            not str(entry["task_id"]).startswith("train_")
            for entry in self.entries
        ):
            raise ValueError("guidance index contains non-Train experience")
        evaluation_summary = _read_json(
            self.evaluation_root / "evaluation_summary.json"
        )
        if (
            evaluation_summary.get("index_split") != "train"
            or evaluation_summary.get("query_split") != "validation"
        ):
            raise ValueError("guidance parameters are not Train/Validation")
        parameters = evaluation_summary["selected_parameters"]["repair"]
        self.k = int(parameters["k"])
        self.group_weights = {
            key: float(value)
            for key, value in parameters["group_weights"].items()
        }
        self.ood_threshold = float(
            parameters["ood_distance_threshold"]
        )

    def _query_case(self, request: dict[str, Any]) -> dict[str, Any]:
        events = request["conflict_events"]
        heatmap, _ = _sparse_heatmap(events)
        metadata = self.task_document["metadata"]
        agents = []
        for agent, path in enumerate(request["paths"]):
            start = self.task_document["starts"][agent]
            goal = self.task_document["goals"][agent]
            agents.append(
                {
                    "agent": agent,
                    "start": start,
                    "goal": goal,
                    "start_zone": _cell_zone(start, self.zone_sets),
                    "goal_zone": _cell_zone(goal, self.zone_sets),
                    "flow_assignment": metadata["flow_assignments"][agent],
                    "shortest_distance": metadata[
                        "actual_shortest_distances"
                    ][agent],
                    "path_before": path,
                    "path_after": None,
                }
            )
        return {
            "map_features": self.map_features,
            "task_features": self.task_features,
            "conflict_events_before": events,
            "conflict_heatmap_before": heatmap,
            "seed_conflict": request["seed_conflict"],
            "agents": agents,
        }

    @staticmethod
    def _role_probabilities(
        template: dict[str, Any],
    ) -> dict[str, float]:
        return {
            item["role"]: float(item["probability"])
            for item in template["additional_role_distribution"]
        }

    @staticmethod
    def _closeness(value: float, target: float | None, scale: float) -> float:
        if target is None:
            return 0.5
        return math.exp(-abs(value - target) / max(scale, 1e-6))

    def _map_agents(
        self,
        request: dict[str, Any],
        query_case: dict[str, Any],
        template: dict[str, Any],
    ) -> list[int]:
        seed_pair = [int(value) for value in request["seed_conflict"]]
        target_size = len(request["baseline_neighborhood"])
        selected = list(dict.fromkeys(seed_pair))
        role_probabilities = self._role_probabilities(template)
        baseline = set(int(value) for value in request["baseline_neighborhood"])
        conflict_graph: collections.defaultdict[int, set[int]] = (
            collections.defaultdict(set)
        )
        for event in request["conflict_events"]:
            first, second = (int(value) for value in event["agents"])
            conflict_graph[first].add(second)
            conflict_graph[second].add(first)
        maximum_degree = max(
            [len(values) for values in conflict_graph.values()] or [1]
        )
        distances: dict[int, int] = {
            agent: 0 for agent in selected
        }
        open_agents = collections.deque(selected)
        while open_agents:
            current = open_agents.popleft()
            for neighbor in sorted(conflict_graph[current]):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    open_agents.append(neighbor)

        heatmap_cells = {
            tuple(item["cell"])
            for item in query_case["conflict_heatmap_before"]
        }
        scored = []
        for agent in query_case["agents"]:
            agent_id = int(agent["agent"])
            if agent_id in selected:
                continue
            signature = (
                f"{agent['start_zone']}->{agent['goal_zone']}"
                f"|{agent['flow_assignment']}"
            )
            shortest = max(1.0, float(agent["shortest_distance"]))
            path = agent["path_before"]
            stretch = (len(path) - 1) / shortest
            overlap = (
                sum(tuple(cell) in heatmap_cells for cell in path)
                / max(1, len(path))
            )
            role_score = (
                0.5 * role_probabilities.get(signature, 0.0)
                + 0.2
                * self._closeness(
                    shortest,
                    template["mean_shortest_distance"],
                    max(5.0, shortest * 0.25),
                )
                + 0.15
                * self._closeness(
                    stretch, template["mean_path_stretch"], 0.25
                )
                + 0.15
                * self._closeness(
                    overlap,
                    template["mean_path_conflict_overlap"],
                    0.1,
                )
            )
            degree_score = len(conflict_graph[agent_id]) / maximum_degree
            proximity_score = (
                1.0 / (1.0 + distances[agent_id])
                if agent_id in distances
                else 0.0
            )
            conflict_score = 0.6 * degree_score + 0.4 * proximity_score
            score = (
                self.role_weight * role_score
                + self.conflict_weight * conflict_score
                + self.baseline_weight * float(agent_id in baseline)
            )
            scored.append((score, agent_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected.extend(
            agent for _, agent in scored[: max(0, target_size - len(selected))]
        )
        return selected

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        query_case = self._query_case(request)
        query_vector = vectorize(
            repair_raw_features(query_case), self.schema
        )
        neighbors = _rank_neighbors(
            query_vector,
            self.entries,
            self.schema,
            self.group_weights,
            self.k,
            "repair",
        )
        probability = _effective_probability(neighbors)
        nearest_distance = (
            float(neighbors[0][1]) if neighbors else math.inf
        )
        out_of_distribution = nearest_distance > self.ood_threshold
        template = _aggregate_role_template(neighbors)
        reason = ""
        if not neighbors:
            reason = "no_neighbors"
        elif out_of_distribution:
            reason = "out_of_distribution"
        elif probability < self.effective_threshold:
            reason = "low_confidence"
        elif template["source_effective_neighbor_count"] == 0:
            reason = "no_effective_neighbors"
        agents = (
            []
            if reason
            else self._map_agents(request, query_case, template)
        )
        if not reason and len(agents) != len(
            request["baseline_neighborhood"]
        ):
            agents = []
            reason = "mapping_failed"
        return {
            "use_guidance": not reason,
            "out_of_distribution": out_of_distribution,
            "effective_probability": probability,
            "nearest_distance": nearest_distance,
            "agents": agents,
            "fallback_reason": reason,
            "neighbors": [
                {
                    "case_id": entry["case_id"],
                    "task_id": entry["task_id"],
                    "distance": distance,
                    "effective": entry["effective"],
                }
                for entry, distance in neighbors
            ],
            "role_template": template,
            "python_runtime_ms": (
                time.perf_counter() - started
            ) * 1000.0,
        }


def run_guided_instance(
    solver: str | Path,
    instance: str | Path,
    trace: str | Path,
    guide: RepairGuide,
    seed: int,
    neighborhood: int = 6,
    iterations: int = 500,
    time_limit_ms: int = 5000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    command = [
        str(Path(solver).resolve()),
        "--instance",
        str(Path(instance).resolve()),
        "--seed",
        str(seed),
        "--neighborhood",
        str(neighborhood),
        "--iterations",
        str(iterations),
        "--time-limit-ms",
        str(time_limit_ms),
        "--trace",
        str(Path(trace).resolve()),
        "--guidance-stdio",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    watchdog = threading.Timer(
        max(30.0, time_limit_ms / 1000.0 + 30.0),
        process.kill,
    )
    watchdog.daemon = True
    watchdog.start()
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("failed to open guidance pipes")
    decisions = []
    result = None
    try:
        for line in process.stdout:
            line = line.strip()
            if line.startswith("GUIDANCE_REQUEST "):
                request = json.loads(line[len("GUIDANCE_REQUEST ") :])
                try:
                    decision = guide.decide(request)
                except Exception as error:
                    decision = {
                        "use_guidance": False,
                        "out_of_distribution": False,
                        "effective_probability": -1.0,
                        "nearest_distance": -1.0,
                        "agents": [],
                        "fallback_reason": "python_error",
                        "neighbors": [],
                        "role_template": {},
                        "python_runtime_ms": 0.0,
                        "error": str(error),
                    }
                decisions.append(
                    {
                        "iteration": request["iteration"],
                        **decision,
                    }
                )
                reason = decision["fallback_reason"] or "-"
                response = [
                    "GUIDANCE",
                    "1" if decision["use_guidance"] else "0",
                    f"{decision['effective_probability']:.12g}",
                    f"{decision['nearest_distance']:.12g}",
                    "1" if decision["out_of_distribution"] else "0",
                    reason,
                    *(str(agent) for agent in decision["agents"]),
                ]
                process.stdin.write(" ".join(response) + "\n")
                process.stdin.flush()
            elif line.startswith("RESULT "):
                result = json.loads(line[len("RESULT ") :])
        process.stdin.close()
        return_code = process.wait()
    finally:
        watchdog.cancel()
    stderr = process.stderr.read() if process.stderr is not None else ""
    if result is None or return_code not in {0, 1}:
        raise RuntimeError(
            f"guided solver failed ({return_code}): {stderr.strip()}"
        )
    result["return_code"] = return_code
    return result, decisions
