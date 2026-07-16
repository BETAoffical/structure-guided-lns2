from __future__ import annotations

import collections
import itertools
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments._common import mean as _mean, quantile as _quantile
from experiments.closed_loop_confirmation import _sha256
from experiments.local_representation_audit import analyze_state
from experiments.realized_neighborhood_ranking_audit import effectiveness_dominates
from experiments.repair_collection import _fingerprint, _read_json, _read_jsonl, _write_json, _write_jsonl


SCHEMA = "lns2.graph_representation_audit.v2"
SCHEMA_VERSION = 2
INDEX_SCHEMA_VERSION = 1
MODEL_NAMES = ("flat_mlp", "agent_deepsets", "conflict_gnn")
NODE_FEATURE_NAMES = (
    "conflict_degree",
    "delay",
    "path_cost",
    "shortest_path_cost",
    "path_stretch",
    "path_wait_ratio",
    "path_length",
    "path_visit_heat_mean",
    "path_visit_heat_max",
    "path_degree_mean",
    "path_low_degree_ratio",
    "path_articulation_ratio",
    "path_obstacle_rate_r2",
    "path_obstacle_rate_r4",
)
FORBIDDEN_INPUT_FRAGMENTS = (
    "outcome",
    "label",
    "conflicts_after",
    "solved_rate",
    "runtime",
    "post_repair",
    "layout_mode",
    "task_variant",
    "agent_density",
)


def input_feature_is_forbidden(name: str) -> bool:
    normalized = name.lower()
    if any(fragment in normalized for fragment in FORBIDDEN_INPUT_FRAGMENTS):
        return True
    return "generated" in normalized and not normalized.startswith(
        "state.low_level_generated"
    )


def _path_wait_ratio(path: list[int]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(left == right for left, right in zip(path, path[1:])) / (len(path) - 1)


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported graph-representation audit config")
    if tuple(map(str, config.get("models", []))) != MODEL_NAMES:
        raise ValueError("graph audit model set differs from preregistration")
    if list(map(int, config.get("random_seeds", []))) != [20260714, 20260715, 20260716]:
        raise ValueError("graph audit random seeds changed")
    expected = {
        "hidden_size": 64,
        "flat_hidden_sizes": [128, 64],
        "message_passing_layers": 2,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "batch_states": 64,
        "minimum_epochs": 20,
        "maximum_epochs": 200,
        "early_stopping_patience": 20,
    }
    if dict(config.get("training", {})) != expected:
        raise ValueError("graph audit training parameters changed")
    if int(config.get("bootstrap_samples", 0)) != 5000:
        raise ValueError("graph audit requires 5,000 map bootstrap samples")


class ProgressLog:
    def __init__(self, path: Path, *, reset: bool) -> None:
        self.path = path
        self.started = time.perf_counter()
        self.rows = [] if reset or not path.exists() else _read_jsonl(path)
        if reset:
            _write_jsonl(path, [])

    def emit(self, event: str, **details: Any) -> None:
        row = {
            "event": event,
            "elapsed_seconds": time.perf_counter() - self.started,
            **details,
        }
        self.rows.append(row)
        _write_jsonl(self.path, self.rows)
        fields = " ".join(f"{name}={value}" for name, value in details.items())
        print(f"[graph-audit] {event}{' ' + fields if fields else ''}", flush=True)


def _registered_paths(project_root: Path) -> dict[str, Path]:
    return {
        "current_lomo_predictions_sha256": project_root / "build/initlns-model-capacity-audit-v1/lomo_predictions__current.jsonl",
        "policy_states_sha256": project_root / "build/initlns-policy-visited-natural-v2-collection/selected_states.jsonl",
        "policy_candidates_sha256": project_root / "build/initlns-policy-visited-natural-v2-collection/candidates.jsonl",
        "historical_candidates_sha256": project_root / "build/realized-neighborhood-stability-probe-v1/candidates.jsonl",
        "aggregate_train_index_sha256": project_root / "build/initlns-policy-visited-natural-v2-training/aggregate_train_index.jsonl",
        "validation_index_sha256": project_root / "build/initlns-policy-visited-natural-v2-training/validation_index.jsonl",
        "validation_gbdt_predictions_sha256": project_root / "build/initlns-policy-visited-natural-v2-offline/offline_predictions__v2_realized_dynamic.jsonl",
    }


def validate_registered_inputs(project_root: Path, config: dict[str, Any]) -> dict[str, str]:
    paths = _registered_paths(project_root)
    actual = {name: _sha256(path) for name, path in paths.items()}
    expected = {str(name): str(value).lower() for name, value in dict(config["registered_inputs"]).items()}
    if actual != expected:
        raise ValueError(f"registered graph-audit inputs changed: {actual}")
    return actual


def _node_features(state: dict[str, Any]) -> tuple[list[int], list[list[float]], list[list[int]]]:
    analysis = analyze_state(state)
    agents = sorted(state["agents"], key=lambda row: int(row["id"]))
    ids = [int(agent["id"]) for agent in agents]
    if len(ids) != len(set(ids)):
        raise ValueError("state has duplicate agent ids")
    position = {agent_id: index for index, agent_id in enumerate(ids)}
    rows: list[list[float]] = []
    for agent in agents:
        path = [int(cell) for cell in agent["path"]]
        if not path or any(cell not in analysis.free_cells for cell in path):
            raise ValueError("agent path contains an invalid cell")
        degree = [float(analysis.degrees.get(cell, 0)) for cell in path]
        heat = [float(analysis.visit_heat.get(cell, 0)) for cell in path]
        obstacle_2 = [float(analysis.obstacle_rate_2.get(cell, 0.0)) for cell in path]
        obstacle_4 = [float(analysis.obstacle_rate_4.get(cell, 0.0)) for cell in path]
        path_cost = float(agent.get("path_cost", max(0, len(path) - 1)))
        shortest = float(agent.get("shortest_path_cost", 0))
        rows.append(
            [
                float(agent.get("conflict_degree", 0)),
                float(agent.get("delay", 0)),
                path_cost,
                shortest,
                path_cost / max(1.0, shortest),
                _path_wait_ratio(path),
                float(len(path)),
                _mean(heat),
                max(heat, default=0.0),
                _mean(degree),
                sum(value <= 2 for value in degree) / max(1, len(degree)),
                sum(cell in analysis.articulation for cell in path) / max(1, len(path)),
                _mean(obstacle_2),
                _mean(obstacle_4),
            ]
        )
    edges = []
    for left, right in sorted(analysis.pair_set):
        if left not in position or right not in position:
            raise ValueError("conflict edge references an unknown agent")
        edges.append([position[left], position[right]])
    if set(tuple(sorted(map(int, edge))) for edge in state.get("conflict_edges", [])) != analysis.pair_set:
        raise ValueError("state conflict edges differ from reconstructed paths")
    return ids, rows, edges


def _raw_states(project_root: Path) -> dict[str, dict[str, Any]]:
    paths = _registered_paths(project_root)
    raw: dict[str, dict[str, Any]] = {}
    for path in (paths["historical_candidates_sha256"], paths["policy_candidates_sha256"]):
        for row in _read_jsonl(path):
            state_id = str(row["state_id"])
            if state_id in raw:
                raise ValueError(f"duplicate raw state: {state_id}")
            raw[state_id] = row
    return raw


def build_graph_indexes(project_root: Path, config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    validate_registered_inputs(project_root, config)
    paths = _registered_paths(project_root)
    train_rows = _read_jsonl(paths["aggregate_train_index_sha256"])
    validation_rows = _read_jsonl(paths["validation_index_sha256"])
    all_rows = train_rows + validation_rows
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in all_rows:
        by_state[str(row["state_id"])].append(row)
    raw = _raw_states(project_root)
    if set(by_state) != set(raw):
        raise ValueError("raw states and registered indexes differ")
    flat_names = sorted({name for row in train_rows for name in row["features"]["realized_dynamic"]})
    metadata_names = [name for name in flat_names if name.startswith(("state.", "proposal."))]
    if len(flat_names) != 139:
        raise ValueError(f"registered realized_dynamic feature count changed: {len(flat_names)}")
    if any(input_feature_is_forbidden(name) for name in flat_names):
        raise ValueError("graph audit input feature leakage detected")
    states = []
    candidates = []
    for state_id in sorted(by_state):
        source = raw[state_id]
        state = source["state"]
        ids, node_rows, edges = _node_features(state)
        raw_candidates = {str(row["candidate_id"]): row for row in source["candidates"]}
        indexed = by_state[state_id]
        if {str(row["candidate_id"]) for row in indexed} != set(raw_candidates):
            raise ValueError(f"candidate ids differ for state {state_id}")
        split_values = {str(row["split"]) for row in indexed}
        map_values = {str(row["map_id"]) for row in indexed}
        if len(split_values) != 1 or len(map_values) != 1:
            raise ValueError("state has inconsistent split or map")
        states.append(
            {
                "schema": "lns2.graph_state.v1",
                "state_id": state_id,
                "map_id": next(iter(map_values)),
                "split": next(iter(split_values)),
                "agent_ids": ids,
                "node_features": node_rows,
                "edges": edges,
            }
        )
        known = set(ids)
        for row in sorted(indexed, key=lambda value: str(value["candidate_id"])):
            candidate_id = str(row["candidate_id"])
            agents = sorted(map(int, row["agents"]))
            if agents != sorted(map(int, raw_candidates[candidate_id]["agents"])):
                raise ValueError(f"candidate agents differ: {state_id}/{candidate_id}")
            if not agents or len(agents) != len(set(agents)) or not set(agents).issubset(known):
                raise ValueError("candidate contains invalid agent ids")
            features = row["features"]["realized_dynamic"]
            candidates.append(
                {
                    "schema": "lns2.graph_candidate.v1",
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    "map_id": str(row["map_id"]),
                    "split": str(row["split"]),
                    "agents": agents,
                    "actual_size": int(row["actual_size"]),
                    "task_id": str(row["task_id"]),
                    "selection_families": list(map(str, row["selection_families"])),
                    "flat_features": [float(features.get(name, 0.0)) for name in flat_names],
                    "metadata_features": [float(features.get(name, 0.0)) for name in metadata_names],
                    "outcome": dict(row["outcome"]),
                    "labels": dict(row["labels"]),
                }
            )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "graph_states.jsonl", states)
    _write_jsonl(output_root / "graph_candidates.jsonl", candidates)
    manifest = {
        "schema": "lns2.graph_index_manifest.v1",
        "schema_version": INDEX_SCHEMA_VERSION,
        "state_count": len(states),
        "candidate_count": len(candidates),
        "train_state_count": sum(row["split"] == "policy_train" for row in states),
        "anchor_state_count": sum(row["split"] != "policy_train" and row["split"] != "policy_validation" for row in states),
        "validation_state_count": sum(row["split"] == "policy_validation" for row in states),
        "map_count": len({row["map_id"] for row in states}),
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "flat_feature_names": flat_names,
        "metadata_feature_names": metadata_names,
        "registered_inputs": validate_registered_inputs(project_root, config),
        "states_sha256": _sha256(output_root / "graph_states.jsonl"),
        "candidates_sha256": _sha256(output_root / "graph_candidates.jsonl"),
    }
    if (manifest["train_state_count"], manifest["anchor_state_count"], manifest["validation_state_count"]) != (288, 23, 154):
        raise ValueError("graph index split counts changed")
    _write_json(output_root / "graph_index_manifest.json", manifest)
    return manifest


@dataclass
class StateExample:
    state_id: str
    map_id: str
    split: str
    agent_ids: list[int]
    node_features: list[list[float]]
    edges: list[list[int]]
    candidates: list[dict[str, Any]]


def load_graph_examples(index_root: Path) -> list[StateExample]:
    states = {str(row["state_id"]): row for row in _read_jsonl(index_root / "graph_states.jsonl")}
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in _read_jsonl(index_root / "graph_candidates.jsonl"):
        grouped[str(row["state_id"])].append(row)
    if set(states) != set(grouped):
        raise ValueError("graph states and candidates differ")
    return [
        StateExample(
            state_id=state_id,
            map_id=str(states[state_id]["map_id"]),
            split=str(states[state_id]["split"]),
            agent_ids=list(map(int, states[state_id]["agent_ids"])),
            node_features=list(states[state_id]["node_features"]),
            edges=list(states[state_id]["edges"]),
            candidates=sorted(grouped[state_id], key=lambda row: str(row["candidate_id"])),
        )
        for state_id in sorted(states)
    ]


def dominance_pairs(candidates: list[dict[str, Any]]) -> list[tuple[int, int]]:
    pairs = []
    for left, right in itertools.combinations(range(len(candidates)), 2):
        if effectiveness_dominates(candidates[left]["outcome"], candidates[right]["outcome"]):
            pairs.append((left, right))
        elif effectiveness_dominates(candidates[right]["outcome"], candidates[left]["outcome"]):
            pairs.append((right, left))
    return pairs


def _import_torch():
    import torch
    from torch import nn

    return torch, nn


class Normalization:
    def __init__(self, mean: Any, std: Any):
        self.mean = mean
        self.std = std


def _normalization(examples: list[StateExample], field: str, device: Any) -> Normalization:
    torch, _ = _import_torch()
    values = []
    if field == "node_features":
        values = [row for example in examples for row in example.node_features]
    else:
        values = [candidate[field] for example in examples for candidate in example.candidates]
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    mean = tensor.mean(dim=0)
    raw_std = tensor.std(dim=0, unbiased=False)
    std = torch.where(raw_std < 1e-6, torch.ones_like(raw_std), raw_std)
    return Normalization(mean, std)


def _masked_pool(values: Any, mask: Any) -> tuple[Any, Any, Any]:
    torch, _ = _import_torch()
    weights = mask.to(values.dtype)
    count = weights.sum(dim=-1, keepdim=True)
    mean = torch.einsum("bcn,bnh->bch", weights, values) / count.clamp_min(1.0)
    expanded = values[:, None, :, :].expand(-1, mask.shape[1], -1, -1)
    maximum = expanded.masked_fill(~mask[..., None], -torch.inf).amax(dim=2)
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    return mean, maximum, (count > 0).to(values.dtype)


def make_model(model_name: str, flat_dim: int, metadata_dim: int, node_dim: int, config: dict[str, Any]):
    torch, nn = _import_torch()
    training = config["training"]
    hidden = int(training["hidden_size"])
    dropout = float(training["dropout"])

    class FlatMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(flat_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
            )

        def forward(self, batch: dict[str, Any]) -> Any:
            return self.network(batch["flat"]).squeeze(-1)

    class SetScorer(nn.Module):
        def __init__(self, use_edges: bool):
            super().__init__()
            self.use_edges = use_edges
            self.node_input = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.self_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(2))
            self.neighbor_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(2))
            self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(2))
            self.dropout = nn.Dropout(dropout)
            readout_dim = hidden * 6 + 3 + metadata_dim
            self.head = nn.Sequential(
                nn.Linear(readout_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
            )

        def forward(self, batch: dict[str, Any]) -> Any:
            h = self.node_input(batch["nodes"])
            if self.use_edges:
                for self_layer, neighbor_layer, norm in zip(self.self_layers, self.neighbor_layers, self.norms):
                    update = torch.relu(self_layer(h) + neighbor_layer(torch.bmm(batch["adjacency"], h)))
                    h = norm(h + self.dropout(update))
                    h = h * batch["node_mask"][..., None]
            selected = _masked_pool(h, batch["candidate_mask"])
            comparison_mask = (
                batch["boundary_mask"]
                if self.use_edges
                else batch["node_mask"][:, None, :]
                & ~batch["candidate_mask"]
            )
            boundary = _masked_pool(h, comparison_mask)
            global_mask = batch["node_mask"][:, None, :].expand(-1, batch["candidate_mask"].shape[1], -1)
            global_pool = _masked_pool(h, global_mask)
            representation = torch.cat((*selected, *boundary, *global_pool, batch["metadata"]), dim=-1)
            return self.head(representation).squeeze(-1)

    if model_name == "flat_mlp":
        return FlatMLP()
    if model_name == "agent_deepsets":
        return SetScorer(False)
    if model_name == "conflict_gnn":
        return SetScorer(True)
    raise ValueError(f"unknown graph audit model: {model_name}")


def collate_examples(examples: list[StateExample], normalizations: dict[str, Normalization], device: Any) -> dict[str, Any]:
    torch, _ = _import_torch()
    batch_size = len(examples)
    max_nodes = max(len(example.node_features) for example in examples)
    max_candidates = max(len(example.candidates) for example in examples)
    node_dim = len(examples[0].node_features[0])
    flat_dim = len(examples[0].candidates[0]["flat_features"])
    metadata_dim = len(examples[0].candidates[0]["metadata_features"])
    nodes = torch.zeros((batch_size, max_nodes, node_dim), device=device)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
    adjacency = torch.zeros((batch_size, max_nodes, max_nodes), device=device)
    candidate_mask = torch.zeros((batch_size, max_candidates, max_nodes), dtype=torch.bool, device=device)
    candidate_valid = torch.zeros((batch_size, max_candidates), dtype=torch.bool, device=device)
    flat = torch.zeros((batch_size, max_candidates, flat_dim), device=device)
    metadata = torch.zeros((batch_size, max_candidates, metadata_dim), device=device)
    pair_lists = []
    for batch_index, example in enumerate(examples):
        n = len(example.node_features)
        c = len(example.candidates)
        node = torch.tensor(example.node_features, dtype=torch.float32, device=device)
        nodes[batch_index, :n] = (node - normalizations["node"].mean) / normalizations["node"].std
        node_mask[batch_index, :n] = True
        adjacency[batch_index, torch.arange(n), torch.arange(n)] = 1.0
        for left, right in example.edges:
            adjacency[batch_index, left, right] = 1.0
            adjacency[batch_index, right, left] = 1.0
        adjacency[batch_index, :n, :n] /= adjacency[batch_index, :n, :n].sum(dim=-1, keepdim=True).clamp_min(1.0)
        if len(example.agent_ids) != n or len(set(example.agent_ids)) != n:
            raise ValueError("graph example has invalid agent ids")
        id_to_position = {
            agent_id: index for index, agent_id in enumerate(example.agent_ids)
        }
        for candidate_index, candidate in enumerate(example.candidates):
            try:
                positions = [id_to_position[int(agent)] for agent in candidate["agents"]]
            except KeyError as error:
                raise ValueError("candidate references an unknown graph agent") from error
            candidate_mask[batch_index, candidate_index, positions] = True
            flat_value = torch.tensor(candidate["flat_features"], dtype=torch.float32, device=device)
            metadata_value = torch.tensor(candidate["metadata_features"], dtype=torch.float32, device=device)
            flat[batch_index, candidate_index] = (flat_value - normalizations["flat"].mean) / normalizations["flat"].std
            metadata[batch_index, candidate_index] = (metadata_value - normalizations["metadata"].mean) / normalizations["metadata"].std
        candidate_valid[batch_index, :c] = True
        pair_lists.append(dominance_pairs(example.candidates))
    incident = torch.bmm(candidate_mask.to(adjacency.dtype), (adjacency > 0).to(adjacency.dtype)) > 0
    boundary_mask = incident & ~candidate_mask & node_mask[:, None, :]
    return {
        "nodes": nodes,
        "node_mask": node_mask,
        "adjacency": adjacency,
        "candidate_mask": candidate_mask,
        "boundary_mask": boundary_mask,
        "candidate_valid": candidate_valid,
        "flat": flat,
        "metadata": metadata,
        "pair_lists": pair_lists,
    }


def _pairwise_loss(scores: Any, pair_lists: list[list[tuple[int, int]]]) -> Any:
    torch, _ = _import_torch()
    losses = []
    for state_index, pairs in enumerate(pair_lists):
        if not pairs:
            continue
        better = torch.tensor([pair[0] for pair in pairs], device=scores.device)
        worse = torch.tensor([pair[1] for pair in pairs], device=scores.device)
        losses.append(torch.nn.functional.softplus(-(scores[state_index, better] - scores[state_index, worse])).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def _set_determinism(seed: int) -> None:
    torch, _ = _import_torch()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def train_model(
    model_name: str,
    fit: list[StateExample],
    config: dict[str, Any],
    seed: int,
    *,
    epochs: int,
    monitor: list[StateExample] | None = None,
) -> tuple[Any, dict[str, Normalization], dict[str, Any]]:
    torch, _ = _import_torch()
    _set_determinism(seed)
    device = torch.device(config["device"])
    norms = {
        "node": _normalization(fit, "node_features", device),
        "flat": _normalization(fit, "flat_features", device),
        "metadata": _normalization(fit, "metadata_features", device),
    }
    model = make_model(
        model_name,
        len(fit[0].candidates[0]["flat_features"]),
        len(fit[0].candidates[0]["metadata_features"]),
        len(fit[0].node_features[0]),
        config,
    ).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    order = list(fit)
    random.Random(seed).shuffle(order)
    batch_size = int(training["batch_states"])
    fit_batches = [
        collate_examples(order[offset : offset + batch_size], norms, device)
        for offset in range(0, len(order), batch_size)
    ]
    monitor_batches = (
        [
            collate_examples(monitor[offset : offset + batch_size], norms, device)
            for offset in range(0, len(monitor), batch_size)
        ]
        if monitor
        else []
    )
    best_epoch = epochs
    best_loss = float("inf")
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in fit_batches:
            optimizer.zero_grad(set_to_none=True)
            loss = _pairwise_loss(model(batch), batch["pair_lists"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        monitor_loss = None
        if monitor:
            model.eval()
            values = []
            with torch.no_grad():
                for batch in monitor_batches:
                    values.append(float(_pairwise_loss(model(batch), batch["pair_lists"]).cpu()))
            monitor_loss = _mean(values)
            if monitor_loss < best_loss - 1e-8:
                best_loss = monitor_loss
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
        history.append({"epoch": epoch, "train_loss": _mean(train_losses), "monitor_loss": monitor_loss})
        if monitor and epoch >= int(training["minimum_epochs"]) and patience >= int(training["early_stopping_patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, norms, {"best_epoch": best_epoch, "epochs_run": len(history), "best_monitor_loss": None if not monitor else best_loss, "history": history}


def predict_scores(model: Any, examples: list[StateExample], norms: dict[str, Normalization], config: dict[str, Any]) -> dict[str, list[float]]:
    torch, _ = _import_torch()
    model.eval()
    device = torch.device(config["device"])
    result = {}
    with torch.no_grad():
        for offset in range(0, len(examples), int(config["training"]["batch_states"])):
            current = examples[offset : offset + int(config["training"]["batch_states"])]
            batch = collate_examples(current, norms, device)
            scores = model(batch).cpu().tolist()
            for example, values in zip(current, scores):
                result[example.state_id] = list(map(float, values[: len(example.candidates)]))
    return result


def selection_records(examples: list[StateExample], scores_by_seed: list[dict[str, list[float]]], selector: str) -> dict[str, dict[str, Any]]:
    records = {}
    for example in examples:
        scores = [
            _mean(seed_scores[example.state_id][index] for seed_scores in scores_by_seed)
            for index in range(len(example.candidates))
        ]
        selected_index = sorted(range(len(scores)), key=lambda index: (-scores[index], str(example.candidates[index]["candidate_id"])))[0]
        selected = example.candidates[selected_index]
        best_conflicts = min(float(row["outcome"]["conflicts_after"]) for row in example.candidates)
        best_solved = max(float(row["outcome"]["solved_rate"]) for row in example.candidates)
        best_auc = min(float(row["outcome"]["conflict_auc"]) for row in example.candidates)
        best_generated = min(float(row["outcome"]["generated"]) for row in example.candidates)
        selected_conflicts = float(selected["outcome"]["conflicts_after"])
        selected_solved = float(selected["outcome"]["solved_rate"])
        selected_auc = float(selected["outcome"]["conflict_auc"])
        selected_generated = float(selected["outcome"]["generated"])
        records[example.state_id] = {
            "state_id": example.state_id,
            "map_id": example.map_id,
            "selector": selector,
            "candidate_id": str(selected["candidate_id"]),
            "task_id": str(selected["task_id"]),
            "selected_size": int(selected["actual_size"]),
            "selection_families": list(selected["selection_families"]),
            "pareto_hit": float(bool(selected["labels"]["effectiveness_pareto"])),
            "compute_aware_pareto_hit": float(bool(selected["labels"]["compute_aware_pareto"])),
            "runtime_sensitive_pareto_hit": float(bool(selected["labels"]["runtime_sensitive_pareto"])),
            "selected_conflicts": selected_conflicts,
            "selected_solved_rate": selected_solved,
            "selected_generated": selected_generated,
            "selected_runtime": float(selected["outcome"]["runtime"]),
            "conflict_regret": (selected_conflicts - best_conflicts) / max(1.0, abs(best_conflicts)),
            "solved_rate_regret": best_solved - selected_solved,
            "auc_regret": (selected_auc - best_auc) / max(1.0, abs(best_auc)),
            "generated_regret": (selected_generated - best_generated) / max(1.0, abs(best_generated)),
            "scores": scores,
        }
    return records


def summarize_records(records: dict[str, dict[str, Any]], examples: list[StateExample], scores_by_seed: list[dict[str, list[float]]]) -> dict[str, Any]:
    pair_correct = 0.0
    pair_count = 0
    by_state = {example.state_id: example for example in examples}
    for state_id, example in by_state.items():
        scores = [_mean(seed[state_id][index] for seed in scores_by_seed) for index in range(len(example.candidates))]
        for better, worse in dominance_pairs(example.candidates):
            pair_correct += 1.0 if scores[better] > scores[worse] else 0.5 if scores[better] == scores[worse] else 0.0
            pair_count += 1
    sizes = collections.Counter(int(row["selected_size"]) for row in records.values())
    return {
        "state_count": len(records),
        "pareto_top1_hit_rate": _mean(row["pareto_hit"] for row in records.values()),
        "mean_conflict_regret": _mean(row["conflict_regret"] for row in records.values()),
        "mean_solved_rate_regret": _mean(row["solved_rate_regret"] for row in records.values()),
        "pairwise_accuracy": pair_correct / pair_count if pair_count else 0.0,
        "selected_size_counts": dict(sorted(sizes.items())),
        "maximum_size_share": max(sizes.values(), default=0) / max(1, len(records)),
    }


def _baseline_records(project_root: Path) -> dict[str, dict[str, Any]]:
    path = _registered_paths(project_root)["current_lomo_predictions_sha256"]
    rows = _read_jsonl(path)
    return {str(row["state_id"]): row for row in rows}


def _validation_baseline_records(project_root: Path) -> dict[str, dict[str, Any]]:
    path = _registered_paths(project_root)["validation_gbdt_predictions_sha256"]
    rows = _read_jsonl(path)
    records = {str(row["state_id"]): row for row in rows}
    if len(records) != 154 or len({str(row["map_id"]) for row in records.values()}) != 6:
        raise ValueError("registered graph Validation baseline changed")
    return records


def _map_bootstrap(baseline: dict[str, dict[str, Any]], challenger: dict[str, dict[str, Any]], samples: int) -> dict[str, Any]:
    by_map: dict[str, list[str]] = collections.defaultdict(list)
    for state_id, row in baseline.items():
        if state_id not in challenger or str(row["map_id"]) != str(challenger[state_id]["map_id"]):
            raise ValueError("bootstrap comparison states differ")
        by_map[str(row["map_id"])].append(state_id)
    maps = sorted(by_map)
    rng = random.Random(20260716)
    top_deltas = []
    regret_improvements = []
    for _ in range(samples):
        sampled = [rng.choice(maps) for _ in maps]
        states = [state for map_id in sampled for state in by_map[map_id]]
        base_top = _mean(baseline[state]["pareto_hit"] for state in states)
        new_top = _mean(challenger[state]["pareto_hit"] for state in states)
        base_regret = _mean(baseline[state]["conflict_regret"] for state in states)
        new_regret = _mean(challenger[state]["conflict_regret"] for state in states)
        top_deltas.append(new_top - base_top)
        regret_improvements.append((base_regret - new_regret) / base_regret if base_regret else 0.0)
    return {
        "map_count": len(maps),
        "samples": samples,
        "top1_delta_95_ci": [_quantile(top_deltas, 0.025), _quantile(top_deltas, 0.975)],
        "conflict_regret_improvement_95_ci": [_quantile(regret_improvements, 0.025), _quantile(regret_improvements, 0.975)],
    }


def acceptance(baseline_summary: dict[str, Any], challenger_summary: dict[str, Any], baseline: dict[str, dict[str, Any]], challenger: dict[str, dict[str, Any]], config: dict[str, Any], *, map_count: int) -> dict[str, Any]:
    thresholds = config["thresholds"]
    top_delta = float(challenger_summary["pareto_top1_hit_rate"]) - float(baseline_summary["pareto_top1_hit_rate"])
    base_regret = float(baseline_summary["mean_conflict_regret"])
    new_regret = float(challenger_summary["mean_conflict_regret"])
    regret_improvement = (base_regret - new_regret) / base_regret if base_regret else 0.0
    bootstrap = _map_bootstrap(baseline, challenger, int(config["bootstrap_samples"]))
    maps_no_worse = 0
    for map_id in sorted({str(row["map_id"]) for row in baseline.values()}):
        states = [state for state, row in baseline.items() if str(row["map_id"]) == map_id]
        if _mean(challenger[state]["conflict_regret"] for state in states) <= _mean(baseline[state]["conflict_regret"] for state in states) + 1e-12:
            maps_no_worse += 1
    minimum_maps = int(thresholds["minimum_train_maps_no_worse" if map_count == 12 else "minimum_validation_maps_no_worse"])
    gates = {
        "top1_or_conflict_regret_improves": top_delta >= thresholds["minimum_top1_improvement"] or regret_improvement >= thresholds["minimum_conflict_regret_improvement"],
        "other_metric_not_degraded": float(challenger_summary["pareto_top1_hit_rate"]) >= float(baseline_summary["pareto_top1_hit_rate"]) - thresholds["maximum_top1_degradation"] and regret_improvement >= -thresholds["maximum_conflict_regret_degradation"],
        "map_bootstrap_not_degraded": bootstrap["top1_delta_95_ci"][0] >= 0.0 or bootstrap["conflict_regret_improvement_95_ci"][0] >= 0.0,
        "minimum_maps_no_worse": maps_no_worse >= minimum_maps,
        "no_size_collapse": challenger_summary["maximum_size_share"] <= thresholds["maximum_single_size_share"],
    }
    return {"passed": all(gates.values()), "gates": gates, "top1_delta": top_delta, "conflict_regret_improvement": regret_improvement, "maps_no_worse": maps_no_worse, "map_count": map_count, "bootstrap": bootstrap}


def _baseline_summary(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sizes = collections.Counter(int(row["selected_size"]) for row in records.values())
    return {
        "state_count": len(records),
        "pareto_top1_hit_rate": _mean(row["pareto_hit"] for row in records.values()),
        "mean_conflict_regret": _mean(row["conflict_regret"] for row in records.values()),
        "mean_solved_rate_regret": _mean(row["solved_rate_regret"] for row in records.values()),
        "pairwise_accuracy": None,
        "selected_size_counts": dict(sorted(sizes.items())),
        "maximum_size_share": max(sizes.values(), default=0) / max(1, len(records)),
    }


def _model_summary(
    records: dict[str, dict[str, Any]],
    examples: list[StateExample],
    scores_by_seed: list[dict[str, list[float]]],
) -> dict[str, Any]:
    summary = summarize_records(records, examples, scores_by_seed)
    seed_top1 = []
    seed_regret = []
    for scores in scores_by_seed:
        seed_records = selection_records(examples, [scores], "seed_diagnostic")
        seed_top1.append(_mean(row["pareto_hit"] for row in seed_records.values()))
        seed_regret.append(_mean(row["conflict_regret"] for row in seed_records.values()))
    summary["seed_top1_standard_deviation"] = statistics.pstdev(seed_top1)
    summary["seed_conflict_regret_standard_deviation"] = statistics.pstdev(seed_regret)
    return summary


def run_cross_validation(
    project_root: Path,
    index_root: Path,
    config: dict[str, Any],
    output_root: Path,
    *,
    run_fingerprint: str,
    progress: ProgressLog,
) -> dict[str, Any]:
    torch, _ = _import_torch()
    if str(torch.__version__) != str(config["torch_version"]) or not torch.cuda.is_available():
        raise ValueError("registered CUDA PyTorch environment is unavailable")
    examples = load_graph_examples(index_root)
    train = [row for row in examples if row.split == "policy_train"]
    anchors = [row for row in examples if row.split not in {"policy_train", "policy_validation"}]
    maps = sorted({row.map_id for row in train})
    if len(maps) != 12 or len(train) != 288 or len(anchors) != 23:
        raise ValueError("registered graph Train design changed")
    baseline = _baseline_records(project_root)
    expected = config["expected_current_lomo"]
    baseline_summary = _baseline_summary(baseline)
    if not math.isclose(baseline_summary["pareto_top1_hit_rate"], expected["pareto_top1_hit_rate"], rel_tol=0.0, abs_tol=expected["absolute_tolerance"]) or not math.isclose(baseline_summary["mean_conflict_regret"], expected["mean_conflict_regret"], rel_tol=0.0, abs_tol=expected["absolute_tolerance"]):
        raise ValueError("current GBDT baseline did not reproduce")
    records: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in MODEL_NAMES}
    all_scores: dict[str, dict[int, dict[str, list[float]]]] = {
        name: {int(seed): {} for seed in config["random_seeds"]}
        for name in MODEL_NAMES
    }
    fold_reports = []
    for fold_number, held_map in enumerate(maps):
        progress.emit("fold_started", fold=fold_number, held_map=held_map)
        outer_fit = anchors + [row for row in train if row.map_id != held_map]
        held = [row for row in train if row.map_id == held_map]
        fit_maps = sorted({row.map_id for row in outer_fit if row.split == "policy_train"})
        if any(row.map_id == held_map for row in outer_fit) or len(fit_maps) != 11:
            raise ValueError("outer LOMO map leakage")
        model_diagnostics = {}
        for model_name in MODEL_NAMES:
            seed_scores = []
            seed_diagnostics = []
            for seed_index, seed in enumerate(map(int, config["random_seeds"])):
                progress.emit(
                    "training_started",
                    fold=fold_number,
                    held_map=held_map,
                    model=model_name,
                    seed=seed,
                )
                inner_map = fit_maps[(fold_number + seed_index) % len(fit_maps)]
                inner_train = anchors + [row for row in outer_fit if row.split != "policy_train" or row.map_id != inner_map]
                inner_monitor = [row for row in outer_fit if row.split == "policy_train" and row.map_id == inner_map]
                _, _, early = train_model(model_name, inner_train, config, seed, epochs=int(config["training"]["maximum_epochs"]), monitor=inner_monitor)
                final_model, norms, _ = train_model(
                    model_name,
                    outer_fit,
                    config,
                    seed,
                    epochs=int(early["best_epoch"]),
                )
                current_scores = predict_scores(final_model, held, norms, config)
                seed_scores.append(current_scores)
                all_scores[model_name][seed].update(current_scores)
                seed_diagnostics.append(
                    {
                        "seed": seed,
                        "inner_map": inner_map,
                        "best_epoch": early["best_epoch"],
                        "early_epochs_run": early["epochs_run"],
                        "parameter_count": sum(
                            parameter.numel() for parameter in final_model.parameters()
                        ),
                        "checkpoint_retained": False,
                    }
                )
                progress.emit(
                    "training_completed",
                    fold=fold_number,
                    held_map=held_map,
                    model=model_name,
                    seed=seed,
                    best_epoch=early["best_epoch"],
                    early_epochs_run=early["epochs_run"],
                )
            selected = selection_records(held, seed_scores, model_name)
            if set(selected) & set(records[model_name]):
                raise ValueError("graph LOMO evaluated a state twice")
            records[model_name].update(selected)
            model_diagnostics[model_name] = seed_diagnostics
        fold_reports.append({"fold": fold_number, "validation_map": held_map, "validation_state_count": len(held), "anchor_state_count": len(anchors), "models": model_diagnostics})
        progress.emit("fold_completed", fold=fold_number, held_map=held_map)
    summaries = {
        model_name: _model_summary(
            records[model_name],
            train,
            [all_scores[model_name][seed] for seed in map(int, config["random_seeds"])],
        )
        for model_name in MODEL_NAMES
    }
    comparisons = {name: acceptance(baseline_summary, summaries[name], baseline, records[name], config, map_count=12) for name in MODEL_NAMES}
    eligible = [name for name in MODEL_NAMES if comparisons[name]["passed"]]
    winner = sorted(eligible, key=lambda name: (-summaries[name]["pareto_top1_hit_rate"], summaries[name]["mean_conflict_regret"], MODEL_NAMES.index(name)))[0] if eligible else None
    edge_increment = acceptance(summaries["agent_deepsets"], summaries["conflict_gnn"], records["agent_deepsets"], records["conflict_gnn"], config, map_count=12)
    for name in MODEL_NAMES:
        _write_jsonl(output_root / f"lomo_predictions__{name}.jsonl", [records[name][key] for key in sorted(records[name])])
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "baseline": baseline_summary,
        "summaries": summaries,
        "comparisons_to_gbdt": comparisons,
        "conflict_edges_increment": edge_increment,
        "folds": fold_reports,
        "eligible_models": eligible,
        "winner": winner,
        "passed": winner is not None,
        "validation_labels_used_for_selection": False,
        "lomo_checkpoints_retained": False,
    }
    _write_json(output_root / "cross_validation_report.json", report)
    return report


def run_validation(
    project_root: Path,
    index_root: Path,
    config: dict[str, Any],
    output_root: Path,
    winner: str,
    *,
    run_fingerprint: str,
    progress: ProgressLog,
) -> dict[str, Any]:
    if winner not in MODEL_NAMES:
        raise ValueError("Validation requires a preregistered Train LOMO winner")
    torch, _ = _import_torch()
    examples = load_graph_examples(index_root)
    train = [row for row in examples if row.split == "policy_train"]
    validation = [row for row in examples if row.split == "policy_validation"]
    anchors = [row for row in examples if row.split not in {"policy_train", "policy_validation"}]
    train_maps = sorted({row.map_id for row in train})
    validation_maps = sorted({row.map_id for row in validation})
    if (len(train), len(train_maps), len(validation), len(validation_maps), len(anchors)) != (
        288,
        12,
        154,
        6,
        23,
    ):
        raise ValueError("registered graph Validation design changed")
    if set(train_maps) & set(validation_maps):
        raise ValueError("graph Train and Validation maps overlap")
    baseline = _validation_baseline_records(project_root)
    if set(baseline) != {row.state_id for row in validation}:
        raise ValueError("graph Validation baseline states differ from graph index")

    seed_scores = []
    diagnostics = []
    trained_models = []
    for seed_index, seed in enumerate(map(int, config["random_seeds"])):
        progress.emit("validation_training_started", model=winner, seed=seed)
        inner_map = train_maps[seed_index % len(train_maps)]
        inner_fit = anchors + [row for row in train if row.map_id != inner_map]
        inner_monitor = [row for row in train if row.map_id == inner_map]
        _, _, early = train_model(
            winner,
            inner_fit,
            config,
            seed,
            epochs=int(config["training"]["maximum_epochs"]),
            monitor=inner_monitor,
        )
        final_model, norms, _ = train_model(
            winner,
            anchors + train,
            config,
            seed,
            epochs=int(early["best_epoch"]),
        )
        scores = predict_scores(final_model, validation, norms, config)
        seed_scores.append(scores)
        final_model.to("cpu")
        trained_models.append((seed, final_model, norms, early))
        diagnostics.append(
            {
                "seed": seed,
                "inner_map": inner_map,
                "best_epoch": early["best_epoch"],
                "early_epochs_run": early["epochs_run"],
                "parameter_count": sum(parameter.numel() for parameter in final_model.parameters()),
                "checkpoint_retained": False,
            }
        )
        progress.emit(
            "validation_training_completed",
            model=winner,
            seed=seed,
            best_epoch=early["best_epoch"],
            early_epochs_run=early["epochs_run"],
        )

    records = selection_records(validation, seed_scores, winner)
    summary = _model_summary(records, validation, seed_scores)
    baseline_summary = _baseline_summary(baseline)
    comparison = acceptance(
        baseline_summary,
        summary,
        baseline,
        records,
        config,
        map_count=6,
    )
    if comparison["passed"]:
        manifest = _read_json(index_root / "graph_index_manifest.json")
        model_root = output_root / "models/full"
        for diagnostic, (seed, model, norms, early) in zip(
            diagnostics, trained_models
        ):
            model_path = model_root / f"{winner}__seed-{seed}.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "normalization": {
                        name: {
                            "mean": value.mean.detach().cpu(),
                            "std": value.std.detach().cpu(),
                        }
                        for name, value in norms.items()
                    },
                    "model_name": winner,
                    "seed": seed,
                    "best_epoch": early["best_epoch"],
                    "train_state_count": len(train),
                    "anchor_state_count": len(anchors),
                    "node_feature_names": manifest["node_feature_names"],
                    "flat_feature_names": manifest["flat_feature_names"],
                    "metadata_feature_names": manifest["metadata_feature_names"],
                    "registered_inputs": manifest["registered_inputs"],
                    "training_configuration": config["training"],
                    "run_fingerprint": run_fingerprint,
                    "validation_labels_used_for_training": False,
                },
                model_path,
            )
            diagnostic["checkpoint_retained"] = True
            diagnostic["model_sha256"] = _sha256(model_path)
    _write_jsonl(
        output_root / f"validation_predictions__{winner}.jsonl",
        [records[key] for key in sorted(records)],
    )
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "winner_frozen_from_train_lomo": winner,
        "baseline": baseline_summary,
        "summary": summary,
        "comparison_to_gbdt": comparison,
        "passed": comparison["passed"],
        "train_state_count": len(train),
        "anchor_state_count": len(anchors),
        "validation_state_count": len(validation),
        "validation_map_count": len(validation_maps),
        "validation_labels_used_for_training": False,
        "validation_labels_used_for_model_selection": False,
        "models": diagnostics,
    }
    _write_json(output_root / "validation_report.json", report)
    return report


def verify_audit_equivalence(
    reference_root: Path, current_root: Path, output_path: Path
) -> dict[str, Any]:
    reference_final = _read_json(reference_root / "graph_representation_audit.json")
    current_final = _read_json(current_root / "graph_representation_audit.json")
    reference_cv = _read_json(reference_root / "cross_validation_report.json")
    current_cv = _read_json(current_root / "cross_validation_report.json")
    prediction_checks = {}
    for model_name in MODEL_NAMES:
        reference = {
            str(row["state_id"]): row
            for row in _read_jsonl(
                reference_root / f"lomo_predictions__{model_name}.jsonl"
            )
        }
        current = {
            str(row["state_id"]): row
            for row in _read_jsonl(
                current_root / f"lomo_predictions__{model_name}.jsonl"
            )
        }
        same_states = set(reference) == set(current)
        candidate_mismatches = (
            sum(
                str(reference[state_id]["candidate_id"])
                != str(current[state_id]["candidate_id"])
                for state_id in reference
            )
            if same_states
            else max(len(reference), len(current))
        )
        prediction_checks[model_name] = {
            "same_states": same_states,
            "state_count": len(current),
            "candidate_mismatch_count": candidate_mismatches,
            "passed": same_states and candidate_mismatches == 0,
        }
    summary_checks = {}
    for model_name in MODEL_NAMES:
        summary_checks[model_name] = {}
        for metric in (
            "pareto_top1_hit_rate",
            "mean_conflict_regret",
            "mean_solved_rate_regret",
            "pairwise_accuracy",
            "maximum_size_share",
        ):
            reference_value = float(reference_cv["summaries"][model_name][metric])
            current_value = float(current_cv["summaries"][model_name][metric])
            summary_checks[model_name][metric] = {
                "reference": reference_value,
                "current": current_value,
                "absolute_delta": abs(current_value - reference_value),
                "passed": math.isclose(
                    current_value, reference_value, rel_tol=0.0, abs_tol=1e-12
                ),
            }
    decision_matches = (
        str(reference_final["decision"]) == str(current_final["decision"])
        and reference_cv.get("winner") == current_cv.get("winner")
        and bool(reference_cv["passed"]) == bool(current_cv["passed"])
    )
    passed = (
        all(row["passed"] for row in prediction_checks.values())
        and all(
            metric["passed"]
            for model in summary_checks.values()
            for metric in model.values()
        )
        and decision_matches
    )
    report = {
        "schema": "lns2.graph_representation_equivalence.v1",
        "reference_root": str(reference_root),
        "current_root": str(current_root),
        "prediction_checks": prediction_checks,
        "summary_checks": summary_checks,
        "decision_matches": decision_matches,
        "passed": passed,
    }
    _write_json(output_path, report)
    if not passed:
        raise ValueError("hardened graph audit did not reproduce the registered result")
    return report


def run_graph_representation_audit(
    project_root: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
    equivalence_reference: str | Path | None = None,
) -> dict[str, Any]:
    if phase not in {"index", "cross_validate", "validate", "all"}:
        raise ValueError("phase must be index, cross_validate, validate, or all")
    root = Path(project_root).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    _validate_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    implementation_sha256 = _sha256(Path(__file__))
    run_fingerprint = _fingerprint(
        {
            "configuration": config,
            "implementation_sha256": implementation_sha256,
        }
    )
    run_config = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "configuration": config,
        "implementation_sha256": implementation_sha256,
        "run_fingerprint": run_fingerprint,
    }
    run_config_path = output_root / "run_config.json"
    if run_config_path.exists():
        existing = _read_json(run_config_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("graph audit output belongs to a different run fingerprint")
        if phase != "validate" and (output_root / "completed.json").exists():
            raise ValueError("graph audit output is already complete")
    else:
        _write_json(run_config_path, run_config)
    progress = ProgressLog(
        output_root / "progress.jsonl", reset=phase != "validate"
    )
    started = time.perf_counter()
    _write_json(
        output_root / "run_status.json",
        {
            "status": "running",
            "phase": phase,
            "pid": os.getpid(),
            "run_fingerprint": run_fingerprint,
        },
    )
    (output_root / "completed.json").unlink(missing_ok=True)
    try:
        progress.emit("index_started")
        manifest = build_graph_indexes(root, config, output_root / "index")
        progress.emit(
            "index_completed",
            states=manifest["state_count"],
            candidates=manifest["candidate_count"],
        )
        if phase == "index":
            final = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": run_fingerprint,
                "index": manifest,
                "phase": phase,
                "elapsed_seconds": time.perf_counter() - started,
            }
        else:
            if phase == "validate":
                cross_validation = _read_json(
                    output_root / "cross_validation_report.json"
                )
                if str(cross_validation.get("run_fingerprint")) != run_fingerprint:
                    raise ValueError(
                        "cross-validation report belongs to a different run fingerprint"
                    )
            else:
                cross_validation = run_cross_validation(
                    root,
                    output_root / "index",
                    config,
                    output_root,
                    run_fingerprint=run_fingerprint,
                    progress=progress,
                )
            validation = None
            if cross_validation["winner"] is not None and phase in {"validate", "all"}:
                validation = run_validation(
                    root,
                    output_root / "index",
                    config,
                    output_root,
                    str(cross_validation["winner"]),
                    run_fingerprint=run_fingerprint,
                    progress=progress,
                )
            decision = (
                "eligible_for_independent_graph_confirmation"
                if validation is not None
                and validation["passed"]
                and cross_validation["winner"] == "conflict_gnn"
                and cross_validation["conflict_edges_increment"]["passed"]
                else "eligible_for_independent_agent_set_confirmation"
                if validation is not None
                and validation["passed"]
                and cross_validation["winner"] in {"agent_deepsets", "conflict_gnn"}
                else "eligible_for_independent_neural_scorer_confirmation"
                if validation is not None
                and validation["passed"]
                and cross_validation["winner"] == "flat_mlp"
                else "stop_after_development_validation"
                if validation is not None
                and not validation["passed"]
                else "train_lomo_winner_awaits_validation"
                if cross_validation["winner"] is not None
                else "stop_supervised_representation_expansion"
            )
            final = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": run_fingerprint,
                "index": manifest,
                "cross_validation": cross_validation,
                "validation": validation,
                "decision": decision,
                "validation_evaluated": validation is not None,
                "new_solver_data_collected": False,
                "static_context_used": False,
                "rl_trained": False,
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(output_root / "graph_representation_audit.json", final)
            if equivalence_reference is not None:
                final["equivalence"] = verify_audit_equivalence(
                    Path(equivalence_reference).resolve(),
                    output_root,
                    output_root / "equivalence_report.json",
                )
                _write_json(output_root / "graph_representation_audit.json", final)
        progress.emit("run_completed", phase=phase)
        completion = {
            "status": "completed",
            "phase": phase,
            "run_fingerprint": run_fingerprint,
            "decision": final.get("decision"),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(output_root / "run_status.json", completion)
        _write_json(output_root / "completed.json", completion)
        return final
    except BaseException as error:
        progress.emit("run_failed", phase=phase, error=repr(error))
        _write_json(
            output_root / "run_status.json",
            {
                "status": "failed",
                "phase": phase,
                "run_fingerprint": run_fingerprint,
                "error": repr(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


__all__ = [
    "MODEL_NAMES",
    "NODE_FEATURE_NAMES",
    "acceptance",
    "build_graph_indexes",
    "collate_examples",
    "dominance_pairs",
    "load_graph_examples",
    "make_model",
    "run_graph_representation_audit",
    "validate_registered_inputs",
]
