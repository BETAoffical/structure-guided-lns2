from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import pickle
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.context_audit import _pair_vector
from experiments.local_representation_audit import analyze_state
from experiments.natural_distribution_confirmation import conflict_density, conflict_severity
from experiments.realized_neighborhood_probe import select_representative_neighborhoods
from experiments.realized_neighborhood_ranking_audit import _feature_profiles
from experiments.realized_ranking_confirmation import _seed_isolation
from experiments.repair_collection import (
    SCHEMA_VERSION,
    CollectionLockError,
    _CollectionRunLock,
    _dataset_fingerprint,
    _fingerprint,
    _load_dataset_rows,
    _low_level_delta,
    _make_environment,
    _plain,
    _qualification_worker,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    select_seed_agents,
    state_fingerprint,
)


CLOSED_LOOP_SCHEMA = "lns2.closed_loop_confirmation.v1"
EPISODE_SCHEMA = "lns2.closed_loop_episode.v1"
POLICIES = ("official_adaptive", "proposal_dynamic", "realized_dynamic")
LEARNED_POLICIES = ("proposal_dynamic", "realized_dynamic")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number_summary(values: Iterable[float | int]) -> dict[str, Any]:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(numbers),
        "min": numbers[0],
        "median": statistics.median(numbers),
        "mean": statistics.fmean(numbers),
        "max": numbers[-1],
    }


def closed_loop_dataset_design(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    expected_tasks = {
        "balanced_80",
        "balanced_100",
        "bottleneck_80",
        "bottleneck_100",
    }
    errors: list[str] = []
    if any(str(row.get("split")) != split for row in rows):
        errors.append("dataset contains a non-closed-loop split")
    by_map: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_map[str(row["map_id"])].append(row)
    layout_counts: collections.Counter[str] = collections.Counter()
    for map_id, tasks in sorted(by_map.items()):
        layouts = {str(row.get("layout_mode")) for row in tasks}
        if len(layouts) != 1:
            errors.append(f"{map_id}: inconsistent layout")
            continue
        layout_counts[next(iter(layouts))] += 1
        if {str(row.get("task_variant")) for row in tasks} != expected_tasks or len(tasks) != 4:
            errors.append(f"{map_id}: incomplete four-task pairing")
        if len({int(row["map_seed"]) for row in tasks}) != 1:
            errors.append(f"{map_id}: inconsistent map seed")
        if len({int(row["task_seed"]) for row in tasks}) != 4:
            errors.append(f"{map_id}: repeated task seed")
    expected_layouts = {
        "regular_beltway": 2,
        "compartmentalized": 2,
        "dead_end_aisles": 2,
    }
    if dict(sorted(layout_counts.items())) != expected_layouts:
        errors.append("layout replication is not the registered 2/2/2 design")
    if len(rows) != 24 or len(by_map) != 6:
        errors.append("dataset is not the registered 6-map/24-task design")
    return {
        "passed": not errors,
        "errors": errors,
        "map_count": len(by_map),
        "task_count": len(rows),
        "layout_counts": dict(sorted(layout_counts.items())),
    }


def closed_loop_qualification_report(
    rows: list[dict[str, Any]],
    qualification: list[dict[str, Any]],
    config: dict[str, Any],
    design: dict[str, Any],
    isolation: dict[str, Any],
    *,
    formal: bool,
) -> dict[str, Any]:
    indexed = {str(row["task_id"]): row for row in qualification}
    errors = [
        str(row.get("error"))
        for row in qualification
        if str(row.get("status")) != "ok"
    ]
    cohort = []
    thresholds = dict(config["severity_thresholds"])
    for source in rows:
        result = indexed.get(str(source["task_id"]))
        if result is None or str(result.get("status")) != "ok":
            continue
        conflicts = int(result["initial_conflicts"])
        agents = int(source["agent_count"])
        density = conflict_density(conflicts, agents)
        cohort.append(
            {
                "map_id": str(source["map_id"]),
                "task_id": str(source["task_id"]),
                "layout_mode": str(source["layout_mode"]),
                "task_variant": str(source["task_variant"]),
                "agent_count": agents,
                "initial_conflicts": conflicts,
                "initial_feasible": conflicts == 0,
                "conflict_density": density,
                "conflict_severity": conflict_severity(density, thresholds),
                "state_fingerprint": str(result["state_fingerprint"]),
            }
        )
    nonzero = [row for row in cohort if int(row["initial_conflicts"]) > 0]
    by_layout = collections.Counter(str(row["layout_mode"]) for row in nonzero)
    active_maps = sorted({str(row["map_id"]) for row in nonzero})
    settings = dict(config["qualification"])
    sample_gates = (
        {
            "minimum_nonzero_states": len(nonzero) >= int(settings["minimum_nonzero_states"]),
            "minimum_nonzero_per_layout": all(
                by_layout.get(layout, 0) >= int(settings["minimum_nonzero_states_per_layout"])
                for layout in ("regular_beltway", "compartmentalized", "dead_end_aisles")
            ),
            "minimum_active_maps": len(active_maps) >= int(settings["minimum_active_maps"]),
        }
        if formal
        else {
            "minimum_nonzero_states": bool(nonzero),
            "minimum_nonzero_per_layout": True,
            "minimum_active_maps": True,
        }
    )
    gates = {
        "dataset_design": bool(design["passed"]) if formal else True,
        "seed_isolation": bool(isolation["passed"]),
        "all_resets_valid": len(cohort) == len(rows) and not errors,
        **sample_gates,
    }
    grouped = {}
    for field in ("layout_mode", "task_variant", "agent_count", "conflict_severity"):
        groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in cohort:
            groups[str(row[field])].append(row)
        grouped[field] = {
            name: {
                "task_count": len(group),
                "initial_feasible_count": sum(int(item["initial_conflicts"]) == 0 for item in group),
                "conflicts": _number_summary(item["initial_conflicts"] for item in group),
                "conflict_density": _number_summary(item["conflict_density"] for item in group),
            }
            for name, group in sorted(groups.items())
        }
    passed = all(gates.values())
    return {
        "schema": CLOSED_LOOP_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal": formal,
        "passed": passed,
        "decision": "eligible_for_closed_loop" if passed else "inconclusive_do_not_resample",
        "gates": gates,
        "valid_count": len(cohort),
        "initial_feasible_count": len(cohort) - len(nonzero),
        "nonzero_state_count": len(nonzero),
        "nonzero_by_layout": dict(sorted(by_layout.items())),
        "active_map_count": len(active_maps),
        "active_maps": active_maps,
        "repairable_task_ids": sorted(str(row["task_id"]) for row in nonzero),
        "zero_conflict_task_ids": sorted(
            str(row["task_id"]) for row in cohort if int(row["initial_conflicts"]) == 0
        ),
        "severity_thresholds": thresholds,
        "natural_distribution": {
            "conflicts": _number_summary(row["initial_conflicts"] for row in cohort),
            "conflict_density": _number_summary(row["conflict_density"] for row in cohort),
            "severity_counts": dict(
                sorted(collections.Counter(str(row["conflict_severity"]) for row in cohort).items())
            ),
            "grouped": grouped,
            "tasks": sorted(cohort, key=lambda row: str(row["task_id"])),
        },
        "errors": errors,
        "dataset_design": design,
        "seed_isolation": isolation,
    }


@dataclass
class FrozenPolicyBundle:
    models: dict[str, Any]
    ranges: dict[str, dict[str, tuple[float, float]]]
    manifest: dict[str, Any]


@dataclass
class PortablePairwiseModel:
    profile: str
    feature_names: list[str]
    baseline: float
    trees: list[list[dict[str, Any]]]

    def predict_positive(self, vectors: list[list[float]]) -> list[float]:
        probabilities = []
        for vector in vectors:
            raw = self.baseline
            for nodes in self.trees:
                index = 0
                while not bool(nodes[index]["is_leaf"]):
                    node = nodes[index]
                    value = float(vector[int(node["feature_idx"])])
                    go_left = (
                        math.isnan(value) and bool(node["missing_go_to_left"])
                    ) or (
                        not math.isnan(value)
                        and value <= float(node["num_threshold"])
                    )
                    index = int(node["left"] if go_left else node["right"])
                raw += float(nodes[index]["value"])
            if raw >= 0.0:
                probabilities.append(1.0 / (1.0 + math.exp(-raw)))
            else:
                exponential = math.exp(raw)
                probabilities.append(exponential / (1.0 + exponential))
        return probabilities


def export_portable_policy_bundle(
    frozen_root: str | Path,
    registration: dict[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    root = Path(frozen_root).resolve()
    freeze_manifest = _read_json(root / "freeze_manifest.json")
    if bool(freeze_manifest.get("confirmation_labels_seen", True)):
        raise ValueError("frozen model manifest has seen confirmation labels")
    model_rows = {str(row["profile"]): row for row in freeze_manifest["models"]}
    expected = {
        str(name): str(value).lower()
        for name, value in dict(registration["model_sha256"]).items()
    }
    output_root = Path(output).resolve()
    exported = []
    for profile in LEARNED_POLICIES:
        source = root / str(model_rows[profile]["model_file"])
        source_sha = _sha256(source)
        if source_sha != expected[profile]:
            raise ValueError(f"frozen model SHA256 mismatch: {profile}")
        with source.open("rb") as stream:
            model = pickle.load(stream)
        estimator = model.estimator
        if list(map(int, estimator.classes_)) != [0, 1]:
            raise ValueError("portable exporter supports only binary pairwise models")
        trees = []
        for stage in estimator._predictors:
            if len(stage) != 1:
                raise ValueError("portable exporter supports one binary tree per stage")
            predictor = stage[0]
            nodes = []
            for node in predictor.nodes:
                if bool(node["is_categorical"]):
                    raise ValueError("portable exporter does not support categorical tree nodes")
                nodes.append(
                    {
                        "value": float(node["value"]),
                        "feature_idx": int(node["feature_idx"]),
                        "num_threshold": float(node["num_threshold"]),
                        "missing_go_to_left": bool(node["missing_go_to_left"]),
                        "left": int(node["left"]),
                        "right": int(node["right"]),
                        "is_leaf": bool(node["is_leaf"]),
                    }
                )
            trees.append(nodes)
        payload = {
            "schema": "lns2.portable_pairwise_hist_gbdt.v1",
            "schema_version": 1,
            "profile": profile,
            "source_model_sha256": source_sha,
            "feature_names": list(model.feature_names),
            "baseline": float(estimator._baseline_prediction[0, 0]),
            "trees": trees,
        }
        path = output_root / f"pairwise__{profile}.json"
        _write_json(path, payload)
        exported.append(
            {
                "profile": profile,
                "file": path.relative_to(output_root).as_posix(),
                "sha256": _sha256(path),
                "source_model_sha256": source_sha,
                "feature_count": len(model.feature_names),
                "tree_count": len(trees),
            }
        )
    manifest = {
        "schema": "lns2.portable_pairwise_bundle.v1",
        "schema_version": 1,
        "models": exported,
        "confirmation_labels_seen": False,
    }
    _write_json(output_root / "portable_manifest.json", manifest)
    return manifest


def _load_portable_models(
    portable_root: Path,
    expected_portable: dict[str, str],
    expected_source: dict[str, str],
) -> dict[str, PortablePairwiseModel]:
    manifest = _read_json(portable_root / "portable_manifest.json")
    if bool(manifest.get("confirmation_labels_seen", True)):
        raise ValueError("portable model manifest has seen confirmation labels")
    rows = {str(row["profile"]): row for row in manifest["models"]}
    models = {}
    for profile in LEARNED_POLICIES:
        row = rows[profile]
        path = portable_root / str(row["file"])
        digest = _sha256(path)
        if digest != expected_portable[profile] or digest != str(row["sha256"]):
            raise ValueError(f"portable model SHA256 mismatch: {profile}")
        payload = _read_json(path)
        if (
            str(payload["profile"]) != profile
            or str(payload["source_model_sha256"]) != expected_source[profile]
        ):
            raise ValueError(f"portable model provenance mismatch: {profile}")
        models[profile] = PortablePairwiseModel(
            profile=profile,
            feature_names=list(map(str, payload["feature_names"])),
            baseline=float(payload["baseline"]),
            trees=list(payload["trees"]),
        )
    return models


def verify_portable_policy_bundle(
    frozen_root: str | Path, registration: dict[str, Any]
) -> dict[str, Any]:
    native_registration = {
        key: value
        for key, value in registration.items()
        if key not in {"portable_models", "portable_model_sha256"}
    }
    native = load_frozen_policy_bundle(frozen_root, native_registration)
    portable = load_frozen_policy_bundle(frozen_root, registration)
    index_path = Path(str(registration["development_index"]))
    if not index_path.is_absolute():
        index_path = Path(__file__).resolve().parents[1] / index_path
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in _read_jsonl(index_path.resolve()):
        grouped[str(row["state_id"])].append(row)
    profiles = {}
    for profile in LEARNED_POLICIES:
        mismatches = 0
        maximum_score_delta = 0.0
        pair_count = 0
        for candidates in grouped.values():
            native_index, native_scores, _ = score_online_candidates(
                candidates, native.models[profile]
            )
            portable_index, portable_scores, _ = score_online_candidates(
                candidates, portable.models[profile]
            )
            mismatches += native_index != portable_index
            maximum_score_delta = max(
                maximum_score_delta,
                max(
                    abs(first - second)
                    for first, second in zip(native_scores, portable_scores)
                ),
            )
            pair_count += len(candidates) * (len(candidates) - 1) // 2
        profiles[profile] = {
            "state_count": len(grouped),
            "pair_count": pair_count,
            "selection_mismatch_count": mismatches,
            "maximum_score_delta": maximum_score_delta,
            "passed": mismatches == 0 and maximum_score_delta <= 1e-12,
        }
    feature_parity = None
    source_value = registration.get("development_candidates")
    if source_value:
        source_path = Path(str(source_value))
        if not source_path.is_absolute():
            source_path = Path(__file__).resolve().parents[1] / source_path
        source_path = source_path.resolve()
        expected_source_sha = str(registration["development_candidates_sha256"]).lower()
        if _sha256(source_path) != expected_source_sha:
            raise ValueError("development candidate source SHA256 mismatch")
        indexed = {
            (str(row["state_id"]), str(row["candidate_id"])): row
            for row in _read_jsonl(index_path.resolve())
        }
        checked = 0
        mismatches = []
        seen = set()
        for source in _read_jsonl(source_path):
            state_id = str(source["state_id"])
            for computed in online_candidate_rows(source["state"], source["candidates"]):
                key = (state_id, str(computed["candidate_id"]))
                checked += 1
                seen.add(key)
                if key not in indexed or computed["features"] != indexed[key]["features"]:
                    mismatches.append(list(key))
        missing = sorted([list(key) for key in set(indexed) - seen])
        feature_parity = {
            "candidate_count": checked,
            "mismatch_count": len(mismatches),
            "missing_count": len(missing),
            "mismatches": mismatches,
            "missing": missing,
            "passed": not mismatches and not missing and checked == len(indexed),
        }
    passed = all(row["passed"] for row in profiles.values()) and (
        feature_parity is None or bool(feature_parity["passed"])
    )
    return {
        "schema": "lns2.portable_pairwise_equivalence.v1",
        "schema_version": 1,
        "passed": passed,
        "profiles": profiles,
        "online_feature_parity": feature_parity,
    }


def load_frozen_policy_bundle(
    frozen_root: str | Path, registration: dict[str, Any]
) -> FrozenPolicyBundle:
    root = Path(frozen_root).resolve()
    manifest = _read_json(root / "freeze_manifest.json")
    if bool(manifest.get("confirmation_labels_seen", True)):
        raise ValueError("frozen model manifest has seen confirmation labels")
    registered_index = registration.get("development_index")
    if registered_index:
        index_path = Path(str(registered_index))
        if not index_path.is_absolute():
            index_path = Path(__file__).resolve().parents[1] / index_path
        index_path = index_path.resolve()
    else:
        index_path = Path(str(manifest["development_index"])).resolve()
    expected_index = str(registration["development_index_sha256"]).lower()
    if _sha256(index_path) != expected_index:
        raise ValueError("development ranking index SHA256 mismatch")
    model_rows = {str(row["profile"]): row for row in manifest.get("models", [])}
    expected_models = {
        str(name): str(value).lower()
        for name, value in dict(registration["model_sha256"]).items()
    }
    if not set(LEARNED_POLICIES).issubset(model_rows):
        raise ValueError("frozen model set is incomplete")
    models: dict[str, Any] = {}
    for profile in LEARNED_POLICIES:
        row = model_rows[profile]
        path = root / str(row["model_file"])
        digest = _sha256(path)
        if digest != expected_models[profile] or digest != str(row["model_sha256"]).lower():
            raise ValueError(f"frozen model SHA256 mismatch: {profile}")
    portable_value = registration.get("portable_models")
    if portable_value:
        portable_root = Path(str(portable_value))
        if not portable_root.is_absolute():
            portable_root = Path(__file__).resolve().parents[1] / portable_root
        expected_portable = {
            str(name): str(value).lower()
            for name, value in dict(registration["portable_model_sha256"]).items()
        }
        models = _load_portable_models(
            portable_root.resolve(), expected_portable, expected_models
        )
    else:
        for profile in LEARNED_POLICIES:
            row = model_rows[profile]
            path = root / str(row["model_file"])
            with path.open("rb") as stream:
                model = pickle.load(stream)
            if str(model.profile) != profile:
                raise ValueError(f"frozen model profile mismatch: {profile}")
            models[profile] = model
    development_rows = _read_jsonl(index_path)
    ranges: dict[str, dict[str, tuple[float, float]]] = {}
    for profile, model in models.items():
        values: dict[str, list[float]] = {name: [] for name in model.feature_names}
        for row in development_rows:
            features = dict(row["features"][profile])
            for name in model.feature_names:
                values[name].append(float(features.get(name, 0.0)))
        ranges[profile] = {
            name: (min(numbers), max(numbers)) for name, numbers in values.items()
        }
    return FrozenPolicyBundle(models=models, ranges=ranges, manifest=manifest)


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
    return int(
        _fingerprint(
            {
                "namespace": "closed-loop-proposal-v1",
                "task_id": task_id,
                "solver_seed": solver_seed,
                "state_fingerprint": state_hash,
                "decision_index": decision_index,
                "seed_agent": seed_agent,
                "heuristic": heuristic,
                "size": size,
                "trial_index": trial_index,
            }
        )[:16],
        16,
    ) % (2**31)


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


def online_candidate_rows(
    state: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    analysis = analyze_state(state)
    rows = []
    state_hash = state_fingerprint(state)
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        rows.append(
            {
                "state_id": state_hash,
                "candidate_id": candidate_id,
                "candidate_key": candidate_id,
                "features": _feature_profiles(state, analysis, candidate),
            }
        )
    return rows


def score_online_candidates(
    rows: list[dict[str, Any]], model: Any
) -> tuple[int, list[float], float]:
    if not rows:
        raise ValueError("cannot score an empty candidate pool")
    scores = [0.0] * len(rows)
    vectors = []
    reverse_vectors = []
    pairs = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            vectors.append(
                _pair_vector(rows[left], rows[right], model.profile, model.feature_names)
            )
            reverse_vectors.append(
                _pair_vector(rows[right], rows[left], model.profile, model.feature_names)
            )
            pairs.append((left, right))
    if vectors:
        if isinstance(model, PortablePairwiseModel):
            forward = model.predict_positive(vectors)
            reverse = model.predict_positive(reverse_vectors)
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
    order = sorted(
        range(len(rows)),
        key=lambda index: (-scores[index], str(rows[index]["candidate_key"])),
    )
    margin = scores[order[0]] - scores[order[1]] if len(order) > 1 else scores[order[0]]
    return order[0], scores, margin


def feature_range_diagnostic(
    row: dict[str, Any], profile: str, ranges: dict[str, tuple[float, float]]
) -> dict[str, Any]:
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state_hash = state_fingerprint(state)
    proposals = []
    proposal_seconds = 0.0
    for seed_agent in select_seed_agents(state, int(proposal_config["max_seed_agents"])):
        for heuristic in map(str, proposal_config["heuristics"]):
            for size in map(int, proposal_config["neighborhood_sizes"]):
                for trial_index in range(int(proposal_config["trials"])):
                    random_seed = proposal_random_seed(
                        task_id,
                        solver_seed,
                        state_hash,
                        decision_index,
                        seed_agent,
                        heuristic,
                        size,
                        trial_index,
                    )
                    started = time.perf_counter()
                    result = _plain(
                        environment.propose(
                            {
                                "mode": "seed",
                                "heuristic": heuristic,
                                "seed_agent": seed_agent,
                                "neighborhood_size": size,
                                "random_seed": random_seed,
                            }
                        )
                    )
                    proposal_seconds += time.perf_counter() - started
                    after = _plain(environment.get_state())
                    if after != state or state_fingerprint(after) != state_hash:
                        raise RuntimeError("proposal changed the closed-loop repair state")
                    if not bool(result.get("action_valid")) or not bool(result.get("generated")):
                        raise RuntimeError("valid online proposal was rejected")
                    agents = sorted(map(int, result.get("neighborhood", [])))
                    if not agents or len(agents) != len(set(agents)):
                        raise RuntimeError("online proposal returned an invalid neighborhood")
                    proposals.append(
                        {
                            "family": f"{heuristic}:{size}",
                            "seed_agent": seed_agent,
                            "proposal_seed": random_seed,
                            "requested_size": size,
                            "agents": agents,
                        }
                    )
    candidates = select_representative_neighborhoods(
        proposals, int(proposal_config["candidates_per_family"])
    )
    if not candidates:
        raise RuntimeError("online proposal stage produced no explicit candidates")
    return candidates, {
        "proposal_count": len(proposals),
        "unique_neighborhood_count": len({tuple(row["agents"]) for row in proposals}),
        "candidate_count": len(candidates),
        "proposal_seconds": proposal_seconds,
    }


def fixed_budget_conflict_auc(
    trajectory: list[int], budget: int, *, success: bool
) -> float:
    if budget <= 0 or not trajectory or len(trajectory) > budget + 1:
        raise ValueError("invalid fixed-budget conflict trajectory")
    values = list(map(int, trajectory))
    pad = 0 if success else values[-1]
    values.extend([pad] * (budget + 1 - len(values)))
    return sum((values[index] + values[index + 1]) / 2.0 for index in range(budget))


def _episode_id(row: dict[str, Any], solver_seed: int, policy: str) -> str:
    return f"{row['task_id']}__seed_{solver_seed:04d}__{policy}"


def _valid_episode_trace(path: Path, run_fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        rows = _read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not rows
        or rows[-1].get("event") != "finish"
        or rows[-1].get("run_fingerprint") != run_fingerprint
    ):
        return None
    return rows[-1].get("summary")


def _emit(stream: Any, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


def _closed_loop_episode_worker(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    policy = str(job["policy"])
    solver_seed = int(job["solver_seed"])
    episode_id = _episode_id(row, solver_seed, policy)
    output_root = Path(job["output_root"])
    trace_path = output_root / "episodes" / str(row["split"]) / policy / f"{episode_id}.jsonl"
    partial_path = trace_path.with_suffix(".partial.jsonl")
    relative_trace = trace_path.relative_to(output_root).as_posix()
    if job["resume"]:
        summary = _valid_episode_trace(trace_path, job["run_fingerprint"])
        if summary is not None:
            return {
                "schema_version": SCHEMA_VERSION,
                "schema": CLOSED_LOOP_SCHEMA,
                "episode_id": episode_id,
                "split": row["split"],
                "map_id": row["map_id"],
                "task_id": row["task_id"],
                "layout_mode": row["layout_mode"],
                "task_variant": row.get("task_variant"),
                "agent_count": int(row["agent_count"]),
                "solver_seed": solver_seed,
                "policy": policy,
                "trace_file": relative_trace,
                "status": "resumed",
                "summary": summary,
                "error": None,
            }
    bundle = None
    if policy in LEARNED_POLICIES:
        bundle = load_frozen_policy_bundle(job["frozen_models"], job["model_registration"])
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.unlink(missing_ok=True)
    started_wall = time.perf_counter()
    try:
        environment = _make_environment(
            job["dataset_root"], row, job["environment"], "Adaptive"
        )
        with partial_path.open("w", encoding="utf-8", newline="\n") as stream:
            state = _plain(environment.reset(seed=solver_seed))
            initial_fingerprint = state_fingerprint(state)
            conflicts = [int(state["num_of_colliding_pairs"])]
            initial_event = {
                "schema": EPISODE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": job["run_fingerprint"],
                "event": "initial",
                "episode_id": episode_id,
                "policy": policy,
                "solver_seed": solver_seed,
                "state_fingerprint": initial_fingerprint,
                "state": state,
            }
            _emit(stream, initial_event)
            controller_totals = collections.Counter()
            selected_sizes: collections.Counter[int] = collections.Counter()
            selected_families: collections.Counter[str] = collections.Counter()
            invalid_actions = 0
            fingerprint_mismatches = 0
            external_timeout = False
            max_decisions = int(job["max_decisions"])
            wall_budget = float(job["wall_time_budget_seconds"])
            while not bool(state["done"]) and len(conflicts) - 1 < max_decisions:
                if time.perf_counter() - started_wall >= wall_budget:
                    external_timeout = True
                    break
                before = state
                before_hash = state_fingerprint(before)
                decision_index = len(conflicts) - 1
                controller: dict[str, Any] = {}
                if policy == "official_adaptive":
                    action = {"mode": "official"}
                else:
                    proposal_started = time.perf_counter()
                    candidates, proposal_metrics = generate_online_candidates(
                        environment,
                        state,
                        task_id=str(row["task_id"]),
                        solver_seed=solver_seed,
                        decision_index=decision_index,
                        proposal_config=job["proposal"],
                    )
                    controller["proposal"] = proposal_metrics
                    feature_started = time.perf_counter()
                    candidate_rows = online_candidate_rows(state, candidates)
                    feature_seconds = time.perf_counter() - feature_started
                    inference_started = time.perf_counter()
                    selected_index, scores, margin = score_online_candidates(
                        candidate_rows, bundle.models[policy]
                    )
                    inference_seconds = time.perf_counter() - inference_started
                    selected = candidates[selected_index]
                    selected_row = candidate_rows[selected_index]
                    random_seed = repair_random_seed(
                        str(row["task_id"]),
                        solver_seed,
                        before_hash,
                        decision_index,
                        str(selected["candidate_id"]),
                        selected["proposal_seeds"],
                    )
                    action = {
                        "mode": "explicit_neighborhood",
                        "agents": selected["agents"],
                        "random_seed": random_seed,
                    }
                    diagnostic = feature_range_diagnostic(
                        selected_row, policy, bundle.ranges[policy]
                    )
                    controller.update(
                        {
                            "candidate_pool": [
                                {
                                    **candidate,
                                    "score": scores[index],
                                    "feature_out_of_range_fraction": feature_range_diagnostic(
                                        candidate_rows[index], policy, bundle.ranges[policy]
                                    )["outside_fraction"],
                                }
                                for index, candidate in enumerate(candidates)
                            ],
                            "selected_candidate_id": selected["candidate_id"],
                            "selected_score": scores[selected_index],
                            "score_margin": margin,
                            "selected_feature_range": diagnostic,
                            "feature_seconds": feature_seconds,
                            "inference_seconds": inference_seconds,
                            "controller_seconds_before_repair": time.perf_counter()
                            - proposal_started,
                        }
                    )
                    selected_sizes[len(selected["agents"])] += 1
                    for family in selected["selection_families"]:
                        selected_families[str(family)] += 1
                    controller_totals["proposal_count"] += int(proposal_metrics["proposal_count"])
                    controller_totals["candidate_count"] += int(proposal_metrics["candidate_count"])
                    controller_totals["proposal_seconds"] += float(proposal_metrics["proposal_seconds"])
                    controller_totals["feature_seconds"] += feature_seconds
                    controller_totals["inference_seconds"] += inference_seconds
                    controller_totals["selected_feature_outside_fraction_sum"] += float(
                        diagnostic["outside_fraction"]
                    )
                    controller_totals["learned_decisions"] += 1
                repair_started = time.perf_counter()
                result = _plain(environment.step(action))
                repair_wall_seconds = time.perf_counter() - repair_started
                state = result["observation"]
                metrics = result["metrics"]
                if policy in LEARNED_POLICIES:
                    actual = sorted(map(int, metrics.get("neighborhood", [])))
                    if not bool(metrics.get("action_valid")) or actual != sorted(action["agents"]):
                        invalid_actions += 1
                        raise RuntimeError("explicit closed-loop action was rejected or changed")
                conflicts.append(int(state["num_of_colliding_pairs"]))
                elapsed_wall = time.perf_counter() - started_wall
                transition = {
                    "schema": EPISODE_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "run_fingerprint": job["run_fingerprint"],
                    "event": "transition",
                    "episode_id": episode_id,
                    "decision_index": decision_index,
                    "action": action,
                    "before_fingerprint": before_hash,
                    "after_fingerprint": state_fingerprint(state),
                    "metrics": metrics,
                    "low_level_delta": _low_level_delta(before, state),
                    "repair_wall_seconds": repair_wall_seconds,
                    "elapsed_wall_seconds": elapsed_wall,
                    "controller": controller,
                    "terminated": bool(result["terminated"]),
                    "truncated": bool(result["truncated"]),
                    "after": state,
                }
                _emit(stream, transition)
                if elapsed_wall >= wall_budget and not bool(state["done"]):
                    external_timeout = True
                    break
            elapsed_wall = time.perf_counter() - started_wall
            if elapsed_wall > wall_budget:
                external_timeout = True
            success = bool(state["feasible"]) and elapsed_wall <= wall_budget
            truncated = not success
            if not state["done"] and len(conflicts) - 1 >= max_decisions:
                truncated = True
            fixed_auc = fixed_budget_conflict_auc(
                conflicts, int(job["metric_iteration_budget"]), success=success
            )
            summary = {
                "initial_fingerprint": initial_fingerprint,
                "initial_conflicts": conflicts[0],
                "final_conflicts": conflicts[-1],
                "repairable": conflicts[0] > 0,
                "success": success,
                "truncated": truncated,
                "external_timeout": external_timeout,
                "repair_iterations": len(conflicts) - 1,
                "conflict_trajectory": conflicts,
                "conflict_auc": sum(
                    (conflicts[index] + conflicts[index + 1]) / 2.0
                    for index in range(len(conflicts) - 1)
                ),
                "fixed_budget_conflict_auc": fixed_auc,
                "wall_time_to_feasible": elapsed_wall if success else None,
                "capped_wall_time_to_feasible": min(elapsed_wall, wall_budget)
                if success
                else wall_budget,
                "native_time_to_feasible": float(state["runtime"]) if success else None,
                "controller_totals": dict(controller_totals),
                "mean_selected_feature_outside_fraction": (
                    float(controller_totals["selected_feature_outside_fraction_sum"])
                    / float(controller_totals["learned_decisions"])
                    if controller_totals["learned_decisions"]
                    else 0.0
                ),
                "selected_size_counts": {
                    str(key): value for key, value in sorted(selected_sizes.items())
                },
                "selected_family_counts": dict(sorted(selected_families.items())),
                "invalid_action_count": invalid_actions,
                "fingerprint_mismatch_count": fingerprint_mismatches,
                "final_sum_of_costs": int(state["sum_of_costs"]),
                "final_low_level": state["low_level"],
            }
            _emit(
                stream,
                {
                    "schema": EPISODE_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "run_fingerprint": job["run_fingerprint"],
                    "event": "finish",
                    "episode_id": episode_id,
                    "policy": policy,
                    "success": success,
                    "final_fingerprint": state_fingerprint(state),
                    "summary": summary,
                },
            )
        os.replace(partial_path, trace_path)
        return {
            "schema_version": SCHEMA_VERSION,
            "schema": CLOSED_LOOP_SCHEMA,
            "episode_id": episode_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row["layout_mode"],
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "policy": policy,
            "trace_file": relative_trace,
            "status": "ok",
            "summary": summary,
            "error": None,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "schema": CLOSED_LOOP_SCHEMA,
            "episode_id": episode_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row.get("layout_mode"),
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "policy": policy,
            "trace_file": None,
            "partial_trace_file": partial_path.relative_to(output_root).as_posix()
            if partial_path.is_file()
            else None,
            "status": "error",
            "summary": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _selected_rows(rows: list[dict[str, Any]], task_ids: list[str] | None) -> list[dict[str, Any]]:
    if task_ids is None:
        return rows
    requested = list(dict.fromkeys(map(str, task_ids)))
    indexed = {str(row["task_id"]): row for row in rows}
    missing = sorted(set(requested) - set(indexed))
    if missing:
        raise ValueError(f"unknown task ids: {missing}")
    return [indexed[task_id] for task_id in requested]


def run_closed_loop_collection(
    dataset: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
    workers: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    phases = {"qualify", "official_adaptive", "proposal_dynamic", "realized_dynamic", "all"}
    if phase not in phases:
        raise ValueError(f"unsupported closed-loop phase: {phase}")
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(dataset).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported closed-loop config")
    split = str(config["split"])
    all_rows = _load_dataset_rows(dataset_root, [split])
    rows = _selected_rows(all_rows, task_ids)
    formal = task_ids is None and bool(config.get("formal", True))
    design = closed_loop_dataset_design(all_rows, split)
    isolation = _seed_isolation(
        all_rows, list(config.get("reference_datasets", [])), project_root
    )
    frozen_root = Path(str(config["frozen_models"]))
    if not frozen_root.is_absolute():
        frozen_root = project_root / frozen_root
    bundle = load_frozen_policy_bundle(frozen_root, dict(config["model_registration"]))
    effective_workers = int(workers or config["workers"])
    dataset_fp = _dataset_fingerprint(dataset_root)
    effective = {**config, "task_ids_override": task_ids}
    config_fp = _fingerprint(effective)
    run_fp = _fingerprint(
        {
            "dataset_fingerprint": dataset_fp,
            "configuration_fingerprint": config_fp,
            "freeze_manifest": bundle.manifest,
        }
    )
    estimate = {
        "task_count": len(rows),
        "policy_episode_count": len(rows) * len(POLICIES),
        "maximum_decisions_per_episode": int(config["max_decisions"]),
        "maximum_proposals_per_decision": int(config["proposal"]["max_seed_agents"])
        * len(config["proposal"]["heuristics"])
        * len(config["proposal"]["neighborhood_sizes"])
        * int(config["proposal"]["trials"]),
        "maximum_candidates_per_decision": len(config["proposal"]["heuristics"])
        * len(config["proposal"]["neighborhood_sizes"])
        * int(config["proposal"]["candidates_per_family"]),
        "workers": effective_workers,
    }
    if dry_run:
        return {
            "schema": CLOSED_LOOP_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "formal": formal,
            "run_fingerprint": run_fp,
            "dataset_design": design,
            "seed_isolation": isolation,
            "frozen_models": bundle.manifest,
            "estimate": estimate,
        }
    run_config = {
        "schema": CLOSED_LOOP_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_fp,
        "configuration": effective,
        "configuration_fingerprint": config_fp,
        "run_fingerprint": run_fp,
        "formal": formal,
        "dataset_design": design,
        "seed_isolation": isolation,
        "frozen_models": bundle.manifest,
    }
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fp:
            raise ValueError("output contains a different closed-loop run")
        if not resume:
            raise ValueError("output already exists; pass resume to continue")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, run_config)
    sequence = POLICIES if phase == "all" else (phase,)
    if phase == "all":
        sequence = ("qualify",) + POLICIES
    summary: dict[str, Any] = {
        "schema": CLOSED_LOOP_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fp,
        "formal": formal,
        "estimate": estimate,
    }
    for current in sequence:
        if current == "qualify":
            jobs = [
                {
                    "row": row,
                    "solver_seed": int(config["solver_seed"]),
                    "dataset_root": str(dataset_root),
                    "environment": config["environment"],
                }
                for row in rows
            ]
            with _CollectionRunLock(output_root, run_fp, "closed-loop-qualification"):
                results = _run_jobs(
                    _qualification_worker,
                    jobs,
                    effective_workers,
                    phase="closed-loop-qualification",
                    output_root=output_root,
                    run_fingerprint=run_fp,
                    timeout_seconds=float(config["episode_process_timeout_seconds"]),
                )
            _write_jsonl(output_root / "qualification_manifest.jsonl", results)
            report = closed_loop_qualification_report(
                rows, results, config, design, isolation, formal=formal
            )
            _write_json(output_root / "qualification_report.json", report)
            summary["qualification"] = report
            if not report["passed"] and phase == "all":
                break
            continue
        qualification = closed_loop_qualification_report(
            rows,
            _read_jsonl(output_root / "qualification_manifest.jsonl"),
            config,
            design,
            isolation,
            formal=formal,
        )
        if not qualification["passed"]:
            raise ValueError("closed-loop qualification failed; policy execution is forbidden")
        jobs = [
            {
                "row": row,
                "policy": current,
                "solver_seed": int(config["solver_seed"]),
                "dataset_root": str(dataset_root),
                "environment": config["environment"],
                "proposal": config["proposal"],
                "max_decisions": int(config["max_decisions"]),
                "metric_iteration_budget": int(config["metric_iteration_budget"]),
                "wall_time_budget_seconds": float(config["wall_time_budget_seconds"]),
                "frozen_models": str(frozen_root.resolve()),
                "model_registration": config["model_registration"],
                "output_root": str(output_root),
                "run_fingerprint": run_fp,
                "resume": resume,
            }
            for row in rows
        ]
        with _CollectionRunLock(output_root, run_fp, f"closed-loop-{current}"):
            results = _run_jobs(
                _closed_loop_episode_worker,
                jobs,
                effective_workers,
                phase=f"closed-loop-{current}",
                output_root=output_root,
                run_fingerprint=run_fp,
                timeout_seconds=float(config["episode_process_timeout_seconds"]),
            )
        _write_jsonl(output_root / f"{current}_manifest.jsonl", results)
        summary[current] = {
            "episode_count": len(results),
            "success_count": sum(bool(row.get("summary", {}).get("success")) for row in results),
            "error_count": sum(str(row.get("status")) not in {"ok", "resumed"} for row in results),
            "timeout_count": sum(str(row.get("status")) == "timeout" for row in results),
        }
        if summary[current]["error_count"] and phase == "all":
            break
    _write_json(output_root / "collection_summary.json", summary)
    return summary


__all__ = [
    "CLOSED_LOOP_SCHEMA",
    "CollectionLockError",
    "LEARNED_POLICIES",
    "POLICIES",
    "closed_loop_dataset_design",
    "closed_loop_qualification_report",
    "feature_range_diagnostic",
    "export_portable_policy_bundle",
    "fixed_budget_conflict_auc",
    "generate_online_candidates",
    "load_frozen_policy_bundle",
    "online_candidate_rows",
    "PortablePairwiseModel",
    "proposal_random_seed",
    "repair_random_seed",
    "run_closed_loop_collection",
    "score_online_candidates",
    "verify_portable_policy_bundle",
]
