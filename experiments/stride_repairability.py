from __future__ import annotations

import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_lns import REQUIRED_POST_STRUCTURE_FIELDS, STRIDE_TRIAL_SCHEMA


CONFIG_SCHEMA = "lns2.stride.repairability_label_config.v1"
LABEL_SCHEMA = "lns2.stride.repairability_label.v1"
CONFLICT_ONLY_LABEL_SCHEMA = "lns2.stride.repairability_conflict_only_ablation.v1"
CANDIDATE_SCHEMA = "lns2.stride.repairability_candidate.v1"
BUILD_SCHEMA = "lns2.stride.repairability_label_build.v1"
CONTROLLER_ID = "stride-augcontrol-v1"
FEATURE_SCHEMA = "lns2.realized_features.v2"
FEATURE_DIMENSION = 124


def validate_repairability_label_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected repairability label config")
    if (
        config.get("scientific_status")
        != "preregistered_before_topology_balanced_outcomes"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("label_schema") != LABEL_SCHEMA
        or config.get("feature_schema") != FEATURE_SCHEMA
        or int(config.get("feature_dimension", -1)) != FEATURE_DIMENSION
    ):
        raise ValueError("repairability label identity changed")
    indices = tuple(map(int, config.get("trial_indices") or ()))
    first = tuple(map(int, config.get("first_half_indices") or ()))
    second = tuple(map(int, config.get("second_half_indices") or ()))
    if (
        indices != tuple(range(16))
        or first != tuple(range(8))
        or second != tuple(range(8, 16))
    ):
        raise ValueError("repairability labels require a registered 8+8 seed product")
    quality = dict(config.get("current_step_quality") or {})
    if quality != {
        "conflict_reduction": "normalized_by_before_conflicts",
        "post_structure_weight": 0.02,
        "post_structure_normalization": "within_state_trial_midrank",
    }:
        raise ValueError("repairability current-step score changed")
    if dict(config.get("candidate_pool") or {}) != {
        "base_families": ["target", "collision", "random"],
        "base_sizes": [4, 8, 16],
        "augmentation": "stride-topoboundary-v1",
        "boundary_size": 16,
        "boundary_core_budget": 4,
        "maximum_boundary_candidates": 2,
        "maximum_total_candidates": 20,
    }:
        raise ValueError("repairability candidate pool changed")
    robust = dict(config.get("robust_pair_rule") or {})
    if robust != {
        "minimum_paired_win_fraction": 0.75,
        "minimum_absolute_mean_effect": 0.02,
        "require_both_half_mean_directions": True,
        "tie_epsilon": 1e-12,
    }:
        raise ValueError("repairability robust-pair rule changed")
    forbidden = set(map(str, config.get("forbidden_label_inputs") or ()))
    if forbidden != {
        "repair_runtime",
        "time_to_feasible",
        "future_repair_rounds",
        "cost_to_go",
        "receding_q",
        "controller_outcome",
    }:
        raise ValueError("repairability forbidden label inputs changed")
    if (
        config.get("sample_weighting") != "equal_total_weight_per_state"
        or bool(config.get("runtime_used_in_label"))
        or bool(config.get("future_trajectory_used_in_label"))
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
    ):
        raise ValueError("repairability evidence boundary changed")


def _midrank_percentile(values: list[float], current: float) -> float:
    if len(values) <= 1:
        return 0.0
    lower = sum(value < current for value in values)
    equal_other = sum(value == current for value in values) - 1
    return (lower + 0.5 * equal_other) / (len(values) - 1)


def _load_trials(
    trial_paths: Iterable[Path], required_indices: tuple[int, ...]
) -> tuple[
    dict[str, dict[str, dict[int, dict[str, Any]]]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str], dict[str, Any]],
]:
    required = set(required_indices)
    states: defaultdict[str, defaultdict[str, dict[int, dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    metadata: dict[str, dict[str, Any]] = {}
    features: dict[tuple[str, str], dict[str, float]] = {}
    candidate_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    expected_feature_names = set(PROFILE_FEATURE_NAMES["realized_dynamic"])
    for path in trial_paths:
        for row in _read_jsonl(path):
            if row.get("schema") != STRIDE_TRIAL_SCHEMA:
                raise ValueError(f"unexpected repair trial schema in {path}")
            state_id = str(row["state_id"])
            candidate_id = str(row["candidate_id"])
            trial_index = int(row["trial_index"])
            if trial_index not in required:
                continue
            if trial_index in states[state_id][candidate_id]:
                raise ValueError(
                    f"duplicate repairability trial: {state_id}/{candidate_id}/{trial_index}"
                )
            if type(row.get("pp_seed")) is not int:
                raise ValueError("repairability trial requires an integer PP seed")
            if type(row.get("feasible")) is not bool:
                raise ValueError("repairability trial requires strict feasible")
            structure = row.get("post_structure")
            if not isinstance(structure, dict) or not REQUIRED_POST_STRUCTURE_FIELDS.issubset(
                structure
            ):
                raise ValueError("repairability trial requires complete post structure")
            if any(
                not math.isfinite(float(structure[name]))
                or float(structure[name]) < 0.0
                for name in REQUIRED_POST_STRUCTURE_FIELDS
            ):
                raise ValueError("repairability post structure is invalid")
            raw_features = row.get("features")
            if not isinstance(raw_features, dict) or set(raw_features) != expected_feature_names:
                raise ValueError("repairability trial requires the exact 124-feature schema")
            normalized_features = {
                str(name): float(value) for name, value in raw_features.items()
            }
            if any(not math.isfinite(value) for value in normalized_features.values()):
                raise ValueError("repairability feature is non-finite")
            feature_key = (state_id, candidate_id)
            if feature_key in features and features[feature_key] != normalized_features:
                raise ValueError(f"candidate features changed across seeds: {feature_key}")
            features[feature_key] = normalized_features
            current_candidate_metadata = {
                "candidate_kind": str(row.get("candidate_kind", "unknown")),
                "actual_size": int(row.get("actual_size", 0)),
                "selection_families": sorted(
                    map(str, row.get("selection_families") or ())
                ),
                "agents": sorted(map(int, row.get("agents") or ())),
            }
            if current_candidate_metadata["actual_size"] <= 0:
                raise ValueError("repairability trial requires candidate actual_size")
            if (
                len(current_candidate_metadata["agents"])
                != current_candidate_metadata["actual_size"]
            ):
                raise ValueError("repairability candidate agents and size differ")
            if (
                feature_key in candidate_metadata
                and candidate_metadata[feature_key] != current_candidate_metadata
            ):
                raise ValueError(
                    f"candidate metadata changed across seeds: {feature_key}"
                )
            candidate_metadata[feature_key] = current_candidate_metadata
            current_metadata = {
                "map_id": str(row["map_id"]),
                "split": str(row["split"]),
                "source_policy": str(row["source_policy"]),
                "decision_stage": str(row["decision_stage"]),
                "before_conflicts": int(row["before_conflicts"]),
                "agent_count": int(row["agent_count"]),
            }
            if current_metadata["before_conflicts"] <= 0:
                raise ValueError("repairability labels require a conflicting state")
            if state_id in metadata and metadata[state_id] != current_metadata:
                raise ValueError(f"inconsistent repairability state metadata: {state_id}")
            metadata[state_id] = current_metadata
            states[state_id][candidate_id][trial_index] = row
    if not states:
        raise ValueError("no complete repairability trials were supplied")
    normalized_states: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for state_id, candidates in sorted(states.items()):
        if len(candidates) < 2:
            raise ValueError(f"repairability state has fewer than two candidates: {state_id}")
        if len(candidates) > 20:
            raise ValueError(f"repairability candidate pool exceeds its cap: {state_id}")
        boundary_count = 0
        base_family_roots: set[str] = set()
        for candidate_id, outcomes in candidates.items():
            if set(outcomes) != required:
                raise ValueError(
                    f"incomplete repairability seed product: {state_id}/{candidate_id}"
                )
            if len({int(row["pp_seed"]) for row in outcomes.values()}) != len(required):
                raise ValueError(
                    f"repairability PP seeds are not distinct: {state_id}/{candidate_id}"
                )
            candidate = candidate_metadata[(state_id, candidate_id)]
            families = set(map(str, candidate["selection_families"]))
            is_boundary = candidate["candidate_kind"] == "boundary_only"
            if is_boundary:
                boundary_count += 1
                if (
                    candidate["actual_size"] != 16
                    or not any(
                        family.startswith("topology-boundary-")
                        for family in families
                    )
                ):
                    raise ValueError(
                        f"invalid repairability boundary candidate: {state_id}/{candidate_id}"
                    )
            else:
                base_families = {
                    family
                    for family in families
                    if not family.startswith("topology-boundary-")
                }
                requested_sizes = {
                    int(family.rsplit(":", 1)[1])
                    for family in base_families
                    if ":" in family
                }
                if (
                    candidate["candidate_kind"] != "base"
                    or not 1 <= candidate["actual_size"] <= 16
                    or not requested_sizes
                    or not requested_sizes <= {4, 8, 16}
                ):
                    raise ValueError(
                        f"invalid repairability base candidate: {state_id}/{candidate_id}"
                    )
                base_family_roots.update(
                    family.split(":", 1)[0] for family in base_families
                )
        if boundary_count > 2:
            raise ValueError(f"repairability boundary candidate cap exceeded: {state_id}")
        if base_family_roots != {"target", "collision", "random"}:
            raise ValueError(f"repairability base candidate families differ: {state_id}")
        for trial_index in required_indices:
            seeds = {
                int(outcomes[trial_index]["pp_seed"])
                for outcomes in candidates.values()
            }
            if len(seeds) != 1:
                raise ValueError(
                    f"PP seeds are not paired within state/trial: {state_id}/{trial_index}"
                )
        normalized_states[state_id] = dict(candidates)
    map_splits: defaultdict[str, set[str]] = defaultdict(set)
    for state_metadata in metadata.values():
        split = str(state_metadata["split"])
        if split not in {"train", "validation"}:
            raise ValueError(f"unsupported repairability research split: {split}")
        map_splits[str(state_metadata["map_id"])].add(split)
    leaking_maps = sorted(
        map_id for map_id, splits in map_splits.items() if len(splits) != 1
    )
    if leaking_maps:
        raise ValueError(f"repairability map split leakage: {leaking_maps}")
    return normalized_states, metadata, features, candidate_metadata


def _state_candidate_scores(
    candidates: dict[str, dict[int, dict[str, Any]]],
    *,
    before_conflicts: int,
    indices: tuple[int, ...],
    structure_weight: float,
) -> dict[str, list[float]]:
    scores = {candidate_id: [] for candidate_id in candidates}
    for trial_index in indices:
        structure_percentiles: dict[str, list[float]] = {
            candidate_id: [] for candidate_id in candidates
        }
        for name in sorted(REQUIRED_POST_STRUCTURE_FIELDS):
            values = [
                float(outcomes[trial_index]["post_structure"][name])
                for outcomes in candidates.values()
            ]
            for candidate_id, outcomes in candidates.items():
                current = float(outcomes[trial_index]["post_structure"][name])
                structure_percentiles[candidate_id].append(
                    _midrank_percentile(values, current)
                )
        for candidate_id, outcomes in candidates.items():
            outcome = outcomes[trial_index]
            reduction_ratio = (
                before_conflicts - int(outcome["conflicts_after"])
            ) / max(1, before_conflicts)
            structural_score = statistics.fmean(structure_percentiles[candidate_id])
            scores[candidate_id].append(
                float(reduction_ratio) - structure_weight * structural_score
            )
    return scores


def _robust_winner(
    left_scores: list[float],
    right_scores: list[float],
    *,
    minimum_win_fraction: float,
    minimum_effect: float,
    tie_epsilon: float,
) -> tuple[int, dict[str, float]]:
    deltas = [left - right for left, right in zip(left_scores, right_scores)]
    mean_effect = statistics.fmean(deltas)
    first_effect = statistics.fmean(deltas[:8])
    second_effect = statistics.fmean(deltas[8:])
    direction = 1 if mean_effect > tie_epsilon else -1 if mean_effect < -tie_epsilon else 0
    aligned = sum(direction * delta > tie_epsilon for delta in deltas) if direction else 0
    win_fraction = aligned / len(deltas)
    passed = bool(
        direction
        and abs(mean_effect) + tie_epsilon >= minimum_effect
        and win_fraction + tie_epsilon >= minimum_win_fraction
        and direction * first_effect > tie_epsilon
        and direction * second_effect > tie_epsilon
    )
    return (direction if passed else 0), {
        "mean_effect": mean_effect,
        "first_half_mean_effect": first_effect,
        "second_half_mean_effect": second_effect,
        "paired_win_fraction": win_fraction,
    }


def build_repairability_labels(
    *, config_path: str | Path, trial_paths: list[Path], output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_repairability_label_config(config)
    paths = [Path(path).resolve() for path in trial_paths]
    indices = tuple(map(int, config["trial_indices"]))
    states, metadata, features, candidate_metadata = _load_trials(paths, indices)
    structure_weight = float(config["current_step_quality"]["post_structure_weight"])
    robust = dict(config["robust_pair_rule"])
    aggregates: list[dict[str, Any]] = []
    oriented_pairs: list[dict[str, Any]] = []
    conflict_only_pairs: list[dict[str, Any]] = []
    possible_pair_count = 0
    uncertain_pair_count = 0
    conflict_only_uncertain_pair_count = 0
    states_with_pairs = 0
    boundary_candidate_count = 0
    for state_id, candidates in sorted(states.items()):
        state_metadata = metadata[state_id]
        before_conflicts = int(state_metadata["before_conflicts"])
        scores = _state_candidate_scores(
            candidates,
            before_conflicts=before_conflicts,
            indices=indices,
            structure_weight=structure_weight,
        )
        conflict_only_scores = {
            candidate_id: [
                (
                    before_conflicts
                    - int(outcomes[trial_index]["conflicts_after"])
                )
                / max(1, before_conflicts)
                for trial_index in indices
            ]
            for candidate_id, outcomes in candidates.items()
        }
        candidate_ids = sorted(candidates)
        for candidate_id in candidate_ids:
            outcomes = candidates[candidate_id]
            candidate_scores = scores[candidate_id]
            reductions = conflict_only_scores[candidate_id]
            aggregates.append(
                {
                    "schema": CANDIDATE_SCHEMA,
                    "label_schema": LABEL_SCHEMA,
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    **state_metadata,
                    "trial_count": len(indices),
                    "pp_seeds": [int(outcomes[index]["pp_seed"]) for index in indices],
                    "mean_repairability_score": statistics.fmean(candidate_scores),
                    "repairability_score_std": statistics.pstdev(candidate_scores),
                    "mean_conflicts_after": statistics.fmean(
                        int(outcomes[index]["conflicts_after"]) for index in indices
                    ),
                    "mean_conflict_reduction_ratio": statistics.fmean(reductions),
                    "conflict_reduction_ratio_std": statistics.pstdev(reductions),
                    "mean_post_structure": {
                        name: statistics.fmean(
                            float(outcomes[index]["post_structure"][name])
                            for index in indices
                        )
                        for name in sorted(REQUIRED_POST_STRUCTURE_FIELDS)
                    },
                    "progress_rate": sum(
                        int(outcomes[index]["conflicts_after"] < before_conflicts)
                        for index in indices
                    )
                    / len(indices),
                    "feasible_rate": sum(
                        int(outcomes[index]["feasible"]) for index in indices
                    )
                    / len(indices),
                    **candidate_metadata[(state_id, candidate_id)],
                    "features": features[(state_id, candidate_id)],
                }
            )
            boundary_candidate_count += int(
                candidate_metadata[(state_id, candidate_id)]["candidate_kind"]
                == "boundary_only"
            )
        state_pairs: list[tuple[str, str, dict[str, float]]] = []
        for left_id, right_id in combinations(candidate_ids, 2):
            possible_pair_count += 1
            direction, diagnostics = _robust_winner(
                scores[left_id],
                scores[right_id],
                minimum_win_fraction=float(robust["minimum_paired_win_fraction"]),
                minimum_effect=float(robust["minimum_absolute_mean_effect"]),
                tie_epsilon=float(robust["tie_epsilon"]),
            )
            if not direction:
                uncertain_pair_count += 1
                continue
            winner, loser = (
                (left_id, right_id) if direction > 0 else (right_id, left_id)
            )
            state_pairs.append((winner, loser, diagnostics))
        if state_pairs:
            states_with_pairs += 1
            sample_weight = 1.0 / (2.0 * len(state_pairs))
            for winner, loser, diagnostics in state_pairs:
                shared = {
                    "schema": LABEL_SCHEMA,
                    "state_id": state_id,
                    "map_id": state_metadata["map_id"],
                    "split": state_metadata["split"],
                    "sample_weight": sample_weight,
                    "paired_win_fraction": diagnostics["paired_win_fraction"],
                    "absolute_mean_effect": abs(diagnostics["mean_effect"]),
                    "first_half_absolute_mean_effect": abs(
                        diagnostics["first_half_mean_effect"]
                    ),
                    "second_half_absolute_mean_effect": abs(
                        diagnostics["second_half_mean_effect"]
                    ),
                }
                oriented_pairs.extend(
                    (
                        {
                            **shared,
                            "left_candidate_id": winner,
                            "right_candidate_id": loser,
                            "label": 1,
                        },
                        {
                            **shared,
                            "left_candidate_id": loser,
                            "right_candidate_id": winner,
                            "label": 0,
                        },
                    )
                )
        conflict_state_pairs: list[tuple[str, str, dict[str, float]]] = []
        for left_id, right_id in combinations(candidate_ids, 2):
            direction, diagnostics = _robust_winner(
                conflict_only_scores[left_id],
                conflict_only_scores[right_id],
                minimum_win_fraction=float(robust["minimum_paired_win_fraction"]),
                minimum_effect=float(robust["minimum_absolute_mean_effect"]),
                tie_epsilon=float(robust["tie_epsilon"]),
            )
            if not direction:
                conflict_only_uncertain_pair_count += 1
                continue
            winner, loser = (
                (left_id, right_id) if direction > 0 else (right_id, left_id)
            )
            conflict_state_pairs.append((winner, loser, diagnostics))
        if conflict_state_pairs:
            conflict_weight = 1.0 / (2.0 * len(conflict_state_pairs))
            for winner, loser, diagnostics in conflict_state_pairs:
                shared = {
                    "schema": CONFLICT_ONLY_LABEL_SCHEMA,
                    "state_id": state_id,
                    "map_id": state_metadata["map_id"],
                    "split": state_metadata["split"],
                    "sample_weight": conflict_weight,
                    "paired_win_fraction": diagnostics["paired_win_fraction"],
                    "absolute_mean_effect": abs(diagnostics["mean_effect"]),
                }
                conflict_only_pairs.extend(
                    (
                        {
                            **shared,
                            "left_candidate_id": winner,
                            "right_candidate_id": loser,
                            "label": 1,
                        },
                        {
                            **shared,
                            "left_candidate_id": loser,
                            "right_candidate_id": winner,
                            "label": 0,
                        },
                    )
                )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "candidate_aggregates.jsonl", aggregates)
    _write_jsonl(output / "dominance_pairs.jsonl", oriented_pairs)
    _write_jsonl(
        output / "conflict_only_dominance_pairs.jsonl", conflict_only_pairs
    )
    summary = {
        "schema": BUILD_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "label_schema": LABEL_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_dimension": FEATURE_DIMENSION,
        "state_count": len(states),
        "map_count_by_split": {
            split: len(
                {
                    str(row["map_id"])
                    for row in metadata.values()
                    if str(row["split"]) == split
                }
            )
            for split in ("train", "validation")
        },
        "states_with_pairs": states_with_pairs,
        "candidate_count": len(aggregates),
        "boundary_candidate_count": boundary_candidate_count,
        "trials_per_candidate": len(indices),
        "possible_pair_count": possible_pair_count,
        "robust_pair_count": len(oriented_pairs) // 2,
        "uncertain_pair_count": uncertain_pair_count,
        "conflict_only_robust_pair_count": len(conflict_only_pairs) // 2,
        "conflict_only_uncertain_pair_count": conflict_only_uncertain_pair_count,
        "oriented_training_row_count": len(oriented_pairs),
        "runtime_used_in_label": False,
        "future_trajectory_used_in_label": False,
        "sample_weighting": config["sample_weighting"],
        "config_sha256": sha256_file(config_path),
        "trial_sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
    }
    _write_json(output / "label_build_summary.json", summary)
    return summary


__all__ = [
    "CONFLICT_ONLY_LABEL_SCHEMA",
    "CONTROLLER_ID",
    "LABEL_SCHEMA",
    "build_repairability_labels",
    "validate_repairability_label_config",
]
