from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.stride_lns import (
    REQUIRED_POST_STRUCTURE_FIELDS,
    STRIDE_TRIAL_SCHEMA,
    assign_structure_scores,
)
from experiments.stride_stage3 import _project_path


CONFIG_SCHEMA = "lns2.stride.robuststep_design_config.v1"
REPORT_SCHEMA = "lns2.stride.robuststep_design_report.v1"
CONTROLLER_ID = "stride-robuststep-v1"
LABEL_SCHEMA = "lns2.stride.robust_step_label.v1"


def validate_robuststep_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected robust-step design config")
    if (
        config.get("scientific_status") != "consumed_design_only"
        or bool(config.get("formal_speed_claim"))
        or not bool(config.get("fresh_confirmation_required"))
    ):
        raise ValueError("robust-step design must remain consumed and non-formal")
    if config.get("controller_id") != CONTROLLER_ID:
        raise ValueError("unexpected robust-step controller id")
    if config.get("label_schema") != LABEL_SCHEMA:
        raise ValueError("unexpected robust-step label schema")
    indices = tuple(map(int, config.get("trial_indices") or ()))
    first = tuple(map(int, config.get("first_half_indices") or ()))
    second = tuple(map(int, config.get("second_half_indices") or ()))
    if indices != tuple(range(8)) or first != tuple(range(4)) or second != tuple(range(4, 8)):
        raise ValueError("robust-step requires registered 4+4 paired trial indices")
    if float(config.get("structure_weight", -1.0)) != 0.02:
        raise ValueError("robust-step structure weight differs")
    variants = list(config.get("variants") or [])
    ids = [str(row.get("id")) for row in variants]
    if len(variants) != 4 or len(set(ids)) != 4:
        raise ValueError("robust-step requires four unique design variants")
    for row in variants:
        win = float(row.get("minimum_paired_win_fraction", 0.0))
        tolerance = float(row.get("maximum_no_progress_disadvantage", -1.0))
        if not 0.5 < win <= 1.0 or not 0.0 <= tolerance <= 0.125:
            raise ValueError("invalid robust-step design variant")
    if bool(config.get("runtime_used_in_label")):
        raise ValueError("runtime cannot enter the robust-step label")


def robust_pair_winner(
    left_scores: list[float],
    right_scores: list[float],
    left_progress: list[bool],
    right_progress: list[bool],
    *,
    minimum_paired_win_fraction: float,
    maximum_no_progress_disadvantage: float,
) -> int:
    """Return 1 for left, -1 for right, or 0 for an uncertain pair."""

    size = len(left_scores)
    if (
        size == 0
        or len(right_scores) != size
        or len(left_progress) != size
        or len(right_progress) != size
    ):
        raise ValueError("robust pair inputs must be nonempty and paired")
    deltas = [left - right for left, right in zip(left_scores, right_scores)]
    left_wins = sum(delta > 1e-12 for delta in deltas)
    ties = sum(abs(delta) <= 1e-12 for delta in deltas)
    left_fraction = (left_wins + 0.5 * ties) / size
    mean_delta = statistics.fmean(deltas)
    direction = 0
    if left_fraction >= minimum_paired_win_fraction and mean_delta > 1e-12:
        direction = 1
    elif left_fraction <= 1.0 - minimum_paired_win_fraction and mean_delta < -1e-12:
        direction = -1
    if direction == 0:
        return 0
    left_no_progress = 1.0 - statistics.fmean(map(float, left_progress))
    right_no_progress = 1.0 - statistics.fmean(map(float, right_progress))
    if direction > 0:
        return (
            1
            if left_no_progress
            <= right_no_progress + maximum_no_progress_disadvantage + 1e-12
            else 0
        )
    return (
        -1
        if right_no_progress
        <= left_no_progress + maximum_no_progress_disadvantage + 1e-12
        else 0
    )


def _checked_input(project_root: Path, specification: dict[str, Any]) -> Path:
    path = _project_path(project_root, str(specification["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != str(specification["sha256"]).lower():
        raise ValueError(f"robust-step input hash differs: {path}")
    return path


def _load_seed_profiles(
    project_root: Path, config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, list[Any]]]], dict[str, Any]]:
    expected_indices = set(map(int, config["trial_indices"]))
    state_metadata: dict[str, dict[str, Any]] = {}
    rows_by_state: defaultdict[str, defaultdict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    seen: set[tuple[str, str, int]] = set()
    source_hashes: dict[str, str] = {}
    read_outcomes = 0
    for specification in config["trial_sources"]:
        path = _checked_input(project_root, dict(specification))
        source_hashes[str(specification["path"])] = sha256_file(path)
        for row in _read_jsonl(path):
            if row.get("schema") != STRIDE_TRIAL_SCHEMA:
                raise ValueError(f"unexpected repair-trial schema: {path}")
            trial_index = int(row["trial_index"])
            if trial_index not in expected_indices:
                continue
            state_id = str(row["state_id"])
            candidate_id = str(row["candidate_id"])
            key = (state_id, candidate_id, trial_index)
            if key in seen:
                raise ValueError(f"duplicate robust-step outcome: {key}")
            seen.add(key)
            structure = row.get("post_structure")
            if not isinstance(structure, dict) or not REQUIRED_POST_STRUCTURE_FIELDS.issubset(structure):
                raise ValueError("robust-step requires complete post structure")
            metadata = {
                "map_id": str(row["map_id"]),
                "split": str(row["split"]),
                "source_policy": str(row["source_policy"]),
                "decision_stage": str(row["decision_stage"]),
                "agent_count": int(row["agent_count"]),
                "before_conflicts": int(row["before_conflicts"]),
            }
            previous = state_metadata.setdefault(state_id, metadata)
            if previous != metadata:
                raise ValueError(f"inconsistent robust-step state metadata: {state_id}")
            rows_by_state[state_id][trial_index][candidate_id] = {
                "candidate_id": candidate_id,
                "pp_seed": int(row["pp_seed"]),
                "conflicts_after": int(row["conflicts_after"]),
                "post_structure": {
                    name: float(structure[name])
                    for name in REQUIRED_POST_STRUCTURE_FIELDS
                },
            }
            read_outcomes += 1

    profiles: dict[str, dict[str, dict[str, list[Any]]]] = {}
    candidate_count = 0
    for state_id in sorted(rows_by_state):
        by_index = rows_by_state[state_id]
        if set(by_index) != expected_indices:
            raise ValueError(f"incomplete robust-step trial indices: {state_id}")
        candidate_sets = [set(by_index[index]) for index in sorted(by_index)]
        if any(candidate_set != candidate_sets[0] for candidate_set in candidate_sets[1:]):
            raise ValueError(f"candidate pool changed across seeds: {state_id}")
        candidates = sorted(candidate_sets[0])
        candidate_count += len(candidates)
        state_profiles = {
            candidate_id: {"scores": [], "progress": []}
            for candidate_id in candidates
        }
        before = int(state_metadata[state_id]["before_conflicts"])
        for trial_index in sorted(expected_indices):
            outcomes = by_index[trial_index]
            seeds = {int(row["pp_seed"]) for row in outcomes.values()}
            if len(seeds) != 1:
                raise ValueError(f"PP seeds are not paired within state/index: {state_id}/{trial_index}")
            scored = [
                {
                    "candidate_id": candidate_id,
                    "mean_post_structure": outcomes[candidate_id]["post_structure"],
                }
                for candidate_id in candidates
            ]
            assign_structure_scores(scored)
            structural = {
                str(row["candidate_id"]): float(row["structural_score"])
                for row in scored
            }
            for candidate_id in candidates:
                after = int(outcomes[candidate_id]["conflicts_after"])
                score = (before - after) / max(1, before) - float(
                    config["structure_weight"]
                ) * structural[candidate_id]
                state_profiles[candidate_id]["scores"].append(score)
                state_profiles[candidate_id]["progress"].append(after < before)
        profiles[state_id] = state_profiles
    integrity = {
        "state_count": len(profiles),
        "map_count": len({row["map_id"] for row in state_metadata.values()}),
        "candidate_count": candidate_count,
        "outcome_count": read_outcomes,
        "trial_source_sha256": source_hashes,
    }
    return state_metadata, profiles, integrity


def _variant_winners(
    candidates: dict[str, dict[str, list[Any]]],
    indices: list[int],
    variant: dict[str, Any],
) -> dict[tuple[str, str], int]:
    winners: dict[tuple[str, str], int] = {}
    for left_id, right_id in combinations(sorted(candidates), 2):
        left = candidates[left_id]
        right = candidates[right_id]
        direction = robust_pair_winner(
            [float(left["scores"][index]) for index in indices],
            [float(right["scores"][index]) for index in indices],
            [bool(left["progress"][index]) for index in indices],
            [bool(right["progress"][index]) for index in indices],
            minimum_paired_win_fraction=float(
                variant["minimum_paired_win_fraction"]
            ),
            maximum_no_progress_disadvantage=float(
                variant["maximum_no_progress_disadvantage"]
            ),
        )
        if direction:
            winners[(left_id, right_id)] = direction
    return winners


def _robust_good_set(
    candidate_ids: list[str], winners: dict[tuple[str, str], int]
) -> set[str]:
    losses = Counter({candidate_id: 0 for candidate_id in candidate_ids})
    for (left_id, right_id), direction in winners.items():
        losses[right_id if direction > 0 else left_id] += 1
    minimum = min(losses.values(), default=0)
    return {candidate_id for candidate_id, count in losses.items() if count == minimum}


def evaluate_robuststep_variant(
    profiles: dict[str, dict[str, dict[str, list[Any]]]],
    variant: dict[str, Any],
    first_indices: list[int],
    second_indices: list[int],
) -> dict[str, Any]:
    union_half_pairs = 0
    agreeing_half_pairs = 0
    full_pairs = 0
    all_pairs = 0
    five_pair_states = 0
    unique_winner_states = 0
    good_set_jaccards: list[float] = []
    full_good_set_sizes: list[int] = []
    state_rows = []
    full_indices = first_indices + second_indices
    for state_id, candidates in sorted(profiles.items()):
        candidate_ids = sorted(candidates)
        first = _variant_winners(candidates, first_indices, variant)
        second = _variant_winners(candidates, second_indices, variant)
        full = _variant_winners(candidates, full_indices, variant)
        pair_union = set(first) | set(second)
        pair_agreement = sum(
            pair in first and pair in second and first[pair] == second[pair]
            for pair in pair_union
        )
        first_good = _robust_good_set(candidate_ids, first)
        second_good = _robust_good_set(candidate_ids, second)
        full_good = _robust_good_set(candidate_ids, full)
        good_union = first_good | second_good
        jaccard = len(first_good & second_good) / len(good_union) if good_union else 1.0
        total = len(candidate_ids) * (len(candidate_ids) - 1) // 2
        union_half_pairs += len(pair_union)
        agreeing_half_pairs += pair_agreement
        full_pairs += len(full)
        all_pairs += total
        five_pair_states += len(full) >= 5
        unique_winner_states += len(full_good) == 1
        good_set_jaccards.append(jaccard)
        full_good_set_sizes.append(len(full_good))
        state_rows.append(
            {
                "state_id": state_id,
                "candidate_count": len(candidate_ids),
                "first_half_pair_count": len(first),
                "second_half_pair_count": len(second),
                "full_pair_count": len(full),
                "half_pair_agreement_count": pair_agreement,
                "half_pair_union_count": len(pair_union),
                "good_set_jaccard": jaccard,
                "full_good_set_size": len(full_good),
            }
        )
    state_count = len(profiles)
    return {
        "id": str(variant["id"]),
        "minimum_paired_win_fraction": float(
            variant["minimum_paired_win_fraction"]
        ),
        "maximum_no_progress_disadvantage": float(
            variant["maximum_no_progress_disadvantage"]
        ),
        "half_pair_union_count": union_half_pairs,
        "half_pair_agreement_count": agreeing_half_pairs,
        "half_pairwise_consistency": (
            agreeing_half_pairs / union_half_pairs if union_half_pairs else 0.0
        ),
        "full_pair_count": full_pairs,
        "all_unordered_pair_count": all_pairs,
        "full_pair_coverage": full_pairs / all_pairs if all_pairs else 0.0,
        "state_five_pair_coverage": five_pair_states / state_count if state_count else 0.0,
        "mean_good_set_jaccard": (
            statistics.fmean(good_set_jaccards) if good_set_jaccards else 0.0
        ),
        "mean_full_good_set_size": (
            statistics.fmean(full_good_set_sizes) if full_good_set_sizes else 0.0
        ),
        "unique_robust_winner_rate": (
            unique_winner_states / state_count if state_count else 0.0
        ),
        "action_uncertainty_rate": (
            1.0 - unique_winner_states / state_count if state_count else 1.0
        ),
        "states": state_rows,
    }


def run_robuststep_design(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robuststep_config(config)
    audit_path = _checked_input(project_root, dict(config["stage3_audit"]))
    audit = _read_json(audit_path)
    formal_path = _checked_input(project_root, dict(config["formal_ood_config"]))
    formal = _read_json(formal_path)
    formal_maps = {str(row["benchmark_id"]) for row in formal.get("cases", [])}
    metadata, profiles, integrity = _load_seed_profiles(project_root, config)
    observed_maps = {str(row["map_id"]) for row in metadata.values()}
    formal_overlap = sorted(observed_maps & formal_maps)
    integrity_gates = {
        "stage3_audit_passed": bool(audit.get("passed")),
        "state_count": integrity["state_count"] == int(config["expected_state_count"]),
        "map_count": integrity["map_count"] == int(config["expected_map_count"]),
        "candidate_count": integrity["candidate_count"] == int(config["expected_candidate_count"]),
        "outcome_count": integrity["outcome_count"] == int(config["expected_outcome_count"]),
        "formal_ood_overlap_zero": not formal_overlap,
    }
    thresholds = dict(config["design_gates"])
    variants = []
    for specification in config["variants"]:
        result = evaluate_robuststep_variant(
            profiles,
            dict(specification),
            list(map(int, config["first_half_indices"])),
            list(map(int, config["second_half_indices"])),
        )
        result["gates"] = {
            "half_pairwise_consistency": result["half_pairwise_consistency"]
            >= float(thresholds["minimum_half_pairwise_consistency"]),
            "mean_good_set_jaccard": result["mean_good_set_jaccard"]
            >= float(thresholds["minimum_mean_good_set_jaccard"]),
            "full_pair_coverage": result["full_pair_coverage"]
            >= float(thresholds["minimum_full_pair_coverage"]),
            "state_five_pair_coverage": result["state_five_pair_coverage"]
            >= float(thresholds["minimum_state_five_pair_coverage"]),
        }
        result["passed"] = all(result["gates"].values())
        variants.append(result)
    eligible = [row for row in variants if row["passed"]]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["full_pair_coverage"]),
            float(row["half_pairwise_consistency"]),
            float(row["mean_good_set_jaccard"]),
            str(row["id"]),
        ),
        default=None,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "consumed_design_only",
        "formal_speed_claim": False,
        "controller_id": CONTROLLER_ID,
        "label_schema": LABEL_SCHEMA,
        "runtime_used_in_label": False,
        "fresh_confirmation_required": True,
        "integrity": integrity,
        "integrity_gates": integrity_gates,
        "formal_ood_overlap": formal_overlap,
        "variants": variants,
        "selected_variant_id": str(selected["id"]) if selected else None,
        "design_passed": all(integrity_gates.values()) and selected is not None,
        "next_decision": (
            "freeze_selected_variant_and_collect_fresh_confirmation"
            if all(integrity_gates.values()) and selected is not None
            else "revise_robust_pair_contract_before_fresh_collection"
        ),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "stage3_audit_sha256": sha256_file(audit_path),
            "formal_ood_config_sha256": sha256_file(formal_path),
        },
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "robuststep_design_report.json", report)
    return report


__all__ = [
    "CONTROLLER_ID",
    "LABEL_SCHEMA",
    "evaluate_robuststep_variant",
    "robust_pair_winner",
    "run_robuststep_design",
    "validate_robuststep_config",
]
