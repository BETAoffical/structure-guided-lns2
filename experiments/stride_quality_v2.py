from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.repair_collection import _read_jsonl, _write_json, _write_jsonl
from experiments.stride_collection import (
    _balanced_result_blind_selection,
    load_stride_selection,
)
from experiments.stride_lns import (
    FROZEN_FEATURE_DIMENSION,
    REQUIRED_POST_STRUCTURE_FIELDS,
    STRIDE_TRIAL_SCHEMA,
    assign_structure_scores,
)
from experiments.stride_stability import collect_stride_extension_trials


STRIDE_QUALITY_V2_CONTROLLER_ID = "stride-quality-v2"
STRIDE_QUALITY_V2_LABEL_SCHEMA = "lns2.stride.quality_label.v2"
STRIDE_QUALITY_V2_CANDIDATE_SCHEMA = "lns2.stride.candidate_aggregate.v2"
STRIDE_QUALITY_V2_DESIGN_EXTENSION_SCHEMA = (
    "lns2.stride.quality_v2.design_extension.v1"
)
STRIDE_QUALITY_V2_CONFIRM_EXTENSION_SCHEMA = (
    "lns2.stride.quality_v2.confirm_extension.v1"
)
STRIDE_QUALITY_V2_SELECTION_SCHEMA = "lns2.stride.quality_v2.confirm_selection.v1"
STRIDE_QUALITY_V2_STABILITY_SCHEMA = "lns2.stride.quality_v2.stability.v1"
STRIDE_QUALITY_V2_BUILD_SCHEMA = "lns2.stride.quality_v2.label_build.v1"
STRIDE_QUALITY_V2_TRIALS_PER_HALF = 8
STRIDE_QUALITY_V2_STRUCTURE_WEIGHT = 0.02
STRIDE_QUALITY_V2_DESIGN_INDICES = tuple(range(8, 16))
STRIDE_QUALITY_V2_CONFIRM_INDICES = tuple(range(4, 16))


def collect_stride_quality_v2_design_trials(
    *, selection_path: Path, collection: Path, output: Path, workers: int = 4
) -> dict[str, Any]:
    return collect_stride_extension_trials(
        selection_path=selection_path,
        collection=collection,
        output=output,
        trial_indices=STRIDE_QUALITY_V2_DESIGN_INDICES,
        artifact_schema=STRIDE_QUALITY_V2_DESIGN_EXTENSION_SCHEMA,
        workers=workers,
    )


def collect_stride_quality_v2_confirmation_trials(
    *, selection_path: Path, collection: Path, output: Path, workers: int = 4
) -> dict[str, Any]:
    return collect_stride_extension_trials(
        selection_path=selection_path,
        collection=collection,
        output=output,
        trial_indices=STRIDE_QUALITY_V2_CONFIRM_INDICES,
        artifact_schema=STRIDE_QUALITY_V2_CONFIRM_EXTENSION_SCHEMA,
        workers=workers,
    )


def select_stride_quality_v2_confirmation_states(
    *, selection_path: Path, design_selection_path: Path, output: Path,
    count_per_policy: int = 24,
) -> dict[str, Any]:
    """Select a result-blind confirmation cohort disjoint by state and episode."""

    selected = load_stride_selection(selection_path)
    design = load_stride_selection(design_selection_path)
    design_state_ids = {str(row["state_id"]) for row in design}
    design_episode_ids = {str(row["episode_id"]) for row in design}
    pool = [
        row
        for row in selected
        if str(row["state_id"]) not in design_state_ids
        and str(row["episode_id"]) not in design_episode_ids
    ]
    cohort, policy_reports = _balanced_result_blind_selection(
        pool, target_per_policy=count_per_policy, max_per_episode=1
    )
    cohort.sort(key=lambda row: (str(row["source_policy"]), str(row["state_id"])))
    state_overlap = sorted(
        {str(row["state_id"]) for row in cohort} & design_state_ids
    )
    episode_overlap = sorted(
        {str(row["episode_id"]) for row in cohort} & design_episode_ids
    )
    gates = {
        "state_disjoint": not state_overlap,
        "episode_disjoint": not episode_overlap,
        "policy_balance": all(
            report["selected_state_count"] == count_per_policy
            for report in policy_reports.values()
        ),
        "one_per_episode": all(
            report["max_states_in_episode"] <= 1
            for report in policy_reports.values()
        ),
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "confirmation_selection.jsonl", cohort)
    report = {
        "schema": STRIDE_QUALITY_V2_SELECTION_SCHEMA,
        "source_state_count": len(selected),
        "design_state_count": len(design),
        "eligible_state_count": len(pool),
        "selected_state_count": len(cohort),
        "count_per_policy": count_per_policy,
        "state_overlap": state_overlap,
        "episode_overlap": episode_overlap,
        "policies": policy_reports,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _write_json(output / "confirmation_selection_report.json", report)
    return report


def aggregate_stride_quality_v2_candidate(
    *, before_conflicts: int, outcomes: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(outcomes) != STRIDE_QUALITY_V2_TRIALS_PER_HALF:
        raise ValueError("STRIDE quality V2 requires exactly eight paired PP outcomes")
    seeds: set[int] = set()
    reductions: list[float] = []
    structures: list[dict[str, float]] = []
    feasible_count = 0
    progress_count = 0
    for outcome in outcomes:
        if type(outcome.get("pp_seed")) is not int:
            raise ValueError("each STRIDE quality V2 outcome requires an integer pp_seed")
        seed = int(outcome["pp_seed"])
        if seed in seeds:
            raise ValueError("STRIDE quality V2 PP seeds must be distinct")
        seeds.add(seed)
        if type(outcome.get("feasible")) is not bool:
            raise ValueError("each STRIDE quality V2 outcome requires strict feasible")
        conflicts_after = int(outcome["conflicts_after"])
        reductions.append(float(before_conflicts - conflicts_after))
        feasible_count += int(outcome["feasible"])
        progress_count += int(conflicts_after < before_conflicts)
        structure = outcome.get("post_structure")
        if not isinstance(structure, dict) or not REQUIRED_POST_STRUCTURE_FIELDS.issubset(
            structure
        ):
            raise ValueError("STRIDE quality V2 requires complete post structure")
        values = {
            name: float(structure[name]) for name in REQUIRED_POST_STRUCTURE_FIELDS
        }
        if any(value < 0.0 for value in values.values()):
            raise ValueError("post-structure metrics must be nonnegative")
        structures.append(values)
    mean_reduction = statistics.fmean(reductions)
    return {
        "trial_count": STRIDE_QUALITY_V2_TRIALS_PER_HALF,
        "pp_seeds": sorted(seeds),
        "feasible_rate": feasible_count / STRIDE_QUALITY_V2_TRIALS_PER_HALF,
        "progress_rate": progress_count / STRIDE_QUALITY_V2_TRIALS_PER_HALF,
        "mean_conflict_reduction": mean_reduction,
        "mean_reduction_ratio": mean_reduction / max(1, before_conflicts),
        "reduction_std": statistics.pstdev(reductions),
        "mean_post_structure": {
            name: statistics.fmean(item[name] for item in structures)
            for name in sorted(REQUIRED_POST_STRUCTURE_FIELDS)
        },
    }


def assign_stride_quality_v2_scores(candidates: list[dict[str, Any]]) -> None:
    assign_structure_scores(candidates)
    for candidate in candidates:
        candidate["quality_score"] = float(candidate["mean_reduction_ratio"]) - (
            STRIDE_QUALITY_V2_STRUCTURE_WEIGHT
            * float(candidate["structural_score"])
        )


def stride_quality_v2_dominates(
    left: dict[str, Any], right: dict[str, Any]
) -> bool:
    return float(left["quality_score"]) > float(right["quality_score"]) + 1e-12


def _load_trial_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in paths:
        for row in _read_jsonl(path):
            if row.get("schema") != STRIDE_TRIAL_SCHEMA:
                raise ValueError(f"unexpected STRIDE trial schema in {path}")
            key = (
                str(row["state_id"]),
                str(row["candidate_id"]),
                int(row["trial_index"]),
            )
            if key in seen:
                raise ValueError(f"duplicate STRIDE quality V2 trial: {key}")
            seen.add(key)
            rows.append(row)
    return rows


def _aggregate_half(
    rows: list[dict[str, Any]], *, expected_indices: set[int],
    state_ids: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            (state_ids is None or str(row["state_id"]) in state_ids)
            and int(row["trial_index"]) in expected_indices
        ):
            grouped[(str(row["state_id"]), str(row["candidate_id"]))].append(row)
    by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for (state_id, candidate_id), outcomes in sorted(grouped.items()):
        indices = {int(row["trial_index"]) for row in outcomes}
        if indices != expected_indices:
            raise ValueError(
                f"incomplete STRIDE quality V2 half: {state_id}/{candidate_id}"
            )
        before = int(outcomes[0]["before_conflicts"])
        aggregate = aggregate_stride_quality_v2_candidate(
            before_conflicts=before, outcomes=outcomes
        )
        aggregate.update(
            {
                "state_id": state_id,
                "candidate_id": candidate_id,
                "before_conflicts": before,
                "map_id": str(outcomes[0]["map_id"]),
                "source_policy": str(outcomes[0]["source_policy"]),
            }
        )
        by_state[state_id].append(aggregate)
    for candidates in by_state.values():
        assign_stride_quality_v2_scores(candidates)
    return dict(by_state)


def _quality_v2_rank(candidates: list[dict[str, Any]]) -> list[str]:
    return [
        str(row["candidate_id"])
        for row in sorted(
            candidates,
            key=lambda row: (-float(row["quality_score"]), str(row["candidate_id"])),
        )
    ]


def analyze_stride_quality_v2_stability(
    *, trial_paths: list[Path], output: Path, expected_state_count: int = 48
) -> dict[str, Any]:
    rows = _load_trial_paths(trial_paths)
    second = _aggregate_half(rows, expected_indices=set(range(8, 16)))
    first = _aggregate_half(
        rows, expected_indices=set(range(0, 8)), state_ids=set(second)
    )
    if set(first) != set(second):
        raise ValueError("STRIDE quality V2 stability halves contain different states")
    total_pairs = 0
    agreed_pairs = 0
    top3_scores: list[float] = []
    state_reports: list[dict[str, Any]] = []
    for state_id in sorted(first):
        first_by_id = {str(row["candidate_id"]): row for row in first[state_id]}
        second_by_id = {str(row["candidate_id"]): row for row in second[state_id]}
        if set(first_by_id) != set(second_by_id):
            raise ValueError(f"candidate mismatch between V2 halves: {state_id}")
        pair_count = 0
        pair_agree = 0
        for left_id, right_id in combinations(sorted(first_by_id), 2):
            first_winner = (
                left_id
                if stride_quality_v2_dominates(
                    first_by_id[left_id], first_by_id[right_id]
                )
                else right_id
            )
            second_winner = (
                left_id
                if stride_quality_v2_dominates(
                    second_by_id[left_id], second_by_id[right_id]
                )
                else right_id
            )
            pair_count += 1
            pair_agree += int(first_winner == second_winner)
        first_top3 = set(_quality_v2_rank(first[state_id])[:3])
        second_top3 = set(_quality_v2_rank(second[state_id])[:3])
        top3_overlap = len(first_top3 & second_top3) / 3.0
        total_pairs += pair_count
        agreed_pairs += pair_agree
        top3_scores.append(top3_overlap)
        state_reports.append(
            {
                "state_id": state_id,
                "map_id": first[state_id][0]["map_id"],
                "source_policy": first[state_id][0]["source_policy"],
                "candidate_count": len(first_by_id),
                "pair_count": pair_count,
                "agreed_pair_count": pair_agree,
                "pairwise_consistency": pair_agree / pair_count,
                "top3_overlap": top3_overlap,
            }
        )
    pairwise = agreed_pairs / total_pairs if total_pairs else 0.0
    top3_mean = statistics.fmean(top3_scores) if top3_scores else 0.0
    gates = {
        "pairwise_consistency_at_least_70_percent": pairwise >= 0.70,
        "mean_top3_overlap_at_least_80_percent": top3_mean >= 0.80,
        "state_coverage": len(state_reports) == expected_state_count,
    }
    report = {
        "schema": STRIDE_QUALITY_V2_STABILITY_SCHEMA,
        "controller_id": STRIDE_QUALITY_V2_CONTROLLER_ID,
        "label_schema": STRIDE_QUALITY_V2_LABEL_SCHEMA,
        "trials_per_half": STRIDE_QUALITY_V2_TRIALS_PER_HALF,
        "structure_weight": STRIDE_QUALITY_V2_STRUCTURE_WEIGHT,
        "runtime_used_in_label": False,
        "state_count": len(state_reports),
        "candidate_count": sum(len(items) for items in first.values()),
        "outcome_count": len(rows),
        "pair_count": total_pairs,
        "pairwise_agreement_count": agreed_pairs,
        "pairwise_consistency": pairwise,
        "mean_top3_overlap": top3_mean,
        "top3_overlap_distribution": dict(sorted(Counter(top3_scores).items())),
        "gates": gates,
        "passed": all(gates.values()),
        "states": state_reports,
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "stability_report.json", report)
    return report


def build_stride_quality_v2_labels(
    *, trial_paths: list[Path], output: Path
) -> dict[str, Any]:
    rows = _load_trial_paths(trial_paths)
    required_indices = set(range(0, 8))
    indices_by_state: defaultdict[str, set[int]] = defaultdict(set)
    for row in rows:
        indices_by_state[str(row["state_id"])].add(int(row["trial_index"]))
    eligible_state_ids = {
        state_id
        for state_id, indices in indices_by_state.items()
        if required_indices.issubset(indices)
    }
    if not eligible_state_ids:
        raise ValueError("no state has a complete eight-seed STRIDE quality V2 label")
    by_state = _aggregate_half(
        rows, expected_indices=required_indices, state_ids=eligible_state_ids
    )
    features: dict[tuple[str, str], dict[str, float]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if int(row["trial_index"]) >= 8:
            continue
        state_id = str(row["state_id"])
        candidate_id = str(row["candidate_id"])
        values = row.get("features")
        if not isinstance(values, dict) or len(values) != FROZEN_FEATURE_DIMENSION:
            raise ValueError("STRIDE quality V2 requires 124 frozen features")
        normalized = {str(name): float(value) for name, value in values.items()}
        key = (state_id, candidate_id)
        if key in features and features[key] != normalized:
            raise ValueError(f"candidate features changed across seeds: {key}")
        features[key] = normalized
        current = {
            "map_id": str(row["map_id"]),
            "split": str(row["split"]),
            "source_policy": str(row["source_policy"]),
            "decision_stage": str(row["decision_stage"]),
            "before_conflicts": int(row["before_conflicts"]),
            "agent_count": int(row["agent_count"]),
        }
        if state_id in metadata and metadata[state_id] != current:
            raise ValueError(f"inconsistent state metadata: {state_id}")
        metadata[state_id] = current
    aggregates: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    states_with_pairs = 0
    for state_id, candidates in sorted(by_state.items()):
        candidates.sort(key=lambda row: str(row["candidate_id"]))
        state_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for candidate in candidates:
            candidate.update(
                {
                    "schema": STRIDE_QUALITY_V2_CANDIDATE_SCHEMA,
                    "label_schema": STRIDE_QUALITY_V2_LABEL_SCHEMA,
                    "features": features[(state_id, str(candidate["candidate_id"]))],
                    **metadata[state_id],
                }
            )
        aggregates.extend(candidates)
        for left, right in combinations(candidates, 2):
            if stride_quality_v2_dominates(left, right):
                state_pairs.append((left, right))
            elif stride_quality_v2_dominates(right, left):
                state_pairs.append((right, left))
        if state_pairs:
            states_with_pairs += 1
            weight = 1.0 / (2.0 * len(state_pairs))
            for winner, loser in state_pairs:
                shared = {
                    "schema": STRIDE_QUALITY_V2_LABEL_SCHEMA,
                    "state_id": state_id,
                    "map_id": metadata[state_id]["map_id"],
                    "split": metadata[state_id]["split"],
                    "sample_weight": weight,
                }
                pairs.extend(
                    [
                        {
                            **shared,
                            "left_candidate_id": winner["candidate_id"],
                            "right_candidate_id": loser["candidate_id"],
                            "label": 1,
                        },
                        {
                            **shared,
                            "left_candidate_id": loser["candidate_id"],
                            "right_candidate_id": winner["candidate_id"],
                            "label": 0,
                        },
                    ]
                )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "candidate_aggregates.jsonl", aggregates)
    _write_jsonl(output / "dominance_pairs.jsonl", pairs)
    summary = {
        "schema": STRIDE_QUALITY_V2_BUILD_SCHEMA,
        "controller_id": STRIDE_QUALITY_V2_CONTROLLER_ID,
        "label_schema": STRIDE_QUALITY_V2_LABEL_SCHEMA,
        "trial_files": [str(path.resolve()) for path in trial_paths],
        "trial_sha256": [sha256_file(path) for path in trial_paths],
        "state_count": len(by_state),
        "states_with_pairs": states_with_pairs,
        "candidate_count": len(aggregates),
        "dominance_pair_count": len(pairs) // 2,
        "oriented_training_row_count": len(pairs),
        "trials_per_candidate": STRIDE_QUALITY_V2_TRIALS_PER_HALF,
        "structure_weight": STRIDE_QUALITY_V2_STRUCTURE_WEIGHT,
        "runtime_used_in_label": False,
    }
    _write_json(output / "label_build_summary.json", summary)
    return summary
