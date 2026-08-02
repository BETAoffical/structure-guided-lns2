from __future__ import annotations

import itertools
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_lns import (
    REQUIRED_POST_STRUCTURE_FIELDS,
    assign_structure_scores,
)
from experiments.stride_stage3 import _project_path
from experiments.stride_stage4 import _control_dominates


STRIDE_STAGE4R_CONFIG_SCHEMA = "lns2.stride.stage4r_diagnostic_config.v1"
STRIDE_STAGE4R_REPORT_SCHEMA = "lns2.stride.stage4r_diagnostic.v1"
STRIDE_STAGE4R_STATE_SCHEMA = "lns2.stride.stage4r_state_diagnostic.v1"


def _number_summary(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(map(float, values))
    if not ordered:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": percentile(0.9),
        "max": ordered[-1],
    }


def _average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda row: (row[1], row[0]))
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(
            ordered[end][1], ordered[index][1], rel_tol=0.0, abs_tol=1e-12
        ):
            end += 1
        rank = 0.5 * (index + end - 1)
        for key, _ in ordered[index:end]:
            result[key] = rank
        index = end
    return result


def _rank_correlation(left: dict[str, float], right: dict[str, float]) -> float:
    if set(left) != set(right) or not left:
        raise ValueError("rank correlation requires equal nonempty keys")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    keys = sorted(left)
    left_mean = statistics.fmean(left_ranks[key] for key in keys)
    right_mean = statistics.fmean(right_ranks[key] for key in keys)
    covariance = math.fsum(
        (left_ranks[key] - left_mean) * (right_ranks[key] - right_mean)
        for key in keys
    )
    left_scale = math.fsum(
        (left_ranks[key] - left_mean) ** 2 for key in keys
    )
    right_scale = math.fsum(
        (right_ranks[key] - right_mean) ** 2 for key in keys
    )
    if left_scale <= 1e-18 or right_scale <= 1e-18:
        return 1.0 if left_scale <= 1e-18 and right_scale <= 1e-18 else 0.0
    return covariance / math.sqrt(left_scale * right_scale)


def _control_order(candidates: list[dict[str, Any]]) -> list[str]:
    dominance = Counter()
    for first, second in itertools.combinations(candidates, 2):
        if _control_dominates(first, second):
            dominance[str(first["candidate_id"])] += 1
        elif _control_dominates(second, first):
            dominance[str(second["candidate_id"])] += 1
    return [
        str(row["candidate_id"])
        for row in sorted(
            candidates,
            key=lambda row: (
                -dominance[str(row["candidate_id"])],
                -float(row["feasible_rate"]),
                -float(row["mean_conflict_reduction"]),
                str(row["candidate_id"]),
            ),
        )
    ]


def _control_front(candidates: list[dict[str, Any]]) -> set[str]:
    return {
        str(candidate["candidate_id"])
        for candidate in candidates
        if not any(
            _control_dominates(other, candidate)
            for other in candidates
            if other is not candidate
        )
    }


def _aggregate_seed_half(
    rows: list[dict[str, Any]], indices: set[int], structure_weight: float
) -> list[dict[str, Any]]:
    by_candidate: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["trial_index"]) in indices:
            by_candidate[str(row["candidate_id"])].append(row)
    candidates = []
    for candidate_id, outcomes in sorted(by_candidate.items()):
        actual_indices = {int(row["trial_index"]) for row in outcomes}
        if actual_indices != indices or len(outcomes) != len(indices):
            raise ValueError(f"incomplete seed half for candidate {candidate_id}")
        before_values = {int(row["before_conflicts"]) for row in outcomes}
        if len(before_values) != 1:
            raise ValueError("seed-half candidate disagrees on conflicts before")
        before = before_values.pop()
        reductions = [before - int(row["conflicts_after"]) for row in outcomes]
        structures = []
        for outcome in outcomes:
            source = outcome.get("post_structure")
            if not isinstance(source, dict) or not REQUIRED_POST_STRUCTURE_FIELDS.issubset(source):
                raise ValueError("seed-half outcome lacks post structure")
            structures.append(
                {name: float(source[name]) for name in REQUIRED_POST_STRUCTURE_FIELDS}
            )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "feasible_rate": statistics.fmean(
                    float(bool(row["feasible"])) for row in outcomes
                ),
                "progress_rate": statistics.fmean(
                    float(int(row["conflicts_after"]) < before) for row in outcomes
                ),
                "mean_conflict_reduction": statistics.fmean(reductions),
                "mean_reduction_ratio": statistics.fmean(reductions) / max(1, before),
                "mean_post_structure": {
                    name: statistics.fmean(row[name] for row in structures)
                    for name in sorted(REQUIRED_POST_STRUCTURE_FIELDS)
                },
            }
        )
    assign_structure_scores(candidates)
    for candidate in candidates:
        candidate["quality_score"] = float(candidate["mean_reduction_ratio"]) - (
            structure_weight * float(candidate["structural_score"])
        )
    return candidates


def _candidate_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        grouped[str(row["state_id"])].append(row)
    for state_id, candidates in grouped.items():
        candidates.sort(key=lambda row: str(row["candidate_id"]))
        if len(candidates) < 2:
            raise ValueError(f"Stage 4R state has fewer than two candidates: {state_id}")
    return dict(grouped)


def _load_oof_predictions(
    rows: list[dict[str, Any]], models: set[str], expected_states: set[str]
) -> dict[str, dict[str, str]]:
    result: defaultdict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        model_id = str(row["model_id"])
        if model_id not in models:
            continue
        state_id = str(row["state_id"])
        if state_id in result[model_id]:
            raise ValueError(f"duplicate OOF prediction: {(model_id, state_id)}")
        result[model_id][state_id] = str(row["selected_candidate_id"])
    if set(result) != models or any(
        set(predictions) != expected_states for predictions in result.values()
    ):
        raise ValueError("Stage 4R OOF prediction coverage differs")
    return dict(result)


def _load_trials(
    paths: list[Path], candidate_keys: set[tuple[str, str]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows_by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    paired_seeds: defaultdict[tuple[str, int], set[int]] = defaultdict(set)
    for path in paths:
        for row in _read_jsonl(path):
            index = int(row["trial_index"])
            key = (str(row["state_id"]), str(row["candidate_id"]))
            if index >= 8 or key not in candidate_keys:
                continue
            unique = (key[0], key[1], index)
            if unique in seen:
                raise ValueError(f"duplicate Stage 4R trial: {unique}")
            seen.add(unique)
            rows_by_state[key[0]].append(row)
            paired_seeds[(key[0], index)].add(int(row["pp_seed"]))
    coverage = Counter((state_id, candidate_id) for state_id, candidate_id, _ in seen)
    invalid_coverage = sum(coverage[key] != 8 for key in candidate_keys)
    paired_seed_error_count = sum(len(values) != 1 for values in paired_seeds.values())
    return dict(rows_by_state), {
        "trial_row_count": len(seen),
        "candidate_trial_coverage_error_count": invalid_coverage,
        "paired_seed_error_count": paired_seed_error_count,
    }


def run_stride_stage4r_diagnostic(
    *, config_path: Path, output: Path, project_root: Path
) -> dict[str, Any]:
    config = _read_json(config_path)
    if config.get("schema") != STRIDE_STAGE4R_CONFIG_SCHEMA:
        raise ValueError("unexpected STRIDE Stage 4R config schema")
    project_root = project_root.resolve()
    labels = _project_path(project_root, str(config["labels"]))
    aggregate_path = labels / "candidate_aggregates.jsonl"
    pair_path = labels / "dominance_pairs.jsonl"
    label_summary_path = labels / "label_build_summary.json"
    protocol_report_path = _project_path(
        project_root, str(config["stage4_protocol_report"])
    )
    fold_path = protocol_report_path.parent / "fold_manifest.jsonl"
    training_report_path = _project_path(
        project_root, str(config["stage4_training_report"])
    )
    oof_path = _project_path(project_root, str(config["oof_predictions"]))
    trial_paths = [
        _project_path(project_root, str(value))
        for value in config.get("trial_sources", [])
    ]
    if not trial_paths:
        raise ValueError("Stage 4R requires trial sources")

    expected_hashes = {str(k): str(v) for k, v in dict(config["expected_sha256"]).items()}
    actual_hashes = {
        "candidate_aggregates": sha256_file(aggregate_path),
        "dominance_pairs": sha256_file(pair_path),
        "label_build_summary": sha256_file(label_summary_path),
        "fold_manifest": sha256_file(fold_path),
        "stage4_training_report": sha256_file(training_report_path),
        "oof_predictions": sha256_file(oof_path),
    }
    label_summary = _read_json(label_summary_path)
    grouped = _candidate_groups(aggregate_path)
    state_ids = set(grouped)
    candidates = [candidate for state in grouped.values() for candidate in state]
    candidate_keys = {
        (str(row["state_id"]), str(row["candidate_id"])) for row in candidates
    }
    models = {str(value) for value in config["diagnostic_models"]}
    required_models = {"v2-full", "stride-control-v1", "stride-quality-v1"}
    if models != required_models:
        raise ValueError(f"Stage 4R diagnostic models must be {sorted(required_models)}")
    oof_rows = _read_jsonl(oof_path)
    predictions = _load_oof_predictions(oof_rows, models, state_ids)
    metadata_by_state = {
        str(row["state_id"]): row
        for row in oof_rows
        if str(row["model_id"]) == "stride-quality-v1"
    }
    trial_rows, trial_diagnostics = _load_trials(trial_paths, candidate_keys)
    actual_trial_hashes = [sha256_file(path) for path in trial_paths]
    label_trial_hashes = [str(value) for value in label_summary["trial_sha256"]]
    trial_source_hashes_match_label_summary = actual_trial_hashes == label_trial_hashes

    structure_weight = float(config["structure_weight"])
    halves = [set(map(int, values)) for values in config["seed_halves"]]
    if len(halves) != 2 or halves[0] | halves[1] != set(range(8)) or halves[0] & halves[1]:
        raise ValueError("Stage 4R requires disjoint seed halves covering indices 0..7")

    global_pairs = Counter()
    penalties = []
    reduction_gaps = []
    penalty_gaps = []
    state_rows = []
    seed_correlations = []
    top3_overlaps = []
    control_top3_overlaps = []
    control_front_sizes = []
    raw_oracle_opportunities = []
    normalized_oracle_opportunities = []
    raw_model_gains = []
    normalized_model_gains = []
    changed_count = 0
    improved_count = 0
    worsened_count = 0
    equal_count = 0
    quality_half_winner_match_count = 0
    control_half_winner_match_count = 0
    reduction_half_winner_match_count = 0
    quality_control_winner_match_count = 0
    quality_reduction_winner_match_count = 0
    quality_top_in_control_front_count = 0

    for state_id, state in sorted(grouped.items()):
        by_id = {str(row["candidate_id"]): row for row in state}
        quality_order = [
            str(row["candidate_id"])
            for row in sorted(
                state,
                key=lambda row: (-float(row["quality_score"]), str(row["candidate_id"])),
            )
        ]
        reduction_order = [
            str(row["candidate_id"])
            for row in sorted(
                state,
                key=lambda row: (
                    -float(row["mean_reduction_ratio"]), str(row["candidate_id"])
                ),
            )
        ]
        control_order = _control_order(state)
        control_front = _control_front(state)
        control_front_sizes.append(len(control_front))
        quality_control_winner_match_count += int(quality_order[0] == control_order[0])
        quality_reduction_winner_match_count += int(quality_order[0] == reduction_order[0])
        quality_top_in_control_front_count += int(quality_order[0] in control_front)

        state_pair_counts = Counter()
        for left, right in itertools.combinations(state, 2):
            state_pair_counts["total"] += 1
            left_control = _control_dominates(left, right)
            right_control = _control_dominates(right, left)
            quality_delta = float(left["quality_score"]) - float(right["quality_score"])
            reduction_delta = float(left["mean_reduction_ratio"]) - float(
                right["mean_reduction_ratio"]
            )
            penalty_delta = structure_weight * (
                float(left["structural_score"]) - float(right["structural_score"])
            )
            reduction_gaps.append(abs(reduction_delta))
            penalty_gaps.append(abs(penalty_delta))
            if left_control != right_control:
                state_pair_counts["control_comparable"] += 1
                control_sign = 1 if left_control else -1
                quality_sign = 1 if quality_delta > 1e-12 else -1 if quality_delta < -1e-12 else 0
                if quality_sign == control_sign:
                    state_pair_counts["control_quality_agree"] += 1
                elif quality_sign == 0:
                    state_pair_counts["control_quality_tie"] += 1
                else:
                    state_pair_counts["control_quality_disagree"] += 1
            else:
                state_pair_counts["control_incomparable"] += 1
            reduction_sign = 1 if reduction_delta > 1e-12 else -1 if reduction_delta < -1e-12 else 0
            quality_sign = 1 if quality_delta > 1e-12 else -1 if quality_delta < -1e-12 else 0
            if reduction_sign == 0:
                state_pair_counts["reduction_tie"] += 1
                if quality_sign != 0:
                    state_pair_counts["structure_resolved_reduction_tie"] += 1
            elif quality_sign == reduction_sign:
                state_pair_counts["reduction_quality_agree"] += 1
            elif quality_sign == 0:
                state_pair_counts["reduction_quality_tie"] += 1
            else:
                state_pair_counts["structure_flipped_reduction_order"] += 1
        global_pairs.update(state_pair_counts)
        penalties.extend(
            structure_weight * float(row["structural_score"]) for row in state
        )

        first_half = _aggregate_seed_half(trial_rows[state_id], halves[0], structure_weight)
        second_half = _aggregate_seed_half(trial_rows[state_id], halves[1], structure_weight)
        first_by_id = {str(row["candidate_id"]): row for row in first_half}
        second_by_id = {str(row["candidate_id"]): row for row in second_half}
        if set(first_by_id) != set(by_id) or set(second_by_id) != set(by_id):
            raise ValueError(f"seed-half candidate coverage differs: {state_id}")
        first_quality = {
            key: float(row["quality_score"]) for key, row in first_by_id.items()
        }
        second_quality = {
            key: float(row["quality_score"]) for key, row in second_by_id.items()
        }
        correlation = _rank_correlation(first_quality, second_quality)
        seed_correlations.append(correlation)
        first_quality_order = sorted(first_quality, key=lambda key: (-first_quality[key], key))
        second_quality_order = sorted(second_quality, key=lambda key: (-second_quality[key], key))
        quality_winner_match = first_quality_order[0] == second_quality_order[0]
        quality_half_winner_match_count += int(quality_winner_match)
        quality_top3_overlap = len(
            set(first_quality_order[:3]) & set(second_quality_order[:3])
        ) / min(3, len(state))
        top3_overlaps.append(quality_top3_overlap)
        first_control = _control_order(first_half)
        second_control = _control_order(second_half)
        control_winner_match = first_control[0] == second_control[0]
        control_half_winner_match_count += int(control_winner_match)
        control_top3_overlap = len(
            set(first_control[:3]) & set(second_control[:3])
        ) / min(3, len(state))
        control_top3_overlaps.append(control_top3_overlap)
        first_reduction = sorted(
            first_by_id,
            key=lambda key: (-float(first_by_id[key]["mean_reduction_ratio"]), key),
        )
        second_reduction = sorted(
            second_by_id,
            key=lambda key: (-float(second_by_id[key]["mean_reduction_ratio"]), key),
        )
        reduction_winner_match = first_reduction[0] == second_reduction[0]
        reduction_half_winner_match_count += int(reduction_winner_match)

        best_quality = float(by_id[quality_order[0]]["quality_score"])
        worst_quality = min(float(row["quality_score"]) for row in state)
        quality_range = best_quality - worst_quality
        control_selected = by_id[predictions["stride-control-v1"][state_id]]
        quality_selected = by_id[predictions["stride-quality-v1"][state_id]]
        frozen_selected = by_id[predictions["v2-full"][state_id]]
        control_regret = max(0.0, best_quality - float(control_selected["quality_score"]))
        quality_regret = max(0.0, best_quality - float(quality_selected["quality_score"]))
        frozen_regret = max(0.0, best_quality - float(frozen_selected["quality_score"]))
        normalized_control = control_regret / quality_range if quality_range > 1e-12 else 0.0
        normalized_quality = quality_regret / quality_range if quality_range > 1e-12 else 0.0
        normalized_frozen = frozen_regret / quality_range if quality_range > 1e-12 else 0.0
        raw_gain = control_regret - quality_regret
        normalized_gain = normalized_control - normalized_quality
        raw_oracle_opportunities.append(control_regret)
        normalized_oracle_opportunities.append(normalized_control)
        raw_model_gains.append(raw_gain)
        normalized_model_gains.append(normalized_gain)
        changed = str(control_selected["candidate_id"]) != str(quality_selected["candidate_id"])
        changed_count += int(changed)
        if raw_gain > 1e-12:
            improved_count += 1
        elif raw_gain < -1e-12:
            worsened_count += 1
        else:
            equal_count += 1

        metadata = metadata_by_state[state_id]
        state_rows.append(
            {
                "schema": STRIDE_STAGE4R_STATE_SCHEMA,
                "state_id": state_id,
                "map_id": str(metadata["map_id"]),
                "fold_index": int(metadata["fold_index"]),
                "layout_family": str(metadata["layout_family"]),
                "source_policy": str(metadata["source_policy"]),
                "agent_band": str(metadata["agent_band"]),
                "decision_stage": str(metadata["decision_stage"]),
                "candidate_count": len(state),
                "control_front_size": len(control_front),
                "quality_winner": quality_order[0],
                "control_oracle_winner": control_order[0],
                "reduction_winner": reduction_order[0],
                "quality_winner_in_control_front": quality_order[0] in control_front,
                "control_quality_selection_changed": changed,
                "control_normalized_regret": normalized_control,
                "quality_normalized_regret": normalized_quality,
                "frozen_normalized_regret": normalized_frozen,
                "quality_model_gain_over_control": raw_gain,
                "quality_model_normalized_gain_over_control": normalized_gain,
                "seed_half_quality_rank_correlation": correlation,
                "seed_half_quality_winner_match": quality_winner_match,
                "seed_half_quality_top3_overlap": quality_top3_overlap,
                "seed_half_control_winner_match": control_winner_match,
                "seed_half_control_top3_overlap": control_top3_overlap,
                "seed_half_reduction_winner_match": reduction_winner_match,
                "pair_counts": dict(state_pair_counts),
            }
        )

    state_count = len(grouped)
    comparable = global_pairs["control_comparable"]
    reduction_ordered = (
        global_pairs["reduction_quality_agree"]
        + global_pairs["structure_flipped_reduction_order"]
        + global_pairs["reduction_quality_tie"]
    )
    total_control_regret = math.fsum(raw_oracle_opportunities)
    total_normalized_control_regret = math.fsum(normalized_oracle_opportunities)
    report = {
        "schema": STRIDE_STAGE4R_REPORT_SCHEMA,
        "diagnostic_only": True,
        "promotion_decision": "not_made",
        "runtime_fields_used": False,
        "formal_ood_data_read": False,
        "test_data_read": False,
        "state_count": state_count,
        "candidate_count": len(candidates),
        "pair_relations": {
            **dict(global_pairs),
            "control_quality_agreement_rate_comparable": (
                global_pairs["control_quality_agree"] / comparable if comparable else 0.0
            ),
            "control_quality_disagreement_rate_comparable": (
                global_pairs["control_quality_disagree"] / comparable if comparable else 0.0
            ),
            "structure_flip_rate_reduction_ordered": (
                global_pairs["structure_flipped_reduction_order"] / reduction_ordered
                if reduction_ordered
                else 0.0
            ),
        },
        "state_winners": {
            "quality_control_exact_agreement_rate": quality_control_winner_match_count / state_count,
            "quality_reduction_exact_agreement_rate": quality_reduction_winner_match_count / state_count,
            "quality_winner_in_control_front_rate": quality_top_in_control_front_count / state_count,
            "control_front_size": _number_summary(control_front_sizes),
        },
        "structure_contribution": {
            "candidate_penalty": _number_summary(penalties),
            "absolute_pair_reduction_ratio_gap": _number_summary(reduction_gaps),
            "absolute_pair_structure_penalty_gap": _number_summary(penalty_gaps),
        },
        "oof_oracle_opportunity": {
            "control_selection_change_rate": changed_count / state_count,
            "quality_better_state_count": improved_count,
            "quality_worse_state_count": worsened_count,
            "quality_equal_state_count": equal_count,
            "control_raw_oracle_opportunity": _number_summary(raw_oracle_opportunities),
            "control_normalized_oracle_opportunity": _number_summary(
                normalized_oracle_opportunities
            ),
            "quality_raw_gain_over_control": _number_summary(raw_model_gains),
            "quality_normalized_gain_over_control": _number_summary(
                normalized_model_gains
            ),
            "aggregate_raw_opportunity_capture_fraction": (
                math.fsum(raw_model_gains) / total_control_regret
                if total_control_regret > 1e-12
                else 0.0
            ),
            "aggregate_normalized_opportunity_capture_fraction": (
                math.fsum(normalized_model_gains) / total_normalized_control_regret
                if total_normalized_control_regret > 1e-12
                else 0.0
            ),
        },
        "seed_half_stability": {
            "quality_exact_winner_agreement_rate": quality_half_winner_match_count / state_count,
            "quality_top3_overlap": _number_summary(top3_overlaps),
            "quality_rank_correlation": _number_summary(seed_correlations),
            "control_exact_winner_agreement_rate": control_half_winner_match_count / state_count,
            "control_top3_overlap": _number_summary(control_top3_overlaps),
            "reduction_exact_winner_agreement_rate": reduction_half_winner_match_count / state_count,
        },
        "integrity": {
            "passed": (
                actual_hashes == expected_hashes
                and state_count == int(config["expected_state_count"])
                and len(candidates) == int(config["expected_candidate_count"])
                and int(label_summary["trials_per_candidate"])
                == int(config["expected_trials_per_candidate"])
                and math.isclose(
                    float(label_summary["structure_weight"]),
                    structure_weight,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and trial_source_hashes_match_label_summary
                and trial_diagnostics["candidate_trial_coverage_error_count"] == 0
                and trial_diagnostics["paired_seed_error_count"] == 0
            ),
            "artifact_hashes_match": actual_hashes == expected_hashes,
            "trial_source_hashes_match_label_summary": (
                trial_source_hashes_match_label_summary
            ),
            "trial_diagnostics": trial_diagnostics,
        },
        "inputs": {
            "config_sha256": _fingerprint(config),
            "artifact_sha256": actual_hashes,
            "trial_source_sha256": actual_trial_hashes,
            "label_trial_source_sha256": label_trial_hashes,
        },
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state_diagnostics.jsonl"
    _write_jsonl(state_path, state_rows)
    report["artifacts"] = {
        "state_diagnostics": "state_diagnostics.jsonl",
        "state_diagnostics_sha256": sha256_file(state_path),
    }
    report["passed"] = bool(report["integrity"]["passed"])
    _write_json(output / "stage4r_diagnostic_report.json", report)
    return report


__all__ = [
    "STRIDE_STAGE4R_CONFIG_SCHEMA",
    "STRIDE_STAGE4R_REPORT_SCHEMA",
    "run_stride_stage4r_diagnostic",
]
