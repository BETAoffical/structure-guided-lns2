from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.stride_successor_repairability import (
    aggregate_candidate,
    paired_auc_direction,
    select_preflight_occurrences,
    semantic_action_id,
    summarize_occurrence,
    validate_registration,
)


ROOT = Path(__file__).resolve().parents[2]


def test_registration_freezes_probe_and_claim_boundary() -> None:
    payload = json.loads(
        (
            ROOT
            / "configs"
            / "stride_successor_repairability_v1_registration.json"
        ).read_text(encoding="utf-8")
    )
    validate_registration(payload)
    assert payload["execution"]["collection_workers"] == 16
    assert payload["execution"]["per_candidate_trial_timeout_seconds"] == 300
    assert payload["candidate_probe"]["structpool_allowed"] is False
    assert payload["candidate_probe"]["neighborhood_sizes"] == [4, 8, 16]
    assert payload["claim_boundary"]["model_training_allowed"] is False
    assert payload["claim_boundary"]["ttf_experiment_allowed"] is False


def test_semantic_action_cache_ignores_occurrence_and_candidate_names() -> None:
    first = semantic_action_id(
        task_id="task-a",
        repair_fingerprint="repair-a",
        repair_semantics="pp-sipps",
        agents=[4, 1, 3],
    )
    second = semantic_action_id(
        task_id="task-a",
        repair_fingerprint="repair-a",
        repair_semantics="pp-sipps",
        agents=[3, 4, 1],
    )
    different = semantic_action_id(
        task_id="task-a",
        repair_fingerprint="repair-a",
        repair_semantics="pp-sipps",
        agents=[3, 4, 2],
    )
    assert first == second
    assert first != different


def test_preflight_selection_is_map_balanced_and_outcome_blind() -> None:
    rows = []
    for map_id in ("map-a", "map-b", "map-c"):
        for index in range(4):
            rows.append(
                {
                    "map_id": map_id,
                    "semantic_state_id": f"{map_id}-state-{index}",
                    "occurrence_id": f"{map_id}-occurrence-{index}",
                    "bounded_success": index % 2 == 0,
                    "bounded_normalized_fixed_auc": float(index),
                }
            )
    first = select_preflight_occurrences(rows, state_count=3)
    changed = [
        {
            **row,
            "bounded_success": not row["bounded_success"],
            "bounded_normalized_fixed_auc": 100.0 - row["bounded_normalized_fixed_auc"],
        }
        for row in reversed(rows)
    ]
    second = select_preflight_occurrences(changed, state_count=3)
    assert [row["occurrence_id"] for row in first] == [
        row["occurrence_id"] for row in second
    ]
    assert {row["map_id"] for row in first} == {"map-a", "map-b", "map-c"}


def _candidate(candidate_id: str, *, anchor: bool) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "semantic_action_id": f"semantic-{candidate_id}",
        "agents": [1, 2, 3],
        "actual_size": 3,
        "selection_families": ["target:4"],
        "is_v2_anchor": anchor,
    }


def _trials(changes: list[bool], drops: list[bool]) -> list[dict[str, object]]:
    return [
        {
            "trial_index": index,
            "repair_state_changed": changes[index],
            "strict_conflict_drop": drops[index],
            "no_progress": not drops[index],
            "replan_success": changes[index],
            "normalized_conflict_reduction": 0.1 if drops[index] else 0.0,
            "after_repair_fingerprint": f"after-{index if changes[index] else 'same'}",
        }
        for index in range(8)
    ]


def test_candidate_distribution_and_half_stability_are_not_binary_tail_labels() -> None:
    anchor = aggregate_candidate(
        _candidate("anchor", anchor=True),
        _trials([True] * 8, [True, True, True, False] * 2),
        first_half=(0, 1, 2, 3),
        second_half=(4, 5, 6, 7),
    )
    challenger = aggregate_candidate(
        _candidate("challenger", anchor=False),
        _trials([True, True, False, False] * 2, [True, False, False, False] * 2),
        first_half=(0, 1, 2, 3),
        second_half=(4, 5, 6, 7),
    )
    occurrence = {
        "occurrence_id": "occurrence",
        "case_id": "case",
        "trial_index": 0,
        "arm": "actual_selected",
        "map_id": "map",
        "task_id": "task",
        "successor_fingerprint": "full",
        "successor_repair_fingerprint": "repair",
        "semantic_state_id": "semantic-state",
        "successor_conflicts": 10,
    }
    summary = summarize_occurrence(occurrence, [anchor, challenger])
    assert anchor["repair_state_change_rate"] == 1.0
    assert anchor["strict_conflict_drop_rate"] == 0.75
    assert challenger["repair_state_change_rate"] == 0.5
    assert summary["v2_anchor"]["candidate_id"] == "anchor"
    assert summary["top3_overlap"] == 1.0
    assert math.isclose(summary["repair_change_rank_correlation"], 1.0)
    assert math.isclose(summary["strict_drop_rank_correlation"], 1.0)
    assert math.isclose(summary["pool_mean_strict_conflict_drop_rate"], 0.5)


def test_paired_auc_direction_uses_only_informative_within_case_pairs() -> None:
    rows = [
        {
            "case_id": "case-a",
            "trial_index": 0,
            "repairability_burden": 0.8,
            "bounded_normalized_fixed_auc": 0.5,
        },
        {
            "case_id": "case-a",
            "trial_index": 0,
            "repairability_burden": 0.2,
            "bounded_normalized_fixed_auc": 0.1,
        },
        {
            "case_id": "case-a",
            "trial_index": 0,
            "repairability_burden": 0.2,
            "bounded_normalized_fixed_auc": 0.4,
        },
    ]
    result = paired_auc_direction(rows)
    assert result["informative_pair_count"] == 2
    assert result["concordant_pair_count"] == 2
    assert result["direction_concordance"] == 1.0
