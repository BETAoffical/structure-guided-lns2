from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.stride_cycletransition import (
    _aggregate_candidate,
    _edge_metrics,
    _quality_admissible,
    _select_pilot_occurrences,
    _spearman,
    validate_registration,
)


ROOT = Path(__file__).resolve().parents[2]


def test_registration_is_frozen() -> None:
    payload = json.loads(
        (ROOT / "configs/stride_cycletransition_pilot_v1_registration.json").read_text(
            encoding="utf-8"
        )
    )
    validate_registration(payload)
    assert payload["decision"]["risk_only_selector_allowed"] is False
    assert payload["decision"]["online_tentative_pp_probe_allowed"] is False
    assert payload["candidate_contract"]["six_candidate_reduction_allowed"] is False
    assert payload["execution"]["job_unit"] == (
        "one state, one candidate, and one paired PP trial"
    )
    assert payload["execution"]["per_job_timeout_seconds"] == 300


def test_result_blind_map_balanced_selection() -> None:
    config = {
        "cohort": {
            "source_checkpoint_kind": "first_structural_selection",
            "maps": ["maze-128-128-1", "maze-128-128-2", "maze-32-32-4"],
            "states_per_map": 6,
            "state_count": 18,
        }
    }
    rows = []
    for map_id in config["cohort"]["maps"]:
        for index in range(8):
            rows.append(
                {
                    "checkpoint_kind": "first_structural_selection",
                    "map_id": map_id,
                    "state_fingerprint": f"{map_id}-{index}",
                    "logical_checkpoint_id": f"{map_id}-{index}",
                    "classification": "must-not-matter",
                }
            )
    first = _select_pilot_occurrences(rows, config)
    second = _select_pilot_occurrences(list(reversed(rows)), config)
    assert [row["state_fingerprint"] for row in first] == [
        row["state_fingerprint"] for row in second
    ]
    assert len(first) == 18


def test_edge_metrics_distinguish_persistence_and_new_edges() -> None:
    metrics = _edge_metrics({(1, 2), (2, 3)}, {(2, 3), (3, 4)})
    assert metrics == {
        "conflict_edge_exact_recurrence": False,
        "unresolved_edge_fraction": 0.5,
        "new_edge_fraction": 0.5,
        "conflict_edge_jaccard": 1 / 3,
    }


def test_quality_is_a_hard_constraint_not_a_risk_tradeoff() -> None:
    quality = {
        "minimum_seed_mean_delta": 0.0,
        "maximum_no_progress_rate_delta": 0.0,
        "minimum_each_fixed_four_seed_half_delta": -0.02,
    }
    anchor = {
        "seed_mean": 0.3,
        "no_progress_rate": 0.25,
        "first_half_mean": 0.3,
        "second_half_mean": 0.3,
    }
    safe_but_weak = {
        "seed_mean": 0.29,
        "no_progress_rate": 0.0,
        "first_half_mean": 0.29,
        "second_half_mean": 0.29,
    }
    admissible = {
        "seed_mean": 0.31,
        "no_progress_rate": 0.25,
        "first_half_mean": 0.28,
        "second_half_mean": 0.31,
    }
    assert not _quality_admissible(safe_but_weak, anchor, quality)
    assert _quality_admissible(admissible, anchor, quality)


def test_candidate_aggregate_and_rank_stability() -> None:
    trials = []
    for index in range(8):
        trials.append(
            {
                "trial_index": index,
                "normalized_conflict_reduction": 0.25,
                "repair_exact_noop": index in {0, 1, 4, 5},
                "conflicts_after": 8,
                "before_conflicts": 10,
                "exact_policy_cycle": False,
                "soft_policy_cycle": index in {0, 1, 4, 5},
                "unresolved_edge_fraction": 0.5,
                "new_edge_fraction": 0.2,
                "next_full_pool_selected_agent_jaccard": 0.1,
            }
        )
    row = _aggregate_candidate(
        {
            "candidate_id": "candidate-a",
            "candidate_kind": "structural",
            "actual_size": 16,
            "nominal_sizes": [16],
        },
        trials,
    )
    assert row["exact_noop_rate"] == 0.5
    assert row["half_exact_noop_class_agreement"] is True
    assert row["soft_policy_cycle_rate"] == 0.5
    assert row["half_soft_policy_cycle_class_agreement"] is True
    assert math.isclose(
        _spearman([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]), 1.0
    )
