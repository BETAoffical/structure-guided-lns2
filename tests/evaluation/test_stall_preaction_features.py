from __future__ import annotations

from experiments.stall_preaction_features import temporal_candidate_features
from experiments.stall_shadow import neighborhood_key


def _history(*, changed: bool, agents: list[int], seed: int = 1) -> dict:
    return {
        "before_repair_fingerprint": "same",
        "after_repair_fingerprint": "other" if changed else "same",
        "state_changed": changed,
        "no_progress": not changed,
        "conflict_reduction": 2 if changed else 0,
        "agents": agents,
        "neighborhood_key": neighborhood_key(agents),
        "pp_random_seed": seed,
        "repair_order": list(reversed(agents)),
    }


def test_temporal_features_use_only_prior_same_state_no_progress() -> None:
    history = [
        _history(changed=True, agents=[8, 9]),
        _history(changed=False, agents=[1, 2], seed=2),
        _history(changed=False, agents=[1, 3], seed=3),
    ]
    values = temporal_candidate_features(
        history=history,
        current_repair_fingerprint="same",
        current_conflicts=10,
        agent_count=100,
        candidate_agents=[1, 2],
        candidate_rank=2,
        candidate_score=4.0,
        rank1_score=5.0,
        current_edges={(1, 2), (2, 4), (8, 9)},
        previous_edge_sets=[{(1, 2), (8, 9)}, {(1, 2), (2, 4)}],
    )
    assert values["history.same_state_no_progress_attempts"] == 2.0
    assert values["history.same_state_distinct_pp_orders"] == 2.0
    assert values["candidate.exact_prior_no_progress_count"] == 1.0
    assert values["candidate.persistent_edge_coverage"] == 1.0
    assert values["state.persistent_edge_fraction"] == 1.0 / 3.0


def test_state_change_breaks_same_state_failure_run() -> None:
    history = [
        _history(changed=False, agents=[1, 2]),
        _history(changed=True, agents=[3, 4]),
    ]
    values = temporal_candidate_features(
        history=history,
        current_repair_fingerprint="same",
        current_conflicts=4,
        agent_count=20,
        candidate_agents=[1, 2],
        candidate_rank=1,
        candidate_score=2.0,
        rank1_score=2.0,
        current_edges={(1, 2)},
        previous_edge_sets=[],
    )
    assert values["history.same_state_no_progress_attempts"] == 0.0
    assert values["candidate.overlap_failed_union"] == 0.0
    assert values["state.persistent_edge_fraction"] == 0.0
