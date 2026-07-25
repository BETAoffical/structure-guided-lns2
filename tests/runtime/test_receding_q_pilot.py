from __future__ import annotations

import copy

import pytest

from experiments import receding_q_pilot as module


def _qualification() -> dict:
    return {
        "decision": {
            "temporal_context": {
                "history.available_steps": 2.0,
            }
        },
        "candidates": [
            {"candidate_id": "a", "agents": [2, 1]},
            {"candidate_id": "b", "agents": [4, 3]},
        ],
        "candidate_rows": [
            {
                "candidate_id": "a",
                "features": {"realized_dynamic": {"x": 1.0}},
            },
            {
                "candidate_id": "b",
                "features": {"realized_dynamic": {"x": 2.0}},
            },
        ],
    }


def test_actual_candidates_use_real_neighborhood_identity_and_features() -> None:
    arms = module.actual_candidate_arms(
        _qualification(),
        feature_names=("x", "history.available_steps"),
    )
    assert [row["candidate_id"] for row in arms] == ["a", "b"]
    assert arms[0]["agents"] == [1, 2]
    assert arms[0]["feature_values"] == [1.0, 2.0]


def test_actual_candidates_reject_duplicate_agent_sets() -> None:
    payload = _qualification()
    payload["candidates"][1]["agents"] = [1, 2]
    with pytest.raises(ValueError, match="duplicate actual neighborhoods"):
        module.actual_candidate_arms(
            payload,
            feature_names=("x", "history.available_steps"),
        )


def test_fixed_horizon_keeps_noop_and_pads_only_terminal_tail() -> None:
    labels = module._fixed_horizon_labels(
        initial_conflicts=10,
        trajectory=[10, 10, 7],
        horizon=3,
        feasible=False,
        selection_wall_seconds=0.1,
        repair_wall_seconds=0.4,
        conflict_wall_auc_seconds=3.5,
    )
    assert labels["padded_steps"] == 1
    assert labels["final_conflicts"] == 7
    assert labels["no_progress"] is False
    assert labels["observed_total_seconds"] == pytest.approx(0.5)


class _FakeEnvironment:
    def __init__(self) -> None:
        self.calls = []

    def step(self, action: dict) -> dict:
        self.calls.append(copy.deepcopy(action))
        before = 5 if len(self.calls) == 1 else 4
        after = 4 if len(self.calls) == 1 else 3
        observation = {
            "num_of_colliding_pairs": after,
            "feasible": False,
            "done": False,
            "revision": len(self.calls),
            "paths": [[0, len(self.calls)]],
            "conflict_edges": [],
            "low_level": {
                "generated": len(self.calls),
                "expanded": len(self.calls),
                "reopened": 0,
                "runs": len(self.calls),
            },
        }
        return {
            "observation": observation,
            "metrics": {
                "replan_success": True,
                "requested_pp_random_seed": action["pp_random_seed"],
                "applied_pp_random_seed": action["pp_random_seed"],
                "repair_order": [0],
                "pp_replan_seconds": 0.01,
                "native_neighborhood_generation_seconds": 0.001,
                "neighborhood": [0],
                "conflicts_before": before,
                "conflicts_after": after,
            },
        }


def test_rollout_executes_root_then_replanned_adaptive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = {
        "num_of_colliding_pairs": 5,
        "feasible": False,
        "done": False,
        "revision": 0,
        "paths": [[0]],
        "conflict_edges": [],
        "low_level": {
            "generated": 0,
            "expanded": 0,
            "reopened": 0,
            "runs": 0,
        },
    }
    environment = _FakeEnvironment()
    monkeypatch.setattr(
        module,
        "_source_replay_job",
        lambda _decision: ({"environment": {}}, {"proposal": {}}),
    )
    monkeypatch.setattr(
        module,
        "_full_candidate_rows",
        lambda _environment, _state, _decision, _proposal: (
            [{"candidate_id": "shadow"}],
            [{}],
            {"full_pool_seconds": 0.03},
        ),
    )
    monkeypatch.setattr(
        module,
        "replay_prefix",
        lambda _replay, _prefix: (environment, copy.deepcopy(initial)),
    )
    monkeypatch.setattr(
        module, "state_fingerprint", lambda state: f"full-{state['revision']}"
    )
    monkeypatch.setattr(
        module,
        "repair_structure_fingerprint",
        lambda state: f"repair-{state['revision']}",
    )
    monkeypatch.setattr(
        module,
        "_paired_seed",
        lambda _fingerprint, trial, step: 1000 + 10 * trial + step,
    )
    monkeypatch.setattr(
        module,
        "_low_level_delta",
        lambda before, after: {
            name: int(after["low_level"][name])
            - int(before["low_level"][name])
            for name in ("generated", "expanded", "reopened", "runs")
        },
    )
    job = {
        "state": {
            "state_id": "state",
            "split": "policy_train",
            "map_id": "map",
            "layout_mode": "layout",
            "agent_count": 100,
            "source_stratum": "ordinary",
            "initial_conflicts": 5,
            "before_fingerprint": "full-0",
            "before_repair_fingerprint": "repair-0",
            "root_selection_seconds": 0.2,
            "v2_candidate_id": "candidate",
            "decision": {"prefix_actions": [], "decision_index": 0},
            "arms": [{"candidate_id": "candidate"}],
        },
        "arm": {
            "candidate_id": "candidate",
            "agents": [1, 2, 3, 4],
            "actual_size": 4,
            "selection_families": ["collision:4"],
            "feature_values": [0.0] * len(module.RECEDING_Q_FEATURE_NAMES),
        },
        "trial_index": 0,
        "horizon": 2,
        "continuation_teacher": "official_adaptive",
    }
    result = module.run_receding_q_rollout(job)
    assert environment.calls[0]["mode"] == "explicit_neighborhood"
    assert environment.calls[0]["agents"] == [1, 2, 3, 4]
    assert environment.calls[1]["mode"] == "official"
    assert [step["requested_pp_seed"] for step in result["steps"]] == [
        1001,
        1002,
    ]
    assert result["executed_steps"] == 2
    assert result["selection_wall_seconds"] == pytest.approx(0.23)


def _analysis_plan() -> dict:
    return {
        "feature_names": ["x"],
        "states": [
            {
                "state_id": "state",
                "before_fingerprint": "before",
                "arms": [
                    {
                        "candidate_id": "a",
                        "agents": [1, 2, 3, 4],
                        "feature_values": [1.0],
                    },
                    {
                        "candidate_id": "b",
                        "agents": [5, 6, 7, 8],
                        "feature_values": [1.0],
                    },
                ],
            }
        ],
    }


def _analysis_row(candidate: str, trial: int, auc: float) -> dict:
    initial_repair = f"repair-{trial}"
    seed = module._paired_seed(initial_repair, trial, 1)
    return {
        "state_id": "state",
        "candidate_id": candidate,
        "agents": [1, 2, 3, 4] if candidate == "a" else [5, 6, 7, 8],
        "trial_index": trial,
        "feature_names": ["x"],
        "feature_values": [1.0],
        "executed_steps": 1,
        "initial_conflicts": 10,
        "initial_fingerprint": "before",
        "initial_repair_fingerprint": initial_repair,
        "conflict_trajectory": [10, int(10 * auc)],
        "steps": [
            {
                "step": 1,
                "route": "explicit_root_candidate",
                "candidate_id": candidate,
                "agents": (
                    [1, 2, 3, 4]
                    if candidate == "a"
                    else [5, 6, 7, 8]
                ),
                "before_fingerprint": "before",
                "after_fingerprint": f"after-{candidate}-{trial}",
                "conflicts_before": 10,
                "conflicts_after": int(10 * auc),
                "requested_pp_seed": seed,
            }
        ],
        "feasible": False,
        "final_conflict_ratio": auc,
        "normalized_step_auc": auc,
        "normalized_wall_auc_seconds": auc,
        "observed_total_seconds": auc,
        "continuation_teacher": "official_adaptive",
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 100,
    }


def test_analysis_requires_complete_candidate_trial_matrix() -> None:
    rows = [
        _analysis_row("a", 0, 0.5),
        _analysis_row("b", 0, 0.7),
        _analysis_row("a", 1, 0.5),
        _analysis_row("b", 1, 0.7),
    ]
    report, states = module.analyze_receding_q_rollouts(
        rows,
        plan=_analysis_plan(),
        trials=2,
        horizon=3,
        smoke_only=True,
    )
    assert report["decision"] == "smoke_completed_not_scientific"
    assert report["winner_seed_agreement"] == 1.0
    assert states[0]["candidate_count"] == 2

    with pytest.raises(ValueError, match="coverage mismatch"):
        module.analyze_receding_q_rollouts(
            rows[:-1],
            plan=_analysis_plan(),
            trials=2,
            horizon=3,
            smoke_only=True,
        )
