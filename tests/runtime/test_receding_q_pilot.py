from __future__ import annotations

import copy

import pytest

from experiments import receding_q_pilot as module


def test_persisted_rollout_schema_is_v2() -> None:
    assert module.RECEDING_Q_PILOT_SCHEMA == "lns2.receding_q_label_pilot.v2"


def test_old_schema_resume_fails_before_writing_artifacts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_plan = {
        "schema": module.RECEDING_Q_PILOT_SCHEMA,
        "feature_names": list(module.RECEDING_Q_FEATURE_NAMES),
        "states": [],
    }
    plan_path = tmp_path / "plan.json"
    config_path = tmp_path / "run_config.json"
    module._write_json(plan_path, requested_plan)
    module._write_json(
        config_path,
        {"schema": "lns2.receding_q_label_pilot.v1"},
    )
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    monkeypatch.setattr(
        module,
        "producer_identity",
        lambda **_kwargs: {
            "schema": "lns2.producer_identity.v1",
            "test": "identity",
        },
    )
    monkeypatch.setattr(
        module,
        "build_receding_q_plan",
        lambda **_kwargs: requested_plan,
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        module.run_receding_q_label_pilot(
            source="unused",
            output=tmp_path,
            state_count=1,
            smoke_only=True,
            resume=True,
        )
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before


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
        neighborhood = list(action.get("agents", ())) or [0]
        return {
            "observation": observation,
            "metrics": {
                "step_applied": True,
                "replan_success": True,
                "requested_pp_random_seed": action["pp_random_seed"],
                "applied_pp_random_seed": action["pp_random_seed"],
                "repair_order": list(neighborhood),
                "pp_replan_seconds": 0.01,
                "native_neighborhood_generation_seconds": 0.001,
                "neighborhood": list(neighborhood),
                "conflicts_before": before,
                "conflicts_after": after,
            },
            "terminated": False,
            "truncated": False,
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
        "producer_identity_fingerprint": "producer",
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
    corrupt = copy.deepcopy(result)
    corrupt["steps"][1]["agents"] = None
    with pytest.raises(ValueError, match="agents must be a non-empty list"):
        module.validate_receding_q_rollout(
            corrupt,
            state_plan=job["state"],
            arm_plan=job["arm"],
            feature_names=module.RECEDING_Q_FEATURE_NAMES,
            horizon=2,
            continuation_teacher="official_adaptive",
            expected_trial_index=0,
            expected_producer_fingerprint="producer",
        )
    corrupt = copy.deepcopy(result)
    corrupt["steps"][1]["repair_order"] = [1]
    with pytest.raises(ValueError, match="repair order does not cover"):
        module.validate_receding_q_rollout(
            corrupt,
            state_plan=job["state"],
            arm_plan=job["arm"],
            feature_names=module.RECEDING_Q_FEATURE_NAMES,
            horizon=2,
            continuation_teacher="official_adaptive",
            expected_trial_index=0,
            expected_producer_fingerprint="producer",
        )


def _analysis_plan() -> dict:
    return {
        "feature_names": ["x"],
        "states": [
            {
                "state_id": "state",
                "split": "policy_train",
                "map_id": "map",
                "layout_mode": "layout",
                "agent_count": 100,
                "source_stratum": "ordinary",
                "initial_conflicts": 10,
                "before_fingerprint": "before",
                "before_repair_fingerprint": "repair-before",
                "root_selection_seconds": 0.02,
                "v2_candidate_id": "a",
                "arms": [
                    {
                        "candidate_id": "a",
                        "agents": [1, 2, 3, 4],
                        "actual_size": 4,
                        "selection_families": ["collision:4"],
                        "feature_values": [1.0],
                    },
                    {
                        "candidate_id": "b",
                        "agents": [5, 6, 7, 8],
                        "actual_size": 4,
                        "selection_families": ["collision:4"],
                        "feature_values": [1.0],
                    },
                ],
            }
        ],
    }


def _analysis_row(candidate: str, trial: int, auc: float) -> dict:
    initial_repair = "repair-before"
    seed = module._paired_seed(initial_repair, trial, 1)
    agents = [1, 2, 3, 4] if candidate == "a" else [5, 6, 7, 8]
    final_conflicts = int(10 * auc)
    selection_seconds = 0.02
    repair_seconds = 0.03
    labels = module._fixed_horizon_labels(
        initial_conflicts=10,
        trajectory=[10, final_conflicts],
        horizon=3,
        feasible=False,
        selection_wall_seconds=selection_seconds,
        repair_wall_seconds=repair_seconds,
        conflict_wall_auc_seconds=10 * (
            selection_seconds + repair_seconds
        ),
    )
    return {
        "schema": module.RECEDING_Q_PILOT_SCHEMA,
        "producer_identity_fingerprint": "producer",
        "complete": True,
        "state_id": "state",
        "candidate_id": candidate,
        "agents": agents,
        "actual_size": 4,
        "selection_families": ["collision:4"],
        "is_v2_candidate": candidate == "a",
        "trial_index": trial,
        "feature_names": ["x"],
        "feature_values": [1.0],
        "executed_steps": 1,
        "initial_conflicts": 10,
        "initial_fingerprint": "before",
        "initial_repair_fingerprint": initial_repair,
        "final_fingerprint": f"after-{candidate}-{trial}",
        "final_repair_fingerprint": f"repair-after-{candidate}-{trial}",
        "conflict_trajectory": [10, final_conflicts],
        "steps": [
            {
                "step": 1,
                "route": "explicit_root_candidate",
                "candidate_id": candidate,
                "agents": agents,
                "repair_order": agents,
                "action": {
                    "mode": "explicit_neighborhood",
                    "agents": agents,
                    "pp_random_seed": seed,
                },
                "before_fingerprint": "before",
                "after_fingerprint": f"after-{candidate}-{trial}",
                "before_repair_fingerprint": initial_repair,
                "after_repair_fingerprint": (
                    f"repair-after-{candidate}-{trial}"
                ),
                "conflicts_before": 10,
                "conflicts_after": final_conflicts,
                "conflict_reduction": 10 - final_conflicts,
                "after_done": True,
                "after_feasible": False,
                "step_applied": True,
                "terminated": False,
                "truncated": True,
                "replan_success": True,
                "requested_pp_seed": seed,
                "applied_pp_seed": seed,
                "selection_seconds": selection_seconds,
                "shadow_candidate_count": 2,
                "solver_step_wall_seconds": 0.02,
                "iteration_wall_seconds": repair_seconds,
                "pp_replan_seconds": 0.01,
                "repair_outcome": "conflict_reduced",
                "low_level": {
                    "generated": 5,
                    "expanded": 4,
                    "reopened": 0,
                    "runs": 1,
                },
            }
        ],
        "stop_reason": "environment_terminal",
        "root_selection_seconds": selection_seconds,
        "selection_wall_seconds": selection_seconds,
        "repair_wall_seconds": repair_seconds,
        "pp_replan_seconds": 0.01,
        "low_level": {
            "generated": 5,
            "expanded": 4,
            "reopened": 0,
            "runs": 1,
        },
        "continuation_teacher": "official_adaptive",
        "split": "policy_train",
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 100,
        "source_stratum": "ordinary",
        **labels,
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


def test_empty_repair_order_is_valid_only_for_a_hard_failure() -> None:
    plan = _analysis_plan()
    state = dict(plan["states"][0])
    arm = dict(state["arms"][0])
    row = _analysis_row("a", 0, 1.0)
    step = row["steps"][0]
    step["repair_order"] = []
    step["replan_success"] = False
    step["repair_outcome"] = "hard_failure"
    step["applied_pp_seed"] = -1
    step["after_repair_fingerprint"] = row["initial_repair_fingerprint"]
    row["final_repair_fingerprint"] = row["initial_repair_fingerprint"]
    module.validate_receding_q_rollout(
        row,
        state_plan=state,
        arm_plan=arm,
        feature_names=["x"],
        horizon=3,
        continuation_teacher="official_adaptive",
        expected_trial_index=0,
        expected_producer_fingerprint="producer",
    )

    step["replan_success"] = True
    step["repair_outcome"] = "accepted_noop"
    with pytest.raises(ValueError, match="only valid for a hard failure"):
        module.validate_receding_q_rollout(
            row,
            state_plan=state,
            arm_plan=arm,
            feature_names=["x"],
            horizon=3,
            continuation_teacher="official_adaptive",
            expected_trial_index=0,
            expected_producer_fingerprint="producer",
        )


def test_resume_loader_binds_producer_and_preserves_invalid_complete(
    tmp_path,
) -> None:
    plan = _analysis_plan()
    state = dict(plan["states"][0])
    arm = dict(state["arms"][0])
    row = _analysis_row("a", 0, 0.5)
    path = tmp_path / "rollout.json"
    module._write_json(path, row)
    loaded = module.load_resumable_receding_q_rollout(
        path,
        state_plan=state,
        arm_plan=arm,
        feature_names=["x"],
        horizon=3,
        continuation_teacher="official_adaptive",
        expected_trial_index=0,
        expected_producer_fingerprint="producer",
    )
    assert loaded == row

    row["producer_identity_fingerprint"] = "wrong"
    module._write_json(path, row)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="producer identity mismatch"):
        module.load_resumable_receding_q_rollout(
            path,
            state_plan=state,
            arm_plan=arm,
            feature_names=["x"],
            horizon=3,
            continuation_teacher="official_adaptive",
            expected_trial_index=0,
            expected_producer_fingerprint="producer",
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda row: row["steps"][0].__setitem__(
                "applied_pp_seed",
                row["steps"][0]["applied_pp_seed"] + 1,
            ),
            "applied PP seed mismatch",
        ),
        (
            lambda row: row.__setitem__(
                "final_repair_fingerprint",
                "corrupt",
            ),
            "final repair fingerprint mismatch",
        ),
        (
            lambda row: row.__setitem__(
                "selection_wall_seconds",
                row["selection_wall_seconds"] + 1.0,
            ),
            "selection_wall_seconds mismatch",
        ),
        (
            lambda row: row.__setitem__(
                "normalized_step_auc",
                row["normalized_step_auc"] + 0.1,
            ),
            "normalized_step_auc label mismatch",
        ),
        (
            lambda row: row["steps"][0].__setitem__("after_done", False),
            "native terminal evidence mismatch",
        ),
        (
            lambda row: (
                row["steps"][0].__setitem__("after_fingerprint", None),
                row.__setitem__("final_fingerprint", None),
            ),
            "after fingerprint",
        ),
        (
            lambda row: row["steps"][0].__setitem__(
                "repair_outcome", "accepted_noop"
            ),
            "repair outcome mismatch",
        ),
        (
            lambda row: row["low_level"].__setitem__("generated", 6),
            "low-level generated mismatch",
        ),
        (
            lambda row: row["steps"][0].__setitem__("agents", None),
            "agents must be a non-empty list",
        ),
        (
            lambda row: row["steps"][0].__setitem__(
                "repair_order", [1, 2, 3]
            ),
            "repair order does not cover",
        ),
    ],
)
def test_analysis_rejects_corrupt_rollout_internals(
    mutate,
    message: str,
) -> None:
    rows = [
        _analysis_row("a", 0, 0.5),
        _analysis_row("b", 0, 0.7),
        _analysis_row("a", 1, 0.5),
        _analysis_row("b", 1, 0.7),
    ]
    mutate(rows[0])
    with pytest.raises(ValueError, match=message):
        module.analyze_receding_q_rollouts(
            rows,
            plan=_analysis_plan(),
            trials=2,
            horizon=3,
            smoke_only=True,
        )
