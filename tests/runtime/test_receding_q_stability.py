from __future__ import annotations

import copy

import pytest

from experiments import receding_q_stability as module
from experiments import receding_q_pilot as pilot
from experiments._common import PRODUCER_IDENTITY_SCHEMA


def test_stability_schema_is_v2() -> None:
    assert (
        module.RECEDING_Q_STABILITY_SCHEMA
        == "lns2.receding_q_label_stability.v2"
    )


def _row(
    state: str,
    candidate: str,
    trial: int,
    auc: float,
    *,
    v2: bool = False,
    producer: str = "producer",
) -> dict:
    agents = [1, 2, 3, 4] if candidate == "a" else [5, 6, 7, 8]
    initial_repair = f"repair-before-{state}"
    seed = pilot._paired_seed(initial_repair, trial, 1)
    final_conflicts = int(10 * auc)
    selection_seconds = 0.02
    repair_seconds = 0.03
    labels = pilot._fixed_horizon_labels(
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
        "schema": pilot.RECEDING_Q_PILOT_SCHEMA,
        "producer_identity_fingerprint": producer,
        "state_id": state,
        "candidate_id": candidate,
        "trial_index": trial,
        "complete": True,
        "is_v2_candidate": v2,
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 100,
        "split": "policy_train",
        "source_stratum": "ordinary",
        "low_level": {
            "generated": 5,
            "expanded": 4,
            "reopened": 0,
            "runs": 1,
        },
        "agents": agents,
        "actual_size": 4,
        "initial_conflicts": 10,
        "initial_fingerprint": f"before-{state}",
        "initial_repair_fingerprint": initial_repair,
        "final_fingerprint": f"after-{state}-{candidate}-{trial}",
        "final_repair_fingerprint": (
            f"repair-after-{state}-{candidate}-{trial}"
        ),
        "conflict_trajectory": [10, final_conflicts],
        "feature_names": ["x"],
        "feature_values": [1.0],
        "selection_families": ["collision:4"],
        "executed_steps": 1,
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
                "requested_pp_seed": seed,
                "applied_pp_seed": seed,
                "selection_seconds": selection_seconds,
                "shadow_candidate_count": 2,
                "solver_step_wall_seconds": 0.02,
                "iteration_wall_seconds": repair_seconds,
                "pp_replan_seconds": 0.01,
                "conflicts_before": 10,
                "conflicts_after": final_conflicts,
                "conflict_reduction": 10 - final_conflicts,
                "after_done": True,
                "after_feasible": False,
                "step_applied": True,
                "terminated": False,
                "truncated": True,
                "replan_success": True,
                "repair_outcome": "conflict_reduced",
                "low_level": {
                    "generated": 5,
                    "expanded": 4,
                    "reopened": 0,
                    "runs": 1,
                },
                "before_fingerprint": f"before-{state}",
                "after_fingerprint": f"after-{state}-{candidate}-{trial}",
                "before_repair_fingerprint": initial_repair,
                "after_repair_fingerprint": (
                    f"repair-after-{state}-{candidate}-{trial}"
                ),
            }
        ],
        "stop_reason": "environment_terminal",
        "root_selection_seconds": selection_seconds,
        "selection_wall_seconds": selection_seconds,
        "repair_wall_seconds": repair_seconds,
        "pp_replan_seconds": 0.01,
        "continuation_teacher": "official_adaptive",
        **labels,
    }


def _plan(state: str = "state") -> dict:
    return {
        "schema": pilot.RECEDING_Q_PILOT_SCHEMA,
        "feature_names": ["x"],
        "states": [
            {
                "state_id": state,
                "split": "policy_train",
                "map_id": "map",
                "layout_mode": "layout",
                "agent_count": 100,
                "source_stratum": "ordinary",
                "initial_conflicts": 10,
                "before_fingerprint": f"before-{state}",
                "before_repair_fingerprint": f"repair-before-{state}",
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


def _producer_identity(name: str) -> dict:
    return {
        "schema": PRODUCER_IDENTITY_SCHEMA,
        "source_sha256": {f"{name}.py": "0" * 64},
        "python": {"implementation": "CPython", "version": "3.10.0"},
        "packages": {"numpy": "1", "scikit-learn": "2"},
        "native_required": True,
        "native": {
            "path": f"{name}.so",
            "sha256": "1" * 64,
            "repair_timing_schema": "lns2.repair_timing.v2",
            "native_semantics_schema": (
                "lns2.native_semantics.upstream_compatible.v1"
            ),
        },
    }


def test_identify_targets_uses_exact_two_seed_winner_change() -> None:
    rows = [
        _row("stable", "a", 0, 0.1),
        _row("stable", "b", 0, 0.2),
        _row("stable", "a", 1, 0.1),
        _row("stable", "b", 1, 0.2),
        _row("unstable", "a", 0, 0.1),
        _row("unstable", "b", 0, 0.2),
        _row("unstable", "a", 1, 0.3),
        _row("unstable", "b", 1, 0.1),
    ]
    targets = module.identify_stability_targets(rows)
    assert targets["target_state_ids"] == ["unstable"]
    assert targets["stable_state_ids"] == ["stable"]


def test_followup_jobs_skip_stable_states_and_add_only_trial_2_3() -> None:
    plan = {
        "states": [
            {
                "state_id": "unstable",
                "arms": [{"candidate_id": "a"}, {"candidate_id": "b"}],
            },
            {
                "state_id": "stable",
                "arms": [{"candidate_id": "a"}],
            },
        ]
    }
    jobs = module.build_followup_jobs(
        plan=plan,
        targets={"target_state_ids": ["unstable"]},
        horizon=3,
        continuation_teacher="official_adaptive",
    )
    assert len(jobs) == 4
    assert {job["trial_index"] for job in jobs} == {2, 3}
    assert {job["state"]["state_id"] for job in jobs} == {"unstable"}


def test_merge_rejects_duplicate_followup_key() -> None:
    source = [_row("state", "a", 0, 0.1)]
    with pytest.raises(ValueError, match="duplicates"):
        module.merge_followup_rollouts(source, [copy.deepcopy(source[0])])


def test_source_configuration_is_bound_to_its_plan() -> None:
    plan = {
        "schema": pilot.RECEDING_Q_PILOT_SCHEMA,
        "feature_names": ["x"],
        "states": [],
    }
    config = {
        "schema": pilot.RECEDING_Q_PILOT_SCHEMA,
        "producer_identity": _producer_identity("source"),
        "plan_fingerprint": module._fingerprint(plan),
        "trials": 2,
    }
    config["producer_identity_fingerprint"] = module._fingerprint(
        config["producer_identity"]
    )
    module._validate_source_configuration(plan, config)

    wrong_schema = {**config, "schema": "wrong"}
    with pytest.raises(ValueError, match="source configuration"):
        module._validate_source_configuration(plan, wrong_schema)

    changed_plan = {**plan, "states": [{"state_id": "changed"}]}
    with pytest.raises(ValueError, match="plan fingerprint mismatch"):
        module._validate_source_configuration(changed_plan, config)


def _write_rollouts(root, rows: list[dict]) -> None:
    rollout_root = root / "rollouts"
    rollout_root.mkdir(parents=True)
    for row in rows:
        module._write_json(
            rollout_root
            / module._rollout_file_name(
                row["state_id"],
                row["candidate_id"],
                row["trial_index"],
            ),
            row,
        )


def _write_valid_stability_artifacts(tmp_path):
    source_root = tmp_path / "source-pilot"
    stability_root = tmp_path / "stability"
    source_root.mkdir()
    stability_root.mkdir()
    plan = _plan()
    source_rows = [
        _row("state", "a", 0, 0.10, v2=True, producer="source-producer"),
        _row("state", "b", 0, 0.20, producer="source-producer"),
        _row("state", "a", 1, 0.30, v2=True, producer="source-producer"),
        _row("state", "b", 1, 0.10, producer="source-producer"),
    ]
    followup_rows = [
        _row("state", "a", 2, 0.10, v2=True, producer="stability-producer"),
        _row("state", "b", 2, 0.20, producer="stability-producer"),
        _row("state", "a", 3, 0.10, v2=True, producer="stability-producer"),
        _row("state", "b", 3, 0.20, producer="stability-producer"),
    ]
    targets = module.identify_stability_targets(source_rows)
    source_config = {
        "schema": pilot.RECEDING_Q_PILOT_SCHEMA,
        "producer_identity": _producer_identity("source"),
        "plan_fingerprint": module._fingerprint(plan),
        "trials": 2,
        "horizon": 3,
        "continuation_teacher": "official_adaptive",
    }
    source_config["producer_identity_fingerprint"] = "source-producer"
    source_config["producer_identity"] = {
        **source_config["producer_identity"],
        "test_identity": "source",
    }
    source_config["producer_identity_fingerprint"] = module._fingerprint(
        source_config["producer_identity"]
    )
    for row in source_rows:
        row["producer_identity_fingerprint"] = source_config[
            "producer_identity_fingerprint"
        ]
    stability_identity = _producer_identity("stability")
    stability_producer = module._fingerprint(stability_identity)
    for row in followup_rows:
        row["producer_identity_fingerprint"] = stability_producer
    stability_config = {
        "schema": module.RECEDING_Q_STABILITY_SCHEMA,
        "producer_identity": stability_identity,
        "producer_identity_fingerprint": stability_producer,
        "source": str(source_root),
        "source_producer_identity_fingerprint": source_config[
            "producer_identity_fingerprint"
        ],
        "source_plan_fingerprint": module._fingerprint(plan),
        "source_rollout_fingerprint": module._fingerprint(
            sorted(
                (module._rollout_key(row), row)
                for row in source_rows
            )
        ),
        "target_fingerprint": module._fingerprint(targets),
        "horizon": 3,
        "continuation_teacher": "official_adaptive",
        "followup_trial_indices": [2, 3],
    }
    module._write_json(source_root / "plan.json", plan)
    module._write_json(source_root / "run_config.json", source_config)
    module._write_json(
        source_root / "status.json",
        {
            "schema": pilot.RECEDING_Q_PILOT_SCHEMA,
            "status": "complete",
            "error_count": 0,
            "completed_rollout_count": 4,
            "total_rollout_count": 4,
        },
    )
    module._write_json(
        stability_root / "run_config.json",
        stability_config,
    )
    module._write_json(stability_root / "targets.json", targets)
    module._write_json(
        stability_root / "status.json",
        {
            "schema": module.RECEDING_Q_STABILITY_SCHEMA,
            "status": "complete",
            "error_count": 0,
            "completed_rollout_count": 4,
            "total_rollout_count": 4,
        },
    )
    _write_rollouts(source_root, source_rows)
    _write_rollouts(stability_root, followup_rows)
    return source_root, stability_root, followup_rows


def test_validated_four_seed_loader_rejects_corrupt_followup(
    tmp_path,
) -> None:
    source_root, stability_root, followup_rows = (
        _write_valid_stability_artifacts(tmp_path)
    )
    loaded = module.load_validated_four_seed_stability(
        stability_root,
        source_root,
    )
    assert len(loaded["merged_rows"]) == 8

    corrupt = copy.deepcopy(followup_rows[0])
    corrupt["normalized_step_auc"] += 0.1
    module._write_json(
        stability_root
        / "rollouts"
        / module._rollout_file_name(
            corrupt["state_id"],
            corrupt["candidate_id"],
            corrupt["trial_index"],
        ),
        corrupt,
    )
    with pytest.raises(ValueError, match="normalized_step_auc label mismatch"):
        module.load_validated_four_seed_stability(
            stability_root,
            source_root,
        )


def test_four_seed_analysis_uses_outcome_stability_not_exact_id() -> None:
    rows = []
    for trial in range(4):
        rows.append(_row("state", "a", trial, 0.10, v2=True))
        rows.append(_row("state", "b", trial, 0.11))
    plan = _plan()
    targets = {"target_state_ids": ["state"]}
    report, states, loo = module.analyze_four_seed_stability(
        rows,
        plan=plan,
        targets=targets,
        horizon=3,
        continuation_teacher="official_adaptive",
    )
    assert states[0]["operationally_stable"] is True
    assert states[0]["half_top3_overlap"] == 2
    assert report["loo"]["losses"] == 0
    assert len(loo) == 4


def test_four_seed_analysis_rejects_corrupt_source_rollout() -> None:
    rows = []
    for trial in range(4):
        rows.append(_row("state", "a", trial, 0.10, v2=True))
        rows.append(_row("state", "b", trial, 0.11))
    rows[0]["steps"][0]["applied_pp_seed"] += 1
    plan = _plan()
    with pytest.raises(ValueError, match="applied PP seed mismatch"):
        module.analyze_four_seed_stability(
            rows,
            plan=plan,
            targets={"target_state_ids": ["state"]},
            horizon=3,
            continuation_teacher="official_adaptive",
        )
