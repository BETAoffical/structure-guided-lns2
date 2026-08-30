from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import pytest

from experiments.stride_fresh_matched_v2_c16_h16_v1 import (
    ARMS,
    CONFIG_SCHEMA,
    H1_STATE_SCHEMA,
    H1_TRIAL_SCHEMA,
    TRIAL_INDICES,
    _h1_failure_result,
    _h1_partial_artifact_valid,
    _h1_state_artifact_valid,
    _preflight_failure_result,
    analyze_h1_payloads,
    build_plan,
    classify_h1_challenger,
    matched_pp_seed,
    select_preflight_states,
    source_schedule,
    validate_config,
)
from experiments.repair_collection import _fingerprint


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "stride_fresh_matched_v2_c16_h16_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _preflight_row(
    source_id: str,
    episode_id: str,
    decision_index: int,
    *,
    eligible: bool = True,
    target_result: str = "unused-a",
) -> dict:
    return {
        "source_id": source_id,
        "episode_id": episode_id,
        "task_id": f"task-{episode_id}",
        "map_id": f"map-{episode_id}",
        "map_family": "maze",
        "solver_seed": 41,
        "decision_index": decision_index,
        "before_fingerprint": f"before-{episode_id}-{decision_index}",
        "eligible": eligible,
        "target_result": target_result,
    }


def _trial(
    state_index: int,
    arm: str,
    trial_index: int,
    reduction: float,
) -> dict:
    state_id = f"state-{state_index}"
    pp_seed = matched_pp_seed(state_id, trial_index, 0)
    return {
        "schema": H1_TRIAL_SCHEMA,
        "trial_identity": _fingerprint(
            {
                "state_occurrence_id": state_id,
                "arm": arm,
                "trial_index": trial_index,
            }
        ),
        "state_occurrence_id": state_id,
        "arm": arm,
        "trial_index": trial_index,
        "pp_seed": pp_seed,
        "requested_random_seed": pp_seed,
        "requested_pp_random_seed": pp_seed,
        "applied_pp_random_seed": pp_seed,
        "repair_order_count": 16,
        "action_valid": True,
        "generated": True,
        "fresh_independent_environment_restore": True,
        "integrity_ok": True,
        "before_conflicts": 100,
        "conflicts_after": 97 if reduction > 0.0 else 100,
        "before_sum_of_costs": 1_000,
        "after_sum_of_costs": 1_000,
        "before_repair_fingerprint": "repair-before",
        "after_repair_fingerprint": "repair-after",
        "replan_success": True,
        "normalized_conflict_reduction": reduction,
        "no_progress": reduction <= 0.0,
        "rollback": False,
        "atomic_rollback": False,
        "failure_reason": "",
        "time_limit": False,
        "pp_seconds": 999.0 if arm != "v2_anchor" else 0.0,
    }


def _h1_payloads(config: dict) -> list[dict]:
    map_family = {
        task["map_id"]: task["map_family"]
        for source in config["sources"].values()
        for task in source["tasks"]
    }
    maps = [
        map_id
        for fold in config["h1_gates"]["map_folds"].values()
        for map_id in fold
    ]
    payloads = []
    for state_index in range(96):
        map_id = maps[state_index // 8]
        opportunity = state_index % 8 < 2
        state_id = f"state-{state_index}"
        anchor_size = 8 if state_index % 2 == 0 else 32
        trials = []
        for arm in ARMS:
            reduction = 0.03 if arm == "component16" and opportunity else 0.0
            trials.extend(
                _trial(state_index, arm, trial_index, reduction)
                for trial_index in TRIAL_INDICES
            )
        payloads.append(
            {
                "schema": H1_STATE_SCHEMA,
                "identity": "frozen-h1-identity",
                "state_occurrence_id": state_id,
                "complete": True,
                "before_repair_fingerprint": "repair-before",
                "before_conflicts": 100,
                "before_sum_of_costs": 1_000,
                "state_row": {
                    "state_occurrence_id": state_id,
                    "source_id": "fixture",
                    "map_id": map_id,
                    "map_family": map_family[map_id],
                    "task_id": f"fixture-{state_index}",
                    "solver_seed": 41,
                    "arms": {
                        "v2_anchor": {
                            "agents": list(range(anchor_size)),
                            "actual_size": anchor_size,
                        },
                        "component16": {
                            "agents": list(range(100, 116)),
                            "actual_size": 16,
                        },
                        "hotspot16": {
                            "agents": list(range(200, 216)),
                            "actual_size": 16,
                        },
                    },
                },
                "trials": trials,
            }
        )
    return payloads


def test_registered_config_freezes_exact_cohort_and_p1_runtime_contract() -> None:
    config = _config()
    validate_config(config, project_root=ROOT)

    assert config["schema"] == CONFIG_SCHEMA
    assert set(config["sources"]) == {"movingai_ood", "balanced_wall_clock"}
    tasks = [task for source in config["sources"].values() for task in source["tasks"]]
    assert len(tasks) == 24
    assert len({task["task_id"] for task in tasks}) == 24
    assert len({task["map_id"] for task in tasks}) == 12
    assert {
        family: len({task["map_id"] for task in tasks if task["map_family"] == family})
        for family in {task["map_family"] for task in tasks}
    } == {"game": 2, "maze": 2, "random": 3, "room": 2, "warehouse": 3}
    assert config["state_selection"]["structural_required_actual_size"] == 16
    assert "required_actual_size" not in config["state_selection"]
    assert config["h1"]["logical_trial_count"] == 4608
    assert config["h1"]["workers"] == 16
    assert config["h1"]["per_action_time_limit_seconds"] == 5.0
    assert config["h1"]["per_state_process_fuse_seconds"] == 420.0


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("state_selection", "structural_required_actual_size", 8),
        ("h1", "per_state_process_fuse_seconds", 360.0),
        ("h8_future_contract", "registered_but_not_executed_in_h1_stage", False),
        ("claim_boundary", "training_allowed", True),
    ],
)
def test_validator_rejects_frozen_contract_mutations(
    section: str, field: str, replacement: object
) -> None:
    config = _config()
    config[section][field] = replacement
    with pytest.raises(ValueError):
        validate_config(config)


def test_plan_and_schedule_freeze_48_fresh_paired_source_episodes() -> None:
    config = _config()
    schedule = source_schedule(config)
    plan = build_plan(CONFIG_PATH)

    assert len(schedule) == 48
    by_task = defaultdict(list)
    for row in schedule:
        by_task[(row["source_id"], row["task_id"])].append(row["solver_seed"])
    assert len(by_task) == 24
    assert all(sorted(seeds) == [41, 42] for seeds in by_task.values())
    assert plan["source_episode_count"] == 48
    assert plan["source_workers"] == 16
    assert plan["target_state_count"] == 96
    assert plan["states_per_episode"] == 2
    assert plan["arms"] == ["v2_anchor", "component16", "hotspot16"]
    assert plan["trial_count_per_arm"] == 16
    assert plan["logical_h1_trial_count"] == 4608
    assert plan["h1_workers"] == 16
    assert plan["h1_per_action_time_limit_seconds"] == 5.0
    assert plan["h8_execution_registered"] is True
    assert plan["h8_executed_by_this_stage"] is False
    assert plan["ttf_or_training_run"] is False


def test_preflight_selection_is_deterministic_and_target_outcome_blind() -> None:
    rows = [
        _preflight_row(source, episode, decision, eligible=decision != 3)
        for source, episode in (("a", "ep-1"), ("b", "ep-2"))
        for decision in range(4)
    ]
    selected, report = select_preflight_states(rows)
    changed = [dict(row, target_result="opposite-outcome") for row in reversed(rows)]
    selected_changed, report_changed = select_preflight_states(changed)

    identities = [row["state_occurrence_id"] for row in selected]
    assert identities == [row["state_occurrence_id"] for row in selected_changed]
    assert len(selected) == 4
    assert all(row["target_outcome_fields_read"] is False for row in selected)
    assert report["status"] == report_changed["status"] == "ok"
    assert report["failed_episode_count"] == 0

    failed_rows = [row for row in rows if row["decision_index"] in {0, 3}]
    _selected, failed_report = select_preflight_states(failed_rows)
    assert failed_report["status"] == "STATE_SUPPLY_FAIL"
    assert failed_report["failed_episode_count"] == 2


def test_matched_pp_seed_is_arm_independent_and_step_scoped() -> None:
    seed = matched_pp_seed("fresh-matched-state", 7, 0)
    assert seed == matched_pp_seed("fresh-matched-state", 7, 0)
    assert 0 <= seed < 2**31
    assert seed != matched_pp_seed("fresh-matched-state", 8, 0)
    assert seed != matched_pp_seed("fresh-matched-state", 7, 1)
    with pytest.raises(ValueError):
        matched_pp_seed("", 0, 0)
    with pytest.raises(ValueError):
        matched_pp_seed("fresh-matched-state", 16, 0)
    with pytest.raises(ValueError):
        matched_pp_seed("fresh-matched-state", 0, -1)


def test_h1_label_and_global_gates_authorize_only_complete_opportunity_evidence() -> None:
    config = _config()
    payloads = _h1_payloads(config)
    report = analyze_h1_payloads(config, payloads)

    assert report["complete"] is True
    assert report["passed"] is True
    assert report["observed_trial_count"] == 4608
    assert report["opportunity_state_count"] == 24
    assert report["opportunity_map_count"] == 12
    assert report["opportunity_family_count"] == 5
    assert report["anchor_actual_size_distribution"] == {8: 48, 32: 48}
    assert all(report["gates"].values())
    assert report["h8_authorized"] is True
    assert report["training_authorized"] is False
    assert report["ttf_or_speed_claim_authorized"] is False

    failed = analyze_h1_payloads(config, payloads, error_count=1)
    assert failed["passed"] is False
    assert failed["gates"]["all_integrity_gates"] is False
    assert failed["h8_authorized"] is False


def test_h1_classifier_requires_paired_seed_identity_and_both_fixed_halves() -> None:
    anchor = [_trial(0, "v2_anchor", index, 0.0) for index in TRIAL_INDICES]
    challenger = [_trial(0, "component16", index, 0.03) for index in TRIAL_INDICES]
    passed = classify_h1_challenger(anchor, challenger)
    assert passed["label"] is True
    assert passed["strict_paired_wins"] == 16
    report_only_timing = copy.deepcopy(challenger)
    for row in report_only_timing:
        row["pp_seconds"] = 1_000_000.0
    timed = classify_h1_challenger(anchor, report_only_timing)
    assert timed["label"] is True
    assert timed["mean_delta"] == passed["mean_delta"]
    assert timed["pp_seconds_report_only"]["challenger"] == 16_000_000.0

    second_half_regression = copy.deepcopy(challenger)
    for row in second_half_regression[8:]:
        row["normalized_conflict_reduction"] = -0.01
    failed = classify_h1_challenger(anchor, second_half_regression)
    assert failed["label"] is False
    assert failed["positive_second_half_mean"] is False

    duplicate = classify_h1_challenger(anchor, challenger, distinct_from_anchor=False)
    assert duplicate["label"] is None
    assert duplicate["label_reason"] == "duplicate_of_anchor"

    wrong_seed = copy.deepcopy(challenger)
    wrong_seed[0]["pp_seed"] += 1
    mismatched = classify_h1_challenger(anchor, wrong_seed)
    assert mismatched["label"] is False
    assert mismatched["paired_seed_identity"] is False


def test_resume_integrity_rejects_seed_action_and_success_conflict_tampering() -> None:
    payload = _h1_payloads(_config())[0]
    state_id = payload["state_occurrence_id"]
    assert _h1_state_artifact_valid(
        payload, identity="frozen-h1-identity", state_occurrence_id=state_id
    )

    for field, replacement in (
        ("requested_random_seed", -1),
        ("requested_pp_random_seed", -1),
        ("applied_pp_random_seed", -1),
        ("action_valid", False),
        ("generated", False),
    ):
        changed = copy.deepcopy(payload)
        changed["trials"][0][field] = replacement
        assert not _h1_state_artifact_valid(
            changed, identity="frozen-h1-identity", state_occurrence_id=state_id
        )

    increased = copy.deepcopy(payload)
    increased["trials"][0]["conflicts_after"] = 101
    increased["trials"][0]["normalized_conflict_reduction"] = -0.01
    assert not _h1_state_artifact_valid(
        increased, identity="frozen-h1-identity", state_occurrence_id=state_id
    )

    partial = {**payload, "complete": False, "trials": payload["trials"][:1]}
    assert _h1_partial_artifact_valid(
        partial,
        identity="frozen-h1-identity",
        state_occurrence_id=state_id,
        state_row=payload["state_row"],
        before_repair_fingerprint="repair-before",
        before_conflicts=100,
        before_sum_of_costs=1_000,
    )


def test_worker_failure_results_preserve_job_identity_without_generic_row() -> None:
    preflight_job = {
        "job_id": "source:episode",
        "episode_key": "source:episode",
        "output_path": "episode.json",
    }
    preflight = _preflight_failure_result(preflight_job, "timeout", "fused")
    assert preflight["job_id"] == "source:episode"
    assert preflight["status"] == "timeout"

    h1_job = {
        "job_id": "state-0",
        "state_row": {"state_occurrence_id": "state-0"},
        "output_path": "state.json",
    }
    h1 = _h1_failure_result(h1_job, "error", "failed")
    assert h1["job_id"] == "state-0"
    assert h1["state_occurrence_id"] == "state-0"
