from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from experiments._common import sha256_file
from experiments.repair_collection import _fingerprint
from experiments.stride_fresh_matched_unique_action_hierarchical_overlay_v1 import (
    CONFIG_SCHEMA,
    EXPERIMENT_ID,
    FOLDS,
    H1_STATE_SCHEMA,
    H1_TRIAL_SCHEMA,
    analyze_payloads,
    build_plan,
    load_config,
    matched_unique_pp_seed,
    run_analysis,
    run_h1,
    run_selection,
    select_unique_action_states,
    validate_config,
)
from experiments.stride_repairability_collection import repairability_restore_seed


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "stride_fresh_matched_unique_action_hierarchical_overlay_v1.json"
)
CONFIG_SHA256 = "80430e09b0ad88f5116908ff1e51a25fe66c9e9165d25ba6b9c28172d0531768"
ROLES = ("v2_anchor", "component16", "hotspot16")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _input_rows(config: dict) -> list[dict]:
    relative = config["active_supply_read_only"]["preflight_rows"]["path"]
    return [
        json.loads(line)
        for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _live_selection() -> tuple[dict, list[dict], dict]:
    _path, _root, config = load_config(CONFIG_PATH)
    selected, report = select_unique_action_states(_input_rows(config), config)
    return config, selected, report


def _map_to_fold() -> dict[str, str]:
    return {map_id: fold for fold, map_ids in FOLDS.items() for map_id in map_ids}


def test_registered_config_and_all_frozen_input_hashes_are_exact() -> None:
    config = _config()
    validate_config(config, project_root=ROOT)
    plan = build_plan(CONFIG_PATH)

    assert config["schema"] == CONFIG_SCHEMA
    assert config["experiment_id"] == EXPERIMENT_ID
    assert sha256_file(CONFIG_PATH) == plan["config_sha256"] == CONFIG_SHA256
    source = config["active_supply_read_only"]
    for name in (
        "config",
        "wave_source_report",
        "preflight_report",
        "preflight_rows",
        "source_episode_audit",
    ):
        pin = source[name]
        assert sha256_file(ROOT / pin["path"]) == pin["sha256"]
    for pin in source["wave_manifests"].values():
        assert sha256_file(ROOT / pin["path"]) == pin["sha256"]

    assert source["expected_source_episode_count"] == 68
    assert source["expected_preaction_row_count"] == 380
    assert source["expected_prior_distinct_eligible_count"] == 286
    assert source["expected_prior_status"] == "STATE_SUPPLY_FAIL"
    assert source["mutation_allowed"] is False
    assert plan == {
        "schema": "lns2.stride.fresh_matched_unique_action_hierarchical_overlay_plan.v1",
        "experiment_id": EXPERIMENT_ID,
        "config_path": str(CONFIG_PATH.resolve()),
        "config_sha256": CONFIG_SHA256,
        "read_only_preaction_row_count": 380,
        "unique_action_eligible_count": 379,
        "selected_state_count": 96,
        "selected_unique_action_count_distribution": {"2": 32, "3": 64},
        "selected_unique_action_count": 256,
        "logical_h1_trial_count": 4096,
        "workers": 16,
        "h1_executed_by_plan": False,
        "h8_or_training_or_ttf_executed": False,
        "selection_status": "ok",
    }

    changed = copy.deepcopy(config)
    changed["active_supply_read_only"]["preflight_rows"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="read-only source contract"):
        validate_config(changed)


def test_selection_is_exact_balanced_and_executes_each_agent_set_once() -> None:
    config, selected, report = _live_selection()
    reversed_selected, reversed_report = select_unique_action_states(
        reversed(_input_rows(config)), config
    )

    assert report["status"] == reversed_report["status"] == "ok"
    assert [row["state_occurrence_id"] for row in selected] == [
        row["state_occurrence_id"] for row in reversed_selected
    ]
    assert len(selected) == report["selected_state_count"] == 96
    assert report["unique_action_eligible_count"] == 379
    assert report["eligible_strata_supply"] == {
        "anchor_component_shared": 1,
        "consensus_structural": 92,
        "three_unique": 286,
    }
    assert report["selected_state_strata"] == {
        "consensus_structural": 32,
        "three_unique": 64,
    }
    assert report["selected_unique_action_count_distribution"] == {"2": 32, "3": 64}
    assert report["selected_unique_action_count"] == 256
    assert report["logical_h1_trial_count"] == 4096
    assert report["selected_map_count"] == 11
    assert report["selected_family_count"] == 4
    assert report["distinct_support_map_count"] == 7
    assert report["distinct_support_family_count"] == 3
    assert report["outcome_fields_read"] is False
    assert report["reserve_backfill_used"] is False

    map_to_fold = _map_to_fold()
    fold_counts = Counter(map_to_fold[row["map_id"]] for row in selected)
    fold_strata = Counter(
        (map_to_fold[row["map_id"]], row["hierarchical_stratum"])
        for row in selected
    )
    assert fold_counts == {fold: 24 for fold in FOLDS}
    for fold in FOLDS:
        assert fold_strata[(fold, "consensus_structural")] == 8
        assert fold_strata[(fold, "three_unique")] == 16
        assert all(
            report["fold_depth_counts"][fold].get(band, 0) >= 4
            for band in ("d0", "d1_3", "d4_plus")
        )

    action_count = 0
    logical_trial_keys = set()
    for state in selected:
        actions = state["unique_actions"]
        action_count += len(actions)
        agent_sets = [tuple(action["agents"]) for action in actions]
        assert len(agent_sets) == len(set(agent_sets)) == state["unique_action_count"]
        assert all(list(agents) == sorted(set(agents)) for agents in agent_sets)
        assert sorted(
            role for action in actions for role in action["role_aliases"]
        ) == sorted(ROLES)
        actions_by_id = {action["action_id"]: action for action in actions}
        for role in ROLES:
            action = actions_by_id[state["role_to_action_id"][role]]
            assert role in action["role_aliases"]
        for action in actions:
            if action["structural_role_aliases"]:
                assert action["actual_size"] == 16
            for trial_index in range(16):
                logical_trial_keys.add(
                    (state["state_occurrence_id"], action["action_id"], trial_index)
                )
        if state["hierarchical_stratum"] == "consensus_structural":
            assert state["role_to_action_id"]["component16"] == state[
                "role_to_action_id"
            ]["hotspot16"]
        else:
            assert len(set(state["role_to_action_id"].values())) == 3
    assert action_count == 256
    assert len(logical_trial_keys) == 4096


def test_selection_dry_run_is_a_zero_write_preview(tmp_path: Path) -> None:
    output = tmp_path / "selection-dry-run"
    report = run_selection(CONFIG_PATH, output, dry_run=True)

    assert report["dry_run"] is True
    assert report["status"] == "ok"
    assert report["selected_state_count"] == 96
    assert report["selected_unique_action_count"] == 256
    assert report["logical_h1_trial_count"] == 4096
    assert report["input_artifacts_mutated"] is False
    assert not output.exists()


def test_h1_and_analysis_dry_runs_write_nothing_and_h1_fails_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "h1-dry-run"
    h1 = run_h1(CONFIG_PATH, output, dry_run=True)
    analysis = run_analysis(CONFIG_PATH, output, dry_run=True)

    assert h1["status"] == "dry_run"
    assert h1["state_job_count"] == 96
    assert h1["unique_action_count"] == 256
    assert h1["logical_trial_count"] == 4096
    assert h1["workers"] == 16
    assert h1["h1_executed"] is False
    assert analysis["status"] == "dry_run"
    assert analysis["expected_state_count"] == 96
    assert analysis["expected_unique_action_count"] == 256
    assert analysis["expected_trial_count"] == 4096
    assert analysis["expected_stage2_three_unique_state_count"] == 64
    assert analysis["analysis_executed"] is False
    assert not output.exists()

    with pytest.raises(ValueError, match="requires frozen selection artifacts"):
        run_h1(CONFIG_PATH, output, dry_run=False)
    assert not output.exists()


def _opportunity_ids(selected: list[dict]) -> set[str]:
    map_to_fold = _map_to_fold()
    by_fold: defaultdict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_fold[map_to_fold[row["map_id"]]].append(row)
    result = set()
    for fold, rows in by_fold.items():
        by_map: defaultdict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_map[row["map_id"]].append(row)
        chosen = [sorted(map_rows, key=lambda item: item["state_occurrence_id"])[0]
                  for map_rows in by_map.values()]
        chosen_ids = {row["state_occurrence_id"] for row in chosen}
        for row in sorted(rows, key=lambda item: item["state_occurrence_id"]):
            if len(chosen_ids) == 6:
                break
            chosen_ids.add(row["state_occurrence_id"])
        assert len(chosen_ids) == 6, fold
        result.update(chosen_ids)
    return result


def _synthetic_payloads(
    selected: list[dict], *, identity: str
) -> list[dict]:
    opportunity_ids = _opportunity_ids(selected)
    three_unique_index = 0
    payloads = []
    for state in selected:
        state_id = state["state_occurrence_id"]
        before_fingerprint = state["before_repair_fingerprint"]
        before_conflicts = int(state["before_conflicts"])
        restore_seed = repairability_restore_seed(before_fingerprint)
        role_map = state["role_to_action_id"]
        anchor_id = role_map["v2_anchor"]
        is_opportunity = state_id in opportunity_ids
        favored_role = "component16"
        if state["hierarchical_stratum"] == "three_unique":
            favored_role = (
                "component16" if three_unique_index % 2 == 0 else "hotspot16"
            )
            three_unique_index += 1
        reductions = {anchor_id: 0.0 if is_opportunity else 1.0}
        if state["hierarchical_stratum"] == "consensus_structural":
            reductions[role_map["component16"]] = 1.0 if is_opportunity else 0.0
        else:
            other_role = (
                "hotspot16" if favored_role == "component16" else "component16"
            )
            reductions[role_map[favored_role]] = 1.0
            reductions[role_map[other_role]] = 0.0

        trials = []
        for action in state["unique_actions"]:
            action_id = action["action_id"]
            conflicts_after = round(
                before_conflicts * (1.0 - reductions[action_id])
            )
            for trial_index in range(16):
                pp_seed = matched_unique_pp_seed(state_id, trial_index)
                trials.append(
                    {
                        "schema": H1_TRIAL_SCHEMA,
                        "trial_identity": _fingerprint(
                            {
                                "state_occurrence_id": state_id,
                                "action_id": action_id,
                                "trial_index": trial_index,
                            }
                        ),
                        "state_occurrence_id": state_id,
                        "action_id": action_id,
                        "role_aliases": action["role_aliases"],
                        "candidate_ids_by_role": action["candidate_ids_by_role"],
                        "agents": action["agents"],
                        "actual_size": action["actual_size"],
                        "trial_index": trial_index,
                        "step_index": 0,
                        "restore_seed": restore_seed,
                        "fresh_independent_environment_restore": True,
                        "pp_seed": pp_seed,
                        "requested_random_seed": pp_seed,
                        "requested_pp_random_seed": pp_seed,
                        "applied_pp_random_seed": pp_seed,
                        "repair_order_count": 1,
                        "before_conflicts": before_conflicts,
                        "conflicts_after": conflicts_after,
                        "before_sum_of_costs": 1000,
                        "after_sum_of_costs": 900,
                        "normalized_conflict_reduction": (
                            before_conflicts - conflicts_after
                        ) / before_conflicts,
                        "replan_success": True,
                        "rollback": False,
                        "atomic_rollback": False,
                        "failure_reason": "",
                        "time_limit": False,
                        "no_progress": conflicts_after >= before_conflicts,
                        "before_repair_fingerprint": before_fingerprint,
                        "after_repair_fingerprint": (
                            f"synthetic-after-{state_id}-{action_id}-{trial_index}"
                        ),
                        "native_step_seconds": 0.01,
                        "pp_seconds": 0.01,
                        "requested_pp_time_limit_seconds": 5.0,
                        "action_valid": True,
                        "generated": True,
                        "runtime_used_in_label": False,
                        "integrity_ok": True,
                    }
                )
        payloads.append(
            {
                "schema": H1_STATE_SCHEMA,
                "identity": identity,
                "complete": True,
                "state_occurrence_id": state_id,
                "state_row": state,
                "source_trace_file": state["source_trace_file"],
                "source_trace_path": state["source_trace_path"],
                "source_trace_sha256": state["source_trace_sha256"],
                "before_repair_fingerprint": before_fingerprint,
                "before_conflicts": before_conflicts,
                "before_sum_of_costs": 1000,
                "restore_seed": restore_seed,
                "trials": trials,
                "runtime_used_in_label": False,
            }
        )
    return payloads


def test_synthetic_stage1_and_conditional_stage2_gates_and_fail_close() -> None:
    config, selected, _selection = _live_selection()
    identity = "synthetic-hierarchical-h1-identity"
    payloads = _synthetic_payloads(selected, identity=identity)
    report = analyze_payloads(
        config,
        payloads,
        identity=identity,
        selected=selected,
    )

    assert sum(len(payload["trials"]) for payload in payloads) == 4096
    assert report["complete"] is True
    assert report["passed"] is True
    assert report["state_count"] == 96
    assert report["observed_unique_action_count"] == 256
    assert report["observed_trial_count"] == 4096
    stage1 = report["stage1_structural_vs_v2"]
    assert stage1["passed"] is True
    assert stage1["opportunity_state_count"] == 24
    assert stage1["opportunity_map_count"] == 11
    assert stage1["opportunity_family_count"] == 4
    assert all(stage1["gates"].values())
    assert all(
        fold["has_opportunity"] and fold["has_nonopportunity"]
        for fold in stage1["folds"].values()
    )
    stage2 = report["stage2_component_vs_hotspot"]
    assert stage2["interpretable"] is True
    assert stage2["raw_gates_passed"] is True
    assert stage2["readiness_passed"] is True
    assert stage2["three_unique_state_count"] == 64
    assert stage2["support_map_count"] == 7
    assert stage2["support_family_count"] == 3
    assert stage2["decisive_state_count"] == 64
    assert stage2["decisive_map_count"] == 7
    assert stage2["decisive_family_count"] == 3
    assert set(stage2["direction_counts"]) == {"component_win", "hotspot_win"}
    assert all(stage2["gates"].values())
    assert stage2["training_or_runtime_authorized"] is False
    assert report["training_authorized"] is False
    assert report["runtime_or_ttf_claim_authorized"] is False

    failed = analyze_payloads(
        config,
        payloads,
        identity=identity,
        selected=selected,
        error_count=1,
    )
    assert failed["complete"] is False
    assert failed["passed"] is False
    assert failed["stage1_structural_vs_v2"]["gates"][
        "all_integrity_gates"
    ] is False
    assert failed["stage2_component_vs_hotspot"]["raw_gates_passed"] is False
    assert failed["stage2_component_vs_hotspot"]["interpretable"] is False
    assert failed["stage2_component_vs_hotspot"]["readiness_passed"] is False
    assert failed["training_authorized"] is False


@pytest.mark.parametrize(
    "tamper",
    ("payload_restore_seed", "trial_restore_seed", "trial_step_index"),
)
def test_synthetic_restore_and_step_identity_tampering_fails_closed(
    tamper: str,
) -> None:
    config, selected, _selection = _live_selection()
    identity = "synthetic-hierarchical-h1-tamper-identity"
    payloads = _synthetic_payloads(selected, identity=identity)
    if tamper == "payload_restore_seed":
        payloads[0]["restore_seed"] += 1
    elif tamper == "trial_restore_seed":
        payloads[0]["trials"][0]["restore_seed"] += 1
    else:
        payloads[0]["trials"][0]["step_index"] = 1

    report = analyze_payloads(
        config,
        payloads,
        identity=identity,
        selected=selected,
    )

    assert report["complete"] is False
    assert report["passed"] is False
    assert report["state_count"] == 95
    assert report["stage1_structural_vs_v2"]["gates"][
        "all_integrity_gates"
    ] is False
    assert report["stage2_component_vs_hotspot"]["raw_gates_passed"] is False
    assert report["stage2_component_vs_hotspot"]["readiness_passed"] is False
    assert report["training_authorized"] is False
