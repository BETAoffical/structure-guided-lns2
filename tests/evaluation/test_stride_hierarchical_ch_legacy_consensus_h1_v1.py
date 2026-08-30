from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_legacy_consensus_h1_v1 as experiment
from experiments.repair_collection import _read_jsonl
from lns2_selector.runtime.unique_action_hierarchy import (
    COMPONENT_ROLE,
    CONSENSUS_STRUCTURAL,
    HOTSPOT_ROLE,
    V2_ANCHOR_ROLE,
    canonicalize_role_actions,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "configs" / "stride_hierarchical_ch_legacy_consensus_h1_v1.json"
)


def _inputs() -> tuple[dict, list[dict], list[dict]]:
    _path, root, config = experiment.load_config(CONFIG_PATH)
    preflight_path = root / config["inputs"]["active_supply_preflight_rows"]["path"]
    prior_path = root / config["inputs"]["prior_selected_states"]["path"]
    return config, _read_jsonl(preflight_path), _read_jsonl(prior_path)


def _identity(row: dict) -> tuple[str, str, str, int, str]:
    return (
        str(row["origin"]),
        str(row["source_id"]),
        str(row["episode_id"]),
        int(row["decision_index"]),
        str(row["before_fingerprint"]),
    )


def test_plan_freezes_exact_training_only_product() -> None:
    plan = experiment.build_plan(CONFIG_PATH)
    assert plan["selection_status"] == "ok"
    assert plan["selected_state_count"] == 60
    assert plan["selected_unique_action_count"] == 120
    assert plan["logical_h1_trial_count"] == 1920
    assert plan["trial_indices"] == list(range(16))
    assert plan["fixed_seed_halves"] == [list(range(8)), list(range(8, 16))]
    assert plan["research_split"] == experiment.RESEARCH_SPLIT
    assert plan["collection_executed_by_plan"] is False
    assert plan["training_authorized"] is False


def test_selection_is_all_and_only_remaining_exact_consensus_actions() -> None:
    config, preflight, prior = _inputs()
    selected, report = experiment.select_legacy_consensus_states(
        preflight, prior, config
    )
    prior_keys = {_identity(row) for row in prior}
    selected_keys = {_identity(row) for row in selected}

    assert report["status"] == "ok"
    assert report["source_consensus_structural_state_count"] == 92
    assert report["prior_selected_consensus_structural_state_count"] == 32
    assert report["remaining_consensus_state_count"] == 60
    assert len(selected) == 60
    assert report["selected_unique_action_count"] == 120
    assert report["logical_h1_trial_count"] == 1920
    assert not selected_keys & prior_keys
    assert report["prior_selected_overlap_count"] == 0

    for row in selected:
        canonical = canonicalize_role_actions(
            row["arms"], agent_count=int(row["agent_count"])
        )
        assert canonical.partition == CONSENSUS_STRUCTURAL
        assert canonical.unique_action_count == 2
        assert len(row["unique_actions"]) == 2
        assert len({tuple(action["agents"]) for action in row["unique_actions"]}) == 2
        role_actions = row["role_to_action_id"]
        assert role_actions[COMPONENT_ROLE] == role_actions[HOTSPOT_ROLE]
        assert role_actions[V2_ANCHOR_ROLE] != role_actions[COMPONENT_ROLE]
        structural = next(
            action
            for action in row["unique_actions"]
            if action["action_id"] == role_actions[COMPONENT_ROLE]
        )
        assert structural["role_aliases"] == [COMPONENT_ROLE, HOTSPOT_ROLE]
        assert set(structural["candidate_ids_by_role"]) == {
            COMPONENT_ROLE,
            HOTSPOT_ROLE,
        }
        assert row["research_split"] == experiment.RESEARCH_SPLIT
        assert row["legacy_sequential_development"] is True
        assert row["training_only"] is True
        assert row["evaluation_eligible"] is False
        assert row["training_authorized"] is False
        assert row["selection_outcome_fields_read"] is False
        assert row["target_outcome_fields_read"] is False


def test_selection_and_collection_dry_runs_never_invoke_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_solver_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry run invoked collection workers")

    monkeypatch.setattr(experiment, "_run_jobs", forbidden_solver_call)
    selection = experiment.run_selection(CONFIG_PATH, tmp_path, dry_run=True)
    collection = experiment.run_collection(CONFIG_PATH, tmp_path, dry_run=True)

    assert selection["status"] == "ok"
    assert selection["dry_run"] is True
    assert selection["outcome_fields_read"] is False
    assert collection["status"] == "dry_run"
    assert collection["state_job_count"] == 60
    assert collection["unique_action_count"] == 120
    assert collection["logical_trial_count"] == 1920
    assert collection["collection_executed"] is False
    assert collection["training_authorized"] is False
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_frozen_selection_products_are_new_and_training_only(tmp_path: Path) -> None:
    report = experiment.run_selection(CONFIG_PATH, tmp_path, dry_run=False)
    selected_path = tmp_path / "selected_states.jsonl"
    report_path = tmp_path / "selection_report.json"
    selected = _read_jsonl(selected_path)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == persisted["status"] == "ok"
    assert len(selected) == 60
    assert persisted["selected_states_sha256"]
    assert persisted["input_artifacts_mutated"] is False
    assert persisted["training_authorized"] is False
    assert {row["schema"] for row in selected} == {experiment.SELECTION_SCHEMA}
    dry_collection = experiment.run_collection(CONFIG_PATH, tmp_path, dry_run=True)
    assert dry_collection["selection_source"] == "frozen_selection_artifact"


def test_config_and_source_tampering_fail_closed() -> None:
    config, preflight, prior = _inputs()

    changed_config = copy.deepcopy(config)
    changed_config["inputs"]["active_supply_preflight_rows"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen inputs changed"):
        experiment.validate_config(changed_config)

    changed_rows = copy.deepcopy(preflight)
    consensus_index = next(
        index
        for index, row in enumerate(changed_rows)
        if canonicalize_role_actions(
            row["arms"], agent_count=int(row["agent_count"])
        ).partition
        == CONSENSUS_STRUCTURAL
    )
    changed_rows[consensus_index]["target_result"] = {"success": True}
    selected, report = experiment.select_legacy_consensus_states(
        changed_rows, prior, config
    )
    assert selected == []
    assert report["status"] == "STATE_SUPPLY_FAIL"
    assert "forbidden_source_outcome_field" in report["failure_reasons"]

    duplicate_agent_rows = copy.deepcopy(preflight)
    agents = duplicate_agent_rows[consensus_index]["arms"][COMPONENT_ROLE]["agents"]
    agents[-1] = agents[0]
    duplicate_agent_rows[consensus_index]["arms"][HOTSPOT_ROLE]["agents"] = list(agents)
    selected, report = experiment.select_legacy_consensus_states(
        duplicate_agent_rows, prior, config
    )
    assert selected == []
    assert report["status"] == "STATE_SUPPLY_FAIL"


def test_paired_seed_depends_on_state_and_trial_not_action() -> None:
    state_id = "legacy-consensus-state-example"
    observed = [
        experiment.matched_legacy_consensus_pp_seed(state_id, trial)
        for trial in range(16)
    ]
    assert len(set(observed)) == 16
    assert observed == [
        experiment.matched_legacy_consensus_pp_seed(state_id, trial)
        for trial in range(16)
    ]
    with pytest.raises(ValueError):
        experiment.matched_legacy_consensus_pp_seed(state_id, 16)
