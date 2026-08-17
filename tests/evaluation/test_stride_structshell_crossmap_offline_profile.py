from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

import experiments.stride_structshell_crossmap_offline_profile as subject
from experiments.repair_collection import state_fingerprint
from experiments.state_analysis import analyze_static_grid


def _state() -> dict:
    agents = [
        {"id": 0, "start": 0, "goal": 2, "path": [0, 1, 2]},
        {"id": 1, "start": 2, "goal": 0, "path": [2, 1, 0]},
        {"id": 2, "start": 3, "goal": 4, "path": [3, 4, 4]},
        {"id": 3, "start": 4, "goal": 3, "path": [4, 3, 3]},
    ]
    for agent in agents:
        agent.update(
            {
                "path_cost": len(agent["path"]) - 1,
                "shortest_path_cost": abs(agent["goal"] - agent["start"]),
                "conflict_degree": 1,
                "delay": 0,
            }
        )
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": False,
        "done": False,
        "iteration": 0,
        "rows": 1,
        "cols": 5,
        "sum_of_costs": 8,
        "num_of_colliding_pairs": 2,
        "low_level": {"generated": 0, "expanded": 0, "reopened": 0, "runs": 0},
        "obstacles": [0, 0, 0, 0, 0],
        "conflict_edges": [[0, 1], [2, 3]],
        "agents": agents,
    }


def test_checkpoint_indices_are_exactly_initial_middle_and_last() -> None:
    assert subject.checkpoint_indices(13) == (0, 6, 12)
    assert subject.checkpoint_indices(258) == (0, 129, 257)
    with pytest.raises(ValueError, match="at least three"):
        subject.checkpoint_indices(2)


def test_profile_state_uses_only_state_and_generates_fixed16_candidates() -> None:
    state = _state()
    row = subject.profile_state(
        state,
        map_id="synthetic",
        checkpoint="initial",
        decision_index=0,
        static_grid=analyze_static_grid(state),
    )
    assert row["state_fingerprint"] == state_fingerprint(state)
    assert row["profile_role"] == "shared_initial_outcome_blind_profile"
    assert row["counterfactual_family_label_allowed"] is False
    assert row["profile"]["conflict_pair_count"] == 2
    assert row["profile"]["conflict_event_count"] == 2
    assert row["profile"]["conflict_component_count"] == 2
    assert row["profile"]["largest_conflict_component_size"] == 2
    assert row["fixed16_candidates"]["component16"] is not None
    assert row["fixed16_candidates"]["hotspot16"] is not None
    assert row["candidate_comparison"]["exact_same_agent_set"] is True


def test_analyze_uses_four_v2_source_traces_and_writes_twelve_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    per_map = {}
    base_state = _state()
    fingerprint = state_fingerprint(base_state)
    for index, map_id in enumerate(subject.MAP_IDS):
        collection = source / "maps" / map_id / subject.SOURCE_CONTROLLER
        collection.mkdir(parents=True)
        manifest = {
            "status": "ok",
            "policy": "realized_dynamic",
            "task_id": f"task-{index}",
            "solver_seed": 17,
            "trace_sha256": f"{index + 1:064x}",
            "summary": {
                "controller_mode": "v2-full",
                "repair_iterations": 3,
                "initial_fingerprint": fingerprint,
            },
        }
        (collection / "realized_dynamic_manifest.jsonl").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )
        per_map[map_id] = {
            "component16": {"success": True, "restricted_ttf": 10.0 + index},
            "hotspot16": {"success": True, "restricted_ttf": 11.0 + index},
            "v2_only": {"success": True, "restricted_ttf": 12.0 + index},
        }
    source_report = {
        "schema": subject.SOURCE_REPORT_SCHEMA,
        "experiment_id": subject.SOURCE_EXPERIMENT_ID,
        "integrity_passed": True,
        "map_count": 4,
        "per_map": per_map,
    }
    source.mkdir(exist_ok=True)
    (source / "quick_screen_report.json").write_text(
        json.dumps(source_report), encoding="utf-8"
    )

    def fake_reconstruct(_collection: Path, _manifest: dict) -> dict:
        states = [copy.deepcopy(base_state) for _ in range(4)]
        transitions = [
            {
                "decision_index": index,
                "before_fingerprint": fingerprint,
            }
            for index in range(3)
        ]
        return {"states": states, "transitions": transitions}

    monkeypatch.setattr(subject, "reconstruct_trace", fake_reconstruct)
    report = subject.analyze(source, output)
    assert report["contract"] == {
        "source_controller": "v2_only",
        "map_count": 4,
        "checkpoints_per_map": 3,
        "checkpoint_names": list(subject.CHECKPOINT_NAMES),
        "checkpoint_roles": dict(subject.CHECKPOINT_ROLES),
        "profile_state_count": 12,
        "fixed_nominal_size": 16,
        "families": ["component16", "hotspot16"],
        "native_solver_imported_or_invoked": False,
        "pp_imported_or_invoked": False,
        "controller_invoked": False,
        "new_reset_or_episode_invoked": False,
        "model_training_invoked": False,
        "auc_computed": False,
        "counterfactual_family_labels_created": False,
        "loaded_native_modules": [],
    }
    assert len(report["profiles"]) == 12
    for row in report["profiles"]:
        if row["checkpoint"] != "initial":
            assert row["profile_role"] == (
                "v2_outcome_conditioned_unlabelled_retrospective_profile"
            )
        assert row["counterfactual_family_label_allowed"] is False
    assert report["source_count_evidence"]["source_count_gate_met"] is False
    assert report["source_count_evidence"]["additional_collection_allowed"] is False
    assert report["source_count_evidence"]["router_training_allowed"] is False
    assert report["router_training_allowed"] is False
    integrity = report["source_integrity"]
    assert integrity["all_v2_profile_trace_and_state_fingerprints_verified"] is True
    assert integrity["component_hotspot_outcomes"] == {
        "source": "quick_screen_report.json",
        "bound_by_source_report_sha256": integrity["source_report_sha256"],
        "challenger_traces_read_or_reverified": False,
    }
    assert set(integrity["profile_implementation_sha256"]) == set(
        subject.PROFILE_IMPLEMENTATION_FILES
    )
    assert all(
        len(value) == 64
        for value in integrity["profile_implementation_sha256"].values()
    )
    assert report["default_replacement_allowed"] is False
    assert report["formal_speed_claim"] is False
    assert (output / subject.REPORT_FILENAME).is_file()


def test_source_only_module_has_no_native_solver_or_pp_entrypoint() -> None:
    module_text = Path(subject.__file__).read_text(encoding="utf-8")
    script_text = (
        Path(subject.__file__).parents[1]
        / "scripts"
        / "run_stride_structshell_crossmap_offline_profile.py"
    ).read_text(encoding="utf-8")
    forbidden_calls = (
        "lns2_selector.solver.native",
        "restore_repair_state",
        "run_closed_loop_collection",
        "environment.step",
    )
    assert all(token not in module_text for token in forbidden_calls)
    assert all(token not in script_text for token in forbidden_calls)
    assert "build\" / \"linux\" / \"project" not in script_text
    assert "lns2_env" not in sys.modules
    assert not any(name.startswith("lns2_selector.solver") for name in sys.modules)


def test_source_count_gate_never_authorizes_router_training() -> None:
    maps = []
    for winner in ("component16", "component16", "hotspot16", "hotspot16"):
        maps.append(
            {
                "informative_for_family_crossover_hypothesis": True,
                "source_outcome": {"point_winner": winner},
            }
        )
    evidence = subject.source_count_evidence(maps)
    assert evidence["source_count_gate_met"] is True
    assert evidence["additional_collection_allowed"] is True
    assert evidence["router_training_allowed"] is False


def test_analyze_rejects_a_process_that_already_loaded_native_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "lns2_env", object())
    with pytest.raises(RuntimeError, match="forbidden native modules"):
        subject.analyze(tmp_path / "source", tmp_path / "output")
