from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _write_json, _write_jsonl
import experiments.stride_hierarchical_ch_compact_flow_h1_collection_v1 as module
from experiments.stride_hierarchical_ch_labels_v1 import (
    build_hierarchical_ch_labels,
)


def _candidate(role: str, agents: list[int], candidate_id: str) -> dict:
    return {
        "role": role,
        "candidate_id": candidate_id,
        "agents": agents,
        "actual_size": len(agents),
        "selection_families": [],
        "features": {},
    }


def _selected(state_id: str = "selected-1") -> dict:
    return {
        "schema": module.SELECTED_STATE_SCHEMA,
        "state_occurrence_id": state_id,
        "state_id": state_id,
        "split": "train",
        "map_id": "map-a",
        "map_family": "dao",
        "agent_count": 64,
        "task_id": "task-a",
        "episode_id": "episode-a",
        "solver_seed": 41,
        "decision_index": 2,
        "before_fingerprint": "before-state",
        "before_conflicts": 10,
        "source_root": "/synthetic/source",
        "source_policy": "v2-full",
        "source_trace_file": "traces/a.jsonl.gz",
        "source_trace_sha256": "a" * 64,
        "prefix_actions": [{"mode": "Adaptive"}],
        "depth_band": "d1_3",
        "target_outcome_fields_read": False,
        "outcome_filtering": False,
        "reserve_or_replacement_backfill": False,
        "sequential_design_only": True,
        "training_authorized": False,
    }


def _fake_preflight(arms: dict[str, dict]):
    def run(decision: dict, *, bundle_root: str) -> dict:
        assert bundle_root
        return {
            **decision,
            "agent_count": 64,
            "before_conflicts": 10,
            "before_repair_fingerprint": "before-repair",
            "source_trace_path": "/synthetic/source/traces/a.jsonl.gz",
            "arms": arms,
            "eligible": len({tuple(row["agents"]) for row in arms.values()}) == 3,
            "ineligibility_reasons": (
                []
                if len({tuple(row["agents"]) for row in arms.values()}) == 3
                else ["arm_agent_sets_not_pairwise_distinct"]
            ),
            "target_outcome_fields_read": False,
            "candidate_repair_actions_executed": False,
        }

    return run


def _arms(partition: str) -> dict[str, dict]:
    anchor = list(range(0, 16))
    component = list(range(16, 32))
    hotspot = list(range(32, 48))
    if partition == "consensus":
        hotspot = list(reversed(component))
    elif partition == "anchor_component":
        component = list(reversed(anchor))
    elif partition == "anchor_hotspot":
        hotspot = list(reversed(anchor))
    elif partition == "all_equal":
        component = list(reversed(anchor))
        hotspot = list(anchor)
    return {
        "v2_anchor": _candidate("v2_anchor", anchor, "v"),
        "component16": _candidate("component16", component, "c"),
        "hotspot16": _candidate("hotspot16", hotspot, "h"),
    }


def test_config_is_frozen_and_workers_are_bounded() -> None:
    config = _read_json(
        Path("configs/stride_hierarchical_ch_compact_flow_h1_collection_v1.json")
    )
    module.validate_config(config)
    assert module._workers(config, None) == 16
    assert module._workers(config, 20) == 20
    with pytest.raises(ValueError, match="1..20"):
        module._workers(config, 21)
    changed = json.loads(json.dumps(config))
    changed["claim_boundary"]["training_authorized"] = True
    with pytest.raises(ValueError, match="claim boundary"):
        module.validate_config(changed)


@pytest.mark.parametrize(
    ("partition", "expected", "unique_count", "stage1_count", "stage2"),
    [
        ("consensus", "consensus_structural", 2, 1, False),
        ("anchor_component", "anchor_component_shared", 2, 1, True),
        ("anchor_hotspot", "anchor_hotspot_shared", 2, 1, True),
        ("distinct", "three_unique", 3, 2, True),
        ("all_equal", "single_unique", 1, 0, False),
    ],
)
def test_preflight_exact_tuple_dedup_and_hierarchy(
    partition: str,
    expected: str,
    unique_count: int,
    stage1_count: int,
    stage2: bool,
) -> None:
    row = module.prepare_selected_state(
        _selected(),
        bundle_root="/synthetic/bundle",
        preflight_fn=_fake_preflight(_arms(partition)),
    )
    assert row["canonical_partition"] == expected
    assert row["unique_action_count"] == unique_count
    assert len(row["stage1_structural_action_ids"]) == stage1_count
    assert row["stage2_eligible"] is stage2
    assert row["all_equal_excluded"] is (partition == "all_equal")
    exact_sets = [tuple(action["agents"]) for action in row["unique_actions"]]
    assert len(exact_sets) == len(set(exact_sets))
    assert row["target_outcome_fields_read"] is False
    assert row["candidate_repair_actions_executed"] is False


def test_prepare_calls_module_preflight_without_solver_in_test(monkeypatch) -> None:
    calls = []

    def fake(decision: dict, *, bundle_root: str) -> dict:
        calls.append((decision["state_occurrence_id"], bundle_root))
        return _fake_preflight(_arms("consensus"))(decision, bundle_root=bundle_root)

    monkeypatch.setattr(module, "_preflight_decision", fake)
    row = module.prepare_selected_state(_selected("s-call"), bundle_root="bundle")
    assert calls == [("s-call", "bundle")]
    assert row["unique_action_count"] == 2


def _write_synthetic_project(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "configs").mkdir()
    controller = tmp_path / "artifacts" / "controller"
    controller.mkdir(parents=True)
    manifest = controller / "controller_manifest.json"
    _write_json(manifest, {"synthetic": True})
    selected_path = tmp_path / "selection" / "selected_states.jsonl"
    selected_path.parent.mkdir()
    _write_jsonl(selected_path, [_selected("dry-state")])
    trust_path = selected_path.parent / "trust.json"
    _write_json(
        trust_path,
        {
            "schema": module.SELECTION_TRUST_SCHEMA,
            "complete": True,
            "selected_state_count": 1,
            "selected_states_sha256": sha256_file(selected_path),
            "target_outcome_fields_read": False,
            "outcome_filtering": False,
            "reserve_or_replacement_backfill": False,
            "training_authorized": False,
        },
    )
    source_config = _read_json(
        Path("configs/stride_hierarchical_ch_compact_flow_h1_collection_v1.json")
    )
    source_config["inputs"]["selected_states"] = "selection/selected_states.jsonl"
    source_config["inputs"]["selection_trust_report"] = "selection/trust.json"
    source_config["inputs"]["controller_bundle"] = {
        "path": "artifacts/controller",
        "manifest_sha256": sha256_file(manifest),
    }
    config_path = tmp_path / "configs" / "config.json"
    _write_json(config_path, source_config)
    return config_path, tmp_path / "output"


def test_dry_run_uses_no_preflight_or_job_runner_and_workers_enter_identity(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, output = _write_synthetic_project(tmp_path)

    def forbidden(*args, **kwargs):  # pragma: no cover - asserted by absence
        raise AssertionError("dry-run must not generate candidates or run workers")

    monkeypatch.setattr(module, "_preflight_decision", forbidden)
    monkeypatch.setattr(module, "_run_jobs", forbidden)
    sixteen = module.run_preflight(config_path, output, dry_run=True, workers=16)
    twenty = module.run_preflight(config_path, output, dry_run=True, workers=20)
    assert sixteen["status"] == "dry_run"
    assert sixteen["candidate_repair_actions_executed"] is False
    assert sixteen["workers_in_fingerprint"] is True
    assert sixteen["run_identity"] != twenty["run_identity"]
    assert not output.exists()


class _FakeEnvironment:
    def __init__(self, calls: list[dict]) -> None:
        self.calls = calls

    def step_with_time_limit(self, action: dict, time_limit: float) -> dict:
        self.calls.append({"action": dict(action), "time_limit": time_limit})
        return {"action": dict(action)}


def _run_synthetic_state_worker(
    tmp_path: Path, monkeypatch, *, partition: str = "distinct"
) -> tuple[dict, dict, list[dict]]:
    state_row = module.prepare_selected_state(
        _selected("worker-state"),
        bundle_root="bundle",
        preflight_fn=_fake_preflight(_arms(partition)),
    )
    calls: list[dict] = []
    snapshot = {
        "source_state": {
            "num_of_colliding_pairs": 10,
            "sum_of_costs": 100,
            "rfp": "before-repair",
        },
        "source_trace_file": state_row["source_trace_file"],
        "source_trace_path": state_row["source_trace_path"],
        "source_trace_sha256": state_row["source_trace_sha256"],
        "before_fingerprint": state_row["before_fingerprint"],
        "before_repair_fingerprint": "before-repair",
        "before_conflicts": 10,
        "before_sum_of_costs": 100,
        "restore_seed": 123,
    }
    monkeypatch.setattr(module, "_source_snapshot", lambda row: snapshot)
    monkeypatch.setattr(module, "_replay_job", lambda row: {"synthetic": True})
    monkeypatch.setattr(
        module,
        "restore_repair_state",
        lambda replay, source, seed: (
            _FakeEnvironment(calls),
            {
                "num_of_colliding_pairs": 10,
                "sum_of_costs": 100,
                "rfp": "before-repair",
            },
        ),
    )
    monkeypatch.setattr(module, "repair_structure_fingerprint", lambda row: row["rfp"])

    def validate(result: dict, *, expected_agents: list[int], expected_seed: int):
        assert result["action"]["agents"] == expected_agents
        assert result["action"]["random_seed"] == expected_seed
        return (
            {
                "num_of_colliding_pairs": 8,
                "sum_of_costs": 100,
                "rfp": f"after-{expected_seed}",
            },
            {
                "replan_success": True,
                "pp_rolled_back": False,
                "pp_failure_reason": "",
                "repair_order": list(expected_agents),
                "requested_random_seed": expected_seed,
                "requested_pp_random_seed": expected_seed,
                "applied_pp_random_seed": expected_seed,
                "native_step_seconds": 0.1,
                "pp_replan_seconds": 0.09,
            },
        )

    monkeypatch.setattr(module, "_validate_native_repair", validate)
    output_path = tmp_path / "h1_states" / "worker-state.json"
    result = module._state_worker(
        {
            "job_id": "worker-state",
            "state_row": state_row,
            "identity": "synthetic-run",
            "per_action_time_limit_seconds": 5.0,
            "output_path": str(output_path),
            "resume": False,
        }
    )
    return result, _read_json(output_path), calls


def test_worker_executes_each_exact_action_once_per_paired_seed(
    tmp_path: Path, monkeypatch
) -> None:
    result, payload, calls = _run_synthetic_state_worker(tmp_path, monkeypatch)
    assert result["trial_count"] == 3 * 16
    assert len(calls) == 3 * 16
    trials = payload["trials"]
    by_index = collections.defaultdict(list)
    for row in trials:
        by_index[row["trial_index"]].append(row["pp_seed"])
    assert set(by_index) == set(range(16))
    assert all(len(seeds) == 3 and len(set(seeds)) == 1 for seeds in by_index.values())
    action_seed_pairs = {(row["action_id"], row["pp_seed"]) for row in trials}
    assert len(action_seed_pairs) == 3 * 16
    assert all(row["runtime_used_in_label"] is False for row in trials)


def test_state_payload_is_direct_label_builder_input(
    tmp_path: Path, monkeypatch
) -> None:
    _result, payload, _calls = _run_synthetic_state_worker(tmp_path, monkeypatch)
    state_path = tmp_path / "h1_states" / "worker-state.json"
    manifest_path = tmp_path / "h1_state_manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "state_occurrence_id": "worker-state",
                "state_file": "h1_states/worker-state.json",
                "state_sha256": sha256_file(state_path),
                "unique_action_count": 3,
                "trial_count": 48,
            }
        ],
    )
    config = {
        "schema": "lns2.stride.hierarchical_ch_labels_config.v1",
        "experiment_id": "compact-flow-synthetic-label-test",
        "source": {
            "manifest": {
                "path": "h1_state_manifest.jsonl",
                "sha256": sha256_file(manifest_path),
            },
            "state_schema": module.H1_STATE_SCHEMA,
        },
        "expected": {
            "state_count": 1,
            "unique_action_count": 3,
            "trial_count": 48,
            "stage1_pair_count": 2,
            "stage2_state_count": 1,
        },
    }
    report = build_hierarchical_ch_labels(
        config, project_root=tmp_path, output=tmp_path / "labels"
    )
    assert report["complete"] is True
    assert report["observed"]["stage1_pair_count"] == 2
    assert report["observed"]["stage2_state_count"] == 1
    assert report["training_authorized"] is False
    assert payload["runtime_used_in_label"] is False


@pytest.mark.parametrize(
    ("expected_state_ids", "manifest_state_ids", "expected"),
    [
        (["state-a", "state-b"], ["state-a", "state-b"], True),
        (["state-a", "state-b"], ["state-a"], False),
        (["state-a", "state-b"], ["state-a", "state-a"], False),
        (["state-a", "state-a"], ["state-a", "state-a"], False),
    ],
)
def test_collection_state_id_inventory_requires_exact_unique_coverage(
    expected_state_ids: list[str],
    manifest_state_ids: list[str],
    expected: bool,
) -> None:
    assert (
        module._state_id_inventory_complete(expected_state_ids, manifest_state_ids)
        is expected
    )
