from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.closed_loop_confirmation import _closed_loop_episode_worker
from experiments.closed_loop_trace_storage import TRACE_FORMAT_FULL_V1
from experiments.stride_v2first_family_rescue_g1 import _v2_first_trace_audit
from lns2_selector.runtime.v2_first_single_family_rescue import (
    v2_first_single_family_rescue_augmentation,
)
from tests.runtime.test_closed_loop_confirmation import (
    RollbackCandidateFixture,
    RollbackDirectCandidateModel,
    RollbackFeatureEngine,
    RollbackPlatformEnvironment,
    RollbackTopologyCache,
    make_candidate,
    rollback_bundle,
    rollback_worker_job,
)


def run_v2_first_fixture(
    directory: str,
    *,
    rescue_candidate: dict | None,
) -> tuple[dict, list[dict], RollbackCandidateFixture]:
    environment = RollbackPlatformEnvironment(solve_anchor=False)
    model = RollbackDirectCandidateModel()
    candidates = RollbackCandidateFixture()
    job = rollback_worker_job(directory, max_decisions=6)
    job["run_fingerprint"] = "v2-first-component16"
    job["trace_format"] = TRACE_FORMAT_FULL_V1
    job["proposal"]["hybridstructpool"] = (
        v2_first_single_family_rescue_augmentation("conflict_component")
    )
    RollbackFeatureEngine.instances.clear()

    with (
        patch(
            "experiments.closed_loop_confirmation._make_environment",
            return_value=environment,
        ),
        patch(
            "experiments.closed_loop_confirmation.load_frozen_policy_bundle",
            return_value=rollback_bundle(model),
        ),
        patch(
            "experiments.closed_loop_confirmation.compact_runtime_model",
            side_effect=lambda value: value,
        ),
        patch(
            "experiments.closed_loop_confirmation.OnlineFeatureEngine",
            RollbackFeatureEngine,
        ),
        patch(
            "experiments.closed_loop_confirmation.TopologyAnalysisCache",
            RollbackTopologyCache,
        ),
        patch(
            "experiments.closed_loop_confirmation.generate_online_candidates",
            side_effect=candidates.generate_base,
        ),
        patch(
            "experiments.closed_loop_confirmation."
            "generate_v2_first_single_family_rescue_candidate",
            return_value=(rescue_candidate, 0.0),
        ) as rescue_generator,
    ):
        result = _closed_loop_episode_worker(job)

    assert rescue_generator.call_count == 1
    events = [
        json.loads(line)
        for line in (Path(directory) / result["trace_file"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return result, events, candidates


def audit_rows(events: list[dict]) -> list[dict]:
    rows = []
    for event in events:
        if event.get("event") != "transition":
            continue
        controller = dict(event["controller"])
        rescue = dict(controller["v2_first_single_family_rescue"])
        selection = dict(rescue["selection"])
        observation = dict(rescue["observation"])
        rows.append(
            {
                "decision_index": int(event["decision_index"]),
                "before_platform_signature": str(
                    selection["before_repair_fingerprint"]
                ),
                "after_platform_signature": str(
                    observation["after_repair_fingerprint"]
                ),
                "before_conflicts": int(controller["route_conflicts"]),
                "after_conflicts": int(
                    event["after"]["num_of_colliding_pairs"]
                ),
                "actual_action": dict(event["action"]),
                "actual_metrics": dict(event["metrics"]),
                "controller": controller,
            }
        )
    return rows


def test_worker_keeps_v2_prefix_then_directly_executes_one_component16() -> None:
    structural = make_candidate(
        "component16",
        [0, 1, 2, 3],
        "structpool-conflict-component:16",
    )
    structural["selection_rank_by_family"] = {
        "structpool-conflict-component:16": 0
    }
    structural["structpool_family_groups"] = ["conflict_component"]
    structural["hybridstructpool_provenance"] = [
        "structshell_equal_four_size",
        "v2_first_single_family_rescue",
    ]

    with tempfile.TemporaryDirectory() as directory:
        result, events, candidates = run_v2_first_fixture(
            directory, rescue_candidate=structural
        )

    assert result["status"] == "ok", result
    transitions = [row for row in events if row["event"] == "transition"]
    assert [
        row["controller"]["selected_candidate_id"] for row in transitions
    ] == [
        "v2-anchor",
        "v2-anchor",
        "v2-anchor",
        "component16",
        "v2-anchor",
        "v2-anchor",
    ]
    # No V2 proposal is generated on the structural decision itself.
    assert candidates.generation_calls == 5
    offered = transitions[3]["controller"][
        "v2_first_single_family_rescue"
    ]
    assert offered["selection"]["decision_index"] == 3
    assert offered["selection"]["offered"]
    assert offered["generation"]["challenger_present"]
    assert offered["generation"]["consumed"]
    assert offered["execution"]["structural_selected"]
    assert offered["observation"]["decision_mode"] == (
        "single_family_rescue"
    )
    assert len(transitions[3]["controller"]["candidate_pool"]) == 1
    assert transitions[3]["controller"]["inference_seconds"] == 0.0
    totals = result["summary"]["controller_totals"]
    assert totals["v2_first_rescue_offer_count"] == 1
    assert totals["v2_first_rescue_consumed_count"] == 1
    assert totals["v2_first_rescue_structural_selected_count"] == 1
    assert totals["v2_first_rescue_executed_count"] == 1
    assert all(
        row["controller"]["v2_first_single_family_rescue"]["selection"][
            "selection_phase"
        ]
        == "v2_only_after_offer"
        for row in transitions[4:]
    )


def test_runner_linear_audit_accepts_the_real_mocked_trace_contract() -> None:
    structural = make_candidate(
        "component16",
        [0, 1, 2, 3],
        "structpool-conflict-component:16",
    )
    structural["selection_rank_by_family"] = {
        "structpool-conflict-component:16": 0
    }
    structural["structpool_family_groups"] = ["conflict_component"]
    structural["hybridstructpool_provenance"] = [
        "structshell_equal_four_size",
        "v2_first_single_family_rescue",
    ]
    with (
        tempfile.TemporaryDirectory() as baseline_directory,
        tempfile.TemporaryDirectory() as rescue_directory,
    ):
        _baseline_result, baseline_events, _baseline_candidates = (
            run_v2_first_fixture(baseline_directory, rescue_candidate=None)
        )
        _rescue_result, rescue_events, _rescue_candidates = (
            run_v2_first_fixture(
                rescue_directory, rescue_candidate=structural
            )
        )
        baseline = audit_rows(baseline_events)
        rescue = audit_rows(rescue_events)
        audit = _v2_first_trace_audit(
            rescue, baseline, "conflict_component"
        )

    assert audit["passed"]
    assert audit["trigger_exact_rollback_count"] == 3
    assert audit["executed"]
    assert audit["baseline_counterfactual"]["seed_parity"]


def test_worker_unavailable_offer_runs_fresh_v2_and_never_reoffers() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result, events, candidates = run_v2_first_fixture(
            directory, rescue_candidate=None
        )

    assert result["status"] == "ok", result
    transitions = [row for row in events if row["event"] == "transition"]
    assert candidates.generation_calls == 6
    assert all(
        row["controller"]["selected_candidate_id"] == "v2-anchor"
        for row in transitions
    )
    offered = transitions[3]["controller"][
        "v2_first_single_family_rescue"
    ]
    assert offered["selection"]["offered"]
    assert not offered["generation"]["challenger_present"]
    assert not offered["generation"]["structural_selected"]
    assert offered["generation"]["consumed"]
    assert offered["execution"] is None
    assert transitions[3]["controller"]["proposal"][
        "v2_first_rescue_mode"
    ] == "fresh_v2_after_unavailable"
    totals = result["summary"]["controller_totals"]
    assert totals["v2_first_rescue_offer_count"] == 1
    assert totals["v2_first_rescue_unavailable_count"] == 1
    assert totals.get("v2_first_rescue_executed_count", 0) == 0
    assert all(
        row["controller"]["v2_first_single_family_rescue"]["selection"][
            "selection_phase"
        ]
        == "v2_only_after_offer"
        for row in transitions[4:]
    )
