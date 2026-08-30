from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.closed_loop_confirmation import _closed_loop_episode_worker
from experiments.closed_loop_trace_storage import TRACE_FORMAT_FULL_V1
from lns2_selector.runtime.v2_first_consensus_rescue import (
    v2_first_consensus_rescue_augmentation,
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


def consensus_audit(
    *, component_agents: list[int], hotspot_agents: list[int]
) -> dict[str, object]:
    exact = sorted(component_agents) == sorted(hotspot_agents) and bool(
        component_agents
    )
    return {
        "component_candidate_id": "consensus" if exact else "component",
        "hotspot_candidate_id": "consensus" if exact else "hotspot",
        "component_agents": sorted(component_agents),
        "hotspot_agents": sorted(hotspot_agents),
        "component_available": True,
        "hotspot_available": True,
        "exact_agent_consensus": exact,
        "consensus_candidate_id": "consensus" if exact else None,
        "generated_candidate_count": 1 if exact else 2,
    }


def run_consensus_fixture(
    directory: str,
    *,
    rescue_candidate: dict | None,
    generation_audit: dict[str, object],
) -> tuple[dict, list[dict], RollbackCandidateFixture]:
    environment = RollbackPlatformEnvironment(solve_anchor=False)
    model = RollbackDirectCandidateModel()
    candidates = RollbackCandidateFixture()
    job = rollback_worker_job(directory, max_decisions=6)
    job["run_fingerprint"] = "v2-first-consensus16"
    job["trace_format"] = TRACE_FORMAT_FULL_V1
    job["proposal"]["hybridstructpool"] = (
        v2_first_consensus_rescue_augmentation()
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
            "generate_v2_first_consensus_rescue_candidate",
            return_value=(rescue_candidate, 0.0, generation_audit),
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


def test_worker_directly_executes_only_the_exact_consensus_candidate() -> None:
    structural = make_candidate(
        "consensus",
        [0, 1, 2, 3],
        "structpool-conflict-component:16",
    )
    structural["selection_families"] = [
        "structpool-conflict-component:16",
        "structpool-spatiotemporal-hotspot:16",
    ]
    structural["selection_rank_by_family"] = {
        "structpool-conflict-component:16": 0,
        "structpool-spatiotemporal-hotspot:16": 0,
    }
    structural["structpool_family_groups"] = [
        "conflict_component",
        "spatiotemporal_hotspot",
    ]
    structural["hybridstructpool_provenance"] = [
        "structshell_equal_four_size",
        "v2_first_consensus_rescue",
    ]
    audit = consensus_audit(
        component_agents=[0, 1, 2, 3],
        hotspot_agents=[0, 1, 2, 3],
    )

    with tempfile.TemporaryDirectory() as directory:
        result, events, candidates = run_consensus_fixture(
            directory,
            rescue_candidate=structural,
            generation_audit=audit,
        )

    assert result["status"] == "ok", result
    transitions = [row for row in events if row["event"] == "transition"]
    assert [
        row["controller"]["selected_candidate_id"] for row in transitions
    ] == [
        "v2-anchor",
        "v2-anchor",
        "v2-anchor",
        "consensus",
        "v2-anchor",
        "v2-anchor",
    ]
    assert candidates.generation_calls == 5
    offered = transitions[3]["controller"]["v2_first_consensus_rescue"]
    assert offered["selection"]["selection_phase"] == "consensus_rescue_due"
    assert offered["generation"]["exact_agent_consensus"] is True
    assert offered["generation"]["component_agents"] == [0, 1, 2, 3]
    assert offered["generation"]["hotspot_agents"] == [0, 1, 2, 3]
    assert offered["generation"]["challenger_present"] is True
    assert offered["generation"]["consumed"] is True
    assert offered["execution"]["structural_selected"] is True
    assert offered["execution"]["consumed"] is True
    assert offered["observation"]["decision_mode"] == "consensus_rescue"
    assert len(transitions[3]["controller"]["candidate_pool"]) == 1
    assert transitions[3]["controller"]["inference_seconds"] == 0.0
    assert all(
        row["controller"]["v2_first_consensus_rescue"]["selection"][
            "selection_phase"
        ]
        == "v2_only_after_offer"
        for row in transitions[4:]
    )


def test_worker_mismatch_consumes_offer_and_runs_fresh_v2_same_decision() -> None:
    audit = consensus_audit(
        component_agents=[0, 1, 2, 3],
        hotspot_agents=[0, 1, 2, 4],
    )
    with tempfile.TemporaryDirectory() as directory:
        result, events, candidates = run_consensus_fixture(
            directory,
            rescue_candidate=None,
            generation_audit=audit,
        )

    assert result["status"] == "ok", result
    transitions = [row for row in events if row["event"] == "transition"]
    assert candidates.generation_calls == 6
    assert all(
        row["controller"]["selected_candidate_id"] == "v2-anchor"
        for row in transitions
    )
    offered = transitions[3]["controller"]["v2_first_consensus_rescue"]
    assert offered["selection"]["offered"] is True
    assert offered["generation"]["exact_agent_consensus"] is False
    assert offered["generation"]["challenger_present"] is False
    assert offered["generation"]["consumed"] is True
    assert offered["generation"]["fallback"] == (
        "fresh_v2_after_no_consensus"
    )
    assert offered["execution"] is None
    assert transitions[3]["controller"]["proposal"][
        "v2_first_rescue_mode"
    ] == "fresh_v2_after_no_consensus"
    assert all(
        row["controller"]["v2_first_consensus_rescue"]["selection"][
            "selection_phase"
        ]
        == "v2_only_after_offer"
        for row in transitions[4:]
    )
