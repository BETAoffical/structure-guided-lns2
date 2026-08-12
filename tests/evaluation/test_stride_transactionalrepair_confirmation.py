from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.stride_transactionalrepair import POLICIES, PRIMARY_POLICY
from experiments.stride_transactionalrepair_confirmation import (
    FORBIDDEN_SELECTION_FIELDS,
    _selected_neighborhood,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stride_transactionalrepair_confirmation_v1_registration.json"
DISCOVERY = ROOT / "configs/stride_transactionalrepair_v1_registration.json"


def test_confirmation_freezes_discovery_policy_gates_and_result_blind_scope() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    discovery = json.loads(DISCOVERY.read_text(encoding="utf-8"))
    assert tuple(config["policies"]["ids"]) == POLICIES
    assert config["policies"]["primary"] == PRIMARY_POLICY
    assert config["policies"] == discovery["policies"]
    assert config["readiness_gates"] == discovery["readiness_gates"]
    assert config["cohort"]["required_state_count"] == 240
    assert config["cohort"]["required_map_count"] == 22
    assert config["cohort"]["required_source_policy_counts"] == {
        "official_adaptive": 120,
        "v2-full": 120,
    }
    assert config["cohort"]["candidate_outcomes_read"] is False
    assert config["cohort"]["no_result_based_exclusion"] is True
    assert config["execution"]["worker_count"] == 16
    assert config["execution"]["stop_on_first_error_or_timeout"] is True
    assert config["claim_boundary"]["model_training_allowed"] is False
    assert config["claim_boundary"]["runtime_integration_allowed"] is False
    assert config["claim_boundary"]["ttf_experiment_allowed"] is False


def test_selected_neighborhood_reads_only_pre_repair_membership(monkeypatch) -> None:
    event = {
        "decision_index": 4,
        "before_fingerprint": "before",
        "action": {"mode": "official"},
        "metrics": {
            "neighborhood": [7, 2, 5],
            "replan_success": False,
            "failure_reason": "must_not_be_used_for_selection",
        },
        "after_fingerprint": "must_not_be_used_for_selection",
    }
    monkeypatch.setattr(
        "experiments.stride_transactionalrepair_confirmation.read_trace_events",
        lambda _path: [{"initial": True}, event, {"finish": True}],
    )
    agents, mode = _selected_neighborhood(
        {"decision_index": 4, "before_fingerprint": "before"}, Path("unused")
    )
    assert agents == [2, 5, 7]
    assert mode == "official"


def test_selected_neighborhood_rejects_state_identity_drift(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiments.stride_transactionalrepair_confirmation.read_trace_events",
        lambda _path: [
            {"initial": True},
            {
                "decision_index": 1,
                "before_fingerprint": "changed",
                "action": {"mode": "official"},
                "metrics": {"neighborhood": [1, 2]},
            },
            {"finish": True},
        ],
    )
    with pytest.raises(ValueError, match="before fingerprint changed"):
        _selected_neighborhood(
            {"decision_index": 1, "before_fingerprint": "registered"},
            Path("unused"),
        )


def test_result_blind_contract_forbids_outcome_and_future_fields() -> None:
    assert {
        "candidate_repair_outcome",
        "after_fingerprint",
        "controller_ttf",
        "future_trajectory",
        "repair_runtime",
    } <= FORBIDDEN_SELECTION_FIELDS
