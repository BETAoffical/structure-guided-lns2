from __future__ import annotations

import json
from pathlib import Path

from experiments.stride_nativeorder_transactionalrepair import (
    AMENDED_CONFIG_SCHEMA,
    AUGMENTED_POLICY,
    BASELINE_POLICY,
    DEPLOYABLE_POLICIES,
    POLICIES,
    SAME_SET_POLICY,
    UPPER_BOUND_POLICY,
    retry_plan,
    retry_pp_seed,
)
from experiments.stride_repairability_collection import repairability_pp_seed


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stride_nativeorder_transactionalrepair_v1_registration.json"
R2_CONFIG = ROOT / "configs/stride_nativeorder_transactionalrepair_v1_r2_registration.json"


def test_registration_freezes_two_native_arms_and_one_order_upper_bound() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert tuple(config["policies"]["ids"]) == POLICIES
    assert tuple(config["policies"]["deployable_ids"]) == DEPLOYABLE_POLICIES
    assert config["policies"]["maximum_attempts"] == 2
    assert config["policies"]["maximum_added_external_blockers"] == 8
    assert config["execution"]["worker_count"] == 16
    assert config["execution"]["per_policy_job_timeout_seconds"] == 300
    assert config["claim_boundary"]["retry_does_not_scan_hybridstructpool"] is True
    assert config["claim_boundary"]["ttf_experiment_allowed"] is False


def test_retry_plans_keep_deployable_orders_native() -> None:
    base = [1, 2, 3]
    first_order = [3, 1, 2]
    blockers = [8, 9, 8, 10]
    assert retry_plan(BASELINE_POLICY, base, first_order, blockers, 2) is None
    assert retry_plan(SAME_SET_POLICY, base, first_order, blockers, 2) == (
        base,
        None,
        [],
        "fresh_native_same_set",
    )
    assert retry_plan(AUGMENTED_POLICY, base, first_order, blockers, 2) == (
        [1, 2, 3, 8, 9],
        None,
        [8, 9],
        "fresh_native_blocker_augmented",
    )
    assert retry_plan(UPPER_BOUND_POLICY, base, first_order, blockers, 2) == (
        [1, 2, 3, 8, 9],
        [3, 1, 2, 8, 9],
        [8, 9],
        "preserved_prefix_blocker_tail",
    )


def test_augmented_retry_reduces_to_same_set_when_no_blocker_is_exposed() -> None:
    agents, order, added, reason = retry_plan(
        AUGMENTED_POLICY, [1, 2], [2, 1], [], 8
    )
    assert agents == [1, 2]
    assert order is None
    assert added == []
    assert reason == "fresh_native_blocker_augmented"


def test_retry_seed_is_fresh_deterministic_and_in_range() -> None:
    fingerprint = "ab" * 32
    for trial in range(16):
        first = repairability_pp_seed(fingerprint, trial)
        retry = retry_pp_seed(fingerprint, trial)
        assert 0 <= retry < 2**31
        assert retry != first
        assert retry_pp_seed(fingerprint, trial) == retry


def test_r2_registration_adds_cooperative_budget_without_extending_process_cap() -> None:
    config = json.loads(R2_CONFIG.read_text(encoding="utf-8"))
    assert config["schema"] == AMENDED_CONFIG_SCHEMA
    assert config["execution"]["transaction_pp_wall_budget_seconds"] == 270
    assert config["execution"]["process_safety_reserve_seconds"] == 30
    assert config["execution"]["per_policy_job_timeout_seconds"] == 300
    assert config["superseded_extension"]["reuse_allowed"] is False
    assert (
        config["registered_initial_reuse"][
            "maximum_observed_total_attempt_wall_seconds"
        ]
        < config["execution"]["transaction_pp_wall_budget_seconds"]
    )
