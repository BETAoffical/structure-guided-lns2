from __future__ import annotations

from experiments.stride_signature_scoped_rescue_continuation import (
    ARMS,
    FROZEN_ARM,
    ONE_SHOT_ARM,
    SIGNATURE_SCOPED_ARM,
    _episode_override,
    continuation_schedule,
    prepare_cases,
    run_collection,
)
from lns2_selector.runtime.failure_informed_rescue import (
    COMPACT_BLOCKER_AUGMENTED_MODE,
    CONTROL_MODE,
)


CONFIG = "configs/stride_signature_scoped_rescue_continuation_v1_registration.json"


def test_registration_and_schedule_are_frozen() -> None:
    loaded, cases = prepare_cases(CONFIG)
    assert loaded[2]["execution"]["worker_count"] == 16
    assert len(cases) == 45
    assert len(continuation_schedule(cases, range(4))) == 540
    assert ARMS == (FROZEN_ARM, ONE_SHOT_ARM, SIGNATURE_SCOPED_ARM)


def test_arms_share_first_action_and_use_distinct_rescue_contracts() -> None:
    _loaded, cases = prepare_cases(CONFIG)
    overrides = {
        arm: _episode_override(cases[0], trial_index=0, arm=arm) for arm in ARMS
    }
    assert len(
        {
            tuple(row["forced_first_action"]["agents"])
            for row in overrides.values()
        }
    ) == 1
    assert len(
        {
            int(row["forced_first_action"]["pp_random_seed"])
            for row in overrides.values()
        }
    ) == 1
    assert overrides[FROZEN_ARM]["failure_informed_rescue"]["mode"] == CONTROL_MODE
    assert (
        overrides[ONE_SHOT_ARM]["failure_informed_rescue"]["mode"]
        == COMPACT_BLOCKER_AUGMENTED_MODE
    )
    scoped = overrides[SIGNATURE_SCOPED_ARM]["signature_scoped_rescue"]
    assert scoped["minimum_consecutive_rollbacks"] == 3
    assert scoped["maximum_interventions"] == 3
    assert scoped["maximum_added_blockers"] == 8
    assert "failure_informed_rescue" not in overrides[SIGNATURE_SCOPED_ARM]


def test_dry_run_has_16_workers_and_three_arm_schedule() -> None:
    result = run_collection(
        CONFIG,
        "build/test-stride-signature-scoped-rescue-dry-run",
        phase="initial",
        dry_run=True,
        limit_cases=1,
    )
    assert result["case_count"] == 1
    assert result["schedule_entry_count"] == 12
    assert result["worker_count"] == 16
