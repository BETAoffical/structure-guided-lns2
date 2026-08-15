from __future__ import annotations

from experiments.stride_exactstate_controller_attribution import (
    ADAPTIVE_ARM,
    ARMS,
    FROZEN_ARM,
    TARGET_ARM,
    _episode_override,
    continuation_schedule,
    prepare_cases,
    run_collection,
)


CONFIG = "configs/stride_exactstate_controller_attribution_v1_registration.json"


def test_registration_and_schedule_are_frozen() -> None:
    loaded, cases = prepare_cases(CONFIG)
    assert loaded[2]["execution"]["worker_count"] == 16
    assert len(cases) == 45
    assert len(continuation_schedule(cases, range(4))) == 540
    assert ARMS == (FROZEN_ARM, ADAPTIVE_ARM, TARGET_ARM)


def test_exact_state_override_has_no_forced_action_or_retry() -> None:
    _loaded, cases = prepare_cases(CONFIG)
    trial0 = _episode_override(cases[0], trial_index=0)
    trial1 = _episode_override(cases[0], trial_index=1)
    assert "forced_first_action" not in trial0
    assert "bounded_native_retry" not in trial0
    assert "failure_informed_rescue" not in trial0
    assert "signature_scoped_rescue" not in trial0
    assert trial0["initial_restore"]["repair_structure_fingerprint"] == trial1[
        "initial_restore"
    ]["repair_structure_fingerprint"]
    assert trial0["initial_restore"]["restore_seed"] != trial1["initial_restore"][
        "restore_seed"
    ]
    assert trial0["pp_replay_seed_salt"] != trial1["pp_replay_seed_salt"]


def test_dry_run_has_16_workers_and_three_arm_schedule() -> None:
    result = run_collection(
        CONFIG,
        "build/test-stride-exactstate-controller-attribution-dry-run",
        phase="initial",
        dry_run=True,
        limit_cases=1,
    )
    assert result["case_count"] == 1
    assert result["schedule_entry_count"] == 12
    assert result["worker_count"] == 16
