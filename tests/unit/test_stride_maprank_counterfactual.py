from __future__ import annotations

from experiments.stride_maprank_counterfactual import (
    STATE_SCHEMA,
    TRIAL_SCHEMA,
    _state_artifact_valid,
)


def _payload() -> dict:
    candidates = [
        {"candidate_id": "v2"},
        {"candidate_id": "maprank"},
    ]
    trials = [
        {
            "schema": TRIAL_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "trial_index": trial_index,
        }
        for candidate in candidates
        for trial_index in range(16)
    ]
    return {
        "schema": STATE_SCHEMA,
        "run_fingerprint": "run",
        "state_id": "state",
        "complete": True,
        "candidates": candidates,
        "trials": trials,
    }


def test_counterfactual_state_requires_exact_two_by_sixteen_product() -> None:
    payload = _payload()
    assert _state_artifact_valid(
        payload, run_fingerprint="run", state_id="state"
    )
    payload["trials"].pop()
    assert not _state_artifact_valid(
        payload, run_fingerprint="run", state_id="state"
    )


def test_counterfactual_state_resume_identity_is_strict() -> None:
    payload = _payload()
    assert not _state_artifact_valid(
        payload, run_fingerprint="other", state_id="state"
    )
    assert not _state_artifact_valid(
        payload, run_fingerprint="run", state_id="other"
    )
