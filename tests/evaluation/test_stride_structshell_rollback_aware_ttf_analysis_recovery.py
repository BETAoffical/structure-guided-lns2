from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.closed_loop_trace_storage import EPISODE_SCHEMA_V2
from experiments.stride_structshell_rollback_aware_ttf_analysis_recovery import (
    observed_decision_rows,
)


def _events(*, after_fingerprint: str = "full-after") -> list[dict]:
    return [
        {"schema": EPISODE_SCHEMA_V2},
        {
            "schema": EPISODE_SCHEMA_V2,
            "decision_index": 0,
            "before_fingerprint": "full-before",
            "after_fingerprint": after_fingerprint,
            "state_delta": {"ignored": True},
            "state_extras_delta": {},
            "action": {"mode": "official"},
            "controller": {"route": "official_adaptive"},
            "metrics": {
                "conflicts_before": 2,
                "conflicts_after": 1,
                "requested_pp_random_seed": -1,
                "applied_pp_random_seed": -1,
                "termination_reason": "success",
                "replan_success": True,
                "pp_rolled_back": False,
            },
        },
        {"schema": EPISODE_SCHEMA_V2, "event": "terminal"},
    ]


def _state_fingerprint(state: dict) -> str:
    return f"full-{state['tag']}"


def _repair_fingerprint(state: dict) -> str:
    return f"repair-{state['tag']}"


def _run(events: list[dict]) -> list[dict]:
    before = {
        "tag": "before",
        "num_of_colliding_pairs": 2,
        "feasible": False,
    }
    after = {
        "tag": "after",
        "num_of_colliding_pairs": 1,
        "feasible": False,
    }
    with patch(
        "experiments.stride_structshell_rollback_aware_ttf_analysis_recovery.contained_file",
        return_value=Path("trace.jsonl.gz"),
    ), patch(
        "experiments.stride_structshell_rollback_aware_ttf_analysis_recovery.sha256_file",
        return_value="trace-sha",
    ), patch(
        "experiments.stride_structshell_rollback_aware_ttf_analysis_recovery.read_trace_events",
        return_value=events,
    ), patch(
        "experiments.stride_structshell_rollback_aware_ttf_analysis_recovery._initial_state",
        return_value=before,
    ), patch(
        "experiments.stride_structshell_rollback_aware_ttf_analysis_recovery.apply_state_delta",
        return_value=after,
    ), patch(
        "experiments.stride_structshell_rollback_aware_ttf_analysis_recovery.apply_extras_delta",
        return_value={},
    ), patch(
        "experiments.stride_structshell_rollback_aware_ttf_analysis_recovery.state_fingerprint",
        side_effect=_state_fingerprint,
    ), patch(
        "experiments.stride_structshell_rollback_aware_ttf_analysis_recovery.repair_structure_fingerprint",
        side_effect=_repair_fingerprint,
    ):
        return observed_decision_rows(
            Path("collection"),
            {"trace_file": "trace.jsonl.gz", "trace_sha256": "trace-sha"},
        )


def test_observed_reader_accepts_official_native_unseeded_pp() -> None:
    rows = _run(_events())
    assert len(rows) == 1
    assert rows[0]["before_platform_signature"] == "repair-before"
    assert rows[0]["after_platform_signature"] == "repair-after"
    assert rows[0]["actual_action"] == {"mode": "official"}
    assert rows[0]["controller"]["route"] == "official_adaptive"


def test_observed_reader_rejects_tampered_after_fingerprint() -> None:
    with pytest.raises(ValueError, match="after fingerprint mismatch"):
        _run(_events(after_fingerprint="tampered"))
