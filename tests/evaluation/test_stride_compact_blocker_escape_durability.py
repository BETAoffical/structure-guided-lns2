from __future__ import annotations

from experiments.repair_collection import _read_json
from experiments.stride_compact_blocker_escape_durability import (
    ARMS,
    _durability_metrics,
    load_audit_registration,
)
from experiments.stride_compact_blocker_rescue_continuation import (
    INITIAL_TRIALS,
    compact_blocker_schedule,
    prepare_factorial_cases,
)


CONFIG = "configs/stride_compact_blocker_escape_durability_v1_registration.json"


def _row(
    before: str,
    after: str,
    *,
    exact: bool = False,
    edges: set[tuple[int, int]] | None = None,
    conflicts: int = 2,
    feasible: bool = False,
) -> dict:
    selected_edges = {(0, 1), (1, 2)} if edges is None else edges
    return {
        "before_repair_fingerprint": before,
        "after_repair_fingerprint": after,
        "after_edges": selected_edges,
        "after_conflicts": conflicts,
        "after_feasible": feasible,
        "exact_rollback": exact,
    }


def _origin() -> dict:
    return _row("origin", "origin", exact=True)


def test_sustained_escape_requires_no_origin_reentry_or_new_platform() -> None:
    rows = [
        _origin(),
        _row("origin", "a", edges={(0, 1)}),
        _row("a", "b", edges={(0, 1)}, conflicts=1),
        _row("b", "c", edges=set(), conflicts=0),
    ]
    metrics = _durability_metrics(rows, 3)
    assert metrics["observed"] is True
    assert metrics["sustained_escape"] is True
    assert metrics["original_platform_reentered"] is False
    assert metrics["new_platform_formed"] is False
    assert metrics["original_edge_retention_auc"] == 1.0 / 3.0


def test_origin_reentry_invalidates_escape() -> None:
    rows = [
        _origin(),
        _row("origin", "a"),
        _row("a", "origin", exact=False),
        _row("origin", "b"),
    ]
    metrics = _durability_metrics(rows, 3)
    assert metrics["sustained_escape"] is False
    assert metrics["original_platform_reentered"] is True
    assert metrics["original_reentry_offset"] == 2


def test_three_exact_rollbacks_form_a_new_platform() -> None:
    rows = [
        _origin(),
        _row("origin", "a"),
        _row("a", "a", exact=True),
        _row("a", "a", exact=True),
        _row("a", "a", exact=True),
    ]
    metrics = _durability_metrics(rows, 4)
    assert metrics["sustained_escape"] is False
    assert metrics["new_platform_formed"] is True
    assert metrics["new_platform_offset"] == 4


def test_success_is_observed_but_short_unsolved_trace_is_censored() -> None:
    successful = _durability_metrics(
        [_origin(), _row("origin", "feasible", conflicts=0, feasible=True)],
        8,
    )
    assert successful["observed"] is True
    assert successful["decision_count"] == 1
    assert successful["sustained_escape"] is True
    censored = _durability_metrics([_origin(), _row("origin", "a")], 3)
    assert censored == {
        "observed": False,
        "right_censored": True,
        "decision_count": 1,
    }


def test_registration_selects_complete_paired_compact_state_cohort() -> None:
    _path, _root, config, inputs = load_audit_registration(CONFIG)
    report = _read_json(inputs["source_report"])
    _loaded, cases = prepare_factorial_cases(inputs["source_registration"])
    schedule = compact_blocker_schedule(cases, INITIAL_TRIALS)
    states = set(map(str, report["compact_eligible_state_fingerprints"]))
    selected = [
        row
        for row in schedule
        if str(row["state_fingerprint"]) in states
        and str(row["arm"]) in ARMS
    ]
    assert len(states) == int(
        config["population"]["required_compact_eligible_state_count"]
    )
    assert len(selected) == len(states) * len(INITIAL_TRIALS) * len(ARMS)
    assert {row["arm"] for row in selected} == set(ARMS)
