from __future__ import annotations

import pytest

from experiments import v3_wall_clock_history_audit as module


def _row(candidate: float, reference: float, *, agents: int = 400) -> dict:
    return {
        "candidate_restricted_time_to_feasible": candidate,
        "reference_restricted_time_to_feasible": reference,
        "candidate_success": True,
        "reference_success": True,
        "common_success": True,
        "agent_count": agents,
    }


def test_pairwise_summary_uses_restricted_time_and_success() -> None:
    report = module.summarize_pairwise_rows(
        [_row(1.0, 2.0), _row(3.0, 2.0)],
        pair="v3_vs_v2",
        group="all",
    )
    assert report["candidate_success_count"] == 2
    assert report["reference_success_count"] == 2
    assert report["capped_ttf_ratio"] == 1.0
    assert report["candidate_faster_count"] == 1
    assert report["reference_faster_count"] == 1


def test_pairwise_summary_keeps_failed_episode_cap() -> None:
    row = _row(300.0, 5.0)
    row["candidate_success"] = False
    row["common_success"] = False
    report = module.summarize_pairwise_rows(
        [row], pair="v3_vs_v2", group="all"
    )
    assert report["candidate_capped_ttf_mean"] == 300.0
    assert report["common_success_count"] == 0
    assert report["common_success_ttf_ratio"] is None


def test_agent_summary_separates_sub600_and_600() -> None:
    rows = []
    for agents, candidate, reference in ((400, 1.0, 2.0), (600, 3.0, 2.0)):
        row = _row(candidate, reference, agents=agents)
        row["pair"] = "v3_vs_v2"
        rows.append(row)
    summaries = {
        row["group"]: row for row in module.pairwise_summary_rows(rows)
    }
    assert summaries["agents_lt_600"]["capped_ttf_ratio"] == 0.5
    assert summaries["agents_600"]["capped_ttf_ratio"] == 1.5


def test_agent_summary_keeps_individual_agent_groups() -> None:
    rows = []
    for agents, candidate, reference in ((200, 0.9, 1.0), (400, 1.1, 1.0)):
        row = _row(candidate, reference, agents=agents)
        row["pair"] = "v3_vs_v2"
        rows.append(row)
    summaries = {
        row["group"]: row for row in module.pairwise_summary_rows(rows)
    }
    assert summaries["agents_200"]["capped_ttf_ratio"] == 0.9
    assert summaries["agents_200"]["candidate_faster_count"] == 1
    assert summaries["agents_400"]["capped_ttf_ratio"] == 1.1


def test_empty_summary_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        module.summarize_pairwise_rows([], pair="pair", group="all")
