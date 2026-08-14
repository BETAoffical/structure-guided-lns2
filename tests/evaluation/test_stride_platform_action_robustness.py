from __future__ import annotations

from experiments.stride_platform_action_robustness import analyze_rows


def _rows(
    case_id: str,
    candidate_id: str,
    platforms: list[bool],
    *,
    historical: bool = False,
    deterministic: bool = False,
    success: bool = True,
    auc: float = 0.2,
) -> list[dict]:
    roles = []
    if historical:
        roles.append("historical_actual")
    if deterministic:
        roles.append("deterministic_compact_augment")
    return [
        {
            "case_id": case_id,
            "candidate_id": candidate_id,
            "candidate_size": 16,
            "logical_roles": roles,
            "frontier_variant": "compact-augment" if deterministic else None,
            "trial_index": trial,
            "entered_platform": platform,
            "success": success,
            "normalized_fixed_auc": auc,
            "repair_iterations": 20,
            "map_id": "maze-a",
        }
        for trial, platform in enumerate(platforms)
    ]


def test_stable_alternative_requires_improvement_in_both_seed_halves() -> None:
    actual = [True] * 8
    one_half_only = [False, True, True, True, True, True, True, True]
    both_halves = [False, True, True, True, False, True, True, True]
    episodes = [
        *_rows("case-a", "actual", actual, historical=True),
        *_rows("case-a", "unstable", one_half_only),
        *_rows("case-a", "stable", both_halves),
    ]
    report = analyze_rows(episodes, {"case-a": "set"})
    case = report["cases"][0]
    assert case["classification"] == "selection_headroom"
    assert case["stable_candidate_ids"] == ["stable"]


def test_deterministic_stable_action_is_reported_separately() -> None:
    episodes = [
        *_rows("case-a", "actual", [True] * 8, historical=True),
        *_rows(
            "case-a",
            "selected",
            [False, True, True, True, False, True, True, True],
            deterministic=True,
        ),
    ]
    report = analyze_rows(episodes, {"case-a": "order"})
    assert report["cases"][0]["classification"] == "deterministic_rule_success"


def test_platform_only_headroom_keeps_quality_failure_visible() -> None:
    episodes = [
        *_rows("case-a", "actual", [True] * 8, historical=True, success=True),
        *_rows(
            "case-a",
            "safer-but-worse",
            [False, True, True, True, False, True, True, True],
            success=False,
        ),
    ]
    report = analyze_rows(episodes, {"case-a": "joint"})
    assert report["cases"][0]["classification"] == "platform_only_headroom"
    action = next(
        row
        for row in report["cases"][0]["actions"]
        if row["candidate_id"] == "safer-but-worse"
    )
    assert action["seed_half_stable_platform_improvement"]
    assert not action["quality_preserving"]
