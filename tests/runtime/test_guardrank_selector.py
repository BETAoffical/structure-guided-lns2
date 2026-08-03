from __future__ import annotations

from types import SimpleNamespace

from lns2_selector.controllers.guardrank import GuardRankSelector
from lns2_selector.runtime.contracts import SelectionRequest
from lns2_selector.runtime.online_selection import pairwise_win_probability


class _ScoreModel:
    feature_names = ["value"]

    def __init__(self, *, prefer_high: bool, profile: str = "realized_dynamic"):
        self.prefer_high = prefer_high
        self.profile = profile

    def score_candidates(self, rows):
        values = [float(row["features"][self.profile]["value"]) for row in rows]
        return values if self.prefer_high else [-value for value in values]

    @staticmethod
    def pair_vector(left, right):
        return [
            float(left["features"]["realized_dynamic"]["value"])
            - float(right["features"]["realized_dynamic"]["value"])
        ]

    def predict_positive(self, vectors):
        return [0.8 if vector[0] > 0.0 else 0.2 for vector in vectors]


def _bundle(threshold: float):
    strategy = {
        "schema": "lns2.stride.guardrank_strategy.v1",
        "strategy_id": "v2_anchor_pairwise_guard",
        "applied_profile": "realized_dynamic",
        "challenger_thresholds": {
            "base": threshold,
            "boundary_only": threshold,
        },
    }
    return SimpleNamespace(
        manifest={
            "controller_id": "stride-guardrank-v1",
            "selection_strategy": strategy,
        },
        main_models={
            "realized_dynamic": _ScoreModel(prefer_high=True),
            "proposal_dynamic": _ScoreModel(
                prefer_high=True, profile="proposal_dynamic"
            ),
        },
        anchor_models={
            "realized_dynamic": _ScoreModel(prefer_high=False),
            "proposal_dynamic": _ScoreModel(
                prefer_high=False, profile="proposal_dynamic"
            ),
        },
    )


def _request(profile: str = "realized_dynamic") -> SelectionRequest:
    candidates = [
        {"candidate_id": "low", "selection_families": ["target:4"]},
        {
            "candidate_id": "high",
            "selection_families": ["topology-boundary-low_degree:16"],
        },
    ]
    rows = [
        {
            "candidate_id": "low",
            "candidate_key": "low",
            "features": {profile: {"value": 0.0}},
        },
        {
            "candidate_id": "high",
            "candidate_key": "high",
            "features": {profile: {"value": 1.0}},
        },
    ]
    return SelectionRequest(
        candidates=candidates,
        candidate_rows=rows,
        before_fingerprint="state",
        profile=profile,
    )


def test_guardrank_overrides_only_above_registered_threshold() -> None:
    accepted = GuardRankSelector(_bundle(0.75)).select(_request())
    rejected = GuardRankSelector(_bundle(0.85)).select(_request())
    assert accepted.candidate_index == 1
    assert accepted.diagnostics["route"] == "guard-override"
    assert accepted.diagnostics["challenger_kind"] == "boundary_only"
    assert rejected.candidate_index == 0
    assert rejected.diagnostics["route"] == "guard-anchor"


def test_guardrank_non_applied_profile_is_exact_anchor() -> None:
    decision = GuardRankSelector(_bundle(0.55)).select(
        _request(profile="proposal_dynamic")
    )
    assert decision.candidate_index == 0
    assert decision.diagnostics["route"] == "anchor-profile"


def test_pairwise_evidence_uses_forward_reverse_symmetry() -> None:
    request = _request()
    probability = pairwise_win_probability(
        list(request.candidate_rows), _ScoreModel(prefer_high=True), 1, 0
    )
    assert probability == 0.8
