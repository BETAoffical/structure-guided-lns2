from __future__ import annotations

from typing import Any

from lns2_selector.runtime.contracts import SelectionDecision, SelectionRequest
from lns2_selector.runtime.online_selection import (
    pairwise_win_probability,
    score_online_candidates,
)


CONTROLLER_ID = "stride-guardrank-v1"
MAPRANK_CONTROLLER_ID = "stride-maprank-v1"
STRATEGY_SCHEMAS = {
    CONTROLLER_ID: "lns2.stride.guardrank_strategy.v1",
    MAPRANK_CONTROLLER_ID: "lns2.stride.maprank_strategy.v1",
}


def candidate_kind(candidate: Any, row: Any) -> str:
    explicit = str(row.get("candidate_kind", ""))
    if explicit in {"base", "boundary_only"}:
        return explicit
    families = list(candidate.get("selection_families") or ())
    return (
        "boundary_only"
        if any(str(value).startswith("topology-boundary-") for value in families)
        else "base"
    )


class GuardRankSelector:
    """Conservative conflict ranker that keeps frozen V2 as its anchor."""

    def __init__(self, bundle: Any, *, controller_id: str = CONTROLLER_ID):
        manifest = dict(getattr(bundle, "manifest", {}) or {})
        strategy = dict(manifest.get("selection_strategy") or {})
        resolved = str(controller_id)
        if (
            resolved not in STRATEGY_SCHEMAS
            or str(manifest.get("controller_id")) != resolved
            or strategy.get("schema") != STRATEGY_SCHEMAS[resolved]
            or strategy.get("strategy_id") != "v2_anchor_pairwise_guard"
        ):
            raise ValueError(f"{resolved} requires its registered strategy")
        self.controller_id = resolved
        self.models = bundle.main_models
        self.anchor_models = bundle.anchor_models
        self.applied_profile = str(strategy["applied_profile"])
        self.thresholds = {
            str(name): float(value)
            for name, value in dict(strategy["challenger_thresholds"]).items()
        }

    def select(self, request: SelectionRequest) -> SelectionDecision:
        if not request.candidates:
            raise ValueError("cannot select from an empty candidate pool")
        profile = str(request.profile)
        rows = list(request.candidate_rows)
        if profile not in self.models or profile not in self.anchor_models:
            raise ValueError(f"guard rank bundle lacks profile: {profile}")
        anchor_index, anchor_scores, anchor_margin = score_online_candidates(
            rows, self.anchor_models[profile]
        )
        if profile != self.applied_profile:
            return SelectionDecision(
                controller_id=self.controller_id,
                candidate_index=anchor_index,
                candidate=request.candidates[anchor_index],
                diagnostics={
                    "route": "anchor-profile",
                    "profile": profile,
                    "scores": anchor_scores,
                    "margin": anchor_margin,
                },
            )
        challenger_index, challenger_scores, challenger_margin = (
            score_online_candidates(rows, self.models[profile])
        )
        if challenger_index == anchor_index:
            selected_index = anchor_index
            evidence = 1.0
            threshold = None
            route = "anchor-agreement"
            kind = candidate_kind(
                request.candidates[selected_index], rows[selected_index]
            )
        else:
            kind = candidate_kind(
                request.candidates[challenger_index], rows[challenger_index]
            )
            threshold = self.thresholds[kind]
            evidence = pairwise_win_probability(
                rows, self.models[profile], challenger_index, anchor_index
            )
            selected_index = (
                challenger_index if evidence + 1e-12 >= threshold else anchor_index
            )
            route = (
                "guard-override" if selected_index == challenger_index else "guard-anchor"
            )
        return SelectionDecision(
            controller_id=self.controller_id,
            candidate_index=selected_index,
            candidate=request.candidates[selected_index],
            diagnostics={
                "route": route,
                "profile": profile,
                "anchor_index": anchor_index,
                "challenger_index": challenger_index,
                "challenger_kind": kind,
                "challenger_evidence": evidence,
                "challenger_threshold": threshold,
                "anchor_scores": anchor_scores,
                "anchor_margin": anchor_margin,
                "challenger_scores": challenger_scores,
                "challenger_margin": challenger_margin,
            },
        )


__all__ = [
    "CONTROLLER_ID",
    "MAPRANK_CONTROLLER_ID",
    "GuardRankSelector",
    "candidate_kind",
]
