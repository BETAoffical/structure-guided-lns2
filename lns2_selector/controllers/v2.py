from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lns2_selector.runtime.contracts import (
    SelectionDecision,
    SelectionRequest,
)
from lns2_selector.runtime.online_selection import score_online_candidates


class PairwiseV2Selector:
    """Adapter shared by pairwise realized-neighborhood ranker bundles."""

    def __init__(self, controller_id: str, bundle: Any):
        if controller_id not in {
            "v2-full",
            "mixed-full-v2",
            "stride-control-v1",
            "stride-quality-v1",
        }:
            raise ValueError("unsupported pairwise V2 controller id")
        models = getattr(bundle, "main_models", None)
        if models is None:
            models = getattr(bundle, "models", None)
        if models is None and isinstance(bundle, Mapping):
            models = bundle
        if not isinstance(models, Mapping):
            raise ValueError("pairwise V2 bundle does not expose model profiles")
        self.controller_id = controller_id
        self.models = models

    def select(self, request: SelectionRequest) -> SelectionDecision:
        if not request.candidates:
            raise ValueError("cannot select from an empty candidate pool")
        profile = str(request.profile)
        if profile not in self.models:
            raise ValueError(f"pairwise V2 bundle lacks profile: {profile}")
        model = self.models[profile]
        index, scores, margin = score_online_candidates(
            list(request.candidate_rows), model
        )
        return SelectionDecision(
            controller_id=self.controller_id,
            candidate_index=index,
            candidate=request.candidates[index],
            diagnostics={
                "route": "model",
                "profile": profile,
                "scores": scores,
                "margin": margin,
            },
        )
