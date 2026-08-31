from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lns2_selector.runtime.contracts import (
    SelectionDecision,
    SelectionRequest,
)
from lns2_selector.runtime.online_selection import score_online_candidates


PAIRWISE_CONTROLLER_IDS = frozenset({"v2-full", "mixed-full-v2"})


def require_pairwise_bundle_identity(
    controller_id: str, manifest: Mapping[str, Any]
) -> str:
    """Return the canonical bundle identity or reject a mislabeled ranker."""

    if controller_id not in PAIRWISE_CONTROLLER_IDS:
        raise ValueError("unsupported pairwise V2 controller id")
    manifest_id = manifest.get("controller_id")
    if manifest_id is None:
        # The frozen v2-full artifact predates the explicit controller_id field;
        # its promoted default is its canonical identity, not a wildcard.
        manifest_id = manifest.get("default_controller")
    resolved = str(manifest_id or "")
    if resolved != controller_id:
        raise ValueError(
            f"{controller_id} requires a matching controller bundle; "
            f"manifest identity is {resolved or 'missing'}"
        )
    return resolved


class PairwiseV2Selector:
    """Adapter shared by pairwise realized-neighborhood ranker bundles."""

    def __init__(self, controller_id: str, bundle: Any):
        if controller_id not in PAIRWISE_CONTROLLER_IDS:
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
