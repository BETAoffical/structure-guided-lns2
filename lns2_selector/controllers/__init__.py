from __future__ import annotations

from pathlib import Path

from experiments.compact_controller_model import load_controller_bundle
from experiments.v3_s3 import load_v3_s3_bundle
from lns2_selector.controllers.official import OfficialAdaptiveSelector
from lns2_selector.controllers.v2 import (
    PAIRWISE_CONTROLLER_IDS,
    PairwiseV2Selector,
    require_pairwise_bundle_identity,
)
from lns2_selector.controllers.v3_s3 import V3S3Selector
from lns2_selector.runtime.contracts import CONTROLLER_IDS, Selector


def load_selector(
    controller_id: str,
    bundle: str | Path | None = None,
) -> Selector:
    """Load an active selector by its canonical identifier."""

    resolved = str(controller_id)
    if resolved not in CONTROLLER_IDS:
        raise ValueError(f"unsupported controller: {resolved}")
    if resolved == "official_adaptive":
        if bundle is not None:
            raise ValueError("official_adaptive does not use a model bundle")
        return OfficialAdaptiveSelector()
    if bundle is None:
        raise ValueError(f"{resolved} requires a controller bundle")
    if resolved in PAIRWISE_CONTROLLER_IDS:
        loaded = load_controller_bundle(bundle)
        require_pairwise_bundle_identity(resolved, loaded.manifest)
        return PairwiseV2Selector(resolved, loaded)
    if resolved == "v3-s3":
        return V3S3Selector(load_v3_s3_bundle(bundle))
    raise AssertionError(f"unhandled active controller: {resolved}")


__all__ = ["CONTROLLER_IDS", "load_selector"]
