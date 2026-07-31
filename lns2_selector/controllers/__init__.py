from __future__ import annotations

from pathlib import Path

from experiments.compact_controller_model import load_controller_bundle
from experiments.v3_s3 import load_v3_s3_bundle
from lns2_selector.controllers.official import OfficialAdaptiveSelector
from lns2_selector.controllers.v2 import PairwiseV2Selector
from lns2_selector.controllers.v3_s3 import V3S3Selector
from lns2_selector.runtime.contracts import Selector


CONTROLLER_IDS = (
    "official_adaptive",
    "v2-full",
    "mixed-full-v2",
    "v3-s3",
)


def load_selector(
    controller_id: str,
    bundle: str | Path | None = None,
) -> Selector:
    """Load one of the four active selectors by its canonical identifier."""

    resolved = str(controller_id)
    if resolved not in CONTROLLER_IDS:
        raise ValueError(f"unsupported controller: {resolved}")
    if resolved == "official_adaptive":
        if bundle is not None:
            raise ValueError("official_adaptive does not use a model bundle")
        return OfficialAdaptiveSelector()
    if bundle is None:
        raise ValueError(f"{resolved} requires a controller bundle")
    if resolved in {"v2-full", "mixed-full-v2"}:
        loaded = load_controller_bundle(bundle)
        manifest_id = str(loaded.manifest.get("controller_id", "v2-full"))
        if resolved == "mixed-full-v2" and manifest_id != resolved:
            raise ValueError("mixed-full-v2 requires a mixed controller bundle")
        if resolved == "v2-full" and manifest_id not in {"", "v2-full"}:
            raise ValueError("v2-full requires the canonical v2 bundle")
        return PairwiseV2Selector(resolved, loaded)
    if resolved == "v3-s3":
        return V3S3Selector(load_v3_s3_bundle(bundle))
    raise AssertionError(f"unhandled active controller: {resolved}")


__all__ = ["CONTROLLER_IDS", "load_selector"]
