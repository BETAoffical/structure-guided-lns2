from __future__ import annotations

from pathlib import Path

from experiments.compact_controller_model import load_controller_bundle
from experiments.v3_s3 import load_v3_s3_bundle
from lns2_selector.controllers.official import OfficialAdaptiveSelector
from lns2_selector.controllers.guardrank import GuardRankSelector
from lns2_selector.controllers.v2 import PairwiseV2Selector
from lns2_selector.controllers.v3_s3 import V3S3Selector
from lns2_selector.runtime.contracts import (
    CONTROLLER_IDS,
    DIAGNOSTIC_CONTROLLER_IDS,
    Selector,
)


def load_selector(
    controller_id: str,
    bundle: str | Path | None = None,
) -> Selector:
    """Load an active or diagnostic selector by its canonical identifier."""

    resolved = str(controller_id)
    if resolved not in (*CONTROLLER_IDS, *DIAGNOSTIC_CONTROLLER_IDS):
        raise ValueError(f"unsupported controller: {resolved}")
    if resolved == "official_adaptive":
        if bundle is not None:
            raise ValueError("official_adaptive does not use a model bundle")
        return OfficialAdaptiveSelector()
    if bundle is None:
        raise ValueError(f"{resolved} requires a controller bundle")
    pairwise_ids = {
        "v2-full",
        "mixed-full-v2",
        "stride-control-v1",
        "stride-quality-v1",
        "stride-augcontrol-v1",
    }
    if resolved == "stride-guardrank-v1":
        loaded = load_controller_bundle(bundle)
        if (
            str(loaded.manifest.get("controller_id")) != resolved
            or loaded.manifest.get("scientific_status") != "diagnostic_only"
            or loaded.manifest.get("default_replacement_allowed") is not False
        ):
            raise ValueError(
                "stride-guardrank-v1 requires an exactly matching diagnostic bundle"
            )
        return GuardRankSelector(loaded)
    if resolved in pairwise_ids:
        loaded = load_controller_bundle(bundle)
        manifest_id = str(loaded.manifest.get("controller_id", "v2-full"))
        if resolved == "mixed-full-v2" and manifest_id != resolved:
            raise ValueError("mixed-full-v2 requires a mixed controller bundle")
        if resolved == "v2-full" and manifest_id not in {"", "v2-full"}:
            raise ValueError("v2-full requires the canonical v2 bundle")
        if resolved.startswith("stride-"):
            if manifest_id != resolved:
                raise ValueError(f"{resolved} requires an exactly matching bundle")
            if (
                loaded.manifest.get("scientific_status") != "diagnostic_only"
                or loaded.manifest.get("default_replacement_allowed") is not False
            ):
                raise ValueError(f"{resolved} requires a diagnostic-only bundle")
        return PairwiseV2Selector(resolved, loaded)
    if resolved == "v3-s3":
        return V3S3Selector(load_v3_s3_bundle(bundle))
    raise AssertionError(f"unhandled active controller: {resolved}")


__all__ = ["CONTROLLER_IDS", "DIAGNOSTIC_CONTROLLER_IDS", "load_selector"]
