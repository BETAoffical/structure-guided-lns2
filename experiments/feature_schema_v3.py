from __future__ import annotations

import hashlib
import json
from typing import Any

from experiments.feature_schema_v2 import (
    FEATURE_SCHEMA_ID as SOURCE_FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256 as SOURCE_FEATURE_SCHEMA_SHA256,
    PROFILE_FEATURE_NAMES,
)


V3_FEATURE_SCHEMA_ID = "lns2.v3_training_features.v1"
V3_FEATURE_SCHEMA_VERSION = 1

# Frozen from the map-group OOF audit in ``v3_feature_audit.v1``.  This is a
# training projection only: feature-v2 remains the canonical extraction schema
# used by v2 and by historical v3 bundles.
V3_REMOVED_TRAINING_FEATURE_NAMES = frozenset(
    {
        "proposal.actual_size=1",
        "proposal.actual_size=10",
        "proposal.actual_size=11",
        "proposal.actual_size=12",
        "proposal.actual_size=13",
        "proposal.actual_size=14",
        "proposal.actual_size=15",
        "proposal.actual_size=16",
        "proposal.actual_size=2",
        "proposal.actual_size=3",
        "proposal.actual_size=5",
        "proposal.actual_size=6",
        "proposal.actual_size=7",
        "proposal.actual_size=9",
        "proposal.family_ratio=collision:16",
        "proposal.family_ratio=collision:4",
        "proposal.family_ratio=collision:8",
        "proposal.family_ratio=random:4",
        "proposal.family_ratio=target:16",
        "proposal.family_ratio=target:4",
        "proposal.family_ratio=target:8",
        "proposal.seed_agent_count",
        "proposal.seed_delay_std",
        "proposal.seed_path_cost_std",
        "proposal.selection_family=collision:4",
        "proposal.selection_family=target:16",
        "proposal.selection_family=target:4",
        "proposal.selection_family=target:8",
        "proposal.selection_family_count",
        "proposal.support_family_count",
    }
)

V3_FEATURE_NAMES = tuple(
    name
    for name in PROFILE_FEATURE_NAMES["realized_dynamic"]
    if name not in V3_REMOVED_TRAINING_FEATURE_NAMES
)


def _schema_payload() -> dict[str, Any]:
    return {
        "schema": V3_FEATURE_SCHEMA_ID,
        "schema_version": V3_FEATURE_SCHEMA_VERSION,
        "source_feature_schema_id": SOURCE_FEATURE_SCHEMA_ID,
        "source_feature_schema_sha256": SOURCE_FEATURE_SCHEMA_SHA256,
        "profile": "realized_dynamic",
        "feature_names": list(V3_FEATURE_NAMES),
        "removed_training_feature_names": sorted(
            V3_REMOVED_TRAINING_FEATURE_NAMES
        ),
        "selection_basis": "policy_train_map_group_oof_only",
        "diagnostic_split_role": "reported_not_used_for_feature_selection",
    }


V3_FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        _schema_payload(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def v3_feature_schema_manifest() -> dict[str, Any]:
    return {**_schema_payload(), "sha256": V3_FEATURE_SCHEMA_SHA256}


def resolve_v3_feature_names(
    feature_schema_id: str, feature_schema_sha256: str
) -> tuple[str, ...]:
    """Resolve current and historical v3 bundle feature declarations."""

    key = (str(feature_schema_id), str(feature_schema_sha256))
    if key == (V3_FEATURE_SCHEMA_ID, V3_FEATURE_SCHEMA_SHA256):
        return V3_FEATURE_NAMES
    if key == (SOURCE_FEATURE_SCHEMA_ID, SOURCE_FEATURE_SCHEMA_SHA256):
        return tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
    raise ValueError("v3 controller feature schema mismatch")


_source_names = set(PROFILE_FEATURE_NAMES["realized_dynamic"])
if len(V3_REMOVED_TRAINING_FEATURE_NAMES) != 30:
    raise RuntimeError("v3 training schema must remove exactly 30 audited features")
if not V3_REMOVED_TRAINING_FEATURE_NAMES <= _source_names:
    raise RuntimeError("v3 training schema removes unknown feature-v2 names")
if len(V3_FEATURE_NAMES) != 94 or len(set(V3_FEATURE_NAMES)) != 94:
    raise RuntimeError("v3 training schema must contain 94 unique features")


__all__ = [
    "SOURCE_FEATURE_SCHEMA_ID",
    "SOURCE_FEATURE_SCHEMA_SHA256",
    "V3_FEATURE_NAMES",
    "V3_FEATURE_SCHEMA_ID",
    "V3_FEATURE_SCHEMA_SHA256",
    "V3_FEATURE_SCHEMA_VERSION",
    "V3_REMOVED_TRAINING_FEATURE_NAMES",
    "resolve_v3_feature_names",
    "v3_feature_schema_manifest",
]
