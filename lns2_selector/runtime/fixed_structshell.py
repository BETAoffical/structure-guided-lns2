"""Registered fixed structural pools accepted by the active runtime."""

from __future__ import annotations

from typing import Any

from lns2_selector.runtime.structshell_component16 import (
    STRUCTSHELL_COMPONENT16_POOL_ID,
    validate_structshell_component16_augmentation,
)
from lns2_selector.runtime.structshell_dual16 import (
    STRUCTSHELL_DUAL16_POOL_ID,
    validate_structshell_dual16_augmentation,
)


def validate_fixed_structshell_augmentation(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("fixed StructShell augmentation must be an object")
    pool_id = str(value.get("pool_id") or "")
    if pool_id == STRUCTSHELL_COMPONENT16_POOL_ID:
        return validate_structshell_component16_augmentation(value)
    if pool_id == STRUCTSHELL_DUAL16_POOL_ID:
        return validate_structshell_dual16_augmentation(value)
    raise ValueError("unsupported or retired structural runtime augmentation")


__all__ = ["validate_fixed_structshell_augmentation"]
