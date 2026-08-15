from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any


NATIVE_SEMANTICS_SCHEMA = "lns2.native_semantics.official_step_timed_extension.v2"


def _validate_native_module(module: ModuleType) -> ModuleType:
    semantics_schema = str(getattr(module, "native_semantics_schema", ""))
    if semantics_schema != NATIVE_SEMANTICS_SCHEMA:
        raise RuntimeError(
            "loaded lns2_env has unsupported native semantics schema: "
            f"{semantics_schema or 'missing'}"
        )
    if not str(getattr(module, "repair_timing_schema", "")):
        raise RuntimeError("loaded lns2_env has no repair_timing_schema")
    if not hasattr(module, "LNS2RepairEnv"):
        raise RuntimeError("lns2_env is missing LNS2RepairEnv")
    return module


def load_native_module() -> ModuleType:
    """Load the existing binding without changing native solver behaviour."""

    return _validate_native_module(importlib.import_module("lns2_env"))


def _binary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_identity(module: ModuleType | None = None) -> dict[str, Any]:
    """Return the identity of the binary actually imported by Python."""

    loaded = _validate_native_module(module) if module is not None else load_native_module()
    path = Path(str(getattr(loaded, "__file__", ""))).resolve()
    if not path.is_file():
        raise RuntimeError("loaded lns2_env has no readable binary path")
    return {
        "path": str(path),
        "sha256": _binary_sha256(path),
        "repair_timing_schema": str(
            getattr(loaded, "repair_timing_schema", "")
        ),
        "native_semantics_schema": str(
            getattr(loaded, "native_semantics_schema", "")
        ),
    }
