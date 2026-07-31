from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any


def load_native_module() -> ModuleType:
    """Load the existing binding without changing native solver behaviour."""

    module = importlib.import_module("lns2_env")
    if not hasattr(module, "LNS2RepairEnv"):
        raise RuntimeError("lns2_env is missing LNS2RepairEnv")
    return module


def _binary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_identity(module: ModuleType | None = None) -> dict[str, Any]:
    """Return the identity of the binary actually imported by Python."""

    loaded = module or load_native_module()
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
