"""Neighborhood-selection layer built on the official MAPF-LNS2 kernel."""

from pathlib import Path

from lns2_selector.runtime.contracts import (
    SelectionDecision,
    SelectionRequest,
    Selector,
)


CONTROLLER_IDS = (
    "official_adaptive",
    "v2-full",
    "mixed-full-v2",
    "v3-s3",
)


def load_selector(controller_id: str, bundle: str | Path | None = None) -> Selector:
    """Load lazily so low-level metric modules never import model runtimes."""

    from lns2_selector.controllers import load_selector as _load_selector

    return _load_selector(controller_id, bundle)

__all__ = [
    "CONTROLLER_IDS",
    "SelectionDecision",
    "SelectionRequest",
    "Selector",
    "load_selector",
]
