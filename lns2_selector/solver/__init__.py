"""Boundary between Python selectors and the official native solver."""

from lns2_selector.solver.native import load_native_module, native_identity

__all__ = ["load_native_module", "native_identity"]
