"""Compatibility symbols for frozen sklearn models created before reorganization.

New research code must import from ``research.studies.context.context_audit``.
This module remains because the canonical v1 pickle records the historical
``experiments.context_audit.PairwiseModel`` qualified name.
"""

from research.studies.context.context_audit import PairwiseModel

__all__ = ["PairwiseModel"]
