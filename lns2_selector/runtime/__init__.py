"""Algorithm-neutral selector runtime contracts and metrics."""

from lns2_selector.runtime.contracts import (
    SelectionDecision,
    SelectionObservation,
    SelectionRequest,
    Selector,
    StatefulSelector,
)
from lns2_selector.runtime.metrics import wall_clock_conflict_auc
from lns2_selector.runtime.online_selection import (
    ClosedLoopExecutionError,
    generate_online_candidates,
    score_online_candidates,
)
from lns2_selector.runtime.fingerprints import (
    repair_structure_fingerprint,
    semantic_fingerprint,
)
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome

__all__ = [
    "SelectionDecision",
    "SelectionObservation",
    "SelectionRequest",
    "Selector",
    "StatefulSelector",
    "ClosedLoopExecutionError",
    "generate_online_candidates",
    "score_online_candidates",
    "repair_structure_fingerprint",
    "semantic_fingerprint",
    "classify_repair_outcome",
    "wall_clock_conflict_auc",
]
