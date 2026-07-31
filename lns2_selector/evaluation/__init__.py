"""Wall-clock evaluation for active controllers."""

from typing import Any

from lns2_selector.evaluation.trace_validation import (
    ClosedLoopTraceError,
    validate_closed_loop_trace,
)
from lns2_selector.runtime.metrics import wall_clock_conflict_auc


def run_balanced_wall_clock(*args: Any, **kwargs: Any) -> Any:
    from experiments.balanced_wall_clock import run_balanced_wall_clock as run

    return run(*args, **kwargs)


def run_closed_loop_collection(*args: Any, **kwargs: Any) -> Any:
    from experiments.closed_loop_confirmation import run_closed_loop_collection as run

    return run(*args, **kwargs)

__all__ = [
    "ClosedLoopTraceError",
    "run_balanced_wall_clock",
    "run_closed_loop_collection",
    "validate_closed_loop_trace",
    "wall_clock_conflict_auc",
]
