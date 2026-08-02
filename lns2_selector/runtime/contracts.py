from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


CONTROLLER_IDS = (
    "official_adaptive",
    "v2-full",
    "mixed-full-v2",
    "v3-s3",
)

DIAGNOSTIC_CONTROLLER_IDS = (
    "stride-control-v1",
    "stride-quality-v1",
)


@dataclass(frozen=True)
class SelectionRequest:
    """State made available to a neighborhood selector before repair."""

    candidates: Sequence[Mapping[str, Any]]
    candidate_rows: Sequence[Mapping[str, Any]]
    before_fingerprint: str
    temporal_context: Mapping[str, Any] = field(default_factory=dict)
    agent_count: int | None = None
    profile: str = "realized_dynamic"

    def __post_init__(self) -> None:
        if len(self.candidates) != len(self.candidate_rows):
            raise ValueError("candidates and candidate_rows differ in length")
        if self.agent_count is not None and int(self.agent_count) <= 0:
            raise ValueError("agent_count must be positive")
        if not str(self.profile):
            raise ValueError("profile must be non-empty")


@dataclass(frozen=True)
class SelectionDecision:
    """A selected candidate or an explicit route to native Adaptive."""

    controller_id: str
    candidate_index: int | None
    candidate: Mapping[str, Any] | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    fallback_reason: str | None = None

    @property
    def uses_native_adaptive(self) -> bool:
        route = str(self.diagnostics.get("route", ""))
        return self.candidate_index is None and route in {
            "native-official-adaptive",
            "official_adaptive",
        }


@runtime_checkable
class Selector(Protocol):
    controller_id: str

    def select(self, request: SelectionRequest) -> SelectionDecision:
        """Select one generated neighborhood or route to native Adaptive."""
