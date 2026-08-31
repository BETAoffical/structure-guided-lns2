from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


CONTROLLER_IDS = (
    "official_adaptive",
    "v2-full",
    "mixed-full-v2",
    "v3-s3",
)


def require_bool(value: Any, *, field: str) -> bool:
    """Return a JSON boolean while rejecting truthy substitutes."""

    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def require_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return a JSON integer without accepting booleans or numeric strings."""

    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return value


def require_nonempty_string(value: Any, *, field: str) -> str:
    """Return a non-empty JSON string without coercing arbitrary objects."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def require_int_list(
    value: Any, *, field: str, minimum: int | None = None
) -> list[int]:
    """Return a JSON integer array with strict element types."""

    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return [
        require_int(item, field=f"{field}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    ]


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
        for index, (candidate, row) in enumerate(
            zip(self.candidates, self.candidate_rows)
        ):
            candidate_id = candidate.get("candidate_id", candidate.get("candidate_key"))
            row_id = row.get("candidate_id", row.get("candidate_key"))
            if (
                candidate_id is not None
                and row_id is not None
                and str(candidate_id) != str(row_id)
            ):
                raise ValueError(
                    f"candidate and feature row identities differ at index {index}"
                )
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
