from __future__ import annotations

import math
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
    candidate_pool_mode: str = "full"
    generation_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.candidates) != len(self.candidate_rows):
            raise ValueError("candidates and candidate_rows differ in length")

        def identity(
            value: Mapping[str, Any], *, kind: str, index: int
        ) -> str:
            raw = [
                value[name]
                for name in ("candidate_id", "candidate_key")
                if name in value
            ]
            if not raw or any(
                not isinstance(item, str) or not item for item in raw
            ):
                raise ValueError(f"{kind} identity is missing at index {index}")
            normalized = set(raw)
            if len(normalized) != 1:
                raise ValueError(
                    f"{kind} identity fields differ at index {index}"
                )
            return normalized.pop()

        for index, (candidate, row) in enumerate(
            zip(self.candidates, self.candidate_rows)
        ):
            candidate_id = identity(candidate, kind="candidate", index=index)
            row_id = identity(row, kind="feature row", index=index)
            if candidate_id != row_id:
                raise ValueError(
                    f"candidate and feature row identities differ at index {index}"
                )
        if self.agent_count is not None:
            require_int(self.agent_count, field="agent_count", minimum=1)
        require_nonempty_string(self.before_fingerprint, field="before_fingerprint")
        require_nonempty_string(self.profile, field="profile")
        if self.candidate_pool_mode not in {"full", "restricted"}:
            raise ValueError("candidate_pool_mode must be full or restricted")
        context_mode = self.generation_context.get("mode")
        if (
            context_mode is not None
            and require_nonempty_string(
                context_mode, field="generation_context.mode"
            )
            != self.candidate_pool_mode
        ):
            raise ValueError(
                "generation_context mode differs from candidate_pool_mode"
            )
        if self.candidate_pool_mode == "restricted" and not isinstance(
            self.generation_context.get("template"), Mapping
        ):
            raise ValueError(
                "restricted candidate pools require a generation template"
            )


@dataclass(frozen=True)
class SelectionObservation:
    """Native outcome bound to the selector decision that produced it."""

    candidate_id: str
    actual_agents: Sequence[int]
    before_fingerprint: str
    after_fingerprint: str
    replan_success: bool
    conflicts_before: int
    conflicts_after: int
    feasible: bool
    terminal: bool
    total_seconds: float

    def __post_init__(self) -> None:
        require_nonempty_string(self.candidate_id, field="candidate_id")
        agents = tuple(self.actual_agents)
        if not agents:
            raise ValueError("actual_agents must be non-empty")
        if any(type(agent) is not int or agent < 0 for agent in agents):
            raise ValueError("actual_agents must contain non-negative integers")
        if len(agents) != len(set(agents)):
            raise ValueError("actual_agents must not contain duplicates")
        require_nonempty_string(
            self.before_fingerprint, field="before_fingerprint"
        )
        require_nonempty_string(
            self.after_fingerprint, field="after_fingerprint"
        )
        if (
            type(self.replan_success) is not bool
            or type(self.feasible) is not bool
            or type(self.terminal) is not bool
        ):
            raise ValueError("selection observation flags must be boolean")
        if self.feasible and not self.terminal:
            raise ValueError("a feasible selection observation must be terminal")
        if type(self.conflicts_before) is not int or self.conflicts_before < 0:
            raise ValueError("conflicts_before must be a non-negative integer")
        if type(self.conflicts_after) is not int or self.conflicts_after < 0:
            raise ValueError("conflicts_after must be a non-negative integer")
        total_seconds = float(self.total_seconds)
        if not math.isfinite(total_seconds) or total_seconds < 0.0:
            raise ValueError("total_seconds must be finite and non-negative")


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


@runtime_checkable
class StatefulSelector(Selector, Protocol):
    def observe(self, observation: SelectionObservation) -> Mapping[str, Any]:
        """Observe the native outcome of the most recent selected action."""
