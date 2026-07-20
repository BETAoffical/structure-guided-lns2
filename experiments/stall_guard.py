from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.repair_collection import _fingerprint, _read_json


REPAIR_STRUCTURE_KEYS = (
    "initialized",
    "initial_solution_complete",
    "feasible",
    "rows",
    "cols",
    "sum_of_costs",
    "num_of_colliding_pairs",
    "obstacles",
    "conflict_edges",
    "agents",
)
STALL_GUARD_SCHEMA = "lns2.stall_guard.v1"
STALL_GUARD_VERSION = 1


def repair_structure_fingerprint(state: dict[str, Any]) -> str:
    """Hash repair-relevant state while excluding attempts and search counters."""

    missing = [key for key in REPAIR_STRUCTURE_KEYS if key not in state]
    if missing:
        raise ValueError(f"repair state is missing structural fields: {missing}")
    return _fingerprint({key: state[key] for key in REPAIR_STRUCTURE_KEYS})


@dataclass(frozen=True)
class StallGuardConfig:
    unchanged_state_attempts_per_level: int
    size_caps: tuple[int, ...]
    terminal_fallback: str
    reset_on_state_fingerprint_change: bool
    source: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.payload())

    def payload(self) -> dict[str, Any]:
        return {
            "schema": STALL_GUARD_SCHEMA,
            "schema_version": STALL_GUARD_VERSION,
            "unchanged_state_attempts_per_level": self.unchanged_state_attempts_per_level,
            "size_caps": list(self.size_caps),
            "terminal_fallback": self.terminal_fallback,
            "reset_on_state_fingerprint_change": self.reset_on_state_fingerprint_change,
        }


def load_stall_guard_config(value: str | Path | dict[str, Any]) -> StallGuardConfig:
    raw = _read_json(Path(value).resolve()) if isinstance(value, (str, Path)) else dict(value)
    if str(raw.get("schema")) != STALL_GUARD_SCHEMA:
        raise ValueError("stall guard config has an invalid schema")
    version = int(raw.get("schema_version", STALL_GUARD_VERSION))
    if version != STALL_GUARD_VERSION:
        raise ValueError("stall guard config has an unsupported schema version")
    attempts = int(raw.get("unchanged_state_attempts_per_level", 0))
    caps = tuple(map(int, raw.get("size_caps", ())))
    fallback = str(raw.get("terminal_fallback", ""))
    reset = bool(raw.get("reset_on_state_fingerprint_change"))
    if attempts <= 0:
        raise ValueError("stall guard attempt threshold must be positive")
    if not caps or any(cap <= 0 for cap in caps):
        raise ValueError("stall guard size caps must be positive")
    if tuple(sorted(set(caps), reverse=True)) != caps:
        raise ValueError("stall guard size caps must be unique and descending")
    if fallback != "official_adaptive":
        raise ValueError("stall guard supports only official_adaptive fallback")
    if not reset:
        raise ValueError("stall guard must reset when repair state changes")
    return StallGuardConfig(
        unchanged_state_attempts_per_level=attempts,
        size_caps=caps,
        terminal_fallback=fallback,
        reset_on_state_fingerprint_change=reset,
        source=raw,
    )


def _ranking_indices(
    candidates: list[dict[str, Any]], scores: list[float]
) -> list[int]:
    if not candidates or len(candidates) != len(scores):
        raise ValueError("stall guard ranking inputs are invalid")
    return sorted(
        range(len(candidates)),
        key=lambda index: (
            -round(float(scores[index]), 12),
            str(candidates[index]["candidate_id"]),
        ),
    )


@dataclass
class StallGuardState:
    config: StallGuardConfig
    state_anchor_fingerprint: str | None = None
    level_index: int = 0
    level_stagnant_attempts: int = 0
    state_stagnant_attempts: int = 0
    current_stagnant_streak: int = 0
    candidate_failures: collections.Counter[str] = field(
        default_factory=collections.Counter
    )
    blacklisted_candidates: set[str] = field(default_factory=set)
    fallback_active: bool = False
    state_guard_triggered: bool = False
    pending_selection: dict[str, Any] | None = None
    totals: collections.Counter[str] = field(default_factory=collections.Counter)

    @property
    def active_size_cap(self) -> int:
        return int(self.config.size_caps[self.level_index])

    def ensure_state(self, before_fingerprint: str) -> None:
        if self.state_anchor_fingerprint is None:
            self.state_anchor_fingerprint = str(before_fingerprint)

    def route_before_selection(self, before_fingerprint: str) -> str:
        self.ensure_state(before_fingerprint)
        return "official_adaptive" if self.fallback_active else "model"

    def select(
        self,
        candidates: list[dict[str, Any]],
        scores: list[float],
        *,
        before_fingerprint: str,
    ) -> tuple[int | None, dict[str, Any]]:
        self.ensure_state(before_fingerprint)
        order = _ranking_indices(candidates, scores)
        base_index = order[0]
        base = candidates[base_index]
        eligible = [
            index
            for index in order
            if int(candidates[index]["actual_size"]) <= self.active_size_cap
            and str(candidates[index]["candidate_id"])
            not in self.blacklisted_candidates
        ]
        fallback_reason = None
        if self.fallback_active or not eligible:
            was_fallback_active = self.fallback_active
            self.fallback_active = True
            self.state_guard_triggered = True
            fallback_reason = (
                "terminal_stagnation"
                if self.level_index == len(self.config.size_caps) - 1
                else "no_eligible_candidate"
            )
            selected_index = None
            self.totals["fallback_activation_count"] += int(
                not was_fallback_active
            )
            self.totals["official_fallback_decision_count"] += 1
        else:
            selected_index = eligible[0]
        selected = candidates[selected_index] if selected_index is not None else None
        diagnostics = {
            "schema": STALL_GUARD_SCHEMA,
            "config_fingerprint": self.config.fingerprint,
            "state_anchor_fingerprint": self.state_anchor_fingerprint,
            "active_size_cap": self.active_size_cap,
            "level_stagnant_attempts_before": self.level_stagnant_attempts,
            "state_stagnant_attempts_before": self.state_stagnant_attempts,
            "blacklisted_candidate_count_before": len(self.blacklisted_candidates),
            "blacklisted_candidates_before": sorted(self.blacklisted_candidates),
            "base_selected_candidate_id": str(base["candidate_id"]),
            "base_selected_rank": 1,
            "base_selected_size": int(base["actual_size"]),
            "effective_selected_candidate_id": (
                str(selected["candidate_id"]) if selected is not None else None
            ),
            "effective_selected_rank": (
                order.index(selected_index) + 1 if selected_index is not None else None
            ),
            "effective_selected_size": (
                int(selected["actual_size"]) if selected is not None else None
            ),
            "base_selection_preserved": selected_index == base_index,
            "fallback_reason": fallback_reason,
            "route": "model" if selected_index is not None else "official_adaptive",
        }
        self.pending_selection = {
            "candidate_id": (
                str(selected["candidate_id"]) if selected is not None else None
            ),
            "route": diagnostics["route"],
            "diagnostics": diagnostics,
        }
        if selected_index == base_index:
            self.totals["base_selection_preserved_count"] += 1
        elif selected_index is not None:
            self.totals["model_override_count"] += 1
        return selected_index, diagnostics

    def observe(
        self,
        *,
        after_fingerprint: str,
        replan_success: bool,
        paths_changed: bool,
        conflict_graph_changed: bool,
        sum_of_costs_changed: bool,
        actual_neighborhood_size: int,
    ) -> dict[str, Any]:
        if self.pending_selection is None:
            raise RuntimeError("stall guard observed a repair without a pending selection")
        pending = self.pending_selection
        diagnostics = dict(pending["diagnostics"])
        route = str(pending["route"])
        candidate_id = pending["candidate_id"]
        # PP success alone is not evidence that the repair state changed: PP can
        # legally return the same paths. Stall memory is reset only by the
        # repair-relevant state, never by attempt counters or state_revision.
        changed = bool(paths_changed or conflict_graph_changed or sum_of_costs_changed)
        diagnostics["replan_success"] = bool(replan_success)
        diagnostics["paths_changed"] = bool(paths_changed)
        diagnostics["conflict_graph_changed"] = bool(conflict_graph_changed)
        diagnostics["final_candidate_id"] = candidate_id
        diagnostics["final_candidate_rank"] = diagnostics.get(
            "effective_selected_rank"
        )
        diagnostics["final_neighborhood_size"] = int(actual_neighborhood_size)
        diagnostics["repair_state_changed"] = changed
        diagnostics["stagnant_attempt"] = not changed
        if changed:
            if self.state_guard_triggered:
                self.totals["rescued_state_count"] += 1
            diagnostics.update(
                {
                    "next_active_size_cap": int(self.config.size_caps[0]),
                    "next_fallback_active": False,
                    "candidate_blacklisted_after": False,
                    "backoff_triggered": False,
                }
            )
            self._reset_for_new_state(str(after_fingerprint))
        else:
            self.state_stagnant_attempts += 1
            self.current_stagnant_streak += 1
            self.totals["stagnant_attempt_count"] += 1
            self.totals["longest_unchanged_state_streak"] = max(
                int(self.totals["longest_unchanged_state_streak"]),
                self.current_stagnant_streak,
            )
            blacklisted_after = False
            backoff = False
            if route == "model":
                self.level_stagnant_attempts += 1
                if candidate_id is not None:
                    self.candidate_failures[str(candidate_id)] += 1
                    if (
                        self.candidate_failures[str(candidate_id)]
                        >= self.config.unchanged_state_attempts_per_level
                    ):
                        if str(candidate_id) not in self.blacklisted_candidates:
                            self.totals["blacklist_addition_count"] += 1
                        self.blacklisted_candidates.add(str(candidate_id))
                        blacklisted_after = True
                if (
                    self.level_stagnant_attempts
                    >= self.config.unchanged_state_attempts_per_level
                ):
                    self.state_guard_triggered = True
                    if self.level_index + 1 < len(self.config.size_caps):
                        self.level_index += 1
                        self.level_stagnant_attempts = 0
                        self.totals["size_backoff_count"] += 1
                        backoff = True
                    else:
                        self.fallback_active = True
                        self.totals["fallback_activation_count"] += 1
            diagnostics.update(
                {
                    "next_active_size_cap": self.active_size_cap,
                    "next_fallback_active": self.fallback_active,
                    "candidate_blacklisted_after": blacklisted_after,
                    "backoff_triggered": backoff,
                }
            )
        diagnostics["candidate_failure_count_after"] = (
            int(self.candidate_failures[str(candidate_id)])
            if candidate_id is not None and not changed
            else 0
        )
        diagnostics["blacklisted_candidate_count_after"] = len(
            self.blacklisted_candidates
        )
        diagnostics["state_stagnant_attempts_after"] = (
            self.state_stagnant_attempts if not changed else 0
        )
        self.pending_selection = None
        return diagnostics

    def _reset_for_new_state(self, anchor: str) -> None:
        self.state_anchor_fingerprint = anchor
        self.level_index = 0
        self.level_stagnant_attempts = 0
        self.state_stagnant_attempts = 0
        self.current_stagnant_streak = 0
        self.candidate_failures.clear()
        self.blacklisted_candidates.clear()
        self.fallback_active = False
        self.state_guard_triggered = False
        self.pending_selection = None

    def summary(self) -> dict[str, Any]:
        return {
            "schema": STALL_GUARD_SCHEMA,
            "config": self.config.payload(),
            "config_fingerprint": self.config.fingerprint,
            "size_backoff_count": int(self.totals["size_backoff_count"]),
            "fallback_activation_count": int(
                self.totals["fallback_activation_count"]
            ),
            "official_fallback_decision_count": int(
                self.totals["official_fallback_decision_count"]
            ),
            "blacklist_addition_count": int(
                self.totals["blacklist_addition_count"]
            ),
            "base_selection_preserved_count": int(
                self.totals["base_selection_preserved_count"]
            ),
            "model_override_count": int(self.totals["model_override_count"]),
            "stagnant_attempt_count": int(self.totals["stagnant_attempt_count"]),
            "longest_unchanged_state_streak": int(
                self.totals["longest_unchanged_state_streak"]
            ),
            "rescued_state_count": int(self.totals["rescued_state_count"]),
        }


__all__ = [
    "REPAIR_STRUCTURE_KEYS",
    "STALL_GUARD_SCHEMA",
    "StallGuardConfig",
    "StallGuardState",
    "load_stall_guard_config",
    "repair_structure_fingerprint",
]
