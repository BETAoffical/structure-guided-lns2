from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from experiments._common import strict_bool as _strict_bool, strict_int as _strict_int
from experiments.repair_collection import _fingerprint, _read_json
from experiments.repair_aware import classify_repair_outcome


STALL_SHADOW_SCHEMA = "lns2.stall_shadow.v2"
STALL_SHADOW_VERSION = 2
STALL_SHADOW_TRANSITION_SCHEMA = "lns2.stall_shadow_transition.v3"
STALL_SHADOW_SUMMARY_SCHEMA = "lns2.stall_shadow_summary.v2"


def _normalized_agents(agents: Iterable[int], *, field: str) -> tuple[int, ...]:
    values = list(agents)
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError(f"{field} must contain non-negative integer agent ids")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must not contain duplicate agent ids")
    return tuple(sorted(values))


@dataclass(frozen=True)
class StallShadowConfig:
    unchanged_attempt_thresholds: tuple[int, ...]
    minimum_distinct_pp_attempts: int
    future_observation_decisions: int
    maximum_false_trigger_rate: float
    post_state_change_cooldown_decisions: int
    deployment_enabled: bool

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.payload())

    def payload(self) -> dict[str, Any]:
        return {
            "schema": STALL_SHADOW_SCHEMA,
            "schema_version": STALL_SHADOW_VERSION,
            "mode": "shadow",
            "unchanged_attempt_thresholds": list(
                self.unchanged_attempt_thresholds
            ),
            "minimum_distinct_pp_attempts": self.minimum_distinct_pp_attempts,
            "future_observation_decisions": self.future_observation_decisions,
            "maximum_false_trigger_rate": self.maximum_false_trigger_rate,
            "post_state_change_cooldown_decisions": (
                self.post_state_change_cooldown_decisions
            ),
            "deployment_enabled": self.deployment_enabled,
        }


def load_stall_shadow_config(
    value: str | Path | dict[str, Any],
) -> StallShadowConfig:
    raw = (
        _read_json(Path(value).resolve())
        if isinstance(value, (str, Path))
        else dict(value)
    )
    if str(raw.get("schema")) != STALL_SHADOW_SCHEMA:
        raise ValueError("stall shadow config has an invalid schema")
    version = _strict_int(
        raw.get("schema_version", STALL_SHADOW_VERSION),
        field="stall shadow schema version",
        minimum=1,
    )
    if version != STALL_SHADOW_VERSION:
        raise ValueError("stall shadow config has an unsupported schema version")
    if str(raw.get("mode", "shadow")) != "shadow":
        raise ValueError("stall shadow supports shadow mode only")
    raw_thresholds = raw.get("unchanged_attempt_thresholds")
    if not isinstance(raw_thresholds, list):
        raise ValueError("stall shadow thresholds must be a list")
    thresholds = tuple(
        _strict_int(
            threshold,
            field="stall shadow threshold",
            minimum=1,
        )
        for threshold in raw_thresholds
    )
    if (
        not thresholds
        or tuple(sorted(set(thresholds))) != thresholds
    ):
        raise ValueError(
            "stall shadow thresholds must be unique positive ascending integers"
        )
    distinct = _strict_int(
        raw.get("minimum_distinct_pp_attempts"),
        field="stall shadow minimum distinct PP attempts",
        minimum=2,
    )
    future = _strict_int(
        raw.get("future_observation_decisions"),
        field="stall shadow future observation decisions",
        minimum=1,
    )
    raw_false_rate = raw.get("maximum_false_trigger_rate")
    if isinstance(raw_false_rate, bool) or not isinstance(
        raw_false_rate, (int, float)
    ):
        raise ValueError("stall shadow false-trigger limit must be numeric")
    false_rate = float(raw_false_rate)
    cooldown = _strict_int(
        raw.get("post_state_change_cooldown_decisions"),
        field="stall shadow cooldown decisions",
        minimum=2,
    )
    deployment_enabled = _strict_bool(
        raw.get("deployment_enabled", False),
        field="stall shadow deployment_enabled",
    )
    if not math.isfinite(false_rate) or not 0.0 <= false_rate <= 1.0:
        raise ValueError("stall shadow false-trigger limit must be in [0, 1]")
    if deployment_enabled:
        raise ValueError(
            "stall shadow v2 is diagnostic-only and cannot enable deployment"
        )
    return StallShadowConfig(
        unchanged_attempt_thresholds=thresholds,
        minimum_distinct_pp_attempts=distinct,
        future_observation_decisions=future,
        maximum_false_trigger_rate=false_rate,
        post_state_change_cooldown_decisions=cooldown,
        deployment_enabled=False,
    )


def neighborhood_key(agents: Iterable[int]) -> str:
    normalized = _normalized_agents(agents, field="stall shadow neighborhood")
    return _fingerprint({"agents": normalized})


def pp_attempt_key(
    *,
    state_fingerprint: str,
    agents: Iterable[int],
    step_random_seed: int,
    requested_pp_seed: int | None,
    applied_pp_seed: int | None,
    repair_order: Iterable[int] | None,
) -> str:
    fingerprint = str(state_fingerprint)
    if not fingerprint:
        raise ValueError("stall shadow requires a non-empty state fingerprint")
    step_seed = _strict_int(step_random_seed, field="stall shadow step random seed")
    requested_seed = (
        None
        if requested_pp_seed is None
        else _strict_int(requested_pp_seed, field="stall shadow requested PP seed")
    )
    applied_seed = (
        None
        if applied_pp_seed is None
        else _strict_int(applied_pp_seed, field="stall shadow applied PP seed")
    )
    if applied_seed is not None and applied_seed != requested_seed:
        raise ValueError("stall shadow applied PP seed differs from its request")
    normalized_agents = _normalized_agents(
        agents, field="stall shadow attempt neighborhood"
    )
    raw_order = list(repair_order or ())
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in raw_order
    ):
        raise ValueError("stall shadow repair order must contain integer agent ids")
    normalized_order = tuple(raw_order)
    if normalized_order and (
        len(normalized_order) != len(set(normalized_order))
        or set(normalized_order) != set(normalized_agents)
    ):
        raise ValueError("stall shadow repair order differs from its neighborhood")
    return _fingerprint(
        {
            "state_fingerprint": fingerprint,
            "neighborhood": neighborhood_key(normalized_agents),
            "step_random_seed": step_seed,
            "requested_pp_seed": requested_seed,
            "applied_pp_seed": applied_seed,
            "repair_order": list(normalized_order),
        }
    )


def _ranked_candidates(
    candidates: list[dict[str, Any]], scores: list[float]
) -> list[dict[str, Any]]:
    if not candidates or len(candidates) != len(scores):
        raise ValueError("stall shadow ranking inputs are invalid")
    normalized: list[dict[str, Any]] = []
    for candidate, raw_score in zip(candidates, scores):
        if not isinstance(candidate, dict):
            raise ValueError("stall shadow candidate must be an object")
        candidate_id = str(candidate.get("candidate_id") or "")
        agents = _normalized_agents(
            candidate.get("agents", ()), field="stall shadow candidate neighborhood"
        )
        actual_size = _strict_int(
            candidate.get("actual_size"), field="stall shadow candidate size", minimum=1
        )
        if actual_size != len(agents):
            raise ValueError("stall shadow candidate size differs from its agents")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError("stall shadow score must be numeric")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("stall shadow score must be finite")
        if not candidate_id:
            raise ValueError("stall shadow candidate id is missing")
        normalized.append(
            {
                "candidate_id": candidate_id,
                "agents": agents,
                "actual_size": actual_size,
                "score": score,
            }
        )
    if len({candidate["candidate_id"] for candidate in normalized}) != len(normalized):
        raise ValueError("stall shadow candidate ids must be unique")
    order = sorted(
        range(len(normalized)),
        key=lambda index: (
            -round(float(normalized[index]["score"]), 12),
            str(normalized[index]["candidate_id"]),
        ),
    )
    return [
        {
            "candidate_id": str(normalized[index]["candidate_id"]),
            "agents": list(normalized[index]["agents"]),
            "neighborhood_key": neighborhood_key(normalized[index]["agents"]),
            "actual_size": int(normalized[index]["actual_size"]),
            "score": float(normalized[index]["score"]),
            "rank": rank,
        }
        for rank, index in enumerate(order, 1)
    ]


@dataclass
class StallShadowState:
    config: StallShadowConfig
    state_anchor_fingerprint: str | None = None
    cooldown_remaining: int = 0
    unchanged_attempts: list[dict[str, Any]] = field(default_factory=list)
    triggered_thresholds: set[int] = field(default_factory=set)
    pending_evaluations: list[dict[str, Any]] = field(default_factory=list)
    pending_selection: dict[str, Any] | None = None
    totals: collections.Counter[str] = field(default_factory=collections.Counter)

    def _begin_state(
        self,
        fingerprint: str,
        *,
        cooldown: int | None = None,
    ) -> None:
        self.state_anchor_fingerprint = str(fingerprint)
        self.unchanged_attempts.clear()
        self.triggered_thresholds.clear()
        self.cooldown_remaining = (
            self.config.post_state_change_cooldown_decisions
            if cooldown is None
            else int(cooldown)
        )

    def _reset_after_change(self, fingerprint: str) -> None:
        self.state_anchor_fingerprint = str(fingerprint)
        self.unchanged_attempts.clear()
        self.triggered_thresholds.clear()
        self.cooldown_remaining = (
            self.config.post_state_change_cooldown_decisions
        )

    def _distinct_attempt_count(self) -> int:
        return len({str(row["attempt_key"]) for row in self.unchanged_attempts})

    def before_selection(
        self,
        candidates: list[dict[str, Any]],
        scores: list[float],
        base_index: int,
        *,
        before_fingerprint: str,
        decision_index: int,
    ) -> tuple[int, dict[str, Any]]:
        if self.pending_selection is not None:
            raise RuntimeError("stall shadow has an unobserved pending selection")
        if (
            isinstance(base_index, bool)
            or not isinstance(base_index, int)
            or not 0 <= base_index < len(candidates)
        ):
            raise ValueError("stall shadow base selection index is invalid")
        if not str(before_fingerprint):
            raise ValueError("stall shadow requires a non-empty state fingerprint")
        if (
            isinstance(decision_index, bool)
            or not isinstance(decision_index, int)
            or decision_index < 0
        ):
            raise ValueError("stall shadow decision index must be non-negative")
        ranked = _ranked_candidates(candidates, scores)
        base_id = str(candidates[base_index]["candidate_id"])
        if ranked[0]["candidate_id"] != base_id:
            raise ValueError("stall shadow base selection is not the frozen v2 winner")
        if self.state_anchor_fingerprint is None:
            self._begin_state(before_fingerprint, cooldown=0)
        elif self.state_anchor_fingerprint != str(before_fingerprint):
            raise RuntimeError(
                "stall shadow saw a new state before the previous repair was observed"
            )
        cooldown_before = self.cooldown_remaining
        distinct_attempts = self._distinct_attempt_count()
        triggered_now: list[int] = []
        if cooldown_before == 0:
            for threshold in self.config.unchanged_attempt_thresholds:
                if (
                    threshold not in self.triggered_thresholds
                    and len(self.unchanged_attempts) >= threshold
                    and distinct_attempts
                    >= self.config.minimum_distinct_pp_attempts
                ):
                    self.triggered_thresholds.add(threshold)
                    triggered_now.append(threshold)
                    self.pending_evaluations.append(
                        {
                            "threshold": threshold,
                            "decision_index": int(decision_index),
                            "observed_decisions": 0,
                        }
                    )
                    self.totals[f"threshold_{threshold}_trigger_count"] += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        self.pending_selection = {
            "candidate_id": base_id,
            "agents": list(ranked[0]["agents"]),
            "action_preserved": True,
            "triggered_thresholds": list(triggered_now),
            "decision_index": int(decision_index),
            "cooldown_before": int(cooldown_before),
        }
        self.totals["decision_count"] += 1
        self.totals["cooldown_decision_count"] += int(cooldown_before > 0)
        return base_index, {
            "schema": STALL_SHADOW_TRANSITION_SCHEMA,
            "config_fingerprint": self.config.fingerprint,
            "mode": "shadow",
            "deployment_enabled": False,
            "state_anchor_fingerprint": self.state_anchor_fingerprint,
            "base_selected_candidate_id": base_id,
            "effective_selected_candidate_id": base_id,
            "base_selection_preserved": True,
            "route": "model",
            "unchanged_attempt_count_before": len(self.unchanged_attempts),
            "distinct_pp_attempt_count_before": distinct_attempts,
            "cooldown_remaining_before": cooldown_before,
            "triggered_thresholds": triggered_now,
        }

    def abort_selection(self) -> None:
        """Roll back a selection when the wall deadline prevents its repair."""

        if self.pending_selection is None:
            raise RuntimeError("stall shadow has no pending selection to abort")
        decision_index = int(self.pending_selection["decision_index"])
        triggered = set(map(int, self.pending_selection["triggered_thresholds"]))
        self.pending_evaluations = [
            row
            for row in self.pending_evaluations
            if not (
                int(row["decision_index"]) == decision_index
                and int(row["threshold"]) in triggered
            )
        ]
        for threshold in triggered:
            self.triggered_thresholds.remove(threshold)
            self.totals[f"threshold_{threshold}_trigger_count"] -= 1
        cooldown_before = int(self.pending_selection["cooldown_before"])
        self.cooldown_remaining = cooldown_before
        self.totals["decision_count"] -= 1
        self.totals["cooldown_decision_count"] -= int(cooldown_before > 0)
        self.pending_selection = None

    def _resolve_pending(self, outcome: str) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        changed = outcome not in {"hard_failure", "accepted_noop"}
        for row in self.pending_evaluations:
            value = dict(row)
            value["observed_decisions"] = int(value["observed_decisions"]) + 1
            threshold = int(value["threshold"])
            if changed:
                value["resolution"] = "premature_trigger"
                self.totals[f"threshold_{threshold}_premature_count"] += 1
                resolved.append(value)
            elif int(value["observed_decisions"]) >= (
                self.config.future_observation_decisions
            ):
                value["resolution"] = "confirmed_stall"
                self.totals[f"threshold_{threshold}_confirmed_count"] += 1
                resolved.append(value)
            else:
                remaining.append(value)
        self.pending_evaluations = remaining
        return resolved

    def observe(
        self,
        *,
        before_fingerprint: str,
        after_fingerprint: str,
        replan_success: bool,
        conflicts_before: int,
        conflicts_after: int,
        feasible: bool,
        candidate_id: str,
        actual_agents: Iterable[int],
        step_random_seed: int,
        requested_pp_seed: int | None,
        applied_pp_seed: int | None,
        repair_order: Iterable[int] | None,
    ) -> dict[str, Any]:
        if self.pending_selection is None:
            raise RuntimeError("stall shadow observed a repair without a selection")
        if not bool(self.pending_selection.get("action_preserved")):
            raise RuntimeError("stall shadow attempted to modify the v2 action")
        if str(self.pending_selection["candidate_id"]) != str(candidate_id):
            raise RuntimeError("stall shadow selected candidate does not match v2")
        agents = _normalized_agents(
            actual_agents, field="stall shadow actual neighborhood"
        )
        if agents != tuple(self.pending_selection["agents"]):
            raise RuntimeError("stall shadow actual neighborhood does not match v2")
        if str(before_fingerprint) != str(self.state_anchor_fingerprint):
            raise RuntimeError("stall shadow observation does not match its state anchor")
        if not str(after_fingerprint):
            raise ValueError("stall shadow requires a non-empty after fingerprint")
        if not isinstance(replan_success, bool) or not isinstance(feasible, bool):
            raise ValueError("stall shadow outcome flags must be boolean")
        conflicts_before = _strict_int(
            conflicts_before, field="stall shadow conflicts_before"
        )
        conflicts_after = _strict_int(
            conflicts_after, field="stall shadow conflicts_after"
        )
        if feasible != (conflicts_after == 0):
            raise ValueError("stall shadow feasible flag differs from conflicts_after")
        outcome = classify_repair_outcome(
            before_fingerprint=before_fingerprint,
            after_fingerprint=after_fingerprint,
            replan_success=replan_success,
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=feasible,
        )
        self.totals[f"outcome_{outcome}_count"] += 1
        resolved = self._resolve_pending(outcome)
        no_progress = outcome in {"hard_failure", "accepted_noop"}
        attempt = None
        if no_progress:
            attempt = {
                "candidate_id": str(candidate_id),
                "agents": list(agents),
                "neighborhood_key": neighborhood_key(agents),
                "attempt_key": pp_attempt_key(
                    state_fingerprint=before_fingerprint,
                    agents=agents,
                    step_random_seed=step_random_seed,
                    requested_pp_seed=requested_pp_seed,
                    applied_pp_seed=applied_pp_seed,
                    repair_order=repair_order,
                ),
                "step_random_seed": int(step_random_seed),
                "requested_pp_seed": requested_pp_seed,
                "applied_pp_seed": applied_pp_seed,
                "outcome": outcome,
            }
            self.unchanged_attempts.append(attempt)
            self.totals["unchanged_attempt_count"] += 1
            self.totals["longest_unchanged_streak"] = max(
                int(self.totals["longest_unchanged_streak"]),
                len(self.unchanged_attempts),
            )
        else:
            if outcome == "state_changed_no_reduction":
                self.totals["state_changed_no_reduction_reset_count"] += 1
            self._reset_after_change(after_fingerprint)
        self.pending_selection = None
        return {
            "repair_outcome": outcome,
            "no_progress": no_progress,
            "state_unchanged": str(before_fingerprint) == str(after_fingerprint),
            "attempt": attempt,
            "unchanged_attempt_count_after": (
                len(self.unchanged_attempts) if no_progress else 0
            ),
            "distinct_pp_attempt_count_after": (
                self._distinct_attempt_count() if no_progress else 0
            ),
            "resolved_triggers": resolved,
            "action_preserved": True,
        }

    def summary(self) -> dict[str, Any]:
        thresholds: dict[str, Any] = {}
        for threshold in self.config.unchanged_attempt_thresholds:
            triggers = int(self.totals[f"threshold_{threshold}_trigger_count"])
            premature = int(
                self.totals[f"threshold_{threshold}_premature_count"]
            )
            confirmed = int(
                self.totals[f"threshold_{threshold}_confirmed_count"]
            )
            resolved = premature + confirmed
            false_rate = premature / resolved if resolved else None
            thresholds[str(threshold)] = {
                "trigger_count": triggers,
                "premature_trigger_count": premature,
                "confirmed_stall_count": confirmed,
                "unresolved_trigger_count": triggers - resolved,
                "false_trigger_rate": false_rate,
                "false_trigger_gate_passed": False,
                "gate_status": "external_audit_required",
            }
        return {
            "schema": STALL_SHADOW_SUMMARY_SCHEMA,
            "config": self.config.payload(),
            "config_fingerprint": self.config.fingerprint,
            "mode": "shadow",
            "deployment_enabled": False,
            "action_override_count": 0,
            "decision_count": int(self.totals["decision_count"]),
            "cooldown_decision_count": int(
                self.totals["cooldown_decision_count"]
            ),
            "outcome_counts": {
                name: int(self.totals[f"outcome_{name}_count"])
                for name in (
                    "hard_failure",
                    "accepted_noop",
                    "state_changed_no_reduction",
                    "conflict_reduced",
                    "feasible",
                )
            },
            "state_changed_no_reduction_reset_count": int(
                self.totals["state_changed_no_reduction_reset_count"]
            ),
            "longest_unchanged_streak": int(
                self.totals["longest_unchanged_streak"]
            ),
            "thresholds": thresholds,
            "most_conservative_passing_threshold": None,
            "promotion_eligible": False,
        }


__all__ = [
    "STALL_SHADOW_SCHEMA",
    "STALL_SHADOW_SUMMARY_SCHEMA",
    "STALL_SHADOW_TRANSITION_SCHEMA",
    "StallShadowConfig",
    "StallShadowState",
    "load_stall_shadow_config",
    "neighborhood_key",
    "pp_attempt_key",
]
