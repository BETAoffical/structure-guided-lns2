from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from lns2_selector.runtime.bounded_native_retry import (
    attempt_snapshot,
    platform_signature,
)
from lns2_selector.runtime.fingerprints import semantic_fingerprint


RESCUE_SCHEMA = "lns2.failure_informed_next_decision_rescue.v1"
CONTROL_MODE = "frozen_controller"
SAME_SET_MODE = "same_set_fresh_seed"
BLOCKER_AUGMENTED_MODE = "blocker_augmented_fresh_seed"
MODES = {CONTROL_MODE, SAME_SET_MODE, BLOCKER_AUGMENTED_MODE}


def rescue_seed(
    *, namespace: str, episode_key: str, trial_index: int, signature: str,
    first_attempt_seed: int,
) -> int:
    seed = int(
        semantic_fingerprint(
            {
                "namespace": str(namespace),
                "episode_key": str(episode_key),
                "trial_index": int(trial_index),
                "platform_signature": str(signature),
                "purpose": "next-decision-rescue",
            }
        )[:16],
        16,
    ) % (2**31)
    return (seed + 1) % (2**31) if seed == int(first_attempt_seed) else seed


def ordered_external_blockers(
    metrics: Mapping[str, Any], neighborhood: list[int], maximum: int
) -> list[int]:
    rows = metrics.get("pp_agent_diagnostics")
    if not isinstance(rows, list):
        raise ValueError("failure-informed rescue requires PP agent diagnostics")
    base = set(map(int, neighborhood))
    seen: set[int] = set()
    blockers: list[int] = []
    for expected_index, raw in enumerate(rows):
        row = dict(raw)
        if int(row.get("order_index", -1)) != expected_index:
            raise ValueError("PP diagnostic order changed")
        for raw_agent in row.get("external_blocker_agents") or ():
            agent = int(raw_agent)
            if agent in base or agent in seen:
                continue
            seen.add(agent)
            blockers.append(agent)
            if len(blockers) >= int(maximum):
                return blockers
    return blockers


@dataclass
class FailureInformedRescueTracker:
    mode: str
    maximum_added_blockers: int
    seed_namespace: str
    episode_key: str
    trial_index: int
    initial_repeat_count: int
    initial_record: dict[str, Any] | None = None
    pending_action: dict[str, Any] | None = None
    rescue_record: dict[str, Any] | None = None
    action_issued: bool = False

    @classmethod
    def from_spec(cls, specification: Mapping[str, Any]) -> "FailureInformedRescueTracker":
        required = {
            "mode",
            "maximum_added_blockers",
            "seed_namespace",
            "episode_key",
            "trial_index",
            "initial_repeat_count",
        }
        if set(specification) != required:
            raise ValueError("failure-informed rescue specification changed")
        mode = str(specification["mode"])
        maximum = int(specification["maximum_added_blockers"])
        initial = int(specification["initial_repeat_count"])
        if mode not in MODES:
            raise ValueError(f"unknown failure-informed rescue mode: {mode}")
        if maximum != 8 or initial != 2:
            raise ValueError("failure-informed rescue limits changed")
        return cls(
            mode=mode,
            maximum_added_blockers=maximum,
            seed_namespace=str(specification["seed_namespace"]),
            episode_key=str(specification["episode_key"]),
            trial_index=int(specification["trial_index"]),
            initial_repeat_count=initial,
        )

    def requires_diagnostics(self, decision_index: int) -> bool:
        # Platform entry is an end-to-end trajectory outcome.  Collect the
        # same native diagnostics on every arm/decision so later exact
        # rollbacks can be audited without inferring them from conflict count.
        return int(decision_index) >= 0

    def action_for_decision(
        self, decision_index: int, state: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if int(decision_index) != 1 or self.pending_action is None:
            return None
        if self.action_issued:
            raise ValueError("failure-informed rescue action was requested twice")
        expected = str(self.pending_action["platform_signature"])
        if platform_signature(state) != expected:
            raise ValueError("failure-informed rescue pending state changed")
        self.action_issued = True
        return dict(self.pending_action)

    def observe_decision(
        self,
        *,
        decision_index: int,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        decision = int(decision_index)
        if decision == 0:
            if self.initial_record is not None:
                raise ValueError("failure-informed trigger was observed twice")
            snapshot = attempt_snapshot(metrics)
            before_signature = platform_signature(before)
            after_signature = platform_signature(after)
            exact_rollback = bool(
                snapshot["failure_reason"] == "conflict_bound_exceeded"
                and snapshot["replan_success"] is False
                and snapshot["rolled_back"] is True
                and before_signature == after_signature
                and int(before["num_of_colliding_pairs"])
                == int(after["num_of_colliding_pairs"])
            )
            base_agents = list(map(int, snapshot["neighborhood"]))
            blockers = ordered_external_blockers(
                metrics, base_agents, self.maximum_added_blockers
            )
            eligible = bool(exact_rollback)
            triggered = bool(eligible and self.mode != CONTROL_MODE)
            selected_blockers = blockers if self.mode == BLOCKER_AUGMENTED_MODE else []
            planned_agents = base_agents + selected_blockers
            next_seed = rescue_seed(
                namespace=self.seed_namespace,
                episode_key=self.episode_key,
                trial_index=self.trial_index,
                signature=before_signature,
                first_attempt_seed=int(snapshot["requested_pp_random_seed"]),
            )
            self.initial_record = {
                "schema": RESCUE_SCHEMA,
                "mode": self.mode,
                "trigger_decision_index": decision,
                "rescue_decision_index": 1 if triggered else None,
                "platform_signature": before_signature,
                "initial_repeat_count": self.initial_repeat_count,
                "exact_rollback": exact_rollback,
                "trigger_eligible": eligible,
                "triggered": triggered,
                "first_attempt": snapshot,
                "observed_external_blockers": blockers,
                "selected_blockers": selected_blockers,
                "planned_agents": planned_agents if triggered else [],
                "rescue_seed": next_seed if triggered else None,
                "rescue_attempt": None,
                "resolved_by_rescue": False,
            }
            if triggered:
                self.pending_action = {
                    "schema": RESCUE_SCHEMA,
                    "platform_signature": before_signature,
                    "mode": self.mode,
                    "agents": planned_agents,
                    "selected_blockers": selected_blockers,
                    "pp_random_seed": next_seed,
                }
            return dict(self.initial_record)
        if decision == 1 and self.action_issued:
            if self.initial_record is None or self.pending_action is None:
                raise ValueError("failure-informed rescue action has no trigger")
            snapshot = attempt_snapshot(metrics)
            expected_agents = list(map(int, self.pending_action["agents"]))
            if sorted(snapshot["neighborhood"]) != sorted(expected_agents):
                raise ValueError("failure-informed rescue neighborhood changed")
            if int(snapshot["requested_pp_random_seed"]) != int(
                self.pending_action["pp_random_seed"]
            ):
                raise ValueError("failure-informed rescue seed changed")
            resolved = platform_signature(after) != str(
                self.pending_action["platform_signature"]
            )
            self.rescue_record = {
                **self.initial_record,
                "rescue_attempt": snapshot,
                "resolved_by_rescue": bool(resolved),
            }
            self.pending_action = None
            return dict(self.rescue_record)
        return None

    def summary(self) -> dict[str, Any]:
        record = self.rescue_record or self.initial_record or {}
        return {
            "schema": RESCUE_SCHEMA,
            "mode": self.mode,
            "maximum_added_blockers": self.maximum_added_blockers,
            "initial_repeat_count": self.initial_repeat_count,
            "trigger_eligible": bool(record.get("trigger_eligible", False)),
            "triggered": bool(record.get("triggered", False)),
            "observed_external_blocker_count": len(
                record.get("observed_external_blockers") or ()
            ),
            "selected_blocker_count": len(record.get("selected_blockers") or ()),
            "rescue_executed": bool(record.get("rescue_attempt")),
            "resolved_by_rescue": bool(record.get("resolved_by_rescue", False)),
        }


__all__ = [
    "BLOCKER_AUGMENTED_MODE",
    "CONTROL_MODE",
    "FailureInformedRescueTracker",
    "MODES",
    "RESCUE_SCHEMA",
    "SAME_SET_MODE",
    "ordered_external_blockers",
    "rescue_seed",
]
