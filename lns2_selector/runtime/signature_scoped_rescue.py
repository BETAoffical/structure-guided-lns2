from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from lns2_selector.runtime.bounded_native_retry import (
    attempt_snapshot,
    platform_signature,
)
from lns2_selector.runtime.contracts import (
    require_bool,
    require_int,
    require_nonempty_string,
)
from lns2_selector.runtime.failure_informed_rescue import (
    COMPACT_BLOCKER_AUGMENTED_MODE,
    ordered_external_blockers,
    rescue_seed,
)
from lns2_selector.runtime.semantic_compaction import semantic_compact_plan


RESCUE_SCHEMA = "lns2.signature_scoped_next_decision_rescue.v1"


@dataclass
class SignatureScopedRescueTracker:
    """Schedule at most one compact+blocker rescue for each exact platform.

    A rescue replaces the action at the *next* controller decision.  It never
    performs a second PP call inside one decision.  A failed rescue therefore
    cannot retry the same signature, while a changed state may earn one later
    rescue only after it independently reaches the three-rollback threshold.
    """

    maximum_added_blockers: int
    minimum_consecutive_rollbacks: int
    maximum_interventions: int
    initial_repeat_count: int
    seed_namespace: str
    episode_key: str
    trial_index: int
    enable_semantic_compaction_audit: bool
    current_signature: str
    current_streak: int
    used_signatures: set[str] = field(default_factory=set)
    intervention_count: int = 0
    resolved_intervention_count: int = 0
    pending_action: dict[str, Any] | None = None
    action_issued: bool = False
    records: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_spec(
        cls, state: Mapping[str, Any], specification: Mapping[str, Any]
    ) -> "SignatureScopedRescueTracker":
        required = {
            "maximum_added_blockers",
            "minimum_consecutive_rollbacks",
            "maximum_interventions",
            "initial_repeat_count",
            "seed_namespace",
            "episode_key",
            "trial_index",
            "enable_semantic_compaction_audit",
        }
        if set(specification) != required:
            raise ValueError("signature-scoped rescue specification changed")
        maximum_blockers = require_int(
            specification["maximum_added_blockers"],
            field="maximum added blockers",
            minimum=0,
        )
        minimum_rollbacks = require_int(
            specification["minimum_consecutive_rollbacks"],
            field="minimum consecutive rollbacks",
            minimum=0,
        )
        maximum_interventions = require_int(
            specification["maximum_interventions"],
            field="maximum interventions",
            minimum=0,
        )
        initial_repeat_count = require_int(
            specification["initial_repeat_count"],
            field="initial repeat count",
            minimum=0,
        )
        compaction = require_bool(
            specification["enable_semantic_compaction_audit"],
            field="enable semantic compaction audit",
        )
        if (
            maximum_blockers != 8
            or minimum_rollbacks != 3
            or maximum_interventions != 3
            or initial_repeat_count != 2
            or not compaction
        ):
            raise ValueError("signature-scoped rescue limits changed")
        return cls(
            maximum_added_blockers=maximum_blockers,
            minimum_consecutive_rollbacks=minimum_rollbacks,
            maximum_interventions=maximum_interventions,
            initial_repeat_count=initial_repeat_count,
            seed_namespace=require_nonempty_string(
                specification["seed_namespace"], field="rescue seed namespace"
            ),
            episode_key=require_nonempty_string(
                specification["episode_key"], field="rescue episode key"
            ),
            trial_index=require_int(
                specification["trial_index"], field="rescue trial index", minimum=0
            ),
            enable_semantic_compaction_audit=compaction,
            current_signature=platform_signature(state),
            current_streak=initial_repeat_count,
        )

    def requires_diagnostics(self, decision_index: int) -> bool:
        return int(decision_index) >= 0

    def action_for_decision(
        self, decision_index: int, state: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if self.pending_action is None:
            return None
        if int(decision_index) != int(self.pending_action["rescue_decision_index"]):
            return None
        if self.action_issued:
            raise ValueError("signature-scoped rescue action was requested twice")
        if platform_signature(state) != str(
            self.pending_action["platform_signature"]
        ):
            raise ValueError("signature-scoped rescue pending state changed")
        self.action_issued = True
        return dict(self.pending_action)

    @staticmethod
    def _exact_rollback(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> bool:
        return bool(
            snapshot["failure_reason"] == "conflict_bound_exceeded"
            and snapshot["replan_success"] is False
            and snapshot["rolled_back"] is True
            and platform_signature(before) == platform_signature(after)
            and int(before["num_of_colliding_pairs"])
            == int(after["num_of_colliding_pairs"])
        )

    def observe_decision(
        self,
        *,
        decision_index: int,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        decision = int(decision_index)
        snapshot = attempt_snapshot(metrics)
        before_signature = platform_signature(before)
        after_signature = platform_signature(after)
        executed = bool(
            self.pending_action is not None
            and self.action_issued
            and decision == int(self.pending_action["rescue_decision_index"])
        )
        executed_action = dict(self.pending_action or {}) if executed else None
        if executed_action is not None:
            if sorted(snapshot["neighborhood"]) != sorted(
                map(int, executed_action["agents"])
            ):
                raise ValueError("signature-scoped rescue neighborhood changed")
            if int(snapshot["requested_pp_random_seed"]) != int(
                executed_action["pp_random_seed"]
            ):
                raise ValueError("signature-scoped rescue seed changed")

        exact_rollback = self._exact_rollback(before, after, snapshot)
        if exact_rollback:
            if self.current_signature == before_signature:
                self.current_streak += 1
            else:
                self.current_signature = before_signature
                self.current_streak = 1
        else:
            self.current_signature = after_signature
            self.current_streak = 0

        resolved = bool(executed and after_signature != before_signature)
        if resolved:
            self.resolved_intervention_count += 1

        eligible = bool(
            exact_rollback
            and self.current_streak >= self.minimum_consecutive_rollbacks
            and before_signature not in self.used_signatures
            and self.intervention_count < self.maximum_interventions
        )
        scheduled_action: dict[str, Any] | None = None
        blockers: list[int] = []
        compact_plan: dict[str, Any] | None = None
        if eligible:
            base_agents = list(map(int, snapshot["neighborhood"]))
            blockers = ordered_external_blockers(
                metrics, base_agents, self.maximum_added_blockers
            )
            compact_plan = semantic_compact_plan(dict(after), base_agents)
            compact_members = set(map(int, compact_plan["compact_agents"]))
            planned_base = [agent for agent in base_agents if agent in compact_members]
            planned_agents = planned_base + [
                agent for agent in blockers if agent not in set(planned_base)
            ]
            next_seed = rescue_seed(
                namespace=self.seed_namespace,
                episode_key=self.episode_key,
                trial_index=self.trial_index,
                signature=before_signature,
                first_attempt_seed=int(snapshot["requested_pp_random_seed"]),
            )
            intervention_index = self.intervention_count
            scheduled_action = {
                "schema": RESCUE_SCHEMA,
                "mode": COMPACT_BLOCKER_AUGMENTED_MODE,
                "intervention_index": intervention_index,
                "trigger_decision_index": decision,
                "rescue_decision_index": decision + 1,
                "platform_signature": before_signature,
                "agents": planned_agents,
                "selected_blockers": blockers,
                "removed_agents": list(map(int, compact_plan["removed_agents"])),
                "compact_plan": compact_plan,
                "pp_random_seed": next_seed,
            }
            self.used_signatures.add(before_signature)
            self.intervention_count += 1

        record = {
            "schema": RESCUE_SCHEMA,
            "decision_index": decision,
            "platform_signature": before_signature,
            "exact_rollback": exact_rollback,
            "streak_after_attempt": self.current_streak,
            "trigger_eligible": eligible,
            "scheduled": scheduled_action is not None,
            "scheduled_rescue_decision_index": (
                int(scheduled_action["rescue_decision_index"])
                if scheduled_action is not None
                else None
            ),
            "intervention_executed": executed,
            "intervention_index": (
                int(executed_action["intervention_index"])
                if executed_action is not None
                else (
                    int(scheduled_action["intervention_index"])
                    if scheduled_action is not None
                    else None
                )
            ),
            "resolved_by_rescue": resolved,
            "first_attempt": snapshot,
            "observed_external_blockers": blockers,
            "selected_blockers": blockers,
            "compact_plan": compact_plan,
            "scheduled_action": scheduled_action,
        }
        self.records.append(record)
        self.pending_action = scheduled_action
        self.action_issued = False
        return dict(record)

    def summary(self) -> dict[str, Any]:
        return {
            "schema": RESCUE_SCHEMA,
            "mode": COMPACT_BLOCKER_AUGMENTED_MODE,
            "maximum_added_blockers": self.maximum_added_blockers,
            "minimum_consecutive_rollbacks": self.minimum_consecutive_rollbacks,
            "maximum_interventions": self.maximum_interventions,
            "initial_repeat_count": self.initial_repeat_count,
            "enable_semantic_compaction_audit": self.enable_semantic_compaction_audit,
            "intervention_count": self.intervention_count,
            "executed_intervention_count": sum(
                int(bool(record["intervention_executed"])) for record in self.records
            ),
            "resolved_intervention_count": self.resolved_intervention_count,
            "used_signature_count": len(self.used_signatures),
            "new_signature_intervention_count": max(0, self.intervention_count - 1),
            "pending_action": self.pending_action is not None,
        }


__all__ = ["RESCUE_SCHEMA", "SignatureScopedRescueTracker"]
