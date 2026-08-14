from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


RETRY_SCHEMA = "lns2.bounded_native_retry.v1"
VALID_FAILURE_REASONS = {"none", "conflict_bound_exceeded", "time_limit"}


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def conflict_edge_signature(state: Mapping[str, Any]) -> list[list[int]]:
    return sorted(
        [sorted(map(int, edge)) for edge in state.get("conflict_edges", ())]
    )


def platform_signature(state: Mapping[str, Any]) -> str:
    return _fingerprint(
        {
            "repair_fingerprint": repair_structure_fingerprint(dict(state)),
            "conflict_edges": conflict_edge_signature(state),
        }
    )


def bounded_retry_seed(
    *,
    namespace: str,
    episode_key: str,
    trial_index: int,
    signature: str,
    intervention_index: int,
    first_attempt_seed: int,
) -> int:
    seed = int(
        _fingerprint(
            {
                "namespace": str(namespace),
                "episode_key": str(episode_key),
                "trial_index": int(trial_index),
                "platform_signature": str(signature),
                "intervention_index": int(intervention_index),
            }
        )[:16],
        16,
    ) % (2**31)
    return (seed + 1) % (2**31) if seed == int(first_attempt_seed) else seed


def attempt_snapshot(metrics: Mapping[str, Any]) -> dict[str, Any]:
    reason = str(metrics.get("pp_failure_reason"))
    if reason not in VALID_FAILURE_REASONS:
        raise ValueError(f"invalid native PP failure reason: {reason}")
    return {
        "requested_random_seed": int(metrics.get("requested_random_seed", -1)),
        "requested_pp_random_seed": int(metrics.get("requested_pp_random_seed", -1)),
        "applied_pp_random_seed": int(metrics.get("applied_pp_random_seed", -1)),
        "requested_collect_pp_diagnostics": bool(
            metrics.get("requested_collect_pp_diagnostics", False)
        ),
        "neighborhood": list(map(int, metrics.get("neighborhood") or ())),
        "repair_order": list(map(int, metrics.get("repair_order") or ())),
        "replan_success": bool(metrics.get("replan_success")),
        "failure_reason": reason,
        "attempted_agent_count": int(metrics.get("pp_attempted_agent_count", -1)),
        "inserted_agent_count": int(metrics.get("pp_inserted_agent_count", -1)),
        "failed_agent": int(metrics.get("pp_failed_agent", -1)),
        "failed_order_index": int(metrics.get("pp_failed_order_index", -1)),
        "rolled_back": bool(metrics.get("pp_rolled_back", False)),
        "conflicts_after": int(metrics.get("conflicts_after", -1)),
        "pp_agent_diagnostics": [
            dict(row) for row in metrics.get("pp_agent_diagnostics") or ()
        ],
    }


TIMING_FIELDS = (
    "step_runtime",
    "episode_runtime_delta_seconds",
    "native_step_seconds",
    "native_neighborhood_generation_seconds",
    "native_replan_seconds",
    "pp_replan_seconds",
    "native_state_snapshot_seconds",
    "binding_solver_call_seconds",
    "binding_state_snapshot_seconds",
    "state_to_python_seconds",
    "metrics_to_python_seconds",
    "binding_total_seconds",
    "binding_residual_seconds",
    "native_repair_bookkeeping_seconds",
    "native_residual_seconds",
)


def merged_retry_metrics(
    first: Mapping[str, Any],
    retry: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(retry)
    for name in TIMING_FIELDS:
        merged[name] = float(first.get(name, 0.0)) + float(retry.get(name, 0.0))
    merged["bounded_native_retry"] = dict(record)
    return merged


@dataclass
class BoundedNativeRetryTracker:
    enabled: bool
    minimum_consecutive_rollbacks: int
    maximum_interventions: int
    initial_repeat_count: int
    seed_namespace: str
    episode_key: str
    trial_index: int
    first_retry_seed: int
    current_signature: str
    current_streak: int
    used_signatures: set[str] = field(default_factory=set)
    intervention_count: int = 0
    platform_seen: bool = False
    persistent_platform_seen: bool = False
    persistent_decision_count: int = 0
    resolved_intervention_count: int = 0

    @classmethod
    def from_spec(
        cls, state: Mapping[str, Any], specification: Mapping[str, Any]
    ) -> "BoundedNativeRetryTracker":
        required = {
            "enabled",
            "minimum_consecutive_rollbacks",
            "maximum_interventions",
            "initial_repeat_count",
            "seed_namespace",
            "episode_key",
            "trial_index",
            "first_retry_seed",
        }
        if set(specification) != required:
            raise ValueError("bounded native retry specification changed")
        minimum = int(specification["minimum_consecutive_rollbacks"])
        maximum = int(specification["maximum_interventions"])
        initial = int(specification["initial_repeat_count"])
        if minimum != 3 or maximum != 3 or initial != 2:
            raise ValueError("bounded native retry limits changed")
        return cls(
            enabled=bool(specification["enabled"]),
            minimum_consecutive_rollbacks=minimum,
            maximum_interventions=maximum,
            initial_repeat_count=initial,
            seed_namespace=str(specification["seed_namespace"]),
            episode_key=str(specification["episode_key"]),
            trial_index=int(specification["trial_index"]),
            first_retry_seed=int(specification["first_retry_seed"]),
            current_signature=platform_signature(state),
            current_streak=initial,
        )

    def observe_first_attempt(
        self,
        *,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        metrics: Mapping[str, Any],
        decision_index: int,
    ) -> dict[str, Any]:
        before_signature = platform_signature(before)
        after_signature = platform_signature(after)
        snapshot = attempt_snapshot(metrics)
        exact_rollback = bool(
            snapshot["failure_reason"] == "conflict_bound_exceeded"
            and snapshot["replan_success"] is False
            and snapshot["rolled_back"] is True
            and before_signature == after_signature
            and int(before["num_of_colliding_pairs"])
            == int(after["num_of_colliding_pairs"])
        )
        if exact_rollback:
            if self.current_signature == before_signature:
                self.current_streak += 1
            else:
                self.current_signature = before_signature
                self.current_streak = 1
        else:
            self.current_signature = after_signature
            self.current_streak = 0
        eligible = bool(
            exact_rollback
            and self.current_streak >= self.minimum_consecutive_rollbacks
            and before_signature not in self.used_signatures
            and self.intervention_count < self.maximum_interventions
        )
        if self.current_streak >= self.minimum_consecutive_rollbacks:
            self.platform_seen = True
        should_retry = bool(self.enabled and eligible)
        retry_seed = None
        intervention_index = None
        if should_retry:
            intervention_index = self.intervention_count
            retry_seed = (
                self.first_retry_seed
                if intervention_index == 0
                else bounded_retry_seed(
                    namespace=self.seed_namespace,
                    episode_key=self.episode_key,
                    trial_index=self.trial_index,
                    signature=before_signature,
                    intervention_index=intervention_index,
                    first_attempt_seed=int(snapshot["requested_pp_random_seed"]),
                )
            )
            if retry_seed == int(snapshot["requested_pp_random_seed"]):
                raise ValueError("bounded retry seed equals first-attempt seed")
            self.used_signatures.add(before_signature)
            self.intervention_count += 1
        return {
            "schema": RETRY_SCHEMA,
            "decision_index": int(decision_index),
            "platform_signature": before_signature,
            "exact_rollback": exact_rollback,
            "streak_after_first_attempt": self.current_streak,
            "platform_threshold_reached": (
                self.current_streak >= self.minimum_consecutive_rollbacks
            ),
            "trigger_eligible": eligible,
            "triggered": should_retry,
            "intervention_index": intervention_index,
            "retry_seed": retry_seed,
            "first_attempt": snapshot,
            "retry_attempt": None,
            "retry_skipped_reason": None,
            "persistent_after_transaction": (
                self.current_streak >= self.minimum_consecutive_rollbacks
            ),
            "resolved_by_retry": False,
        }

    def cancel_retry(
        self, record: dict[str, Any], *, reason: str
    ) -> dict[str, Any]:
        """Cancel a reserved retry before native PP is called.

        A trigger reserves the signature and intervention slot atomically.  If
        the episode wall budget is already exhausted, release both so the
        trace does not claim that an intervention was executed.
        """
        if record.get("triggered") is not True:
            raise ValueError("cannot cancel an untriggered bounded retry")
        signature = str(record["platform_signature"])
        if signature not in self.used_signatures or self.intervention_count <= 0:
            raise ValueError("bounded retry reservation is missing")
        self.used_signatures.remove(signature)
        self.intervention_count -= 1
        return {
            **record,
            "triggered": False,
            "intervention_index": None,
            "retry_seed": None,
            "retry_skipped_reason": str(reason),
        }

    def observe_retry(
        self,
        record: dict[str, Any],
        *,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        if record.get("triggered") is not True:
            raise ValueError("cannot observe an untriggered bounded retry")
        snapshot = attempt_snapshot(metrics)
        if int(snapshot["requested_pp_random_seed"]) != int(record["retry_seed"]):
            raise ValueError("bounded retry seed was not retained")
        after_signature = platform_signature(after)
        changed = after_signature != str(record["platform_signature"])
        # A native PP call may report success while accepting an exact no-op.
        # Platform escape is a trajectory property, so only an observable
        # repair/conflict signature change resolves the intervention.
        resolved = bool(changed)
        if resolved:
            self.current_signature = after_signature
            self.current_streak = 0
            self.resolved_intervention_count += 1
        record = {
            **record,
            "retry_attempt": snapshot,
            "retry_skipped_reason": None,
            "persistent_after_transaction": not resolved,
            "resolved_by_retry": resolved,
        }
        return record

    def finalize_decision(self, record: Mapping[str, Any]) -> None:
        if bool(record["persistent_after_transaction"]):
            self.persistent_platform_seen = True
            self.persistent_decision_count += 1

    def summary(self) -> dict[str, Any]:
        return {
            "schema": RETRY_SCHEMA,
            "enabled": self.enabled,
            "minimum_consecutive_rollbacks": self.minimum_consecutive_rollbacks,
            "maximum_interventions": self.maximum_interventions,
            "initial_repeat_count": self.initial_repeat_count,
            "intervention_count": self.intervention_count,
            "used_platform_signature_count": len(self.used_signatures),
            "resolved_intervention_count": self.resolved_intervention_count,
            "platform_seen": self.platform_seen,
            "persistent_platform_seen": self.persistent_platform_seen,
            "persistent_platform_decision_count": self.persistent_decision_count,
        }
