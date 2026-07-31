from __future__ import annotations

from typing import Any, Mapping

from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


STALL_SHADOW_TRANSITION_SCHEMA = "lns2.stall_shadow_transition.v3"
LEGACY_CONTROLLER_MODES = frozenset(
    {
        "v2-stall-safe",
        "v2-stall-shadow",
        "v2-repair-aware",
        "v2-critical",
        "v2-cost-top3-frozen",
        "v3-full",
        "v3-h3",
    }
)


class LegacyControllerDiagnosticError(ValueError):
    pass


def _mapping(value: Any, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LegacyControllerDiagnosticError(message)
    return value


def validate_legacy_controller_diagnostics(
    controller: Mapping[str, Any],
    metrics: Mapping[str, Any],
    after: Mapping[str, Any],
    route: str,
) -> bool:
    """Validate diagnostics for a retired controller without executing it.

    Returns ``True`` when the controller mode is a recognized historical mode
    and ``False`` for an active or unknown mode.
    """

    mode = str(controller.get("controller_mode", ""))
    if mode not in LEGACY_CONTROLLER_MODES:
        return False

    if mode == "v2-stall-safe":
        guard = _mapping(
            controller.get("stall_guard"),
            "stall-safe transition is missing guard diagnostics",
        )
        if str(guard.get("route")) != route:
            raise LegacyControllerDiagnosticError("stall guard route mismatch")

    elif mode == "v2-stall-shadow":
        shadow = _mapping(
            controller.get("stall_shadow"),
            "stall-shadow transition is missing diagnostics",
        )
        if route != "model" or str(shadow.get("route")) != "model":
            raise LegacyControllerDiagnosticError("stall shadow changed the v2 route")
        if str(shadow.get("schema")) != STALL_SHADOW_TRANSITION_SCHEMA:
            raise LegacyControllerDiagnosticError(
                "stall shadow transition schema mismatch"
            )
        if shadow.get("base_selection_preserved") is not True or shadow.get(
            "action_preserved"
        ) is not True:
            raise LegacyControllerDiagnosticError("stall shadow changed the v2 action")
        if str(shadow.get("effective_selected_candidate_id")) != str(
            controller.get("selected_candidate_id")
        ):
            raise LegacyControllerDiagnosticError(
                "stall shadow selected candidate does not match v2"
            )
        state_unchanged = shadow.get("state_unchanged")
        replan_success = metrics.get("replan_success")
        if not isinstance(state_unchanged, bool) or not isinstance(
            replan_success, bool
        ):
            raise LegacyControllerDiagnosticError(
                "stall shadow transition has non-boolean outcome evidence"
            )
        try:
            expected_outcome = classify_repair_outcome(
                before_fingerprint="state",
                after_fingerprint=("state" if state_unchanged else "changed"),
                replan_success=replan_success,
                conflicts_before=int(metrics["conflicts_before"]),
                conflicts_after=int(metrics["conflicts_after"]),
                feasible=bool(after.get("feasible")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LegacyControllerDiagnosticError(
                "stall shadow transition outcome evidence is inconsistent"
            ) from error
        if str(shadow.get("repair_outcome")) != expected_outcome:
            raise LegacyControllerDiagnosticError(
                "stall shadow transition outcome mismatch"
            )

    elif mode == "v2-repair-aware":
        repair = _mapping(
            controller.get("repair_aware"),
            "repair-aware transition is missing diagnostics",
        )
        if str(repair.get("route")) != route:
            raise LegacyControllerDiagnosticError("repair-aware route mismatch")
        if str(repair.get("repair_outcome")) not in {
            "hard_failure",
            "accepted_noop",
            "state_changed_no_reduction",
            "conflict_reduced",
            "feasible",
        }:
            raise LegacyControllerDiagnosticError(
                "repair-aware transition has an invalid outcome"
            )

    elif mode == "v2-critical":
        critical = _mapping(
            controller.get("critical_seed"),
            "v2-critical transition is missing seed diagnostics",
        )
        selected_seeds = list(map(int, critical.get("selected_seed_agents", [])))
        if route != "model" or not 1 <= len(selected_seeds) <= 4:
            raise LegacyControllerDiagnosticError(
                "v2-critical transition has invalid seed routing"
            )
        for candidate in controller.get("candidate_pool", []):
            if not set(map(int, candidate.get("seed_agents", []))) <= set(
                selected_seeds
            ):
                raise LegacyControllerDiagnosticError(
                    "v2-critical candidate escaped the retained seed set"
                )

    elif mode == "v2-cost-top3-frozen":
        cost_top3 = _mapping(
            controller.get("cost_top3"),
            "cost Top-3 transition is missing diagnostics",
        )
        if str(cost_top3.get("route")) != route or route != "model":
            raise LegacyControllerDiagnosticError("cost Top-3 route mismatch")
        if str(cost_top3.get("selected_candidate_id")) != str(
            controller.get("selected_candidate_id")
        ):
            raise LegacyControllerDiagnosticError(
                "cost Top-3 selected candidate mismatch"
            )
        rank = int(cost_top3.get("selected_v2_rank", 0))
        if rank < 1 or rank > 3:
            raise LegacyControllerDiagnosticError(
                "cost Top-3 selected rank is outside the frozen Top-3"
            )

    elif mode in {"v3-full", "v3-h3"}:
        v3 = _mapping(
            controller.get("v3"),
            "v3 transition is missing controller diagnostics",
        )
        if str(v3.get("route")) != route:
            raise LegacyControllerDiagnosticError("v3 route mismatch")
        if str(v3.get("repair_outcome")) not in {
            "hard_failure",
            "accepted_noop",
            "state_changed_no_reduction",
            "conflict_reduced",
            "feasible",
        }:
            raise LegacyControllerDiagnosticError(
                "v3 transition has an invalid outcome"
            )

    return True


__all__ = [
    "LEGACY_CONTROLLER_MODES",
    "LegacyControllerDiagnosticError",
    "STALL_SHADOW_TRANSITION_SCHEMA",
    "validate_legacy_controller_diagnostics",
]
