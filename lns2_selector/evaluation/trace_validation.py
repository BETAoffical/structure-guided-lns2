from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Any

from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V1,
    EPISODE_SCHEMA_V2,
    TRACE_FORMAT_DELTA_GZIP_V2,
    TRACE_FORMAT_FULL_V1,
    TraceStorageError,
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
    storage_fingerprint,
)
from experiments.repair_collection import (
    SCHEMA_VERSION,
    _low_level_delta,
    state_fingerprint,
)
from lns2_selector.compatibility.controller_diagnostics import (
    LegacyControllerDiagnosticError,
    validate_legacy_controller_diagnostics,
)
from lns2_selector.compatibility.metrics import fixed_budget_conflict_auc
from lns2_selector.runtime.metrics import wall_clock_conflict_auc


LEARNED_POLICIES = ("proposal_dynamic", "realized_dynamic")
REPAIR_TIMING_SCHEMA_V1 = "lns2.repair_timing.v1"
REPAIR_TIMING_SCHEMA_V2 = "lns2.repair_timing.v2"
REPAIR_TIMING_SCHEMAS = (REPAIR_TIMING_SCHEMA_V1, REPAIR_TIMING_SCHEMA_V2)
REPAIR_TIMING_SCHEMA = REPAIR_TIMING_SCHEMA_V2
_NATIVE_REPAIR_TIMING_KEYS = frozenset(
    {
        "native_step_seconds",
        "native_neighborhood_generation_seconds",
        "native_replan_seconds",
        "pp_replan_seconds",
        "native_state_snapshot_seconds",
        "native_repair_bookkeeping_seconds",
        "native_residual_seconds",
        "binding_solver_call_seconds",
        "binding_state_snapshot_seconds",
        "state_to_python_seconds",
        "metrics_to_python_seconds",
        "binding_total_seconds",
        "binding_residual_seconds",
    }
)


class ClosedLoopTraceError(ValueError):
    pass


def native_repair_timing_schema(metrics: dict[str, Any]) -> str | None:
    if "episode_runtime_delta_seconds" in metrics:
        missing = _NATIVE_REPAIR_TIMING_KEYS.difference(metrics)
        if missing:
            raise ValueError(
                "repair timing v2 lacks native timing fields: "
                + ", ".join(sorted(missing))
            )
        if "step_runtime" not in metrics:
            raise ValueError("repair timing v2 lacks step_runtime")
        return REPAIR_TIMING_SCHEMA_V2
    present = _NATIVE_REPAIR_TIMING_KEYS.intersection(metrics)
    if not present:
        return None
    missing = _NATIVE_REPAIR_TIMING_KEYS.difference(metrics)
    if missing:
        raise ValueError(
            "repair timing v1 lacks native timing fields: "
            + ", ".join(sorted(missing))
        )
    if "step_runtime" not in metrics:
        raise ValueError("repair timing v1 lacks step_runtime")
    return REPAIR_TIMING_SCHEMA_V1


def validate_closed_loop_trace(
    path: str | Path,
    run_fingerprint: str,
    *,
    expected_episode_id: str | None = None,
    expected_policy: str | None = None,
    expected_solver_seed: int | None = None,
    metric_iteration_budget: int | None = None,
    collection_root: str | Path | None = None,
) -> dict[str, Any]:
    trace_path = Path(path)
    try:
        rows = read_trace_events(trace_path)
    except TraceStorageError as error:
        raise ClosedLoopTraceError(f"cannot read trace: {error}") from error
    if len(rows) < 2:
        raise ClosedLoopTraceError("trace must contain initial and finish events")
    event_schema = str(rows[0].get("schema"))
    if event_schema == EPISODE_SCHEMA_V1:
        trace_format = TRACE_FORMAT_FULL_V1
        expected_schema_version = SCHEMA_VERSION
    elif event_schema == EPISODE_SCHEMA_V2:
        trace_format = TRACE_FORMAT_DELTA_GZIP_V2
        expected_schema_version = 2
    else:
        raise ClosedLoopTraceError("trace contains an unexpected event schema")
    if any(str(row.get("schema")) != event_schema for row in rows):
        raise ClosedLoopTraceError("trace contains an unexpected event schema")
    if any(int(row.get("schema_version", -1)) != expected_schema_version for row in rows):
        raise ClosedLoopTraceError("trace contains an unsupported schema version")
    if trace_format == TRACE_FORMAT_DELTA_GZIP_V2:
        expected_storage = storage_fingerprint(trace_format)
        if any(str(row.get("trace_format")) != trace_format for row in rows):
            raise ClosedLoopTraceError("compact trace format marker mismatch")
        if any(str(row.get("storage_fingerprint")) != expected_storage for row in rows):
            raise ClosedLoopTraceError("compact trace storage fingerprint mismatch")
    if any(str(row.get("run_fingerprint")) != run_fingerprint for row in rows):
        raise ClosedLoopTraceError("trace run fingerprint mismatch")
    if rows[0].get("event") != "initial" or rows[-1].get("event") != "finish":
        raise ClosedLoopTraceError("trace event boundaries are invalid")
    if any(row.get("event") != "transition" for row in rows[1:-1]):
        raise ClosedLoopTraceError("trace contains a non-transition event before finish")

    initial = rows[0]
    finish = rows[-1]
    episode_id = str(initial.get("episode_id"))
    policy = str(initial.get("policy"))
    solver_seed = int(initial.get("solver_seed", -1))
    if expected_episode_id is not None and episode_id != expected_episode_id:
        raise ClosedLoopTraceError("trace episode id mismatch")
    if expected_policy is not None and policy != expected_policy:
        raise ClosedLoopTraceError("trace policy mismatch")
    if expected_solver_seed is not None and solver_seed != expected_solver_seed:
        raise ClosedLoopTraceError("trace solver seed mismatch")
    if str(finish.get("episode_id")) != episode_id or str(finish.get("policy")) != policy:
        raise ClosedLoopTraceError("finish metadata mismatch")

    initial_state_ref = None
    if trace_format == TRACE_FORMAT_FULL_V1:
        state = initial.get("state")
        if not isinstance(state, dict):
            raise ClosedLoopTraceError("initial event is missing state")
    else:
        initial_state_ref = str(initial.get("state_blob", ""))
        if not initial_state_ref:
            raise ClosedLoopTraceError("compact initial event is missing state blob")
        try:
            blob_path = resolve_state_blob(
                trace_path,
                initial_state_ref,
                Path(collection_root).resolve() if collection_root is not None else None,
            )
            state = read_state_blob(blob_path)
        except TraceStorageError as error:
            raise ClosedLoopTraceError(str(error)) from error
        extras = initial.get("state_extras")
        if not isinstance(extras, dict):
            raise ClosedLoopTraceError("compact initial event has invalid state extras")
        if any(key in state for key in extras):
            raise ClosedLoopTraceError("compact initial state extras overlap fingerprint fields")
        state.update(extras)
    initial_hash = state_fingerprint(state)
    if str(initial.get("state_fingerprint")) != initial_hash:
        raise ClosedLoopTraceError("initial state fingerprint mismatch")
    conflicts = [int(state.get("num_of_colliding_pairs", -1))]
    learned_policy = policy in LEARNED_POLICIES
    route_counts: collections.Counter[str] = collections.Counter()
    previous_route: str | None = None
    route_switch_count = 0
    transition_elapsed_seconds: list[float] = []
    for decision_index, event in enumerate(rows[1:-1]):
        if str(event.get("episode_id")) != episode_id:
            raise ClosedLoopTraceError("transition episode id mismatch")
        if int(event.get("decision_index", -1)) != decision_index:
            raise ClosedLoopTraceError("transition decision indexes are not contiguous")
        before_hash = state_fingerprint(state)
        if str(event.get("before_fingerprint")) != before_hash:
            raise ClosedLoopTraceError("transition before fingerprint mismatch")
        if trace_format == TRACE_FORMAT_FULL_V1:
            after = event.get("after")
            if not isinstance(after, dict):
                raise ClosedLoopTraceError("transition is missing after state")
        else:
            try:
                after = apply_state_delta(state, event.get("state_delta"))
                after.update(
                    apply_extras_delta(state, event.get("state_extras_delta"))
                )
            except (TraceStorageError, TypeError, ValueError) as error:
                raise ClosedLoopTraceError(
                    f"transition state delta is invalid: {error}"
                ) from error
        after_hash = state_fingerprint(after)
        if str(event.get("after_fingerprint")) != after_hash:
            raise ClosedLoopTraceError("transition after fingerprint mismatch")
        metrics = event.get("metrics")
        action = event.get("action")
        if not isinstance(metrics, dict) or not isinstance(action, dict):
            raise ClosedLoopTraceError("transition is missing action or metrics")
        declared_native_timing_schema = event.get("native_timing_schema")
        if (
            declared_native_timing_schema is not None
            and str(declared_native_timing_schema) not in REPAIR_TIMING_SCHEMAS
        ):
            raise ClosedLoopTraceError(
                "transition has an unsupported native timing schema"
            )
        try:
            detected_native_timing_schema = native_repair_timing_schema(
                metrics
            )
        except (TypeError, ValueError) as error:
            raise ClosedLoopTraceError(
                f"transition native timing metrics are incomplete: {error}"
            ) from error
        if declared_native_timing_schema is None:
            if detected_native_timing_schema is not None:
                raise ClosedLoopTraceError(
                    "transition native timing metrics are missing their schema"
                )
        elif detected_native_timing_schema != str(declared_native_timing_schema):
            raise ClosedLoopTraceError(
                "transition native timing schema does not match its metrics"
            )
        native_timing_schema = detected_native_timing_schema
        if native_timing_schema is not None:
            try:
                native_values = {
                    name: float(metrics[name])
                    for name in _NATIVE_REPAIR_TIMING_KEYS
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ClosedLoopTraceError(
                    "transition native timing metrics are incomplete"
                ) from error
            if any(
                not math.isfinite(value) or value < 0.0
                for value in native_values.values()
            ):
                raise ClosedLoopTraceError(
                    "transition native timing metrics must be non-negative"
                )
            native_step = native_values["native_step_seconds"]
            native_partition = sum(
                native_values[name]
                for name in (
                    "native_neighborhood_generation_seconds",
                    "native_replan_seconds",
                    "native_state_snapshot_seconds",
                    "native_repair_bookkeeping_seconds",
                    "native_residual_seconds",
                )
            )
            if not math.isclose(
                native_partition,
                native_step,
                rel_tol=0.01,
                abs_tol=max(1e-6, 0.01 * native_step),
            ):
                raise ClosedLoopTraceError("native step timing does not close")
            if not math.isclose(
                native_values["native_replan_seconds"],
                native_values["pp_replan_seconds"],
                rel_tol=0.01,
                abs_tol=max(
                    1e-6,
                    0.01
                    * max(
                        native_values["native_replan_seconds"],
                        native_values["pp_replan_seconds"],
                    ),
                ),
            ):
                raise ClosedLoopTraceError(
                    "native replan timing does not match PP timing"
                )
            binding_partition = sum(
                native_values[name]
                for name in (
                    "binding_solver_call_seconds",
                    "binding_state_snapshot_seconds",
                    "state_to_python_seconds",
                    "metrics_to_python_seconds",
                    "binding_residual_seconds",
                )
            )
            binding_total = native_values["binding_total_seconds"]
            binding_tolerance = max(
                1e-6,
                0.01 * max(binding_total, binding_partition, 1e-6),
            )
            if not math.isclose(
                binding_partition,
                binding_total,
                rel_tol=0.01,
                abs_tol=binding_tolerance,
            ):
                raise ClosedLoopTraceError("binding timing does not close")
            if (
                native_values["binding_solver_call_seconds"]
                + max(1e-5, 0.01 * max(native_step, 1e-6))
                < native_step
            ):
                raise ClosedLoopTraceError(
                    "binding solver timing is below native step"
                )
            if native_timing_schema == REPAIR_TIMING_SCHEMA_V2:
                try:
                    step_runtime = float(metrics["step_runtime"])
                    episode_runtime_delta = float(
                        metrics["episode_runtime_delta_seconds"]
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ClosedLoopTraceError(
                        "repair timing v2 metrics are incomplete"
                    ) from error
                if (
                    not math.isfinite(step_runtime)
                    or step_runtime < 0.0
                    or not math.isfinite(episode_runtime_delta)
                    or episode_runtime_delta < 0.0
                ):
                    raise ClosedLoopTraceError(
                        "repair timing v2 metrics must be non-negative"
                    )
                native_tolerance = max(
                    1e-6, 0.01 * max(native_step, 1e-6)
                )
                if not math.isclose(
                    step_runtime,
                    native_step,
                    rel_tol=0.01,
                    abs_tol=native_tolerance,
                ):
                    raise ClosedLoopTraceError(
                        "repair timing v2 step_runtime does not match native step"
                    )
                # episode_runtime_delta ends before InitLNS takes its final
                # post-step snapshot, while native_step includes that snapshot.
                # Conversely, the episode delta can include Python controller
                # time between native calls.  Both values are valid diagnostics,
                # but neither is an ordering bound for the other.
            else:
                try:
                    legacy_step_runtime = float(metrics["step_runtime"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ClosedLoopTraceError(
                        "repair timing v1 metrics are incomplete"
                    ) from error
                if (
                    not math.isfinite(legacy_step_runtime)
                    or legacy_step_runtime < 0.0
                ):
                    raise ClosedLoopTraceError(
                        "repair timing v1 metrics must be non-negative"
                    )
        action_pp_seed = int(action.get("pp_random_seed", -1))
        retry_record = metrics.get("bounded_native_retry")
        if isinstance(retry_record, dict):
            first_attempt = retry_record.get("first_attempt")
            if not isinstance(first_attempt, dict):
                raise ClosedLoopTraceError(
                    "bounded retry transition is missing first-attempt evidence"
                )
            first_requested = int(
                first_attempt.get("requested_pp_random_seed", -1)
            )
            first_applied = int(first_attempt.get("applied_pp_random_seed", -1))
            if action_pp_seed != first_requested:
                raise ClosedLoopTraceError("transition requested PP seed mismatch")
            if first_attempt.get("repair_order") and action_pp_seed >= 0:
                if first_applied != action_pp_seed:
                    raise ClosedLoopTraceError("transition applied PP seed mismatch")
            elif first_applied >= 0:
                raise ClosedLoopTraceError(
                    "transition applied an unexpected PP seed"
                )
            if retry_record.get("triggered"):
                retry_attempt = retry_record.get("retry_attempt")
                if not isinstance(retry_attempt, dict):
                    raise ClosedLoopTraceError(
                        "bounded retry transition is missing retry-attempt evidence"
                    )
                retry_seed = int(retry_record.get("retry_seed", -1))
                retry_requested = int(
                    retry_attempt.get("requested_pp_random_seed", -1)
                )
                retry_applied = int(
                    retry_attempt.get("applied_pp_random_seed", -1)
                )
                if retry_seed < 0 or retry_seed != retry_requested:
                    raise ClosedLoopTraceError(
                        "bounded retry requested PP seed mismatch"
                    )
                if retry_attempt.get("repair_order"):
                    if retry_applied != retry_seed:
                        raise ClosedLoopTraceError(
                            "bounded retry applied PP seed mismatch"
                        )
                elif retry_applied >= 0:
                    raise ClosedLoopTraceError(
                        "bounded retry applied an unexpected PP seed"
                    )
                if (
                    int(metrics.get("requested_pp_random_seed", -1))
                    != retry_requested
                    or int(metrics.get("applied_pp_random_seed", -1))
                    != retry_applied
                ):
                    raise ClosedLoopTraceError(
                        "bounded retry top-level PP seed evidence mismatch"
                    )
            elif (
                int(metrics.get("requested_pp_random_seed", -1))
                != first_requested
                or int(metrics.get("applied_pp_random_seed", -1))
                != first_applied
            ):
                raise ClosedLoopTraceError(
                    "bounded retry first-attempt PP seed evidence mismatch"
                )
        else:
            requested_pp_seed = int(metrics.get("requested_pp_random_seed", -1))
            applied_pp_seed = int(metrics.get("applied_pp_random_seed", -1))
            if action_pp_seed != requested_pp_seed:
                raise ClosedLoopTraceError("transition requested PP seed mismatch")
            if metrics.get("repair_order") and action_pp_seed >= 0:
                if applied_pp_seed != action_pp_seed:
                    raise ClosedLoopTraceError("transition applied PP seed mismatch")
            elif applied_pp_seed >= 0:
                raise ClosedLoopTraceError("transition applied an unexpected PP seed")
        if int(metrics.get("conflicts_before", -1)) != conflicts[-1]:
            raise ClosedLoopTraceError("transition conflicts_before mismatch")
        after_conflicts = int(after.get("num_of_colliding_pairs", -1))
        if int(metrics.get("conflicts_after", -1)) != after_conflicts:
            raise ClosedLoopTraceError("transition conflicts_after mismatch")
        if event.get("low_level_delta") != _low_level_delta(state, after):
            raise ClosedLoopTraceError("transition low-level delta mismatch")
        if bool(event.get("terminated")) != bool(after.get("feasible")):
            raise ClosedLoopTraceError("transition terminated flag mismatch")
        if bool(event.get("truncated")) != (
            bool(after.get("done")) and not bool(after.get("feasible"))
        ):
            raise ClosedLoopTraceError("transition truncated flag mismatch")
        elapsed_seconds = float(event.get("elapsed_wall_seconds", -1.0))
        if (
            not math.isfinite(elapsed_seconds)
            or elapsed_seconds < 0.0
            or (
                transition_elapsed_seconds
                and elapsed_seconds < transition_elapsed_seconds[-1]
            )
        ):
            raise ClosedLoopTraceError("transition wall times are invalid")
        transition_elapsed_seconds.append(elapsed_seconds)
        timings = event.get("timings")
        if native_timing_schema is not None and timings is None:
            raise ClosedLoopTraceError(
                "instrumented transition is missing timings"
            )
        if timings is not None:
            if not isinstance(timings, dict):
                raise ClosedLoopTraceError("transition timings are invalid")
            numeric_timings = {
                str(name): float(value) for name, value in timings.items()
            }
            if any(
                not math.isfinite(value) or value < 0.0
                for value in numeric_timings.values()
            ):
                raise ClosedLoopTraceError("transition timings must be non-negative")
            if native_timing_schema is not None:
                timing_metric_pairs = {
                    "native_neighborhood_generation_seconds": native_values[
                        "native_neighborhood_generation_seconds"
                    ],
                    "pp_replan_seconds": native_values[
                        "pp_replan_seconds"
                    ],
                    "repair_bookkeeping_seconds": native_values[
                        "native_repair_bookkeeping_seconds"
                    ],
                    "native_residual_seconds": native_values[
                        "native_residual_seconds"
                    ],
                    "state_export_seconds": sum(
                        native_values[name]
                        for name in (
                            "native_state_snapshot_seconds",
                            "binding_state_snapshot_seconds",
                            "state_to_python_seconds",
                        )
                    ),
                }
                if native_timing_schema == REPAIR_TIMING_SCHEMA_V2:
                    timing_metric_pairs.update(
                        {
                            "native_step_seconds": native_values[
                                "native_step_seconds"
                            ],
                            "episode_runtime_delta_seconds": float(
                                metrics["episode_runtime_delta_seconds"]
                            ),
                        }
                    )
                else:
                    # Timing-v1 traces produced before the wall-clock
                    # decomposition did not copy these two aggregate values
                    # into event["timings"].  They remain fully represented
                    # in the native metrics, so accept their absence while
                    # still validating them when a later v1 producer included
                    # the optional copies.
                    optional_v1_pairs = {
                        "native_step_seconds": native_values[
                            "native_step_seconds"
                        ],
                        "episode_runtime_delta_seconds": float(
                            metrics["step_runtime"]
                        ),
                    }
                    timing_metric_pairs.update(
                        {
                            name: value
                            for name, value in optional_v1_pairs.items()
                            if name in numeric_timings
                        }
                    )
                for name, expected_value in timing_metric_pairs.items():
                    if name not in numeric_timings:
                        raise ClosedLoopTraceError(
                            f"transition timings are missing {name}"
                        )
                    actual_value = numeric_timings[name]
                    field_tolerance = max(
                        1e-6,
                        0.01
                        * max(
                            actual_value,
                            expected_value,
                            1e-6,
                        ),
                    )
                    if not math.isclose(
                        actual_value,
                        expected_value,
                        rel_tol=0.01,
                        abs_tol=field_tolerance,
                    ):
                        raise ClosedLoopTraceError(
                            f"transition {name} does not match metrics"
                        )
                if "environment_step_wall_seconds" not in numeric_timings:
                    raise ClosedLoopTraceError(
                        "transition timings are missing environment_step_wall_seconds"
                    )
                repair_wall_seconds = float(
                    event.get("repair_wall_seconds", -1.0)
                )
                step_wall_seconds = numeric_timings[
                    "environment_step_wall_seconds"
                ]
                wall_tolerance = max(
                    1e-6,
                    0.01
                    * max(
                        repair_wall_seconds,
                        step_wall_seconds,
                        1e-6,
                    ),
                )
                if (
                    not math.isfinite(repair_wall_seconds)
                    or repair_wall_seconds < 0.0
                    or not math.isclose(
                        step_wall_seconds,
                        repair_wall_seconds,
                        rel_tol=0.01,
                        abs_tol=wall_tolerance,
                    )
                ):
                    raise ClosedLoopTraceError(
                        "transition environment step timing does not match event"
                    )
            selection_expected = float(
                numeric_timings.get("controller_before_repair_seconds", 0.0)
            ) + float(
                numeric_timings.get(
                    "native_neighborhood_generation_seconds", 0.0
                )
            )
            tolerance = max(1e-6, 0.01 * max(selection_expected, 1e-6))
            if not math.isclose(
                float(numeric_timings.get("neighborhood_selection_seconds", 0.0)),
                selection_expected,
                rel_tol=0.01,
                abs_tol=tolerance,
            ):
                raise ClosedLoopTraceError("neighborhood selection timing does not close")
            step_partition = sum(
                float(numeric_timings.get(name, 0.0))
                for name in (
                    "native_neighborhood_generation_seconds",
                    "pp_replan_seconds",
                    "repair_bookkeeping_seconds",
                    "state_export_seconds",
                    "environment_step_residual_seconds",
                )
            )
            step_wall = float(
                numeric_timings.get(
                    "environment_step_wall_seconds",
                    event.get("repair_wall_seconds", 0.0),
                )
            )
            if step_partition > step_wall + max(1e-5, 0.01 * step_wall):
                raise ClosedLoopTraceError("environment step timing exceeds its parent")
        if learned_policy:
            controller = event.get("controller")
            if not isinstance(controller, dict):
                raise ClosedLoopTraceError("learned transition is missing controller data")
            route = str(controller.get("route", "model"))
            if route not in {"model", "official_adaptive"}:
                raise ClosedLoopTraceError("learned transition has an invalid route")
            route_counts[route] += 1
            try:
                validate_legacy_controller_diagnostics(
                    controller, metrics, after, route
                )
            except LegacyControllerDiagnosticError as error:
                raise ClosedLoopTraceError(str(error)) from error
            if str(controller.get("controller_mode")) == "v3-s3":
                v3_s3 = controller.get("v3_s3")
                if not isinstance(v3_s3, dict):
                    raise ClosedLoopTraceError(
                        "v3-S3 transition is missing controller diagnostics"
                    )
                if str(v3_s3.get("route")) != "v3-s3" or route != "model":
                    raise ClosedLoopTraceError("v3-S3 route mismatch")
                if str(v3_s3.get("repair_outcome")) not in {
                    "hard_failure",
                    "accepted_noop",
                    "state_changed_no_reduction",
                    "conflict_reduced",
                    "feasible",
                }:
                    raise ClosedLoopTraceError(
                        "v3-S3 transition has an invalid outcome"
                    )
            if previous_route is not None and previous_route != route:
                route_switch_count += 1
            previous_route = route
            actual = sorted(map(int, metrics.get("neighborhood", [])))
            if route == "official_adaptive":
                if action.get("mode") != "official":
                    raise ClosedLoopTraceError("official route did not use an official action")
                if controller.get("selected_candidate_id") is not None:
                    raise ClosedLoopTraceError("official route unexpectedly selected a candidate")
            else:
                requested = sorted(map(int, action.get("agents", [])))
                if action.get("mode") != "explicit_neighborhood" or requested != actual:
                    raise ClosedLoopTraceError("learned transition neighborhood mismatch")
                if int(action.get("random_seed", -1)) < 0:
                    raise ClosedLoopTraceError("learned transition is missing explicit random seed")
                bounded_record = metrics.get("bounded_native_retry")
                if isinstance(bounded_record, dict):
                    first_attempt = bounded_record.get("first_attempt")
                    if (
                        not isinstance(first_attempt, dict)
                        or int(
                            first_attempt.get(
                                "requested_random_seed",
                                action.get("random_seed", -1),
                            )
                        )
                        != int(action["random_seed"])
                    ):
                        raise ClosedLoopTraceError(
                            "learned transition repair seed mismatch"
                        )
                    if bounded_record.get("triggered"):
                        retry_attempt = bounded_record.get("retry_attempt")
                        if (
                            not isinstance(retry_attempt, dict)
                            or int(
                                retry_attempt.get(
                                    "requested_random_seed",
                                    retry_attempt.get(
                                        "requested_pp_random_seed", -1
                                    ),
                                )
                            )
                            != int(bounded_record.get("retry_seed", -1))
                            or int(metrics.get("requested_random_seed", -1))
                            != int(
                                retry_attempt.get(
                                    "requested_random_seed",
                                    retry_attempt.get(
                                        "requested_pp_random_seed", -1
                                    ),
                                )
                            )
                        ):
                            raise ClosedLoopTraceError(
                                "learned bounded retry repair seed mismatch"
                            )
                    elif int(metrics.get("requested_random_seed", -1)) != int(
                        first_attempt.get(
                            "requested_random_seed",
                            action.get("random_seed", -1),
                        )
                    ):
                        raise ClosedLoopTraceError(
                            "learned bounded retry top-level repair seed mismatch"
                        )
                elif int(metrics.get("requested_random_seed", -1)) != int(
                    action["random_seed"]
                ):
                    raise ClosedLoopTraceError("learned transition repair seed mismatch")
                selected_id = str(controller.get("selected_candidate_id", ""))
                matching = [
                    candidate
                    for candidate in controller.get("candidate_pool", [])
                    if str(candidate.get("candidate_id")) == selected_id
                ]
                if len(matching) != 1 or sorted(
                    map(int, matching[0].get("agents", []))
                ) != requested:
                    raise ClosedLoopTraceError("learned transition selected candidate mismatch")
        conflicts.append(after_conflicts)
        state = after

    summary = finish.get("summary")
    if not isinstance(summary, dict):
        raise ClosedLoopTraceError("finish event is missing summary")
    final_hash = state_fingerprint(state)
    if str(finish.get("final_fingerprint")) != final_hash:
        raise ClosedLoopTraceError("finish state fingerprint mismatch")
    if str(summary.get("initial_fingerprint")) != initial_hash:
        raise ClosedLoopTraceError("summary initial fingerprint mismatch")
    expected_values = {
        "initial_conflicts": conflicts[0],
        "final_conflicts": conflicts[-1],
        "repair_iterations": len(conflicts) - 1,
        "conflict_trajectory": conflicts,
        "final_sum_of_costs": int(state.get("sum_of_costs", -1)),
        "final_low_level": state.get("low_level"),
    }
    for name, value in expected_values.items():
        if summary.get(name) != value:
            raise ClosedLoopTraceError(f"summary {name} mismatch")
    if summary.get("transition_elapsed_seconds") is not None and summary.get(
        "transition_elapsed_seconds"
    ) != transition_elapsed_seconds:
        raise ClosedLoopTraceError("summary transition_elapsed_seconds mismatch")
    route_summary_fields = (
        "model_decision_count",
        "official_decision_count",
        "route_switch_count",
        "model_route_fraction",
    )
    legacy_route_summary_omitted = (
        event_schema == EPISODE_SCHEMA_V1
        and not any(name in summary for name in route_summary_fields)
    )
    if learned_policy and not legacy_route_summary_omitted:
        expected_routes = {
            "model_decision_count": int(route_counts["model"]),
            "official_decision_count": int(route_counts["official_adaptive"]),
            "route_switch_count": route_switch_count,
        }
        for name, value in expected_routes.items():
            if int(summary.get(name, -1)) != value:
                raise ClosedLoopTraceError(f"summary {name} mismatch")
        total_routes = sum(expected_routes[name] for name in (
            "model_decision_count", "official_decision_count"
        ))
        expected_fraction = (
            expected_routes["model_decision_count"] / total_routes
            if total_routes
            else 0.0
        )
        if not math.isclose(
            float(summary.get("model_route_fraction", -1.0)), expected_fraction
        ):
            raise ClosedLoopTraceError("summary model_route_fraction mismatch")
    raw_auc = sum(
        (conflicts[index] + conflicts[index + 1]) / 2.0
        for index in range(len(conflicts) - 1)
    )
    if not math.isclose(float(summary.get("conflict_auc", -1.0)), raw_auc):
        raise ClosedLoopTraceError("summary conflict AUC mismatch")
    if metric_iteration_budget is not None:
        expected_fixed_auc = fixed_budget_conflict_auc(
            conflicts,
            metric_iteration_budget,
            success=bool(summary.get("success")),
        )
        if not math.isclose(
            float(summary.get("fixed_budget_conflict_auc", -1.0)), expected_fixed_auc
        ):
            raise ClosedLoopTraceError("summary fixed-budget conflict AUC mismatch")
        normalized_fixed = summary.get("normalized_fixed_budget_conflict_auc")
        if normalized_fixed is not None and conflicts[0] > 0:
            expected_normalized_fixed = expected_fixed_auc / (
                float(conflicts[0]) * metric_iteration_budget
            )
            if not math.isclose(
                float(normalized_fixed),
                expected_normalized_fixed,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ClosedLoopTraceError(
                    "summary normalized fixed-budget conflict AUC mismatch"
                )
    wall_budget = summary.get("wall_time_budget_seconds")
    if wall_budget is not None and summary.get("wall_clock_conflict_auc") is not None:
        expected_wall_auc = wall_clock_conflict_auc(
            conflicts, transition_elapsed_seconds, float(wall_budget)
        )
        if not math.isclose(
            float(summary["wall_clock_conflict_auc"]),
            expected_wall_auc,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ClosedLoopTraceError("summary wall-clock conflict AUC mismatch")
    if bool(finish.get("success")) != bool(summary.get("success")):
        raise ClosedLoopTraceError("finish success flag mismatch")
    return {
        "events": rows,
        "summary": summary,
        "trace_format": trace_format,
        "initial_state_ref": initial_state_ref,
        "event_count": len(rows),
    }


def _valid_episode_trace(
    path: Path,
    run_fingerprint: str,
    *,
    expected_episode_id: str,
    expected_policy: str,
    expected_solver_seed: int,
    metric_iteration_budget: int | None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        validated = validate_closed_loop_trace(
            path,
            run_fingerprint,
            expected_episode_id=expected_episode_id,
            expected_policy=expected_policy,
            expected_solver_seed=expected_solver_seed,
            metric_iteration_budget=metric_iteration_budget,
        )
    except (ClosedLoopTraceError, KeyError, TypeError, ValueError):
        return None
    return validated

__all__ = [
    "ClosedLoopTraceError",
    "REPAIR_TIMING_SCHEMA",
    "REPAIR_TIMING_SCHEMA_V1",
    "REPAIR_TIMING_SCHEMA_V2",
    "REPAIR_TIMING_SCHEMAS",
    "_valid_episode_trace",
    "native_repair_timing_schema",
    "validate_closed_loop_trace",
]
