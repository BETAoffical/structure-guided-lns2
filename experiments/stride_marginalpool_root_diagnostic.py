from __future__ import annotations

import gzip
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import contained_file, registered_input, sha256_file
from experiments.closed_loop_trace_storage import (
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    resolve_state_blob,
    write_state_blob,
)
from experiments.feature_schema_v2 import (
    FEATURE_SCHEMA_ID,
    PROFILE_FEATURE_NAMES,
    canonicalize_features,
)
from experiments.neighborhood_features import _feature_profiles_from_shared
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import (
    StaticGridAnalysis,
    analyze_state,
    analyze_static_grid,
)
from experiments.stride_productivityguard_trigger_audit import (
    load_registration as load_productivityguard_registration,
)
from experiments.stride_tailswitch_sequence_forensics import (
    _jaccard,
    load_registration as load_sequence_registration,
    original_pool_anchor,
)


CONFIG_SCHEMA = "lns2.stride.marginalpool_root_diagnostic_registration.v1"
REPORT_SCHEMA = "lns2.stride.marginalpool_root_diagnostic_report.v1"
STATUS_SCHEMA = "lns2.stride.marginalpool_root_diagnostic_status.v1"
CHECKPOINT_SCHEMA = "lns2.stride.marginalpool_root_checkpoint.v1"
CASE_SCHEMA = "lns2.stride.marginalpool_root_case.v1"
EXPERIMENT_ID = "stride-marginalpool-root-diagnostic-v1"
SCIENTIFIC_STATUS = "preregistered_existing_trajectory_root_mechanism_diagnostic"
PARENT_COMMIT = "4ca81c53b547b62b0ac9cc1c49db07e7b8fce2e6"
FEATURE_PROFILE = "realized_dynamic"


def _mean(values: Iterable[float | int]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def _median(values: Iterable[float | int]) -> float:
    rows = [float(value) for value in values]
    return float(statistics.median(rows)) if rows else 0.0


def _structural(candidate: dict[str, Any]) -> bool:
    return any(
        str(family).startswith("structpool-")
        for family in candidate.get("selection_families") or ()
    )


def _candidate_score(candidate: dict[str, Any]) -> float:
    value = candidate.get("score")
    return float(value) if value is not None else 0.0


def _candidate_rank_key(candidate: dict[str, Any]) -> tuple[float, str]:
    return (
        -round(_candidate_score(candidate), 12),
        str(candidate["candidate_id"]),
    )


def candidate_pool_diagnostic(
    candidate_pool: list[dict[str, Any]],
    selected_candidate_id: str,
    *,
    maximum_diverse_jaccard: float = 0.8,
) -> dict[str, Any]:
    pool = [dict(candidate) for candidate in candidate_pool]
    matches = [
        candidate
        for candidate in pool
        if str(candidate.get("candidate_id")) == str(selected_candidate_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"selected candidate coverage changed: {selected_candidate_id}")
    selected = matches[0]
    selected_agents = set(map(int, selected.get("agents") or ()))
    ranked = sorted(pool, key=_candidate_rank_key)
    selected_rank = next(
        index + 1
        for index, candidate in enumerate(ranked)
        if str(candidate["candidate_id"]) == str(selected_candidate_id)
    )
    alternatives = []
    for candidate in pool:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id == str(selected_candidate_id):
            continue
        agents = set(map(int, candidate.get("agents") or ()))
        jaccard = _jaccard(selected_agents, agents)
        alternatives.append(
            {
                "candidate_id": candidate_id,
                "structural": _structural(candidate),
                "score": _candidate_score(candidate),
                "actual_size": len(agents),
                "selected_jaccard": jaccard,
                "diverse": jaccard <= maximum_diverse_jaccard,
            }
        )
    diverse = [row for row in alternatives if row["diverse"]]
    diverse_structural = [
        row for row in alternatives if row["diverse"] and row["structural"]
    ]
    anchor = original_pool_anchor(pool)
    return {
        "candidate_count": len(pool),
        "base_candidate_count": sum(not _structural(row) for row in pool),
        "structural_candidate_count": sum(_structural(row) for row in pool),
        "unique_agent_set_count": len(
            {tuple(map(int, row.get("agents") or ())) for row in pool}
        ),
        "selected_candidate_id": str(selected_candidate_id),
        "selected_structural": _structural(selected),
        "selected_rank": selected_rank,
        "selected_score": _candidate_score(selected),
        "score_margin_over_second": (
            _candidate_score(ranked[0])
            - _candidate_score(ranked[1])
            if len(ranked) > 1 and selected_rank == 1
            else 0.0
        ),
        "diverse_alternative_count": len(diverse),
        "diverse_structural_alternative_count": len(diverse_structural),
        "generator_collapse": len(diverse_structural) == 0,
        "ranker_lock": bool(diverse) and selected_rank == 1,
        "original_pool_anchor_candidate_id": (
            str(anchor["candidate_id"]) if anchor is not None else None
        ),
        "selected_vs_original_anchor_jaccard": (
            _jaccard(
                selected_agents,
                set(map(int, anchor.get("agents") or ())),
            )
            if anchor is not None
            else None
        ),
        "selected_minus_original_anchor_score": (
            _candidate_score(selected) - _candidate_score(anchor)
            if anchor is not None
            else None
        ),
        "alternatives": sorted(
            alternatives,
            key=lambda row: (-float(row["score"]), str(row["candidate_id"])),
        ),
    }


def time_aligned_path_change_ratio(left: list[int], right: list[int]) -> float:
    if not left or not right:
        raise ValueError("path comparison requires two non-empty paths")
    horizon = max(len(left), len(right))
    mismatches = 0
    for time_index in range(horizon):
        left_cell = int(left[min(time_index, len(left) - 1)])
        right_cell = int(right[min(time_index, len(right) - 1)])
        mismatches += left_cell != right_cell
    return mismatches / horizon


def _wait_count(path: list[int]) -> int:
    return sum(left == right for left, right in zip(path, path[1:]))


def _agent_rows(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {int(agent["id"]): dict(agent) for agent in state["agents"]}
    if len(rows) != len(state["agents"]):
        raise ValueError("state contains duplicate agent ids")
    return rows


def bottleneck_order_change_fraction(
    before: dict[str, Any],
    after: dict[str, Any],
    agents: set[int],
    bottleneck_cells: set[int],
) -> float:
    before_rows = _agent_rows(before)
    after_rows = _agent_rows(after)

    def arrivals(rows: dict[int, dict[str, Any]]) -> dict[int, dict[int, int]]:
        result: dict[int, dict[int, int]] = {}
        for agent_id in sorted(agents):
            path = list(map(int, rows[agent_id]["path"]))
            for time_index, cell in enumerate(path):
                if cell in bottleneck_cells:
                    result.setdefault(cell, {}).setdefault(agent_id, time_index)
        return result

    left = arrivals(before_rows)
    right = arrivals(after_rows)
    comparisons = 0
    reversals = 0
    for cell in sorted(set(left) & set(right)):
        common = sorted(set(left[cell]) & set(right[cell]))
        for first_index, first in enumerate(common):
            for second in common[first_index + 1 :]:
                left_delta = left[cell][first] - left[cell][second]
                right_delta = right[cell][first] - right[cell][second]
                if left_delta == 0 or right_delta == 0:
                    continue
                comparisons += 1
                reversals += (left_delta < 0) != (right_delta < 0)
    return reversals / comparisons if comparisons else 0.0


def path_response_diagnostic(
    before: dict[str, Any],
    after: dict[str, Any],
    selected_agents: set[int],
    bottleneck_cells: set[int],
) -> dict[str, Any]:
    left = _agent_rows(before)
    right = _agent_rows(after)
    if set(left) != set(right):
        raise ValueError("agent identity changed across a repair")
    changed_agents = []
    selected_changed_agents = []
    aligned_change = []
    wait_absolute_delta = 0
    path_cost_absolute_delta = 0.0
    delay_absolute_delta = 0.0
    for agent_id in sorted(left):
        before_path = list(map(int, left[agent_id]["path"]))
        after_path = list(map(int, right[agent_id]["path"]))
        ratio = time_aligned_path_change_ratio(before_path, after_path)
        if ratio > 0.0:
            changed_agents.append(agent_id)
            if agent_id in selected_agents:
                selected_changed_agents.append(agent_id)
        if agent_id in selected_agents:
            aligned_change.append(ratio)
            wait_absolute_delta += abs(
                _wait_count(after_path) - _wait_count(before_path)
            )
            path_cost_absolute_delta += abs(
                float(right[agent_id].get("path_cost", len(after_path) - 1))
                - float(left[agent_id].get("path_cost", len(before_path) - 1))
            )
            delay_absolute_delta += abs(
                float(right[agent_id].get("delay", 0.0))
                - float(left[agent_id].get("delay", 0.0))
            )
    order_change = bottleneck_order_change_fraction(
        before, after, selected_agents, bottleneck_cells
    )
    descriptor_changed = bool(
        changed_agents
        or wait_absolute_delta
        or path_cost_absolute_delta
        or delay_absolute_delta
        or order_change
    )
    return {
        "changed_agent_count": len(changed_agents),
        "changed_agents": changed_agents,
        "selected_changed_agent_count": len(selected_changed_agents),
        "selected_changed_agents": selected_changed_agents,
        "selected_changed_agent_fraction": (
            len(selected_changed_agents) / len(selected_agents)
            if selected_agents
            else 0.0
        ),
        "selected_time_aligned_path_change_mean": _mean(aligned_change),
        "selected_time_aligned_path_change_max": max(aligned_change, default=0.0),
        "selected_wait_count_absolute_delta": wait_absolute_delta,
        "selected_path_cost_absolute_delta": path_cost_absolute_delta,
        "selected_delay_absolute_delta": delay_absolute_delta,
        "bottleneck_order_change_fraction": order_change,
        "path_descriptor_changed": descriptor_changed,
    }


def _state_path_summary(
    state: dict[str, Any],
    selected_agents: set[int],
    analysis: Any,
    bottleneck_cells: set[int],
) -> dict[str, Any]:
    rows = _agent_rows(state)
    selected = [rows[agent_id] for agent_id in sorted(selected_agents)]
    paths = [list(map(int, row["path"])) for row in selected]
    event_times = [int(event.time) for event in analysis.events]
    first_arrivals = 0
    active_bottlenecks = set()
    for path in paths:
        visited = set()
        for cell in path:
            if cell in bottleneck_cells and cell not in visited:
                visited.add(cell)
                active_bottlenecks.add(cell)
                first_arrivals += 1
    return {
        "selected_agent_count": len(selected),
        "selected_path_cost_sum": sum(
            float(row.get("path_cost", len(path) - 1))
            for row, path in zip(selected, paths)
        ),
        "selected_delay_sum": sum(float(row.get("delay", 0.0)) for row in selected),
        "selected_wait_count_sum": sum(_wait_count(path) for path in paths),
        "conflict_event_count": len(event_times),
        "conflict_time_mean": _mean(event_times),
        "conflict_time_median": _median(event_times),
        "conflict_time_max": max(event_times, default=0),
        "selected_bottleneck_first_arrival_count": first_arrivals,
        "selected_active_bottleneck_cell_count": len(active_bottlenecks),
    }


def closure_diagnostic(
    repeated_start_state: dict[str, Any],
    checkpoint_state: dict[str, Any],
    selected_agents: set[int],
    analysis: Any,
) -> dict[str, Any]:
    start_edges = {
        tuple(sorted(map(int, edge)))
        for edge in repeated_start_state.get("conflict_edges") or ()
    }
    current_edges = {
        tuple(sorted(map(int, edge)))
        for edge in checkpoint_state.get("conflict_edges") or ()
    }
    persistent = start_edges & current_edges
    full = {edge for edge in persistent if set(edge) <= selected_agents}
    touched = {edge for edge in persistent if set(edge) & selected_agents}
    boundary = {
        edge
        for edge in persistent
        if len(set(edge) & selected_agents) == 1
    }
    deficit = sorted(
        {agent for edge in persistent for agent in edge} - selected_agents
    )
    touched_components = []
    for component_id, members in sorted(analysis.component_members.items()):
        overlap = members & selected_agents
        if not overlap:
            continue
        missing = members - selected_agents
        touched_components.append(
            {
                "component_id": int(component_id),
                "component_size": len(members),
                "selected_count": len(overlap),
                "missing_count": len(missing),
                "missing_agents": sorted(missing),
                "truncated": bool(missing),
            }
        )
    return {
        "persistent_edge_count": len(persistent),
        "persistent_edges": [list(edge) for edge in sorted(persistent)],
        "persistent_touch_coverage": len(touched) / len(persistent) if persistent else 0.0,
        "persistent_full_coverage": len(full) / len(persistent) if persistent else 0.0,
        "persistent_boundary_edge_count": len(boundary),
        "persistent_boundary_edges": [list(edge) for edge in sorted(boundary)],
        "closure_deficit_count": len(deficit),
        "closure_deficit_agents": deficit,
        "touched_component_count": len(touched_components),
        "truncated_component_count": sum(
            bool(row["truncated"]) for row in touched_components
        ),
        "maximum_component_missing_count": max(
            (int(row["missing_count"]) for row in touched_components),
            default=0,
        ),
        "components": touched_components,
    }


def _feature_payload(
    state: dict[str, Any],
    candidate: dict[str, Any],
    analysis: Any,
) -> dict[str, Any]:
    profiles = _feature_profiles_from_shared(state, analysis, candidate)
    features = canonicalize_features(profiles[FEATURE_PROFILE], FEATURE_PROFILE)
    names = PROFILE_FEATURE_NAMES[FEATURE_PROFILE]
    if len(features) != 124 or tuple(features) != tuple(names):
        raise ValueError("MarginalPool diagnostic feature schema changed")
    digest = hashlib.sha256(
        json.dumps(
            features,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "feature_schema": FEATURE_SCHEMA_ID,
        "feature_profile": FEATURE_PROFILE,
        "feature_count": len(features),
        "feature_sha256": digest,
        "features": features,
    }


def _feature_drift(
    left: dict[str, float], right: dict[str, float]
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("feature drift inputs have different schemas")
    deltas = {name: float(right[name]) - float(left[name]) for name in left}
    changed = {name: value for name, value in deltas.items() if value != 0.0}
    return {
        "changed_feature_count": len(changed),
        "absolute_l1": sum(abs(value) for value in deltas.values()),
        "maximum_absolute_delta": max(map(abs, deltas.values()), default=0.0),
        "state_absolute_l1": sum(
            abs(value) for name, value in deltas.items() if name.startswith("state.")
        ),
        "proposal_absolute_l1": sum(
            abs(value)
            for name, value in deltas.items()
            if name.startswith("proposal.")
        ),
        "realized_absolute_l1": sum(
            abs(value)
            for name, value in deltas.items()
            if name.startswith("realized.")
        ),
        "changed_features": changed,
    }


def _trace_prefix(
    trace_path: Path,
    collection: Path,
    *,
    stop_decision: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt", encoding="utf-8") as stream:
        initial = json.loads(next(stream))
        if initial.get("event") != "initial":
            raise ValueError("TailSwitch trace no longer starts with an initial event")
        state = read_state_blob(
            resolve_state_blob(trace_path, str(initial["state_blob"]), collection)
        )
        state.update(dict(initial.get("state_extras") or {}))
        transitions: list[dict[str, Any]] = []
        for line in stream:
            event = json.loads(line)
            if event.get("event") != "transition":
                raise ValueError("TailSwitch trace ended before registered checkpoint")
            decision = int(event["decision_index"])
            if state_fingerprint(state) != str(event.get("before_fingerprint")):
                raise ValueError(f"before fingerprint changed at decision {decision}")
            action = dict(event.get("action") or {})
            controller = dict(event.get("controller") or {})
            pool = [dict(candidate) for candidate in controller.get("candidate_pool") or ()]
            selected_id = str(controller.get("selected_candidate_id") or "")
            diagnostic = candidate_pool_diagnostic(pool, selected_id)
            selected = next(
                candidate
                for candidate in pool
                if str(candidate["candidate_id"]) == selected_id
            )
            if list(map(int, selected.get("agents") or ())) != list(
                map(int, action.get("agents") or ())
            ):
                raise ValueError(f"selected action changed at decision {decision}")
            transition = {
                "decision_index": decision,
                "before_state": state,
                "before_fingerprint": str(event["before_fingerprint"]),
                "action": action,
                "controller": controller,
                "candidate_pool": pool,
                "selected_candidate": selected,
                "selected_structural": _structural(selected),
                "candidate_pool_diagnostic": diagnostic,
            }
            transitions.append(transition)
            if decision == stop_decision:
                return initial, transitions
            if decision > stop_decision:
                raise ValueError("registered checkpoint decision was skipped")
            after = apply_state_delta(state, dict(event["state_delta"]))
            after.update(apply_extras_delta(state, dict(event["state_extras_delta"])))
            if state_fingerprint(after) != str(event.get("after_fingerprint")):
                raise ValueError(f"after fingerprint changed at decision {decision}")
            transition["after_state"] = after
            transition["after_fingerprint"] = str(event["after_fingerprint"])
            metrics = dict(event.get("metrics") or {})
            transition["prior_repair_metadata"] = {
                "conflicts_before": int(metrics.get("conflicts_before", -1)),
                "conflicts_after": int(metrics.get("conflicts_after", -1)),
                "replan_success": bool(metrics.get("replan_success", False)),
                "step_applied": bool(metrics.get("step_applied", False)),
                "applied_pp_random_seed": int(
                    metrics.get("applied_pp_random_seed", action.get("pp_random_seed", -1))
                ),
            }
            state = after
    raise ValueError("registered checkpoint was not found in TailSwitch trace")


def _checkpoint_record(
    *,
    output: Path,
    case_id: str,
    checkpoint_kind: str,
    transition: dict[str, Any],
    static_grid: StaticGridAnalysis,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    state = dict(transition["before_state"])
    selected = dict(transition["selected_candidate"])
    selected_agents = set(map(int, selected["agents"]))
    analysis = analyze_state(state, static_grid=static_grid)
    bottlenecks = set(static_grid.articulation) | {
        cell for cell, degree in static_grid.degrees.items() if degree <= 2
    }
    blob_ref, blob_path = write_state_blob(output, state)
    feature = _feature_payload(state, selected, analysis)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "case_id": case_id,
        "checkpoint_kind": checkpoint_kind,
        **metadata,
        "decision_index": int(transition["decision_index"]),
        "state_fingerprint": state_fingerprint(state),
        "state_blob": blob_ref,
        "state_blob_sha256": sha256_file(blob_path),
        "state_context": dict(state.get("context") or {}),
        "conflict_pair_count": len(analysis.pair_set),
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_agents": sorted(selected_agents),
        "selected_families": list(map(str, selected.get("selection_families") or ())),
        "selected_feature": feature,
        "candidate_pool_diagnostic": dict(transition["candidate_pool_diagnostic"]),
        "candidate_pool": [dict(row) for row in transition["candidate_pool"]],
        "path_summary": _state_path_summary(
            state, selected_agents, analysis, bottlenecks
        ),
        "current_action_outcome_used": False,
    }


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status") != SCIENTIFIC_STATUS
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit") != PARENT_COMMIT
    ):
        raise ValueError("MarginalPool root-diagnostic registration changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    expected_boundary = {
        "existing_trajectory_diagnostic_only": True,
        "model_training_allowed": False,
        "solver_modification_allowed": False,
        "new_solver_runs_allowed": False,
        "candidate_quality_claim": False,
        "ttf_improvement_claim": False,
        "generalization_claim": False,
        "causal_claim": False,
        "default_controller_replacement_allowed": False,
        "no_result_based_exclusion": True,
    }
    if dict(config.get("claim_boundary") or {}) != expected_boundary:
        raise ValueError("MarginalPool root-diagnostic claim boundary changed")
    return path, root, config, inputs


def analyze_marginalpool_root_diagnostic(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, root, config, inputs = load_registration(config_path)
    productivity_status = _read_json(inputs["productivityguard_status"])
    productivity_report = _read_json(inputs["productivityguard_report"])
    sequence_status = _read_json(inputs["sequence_status"])
    sequence_report = _read_json(inputs["sequence_report"])
    if (
        productivity_status.get("complete") is not True
        or productivity_status.get("integrity_passed") is not True
        or productivity_status.get("trigger_readiness_passed") is not False
        or productivity_report.get("integrity_passed") is not True
        or sequence_status.get("complete") is not True
        or sequence_status.get("integrity_passed") is not True
        or sequence_report.get("integrity_passed") is not True
    ):
        raise ValueError("completed upstream diagnostic contract changed")
    load_productivityguard_registration(inputs["productivityguard_registration"])
    _, _, _, sequence_inputs = load_sequence_registration(
        inputs["sequence_registration"]
    )
    tailswitch_report = _read_json(sequence_inputs["tailswitch_report"])
    states_by_id = {
        str(state["state_id"]): dict(state)
        for state in tailswitch_report.get("states") or ()
    }
    variant = next(
        (
            dict(row)
            for row in productivity_report.get("variant_results") or ()
            if str(dict(row.get("variant") or {}).get("id"))
            == str(config["cohort"]["source_variant"])
        ),
        None,
    )
    if variant is None:
        raise ValueError("registered ProductivityGuard source variant is missing")
    selected_pairs = [
        dict(row)
        for row in variant.get("pair_results") or ()
        if row.get("early_triggered") is True
    ]
    expected_counts = dict(
        config["cohort"]["classification_counts_for_descriptive_stratification_only"]
    )
    observed_counts = {
        label: sum(str(row["classification"]) == label for row in selected_pairs)
        for label in ("adverse", "beneficial", "neutral")
    }
    if (
        len(selected_pairs) != int(config["cohort"]["triggered_contrast_count"])
        or observed_counts != expected_counts
    ):
        raise ValueError("registered MarginalPool cohort changed")

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tailswitch_root = sequence_inputs["tailswitch_report"].parent
    expected_trace_hashes = dict(sequence_report["inputs"]["trace_sha256"])
    expected_manifest_hashes = dict(sequence_report["inputs"]["manifest_sha256"])
    case_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    trace_hashes: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}

    for pair in sorted(
        selected_pairs,
        key=lambda row: (str(row["state_id"]), str(row["contrast"])),
    ):
        state_id = str(pair["state_id"])
        contrast = str(pair["contrast"])
        policy = str(pair["treatment_policy"])
        trigger_decision = int(pair["first_early_trigger_decision"])
        state_metadata = states_by_id.get(state_id)
        if state_metadata is None:
            raise ValueError(f"TailSwitch state disappeared: {state_id}")
        state_dir = tailswitch_root / "states" / _fingerprint(
            {"state_id": state_id}
        )[:20]
        collection = state_dir / policy
        manifest_path = collection / "realized_dynamic_manifest.jsonl"
        manifest_rows = _read_jsonl(manifest_path)
        if len(manifest_rows) != 1 or manifest_rows[0].get("status") != "ok":
            raise ValueError(f"TailSwitch manifest changed: {state_id}/{policy}")
        manifest = dict(manifest_rows[0])
        trace_path = contained_file(
            collection, manifest["trace_file"], field="TailSwitch trace_file"
        )
        identity_key = f"{state_id}/{policy}"
        manifest_hashes[identity_key] = sha256_file(manifest_path)
        trace_hashes[identity_key] = sha256_file(trace_path)
        if (
            manifest_hashes[identity_key] != expected_manifest_hashes.get(identity_key)
            or trace_hashes[identity_key] != expected_trace_hashes.get(identity_key)
        ):
            raise ValueError(f"TailSwitch input identity changed: {identity_key}")

        _, transitions = _trace_prefix(
            trace_path, collection, stop_decision=trigger_decision
        )
        first_structural = next(
            (
                row
                for row in transitions
                if int(row["decision_index"]) >= 1
                and row["selected_structural"] is True
            ),
            None,
        )
        trigger = next(
            row
            for row in transitions
            if int(row["decision_index"]) == trigger_decision
        )
        if first_structural is None:
            raise ValueError(f"first structural checkpoint is missing: {identity_key}")
        trigger_position = transitions.index(trigger)
        if trigger_position < 2:
            raise ValueError(f"repeat/stall checkpoint has insufficient history: {identity_key}")
        repeat_run = transitions[trigger_position - 2 : trigger_position + 1]
        repeated_ids = {
            str(row["selected_candidate"]["candidate_id"]) for row in repeat_run
        }
        if len(repeated_ids) != 1 or any(
            row["selected_structural"] is not True for row in repeat_run
        ):
            raise ValueError(f"registered exact triplet changed: {identity_key}")
        prior = repeat_run[:2]
        if any("after_state" not in row for row in prior):
            raise ValueError(f"prior repair state is missing: {identity_key}")
        if any(
            int(row["prior_repair_metadata"]["conflicts_before"])
            != int(row["prior_repair_metadata"]["conflicts_after"])
            for row in prior
        ):
            raise ValueError(f"registered zero-progress triplet changed: {identity_key}")

        static_grid = analyze_static_grid(first_structural["before_state"])
        metadata = {
            "state_id": state_id,
            "contrast": contrast,
            "classification": str(pair["classification"]),
            "map_id": str(pair["map_id"]),
            "task_id": str(pair["task_id"]),
            "solver_seed": int(pair["solver_seed"]),
            "challenger": str(pair["challenger"]),
            "treatment_policy": policy,
            "trace_identity_key": identity_key,
            "trace_sha256": trace_hashes[identity_key],
        }
        case_id = _fingerprint(
            {"state_id": state_id, "contrast": contrast, "policy": policy}
        )
        first_checkpoint = _checkpoint_record(
            output=output,
            case_id=case_id,
            checkpoint_kind="first_structural_selection",
            transition=first_structural,
            static_grid=static_grid,
            metadata=metadata,
        )
        trigger_checkpoint = _checkpoint_record(
            output=output,
            case_id=case_id,
            checkpoint_kind="first_repeat_stall",
            transition=trigger,
            static_grid=static_grid,
            metadata=metadata,
        )
        checkpoint_rows.extend((first_checkpoint, trigger_checkpoint))

        selected_agents = set(map(int, trigger["selected_candidate"]["agents"]))
        bottlenecks = set(static_grid.articulation) | {
            cell for cell, degree in static_grid.degrees.items() if degree <= 2
        }
        prior_responses = [
            {
                "decision_index": int(row["decision_index"]),
                "candidate_id": str(row["selected_candidate"]["candidate_id"]),
                "repair_metadata": dict(row["prior_repair_metadata"]),
                **path_response_diagnostic(
                    row["before_state"],
                    row["after_state"],
                    selected_agents,
                    bottlenecks,
                ),
            }
            for row in prior
        ]
        latent = any(row["path_descriptor_changed"] for row in prior_responses)
        trigger_analysis = analyze_state(
            trigger["before_state"], static_grid=static_grid
        )
        closure = closure_diagnostic(
            repeat_run[0]["before_state"],
            trigger["before_state"],
            selected_agents,
            trigger_analysis,
        )
        selected_feature_drift = _feature_drift(
            dict(first_checkpoint["selected_feature"]["features"]),
            dict(trigger_checkpoint["selected_feature"]["features"]),
        )
        repeated_candidate_id = str(trigger["selected_candidate"]["candidate_id"])
        first_pool_match = next(
            (
                dict(candidate)
                for candidate in first_structural["candidate_pool"]
                if str(candidate["candidate_id"]) == repeated_candidate_id
            ),
            None,
        )
        same_candidate_drift = None
        if first_pool_match is not None:
            first_analysis = analyze_state(
                first_structural["before_state"], static_grid=static_grid
            )
            first_same = _feature_payload(
                first_structural["before_state"], first_pool_match, first_analysis
            )
            same_candidate_drift = _feature_drift(
                dict(first_same["features"]),
                dict(trigger_checkpoint["selected_feature"]["features"]),
            )
        pool_diagnostic = dict(trigger["candidate_pool_diagnostic"])
        case_rows.append(
            {
                "schema": CASE_SCHEMA,
                "case_id": case_id,
                **metadata,
                "first_structural_decision": int(first_structural["decision_index"]),
                "first_repeat_stall_decision": trigger_decision,
                "repeated_candidate_id": repeated_candidate_id,
                "selection_mechanism": {
                    "generator_collapse": bool(pool_diagnostic["generator_collapse"]),
                    "ranker_lock": bool(pool_diagnostic["ranker_lock"]),
                    "diverse_alternative_count": int(
                        pool_diagnostic["diverse_alternative_count"]
                    ),
                    "diverse_structural_alternative_count": int(
                        pool_diagnostic["diverse_structural_alternative_count"]
                    ),
                    "selected_rank": int(pool_diagnostic["selected_rank"]),
                },
                "pp_response": {
                    "latent_path_change": latent,
                    "true_pp_noop": not latent,
                    "prior_repairs": prior_responses,
                },
                "closure": closure,
                "selected_checkpoint_feature_drift": selected_feature_drift,
                "same_candidate_state_feature_drift": same_candidate_drift,
                "current_third_action_outcome_used": False,
                "future_transition_used": False,
            }
        )

    checkpoint_path = output / "root_diagnostic_checkpoints.jsonl"
    case_path = output / "root_diagnostic_cases.jsonl"
    _write_jsonl(checkpoint_path, checkpoint_rows)
    _write_jsonl(case_path, case_rows)
    classification_counts = {
        label: sum(str(row["classification"]) == label for row in case_rows)
        for label in ("adverse", "beneficial", "neutral")
    }
    integrity = {
        "triggered_contrast_count": len(case_rows) == 45,
        "unique_case_count": len({row["case_id"] for row in case_rows}) == 45,
        "checkpoint_count": len(checkpoint_rows) == 90,
        "classification_counts": classification_counts
        == {"adverse": 17, "beneficial": 8, "neutral": 20},
        "feature_count": all(
            int(row["selected_feature"]["feature_count"]) == 124
            for row in checkpoint_rows
        ),
        "trace_hashes": len(trace_hashes) == 45,
        "manifest_hashes": len(manifest_hashes) == 45,
        "state_fingerprints": all(
            str(row["state_fingerprint"])
            == Path(str(row["state_blob"])).stem.split(".")[0]
            for row in checkpoint_rows
        ),
        "selected_candidate_coverage": all(
            sum(
                str(candidate["candidate_id"]) == str(row["selected_candidate_id"])
                for candidate in row["candidate_pool"]
            )
            == 1
            for row in checkpoint_rows
        ),
        "current_action_outcome_read_count": sum(
            bool(row["current_action_outcome_used"]) for row in checkpoint_rows
        )
        == 0,
        "future_transition_use_count": sum(
            bool(row["future_transition_used"]) for row in case_rows
        )
        == 0,
        "new_solver_run_count": True,
    }
    integrity_passed = all(integrity.values())

    def grouped(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "case_count": len(rows),
            "generator_collapse_count": sum(
                bool(row["selection_mechanism"]["generator_collapse"])
                for row in rows
            ),
            "ranker_lock_count": sum(
                bool(row["selection_mechanism"]["ranker_lock"])
                for row in rows
            ),
            "latent_path_change_count": sum(
                bool(row["pp_response"]["latent_path_change"]) for row in rows
            ),
            "true_pp_noop_count": sum(
                bool(row["pp_response"]["true_pp_noop"]) for row in rows
            ),
            "mean_persistent_full_coverage": _mean(
                row["closure"]["persistent_full_coverage"] for row in rows
            ),
            "mean_closure_deficit_count": _mean(
                row["closure"]["closure_deficit_count"] for row in rows
            ),
            "mean_truncated_component_count": _mean(
                row["closure"]["truncated_component_count"] for row in rows
            ),
            "mean_selected_feature_changed_count": _mean(
                row["selected_checkpoint_feature_drift"]["changed_feature_count"]
                for row in rows
            ),
        }

    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity_passed": integrity_passed,
        "stage2_ready": integrity_passed,
        "case_count": len(case_rows),
        "checkpoint_count": len(checkpoint_rows),
        "classification_counts": classification_counts,
        "overall": grouped(case_rows),
        "by_classification": {
            label: grouped(
                [row for row in case_rows if str(row["classification"]) == label]
            )
            for label in ("adverse", "beneficial", "neutral")
        },
        "by_challenger": {
            challenger: grouped(
                [row for row in case_rows if str(row["challenger"]) == challenger]
            )
            for challenger in sorted({str(row["challenger"]) for row in case_rows})
        },
        "integrity_gates": integrity,
        "selection_and_pp_response_are_separate_axes": True,
        "current_action_outcome_used": False,
        "future_trajectory_used": False,
        "new_solver_run_count": 0,
        "inputs": {
            "registration_sha256": sha256_file(config_path),
            "productivityguard_registration_sha256": sha256_file(
                inputs["productivityguard_registration"]
            ),
            "productivityguard_status_sha256": sha256_file(
                inputs["productivityguard_status"]
            ),
            "productivityguard_report_sha256": sha256_file(
                inputs["productivityguard_report"]
            ),
            "sequence_registration_sha256": sha256_file(
                inputs["sequence_registration"]
            ),
            "sequence_status_sha256": sha256_file(inputs["sequence_status"]),
            "sequence_report_sha256": sha256_file(inputs["sequence_report"]),
            "manifest_sha256": manifest_hashes,
            "trace_sha256": trace_hashes,
        },
        "artifacts": {
            "checkpoint_manifest": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_path),
            "case_manifest": case_path.relative_to(root).as_posix(),
            "case_manifest_sha256": sha256_file(case_path),
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    report_path = output / "root_diagnostic_report.json"
    _write_json(report_path, report)
    _write_json(
        output / "root_diagnostic_status.json",
        {
            "schema": STATUS_SCHEMA,
            "complete": True,
            "integrity_passed": integrity_passed,
            "stage2_ready": integrity_passed,
            "case_count": len(case_rows),
            "checkpoint_count": len(checkpoint_rows),
            "report_sha256": sha256_file(report_path),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_path),
            "case_manifest_sha256": sha256_file(case_path),
        },
    )
    return report


__all__ = [
    "analyze_marginalpool_root_diagnostic",
    "bottleneck_order_change_fraction",
    "candidate_pool_diagnostic",
    "closure_diagnostic",
    "load_registration",
    "path_response_diagnostic",
    "time_aligned_path_change_ratio",
]
