from __future__ import annotations

import collections
import concurrent.futures
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    contained_file,
    json_fingerprint,
    producer_identity,
    registered_input,
    sha256_file,
)
from experiments.closed_loop_trace_storage import read_trace_events
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.run_output_guard import prepare_run_output
from experiments.state_analysis import (
    StateAnalysis,
    StaticGridAnalysis,
    analyze_state,
    analyze_static_grid,
)
from experiments.stride_safeslot_gate import (
    FEATURE_NAMES as STATIC_FEATURE_NAMES,
    _fit_model,
    _map_folds,
    _parameters,
    build_safeslot_gate_rows,
    candidate_classification_metrics,
    validate_safeslot_gate_config,
)
from experiments.stride_temporal_anchor_incremental_readiness import (
    _candidate_agent_sets,
    _manifest_input_name,
    _reproduction_error,
    _static_oof,
    _unique,
)
from experiments.trace_replay import _initial_state, state_before_decision


CONFIG_SCHEMA = "lns2.stride.safeslot_pair_motif_sequential_ablation_config.v1"
ROW_SCHEMA = "lns2.stride.safeslot_pair_motif_sequential_ablation_row.v1"
PREDICTION_SCHEMA = "lns2.stride.safeslot_pair_motif_sequential_ablation_prediction.v1"
REPORT_SCHEMA = "lns2.stride.safeslot_pair_motif_sequential_ablation_report.v1"
EXPERIMENT_ID = "stride-safeslot-pair-motif-sequential-ablation-v1"
MOTIF_FEATURE_NAMES = (
    "ca.delta_incident_event_per_agent",
    "ca.delta_partner_count_per_agent",
    "ca.delta_retained_cross_event_coverage",
    "ca.delta_outsider_cross_event_coverage",
    "ca.delta_repeated_pair_excess_coverage",
    "ca.delta_vertex_event_coverage",
    "ca.delta_edge_event_coverage",
    "ca.delta_early_event_coverage",
    "ca.delta_late_event_coverage",
    "ca.delta_event_time_centroid",
    "ca.delta_event_time_spread",
    "ca.delta_articulation_event_coverage",
    "ca.delta_low_degree_event_coverage",
    "ca.delta_lag1_same_cell_pressure",
    "ca.delta_same_time_adjacent_pressure",
    "ca.delta_lag1_opposite_edge_pressure",
    "ca.internal_event_jaccard",
    "ca.path_time_occupancy_jaccard",
)
FEATURE_NAMES = (*STATIC_FEATURE_NAMES, *MOTIF_FEATURE_NAMES)
PRODUCER_FILES = (
    "experiments/stride_safeslot_pair_motif_sequential_ablation.py",
    "experiments/stride_safeslot_gate.py",
    "experiments/stride_temporal_anchor_incremental_readiness.py",
    "experiments/state_analysis.py",
    "experiments/trace_replay.py",
    "scripts/run_stride_safeslot_pair_motif_sequential_ablation.py",
)


def _registered(project_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    return {
        name: registered_input(project_root, dict(spec), label=f"{EXPERIMENT_ID}:{name}")
        for name, spec in dict(config["inputs"]).items()
    }


def validate_config(config: dict[str, Any], *, project_root: Path | None = None) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "post_history_sequential_exploratory_representation_ablation"
    ):
        raise ValueError("pair-motif ablation identity changed")
    expected_inputs = {
        "safeslot_config",
        "safeslot_oof_predictions",
        "state_selection",
        "state_selection_report",
        "source_v4_official_manifest",
        "source_v4_v2_manifest",
        "da2_v2_official_manifest",
        "da2_v2_v2_manifest",
        "prior_temporal_config",
        "prior_temporal_report",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("pair-motif input registry changed")
    if dict(config.get("source_roots") or {}) != {
        "source-v4": "build/stride-robustaction-structpool-source-episodes-v4",
        "da2-stability-v2": "build/stride-robustaction-da2-source-stability-episodes-v2",
    }:
        raise ValueError("pair-motif source roots changed")
    if dict(config.get("cohort") or {}) != {
        "state_count": 98,
        "map_count": 16,
        "candidate_count": 587,
        "positive_candidate_count": 201,
        "negative_candidate_count": 386,
    }:
        raise ValueError("pair-motif cohort changed")
    features = dict(config.get("features") or {})
    expected_forbidden = {
        "history", "label", "residual_teacher", "repair_trial", "target_action",
        "target_metrics", "target_state_delta", "target_outcome", "after_state",
        "future_transition", "pp_seed", "pp_time", "runtime", "ttf",
    }
    if (
        int(features.get("baseline_dimension", -1)) != 193
        or int(features.get("motif_dimension", -1)) != 18
        or int(features.get("augmented_dimension", -1)) != 211
        or tuple(features.get("motif_feature_names") or ()) != MOTIF_FEATURE_NAMES
        or features.get("roles")
        != "A=anchor,C=challenger,K=A_intersection_C,E=C_minus_A,D=A_minus_C,R=N_minus_A_union_C,F=N_minus_E_union_D"
        or features.get("path_contract")
        != "terminal_pad;H_raw=max_path_length_minus_1;H=max(1,H_raw);tau=t/H"
        or features.get("event_contract")
        != "Q=current_pre_action_vertex_and_opposing_edge_conflict_event_multiset;M=max(1,abs(Q));canonical_identity=(t,kind,min_agent,max_agent,sorted_cells)"
        or features.get("role_metric_contract")
        != "partners(S)=union_of_all_distinct_other_endpoints_over_incident_events;partner_metric=abs(partners(S))/max(1,abs(S));repeat_metric=sum_over_unique_incident_pairs(max(0,pair_event_count-1))/M;vertex_edge_early_late_each_divide_by_M"
        or features.get("time_windows") != "early_tau_le_one_third;late_tau_ge_two_thirds"
        or features.get("pressure_contract")
        != "F=N_minus_E_union_D;all_pair_exposures;denominator=max(1,abs(S)*(H_raw+1));lag1_same_cell_uses_t_plus_or_minus_1_and_excludes_same_time_vertex_conflict;same_time_adjacent_uses_free_grid_4_neighbors;lag1_opposite_edge_uses_transition_t_plus_or_minus_1_and_excludes_same_transition"
        or features.get("jaccard_contract")
        != "set_jaccard_without_agent_token;empty_union_zero;occupancy_tokens=(t,cell)"
        or features.get("delta_contract") != "first_16_equal_metric(E)_minus_metric(D)"
        or set(features.get("forbidden_inputs") or ()) != expected_forbidden
    ):
        raise ValueError("pair-motif feature contract changed")
    model = dict(config.get("model") or {})
    expected_folds = [
        ["ca_cave", "ca_caverns2", "dr_primevalentrance", "lak106d", "lt_hangedman"],
        ["den900d", "lak203d", "lgt604d", "lt_undercitydungeon", "orz201d"],
        ["hrt001d", "lak250d", "orz601d"],
        ["ht_bartrand_n", "lak526d", "w_encounter3"],
    ]
    if (
        model.get("class") != "sklearn.ensemble.HistGradientBoostingClassifier"
        or int(model.get("parameter_index", -1)) != 0
        or dict(model.get("parameters") or {}) != {
            "early_stopping": False,
            "learning_rate": 0.05,
            "max_iter": 100,
            "min_samples_leaf": 20,
            "random_state": 20260809,
            "max_leaf_nodes": 7,
            "l2_regularization": 0.1,
        }
        or list(model.get("outer_map_folds") or ()) != expected_folds
        or model.get("sample_weighting")
        != "same_registered_equal_total_weight_per_state"
        or int(model.get("workers", 0)) != 16
    ):
        raise ValueError("pair-motif model/fold contract changed")
    if dict(config.get("identity_reproduction") or {}) != {
        "registered_row_count": 587,
        "maximum_probability_absolute_error": 1e-12,
        "require_exact_rows_weights_folds_and_old_oof_protocol": True,
        "legacy_identity_reproduction_replays_registered_static_calibration": True,
        "legacy_calibration_not_interpreted_or_exported": True,
    }:
        raise ValueError("pair-motif identity reproduction contract changed")
    if dict(config.get("offline_acceptance") or {}) != {
        "minimum_overall_roc_auc_gain": 0.03,
        "minimum_overall_average_precision_gain": 0.02,
        "minimum_each_fold_augmented_roc_auc": 0.55,
        "minimum_each_fold_roc_auc_gain": -0.02,
        "minimum_positive_gain_fold_count": 3,
    }:
        raise ValueError("pair-motif hard gates changed")
    if dict(config.get("prior_sequential_evidence") or {}) != {
        "experiment": "stride-temporal-anchor-incremental-readiness-v1",
        "offline_passed": False,
        "baseline_roc_auc": 0.6399471379337135,
        "augmented_roc_auc": 0.6535202508356857,
        "roc_auc_gain": 0.013573112901972162,
    }:
        raise ValueError("prior sequential evidence changed")
    if dict(config.get("claim_boundary") or {}) != {
        "post_history_sequential_exploratory": True,
        "same_safeslot_labels_reused": True,
        "independent_confirmation": False,
        "promotion_allowed": False,
        "threshold_or_policy_analysis_allowed": False,
        "model_export_allowed": False,
        "solver_native_or_pp_executed": False,
        "runtime_or_ttf_claim": False,
        "pass_allows_only": "register_fresh_map_paired_label_collection",
        "fail_action": "permanently_stop_pair_motif_representation_route",
    }:
        raise ValueError("pair-motif claim boundary changed")
    if (
        int(config.get("analysis_wall_fuse_seconds", 0)) != 600
        or dict(config.get("outputs") or {})
        != {"ablation": "build/stride-safeslot-pair-motif-sequential-ablation-v1"}
    ):
        raise ValueError("pair-motif execution/output contract changed")
    if project_root is not None:
        _registered(project_root, config)


def _event_identity(event: Any) -> tuple[Any, ...]:
    return (
        int(event.time),
        str(event.kind),
        min(int(event.left), int(event.right)),
        max(int(event.left), int(event.right)),
        tuple(sorted(map(int, event.cells))),
    )


def _set_jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _position(paths: dict[int, tuple[int, ...]], agent: int, time_index: int) -> int:
    path = paths[agent]
    return path[min(max(0, int(time_index)), len(path) - 1)]


@dataclass(frozen=True)
class PairMotifCache:
    analysis: StateAnalysis
    paths: dict[int, tuple[int, ...]]
    known_agents: frozenset[int]
    horizon_raw: int
    horizon: int
    occupancy: tuple[dict[int, frozenset[int]], ...]
    transitions: tuple[dict[tuple[int, int], frozenset[int]], ...]
    incident_by_agent: dict[int, frozenset[int]]
    partners_by_agent: dict[int, frozenset[int]]
    pair_counts: dict[tuple[int, int], int]
    vertex_indices: frozenset[int]
    edge_indices: frozenset[int]
    early_indices: frozenset[int]
    late_indices: frozenset[int]
    articulation_indices: frozenset[int]
    low_degree_indices: frozenset[int]
    same_time_conflict_pairs: frozenset[tuple[int, int, int]]


def _pair_motif_cache(
    state: dict[str, Any], *, static_grid: StaticGridAnalysis | None = None
) -> PairMotifCache:
    agents = list(state.get("agents") or ())
    paths = {
        int(agent["id"]): tuple(map(int, agent.get("path") or ()))
        for agent in agents
    }
    if len(paths) != len(agents) or any(not path for path in paths.values()):
        raise ValueError("pair-motif state paths are invalid")
    analysis = analyze_state(state, static_grid=static_grid)
    reported_pairs = {
        tuple(sorted(map(int, edge))) for edge in state.get("conflict_edges", ())
    }
    if (
        analysis.pair_set != reported_pairs
        or len(reported_pairs) != int(state.get("num_of_colliding_pairs", -1))
    ):
        raise ValueError("pair-motif reconstructed conflicts changed")
    horizon_raw = max((len(path) - 1 for path in paths.values()), default=0)
    horizon = max(1, horizon_raw)
    occupancy: list[dict[int, frozenset[int]]] = []
    for time_index in range(horizon_raw + 1):
        by_cell: dict[int, set[int]] = collections.defaultdict(set)
        for agent in paths:
            by_cell[_position(paths, agent, time_index)].add(agent)
        occupancy.append({cell: frozenset(ids) for cell, ids in by_cell.items()})
    transitions: list[dict[tuple[int, int], frozenset[int]]] = []
    for time_index in range(horizon_raw):
        by_edge: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
        for agent in paths:
            start = _position(paths, agent, time_index)
            end = _position(paths, agent, time_index + 1)
            if start != end:
                by_edge[(start, end)].add(agent)
        transitions.append({edge: frozenset(ids) for edge, ids in by_edge.items()})
    incident: dict[int, set[int]] = collections.defaultdict(set)
    partners: dict[int, set[int]] = collections.defaultdict(set)
    pair_counts: collections.Counter[tuple[int, int]] = collections.Counter()
    vertex_indices: set[int] = set()
    edge_indices: set[int] = set()
    early_indices: set[int] = set()
    late_indices: set[int] = set()
    articulation_indices: set[int] = set()
    low_degree_indices: set[int] = set()
    same_time_conflict_pairs: set[tuple[int, int, int]] = set()
    for index, event in enumerate(analysis.events):
        left, right = int(event.left), int(event.right)
        pair = (min(left, right), max(left, right))
        incident[left].add(index)
        incident[right].add(index)
        partners[left].add(right)
        partners[right].add(left)
        pair_counts[pair] += 1
        if str(event.kind) == "vertex":
            vertex_indices.add(index)
            same_time_conflict_pairs.add((int(event.time), *pair))
        elif str(event.kind) == "edge":
            edge_indices.add(index)
        tau = float(event.time) / horizon
        if tau <= 1.0 / 3.0:
            early_indices.add(index)
        if tau >= 2.0 / 3.0:
            late_indices.add(index)
        if any(int(cell) in analysis.articulation for cell in event.cells):
            articulation_indices.add(index)
        if any(analysis.degrees.get(int(cell), 5) <= 2 for cell in event.cells):
            low_degree_indices.add(index)
    return PairMotifCache(
        analysis=analysis,
        paths=paths,
        known_agents=frozenset(paths),
        horizon_raw=horizon_raw,
        horizon=horizon,
        occupancy=tuple(occupancy),
        transitions=tuple(transitions),
        incident_by_agent={agent: frozenset(indices) for agent, indices in incident.items()},
        partners_by_agent={agent: frozenset(ids) for agent, ids in partners.items()},
        pair_counts=dict(pair_counts),
        vertex_indices=frozenset(vertex_indices),
        edge_indices=frozenset(edge_indices),
        early_indices=frozenset(early_indices),
        late_indices=frozenset(late_indices),
        articulation_indices=frozenset(articulation_indices),
        low_degree_indices=frozenset(low_degree_indices),
        same_time_conflict_pairs=frozenset(same_time_conflict_pairs),
    )


def _role_metrics(
    role: set[int],
    *,
    retained: set[int],
    outsider: set[int],
    pressure_reference: set[int],
    cache: PairMotifCache,
) -> list[float]:
    analysis = cache.analysis
    events = list(analysis.events)
    event_denominator = max(1, len(events))
    role_denominator = max(1, len(role))
    incident_indices: set[int] = set()
    partners: set[int] = set()
    for agent in role:
        incident_indices.update(cache.incident_by_agent.get(agent, ()))
        partners.update(cache.partners_by_agent.get(agent, ()))
    incident = [events[index] for index in sorted(incident_indices)]
    incident_pairs = {
        pair for pair in cache.pair_counts if pair[0] in role or pair[1] in role
    }
    repeated_excess = sum(max(0, cache.pair_counts[pair] - 1) for pair in incident_pairs)
    normalized_times = [float(event.time) / cache.horizon for event in incident]

    pressure_denominator = max(1, len(role) * (cache.horizon_raw + 1))
    lag_same_cell = 0
    same_time_adjacent = 0
    lag_opposite_edge = 0
    for agent in role:
        for time_index in range(cache.horizon_raw + 1):
            cell = _position(cache.paths, agent, time_index)
            same_time_conflict = cache.occupancy[time_index].get(cell, frozenset())
            for shifted in (time_index - 1, time_index + 1):
                if 0 <= shifted <= cache.horizon_raw:
                    lag_same_cell += len(
                        (
                            cache.occupancy[shifted].get(cell, frozenset())
                            & pressure_reference
                        )
                        - same_time_conflict
                    )
            row, column = divmod(cell, analysis.cols)
            for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                near_row, near_column = row + delta_row, column + delta_column
                if 0 <= near_row < analysis.rows and 0 <= near_column < analysis.cols:
                    near = near_row * analysis.cols + near_column
                    if near in analysis.free_cells:
                        same_time_adjacent += len(
                            cache.occupancy[time_index].get(near, frozenset())
                            & pressure_reference
                        )
        for transition in range(cache.horizon_raw):
            start = _position(cache.paths, agent, transition)
            end = _position(cache.paths, agent, transition + 1)
            if start == end:
                continue
            for shifted in (transition - 1, transition + 1):
                if 0 <= shifted < cache.horizon_raw:
                    lag_opposite_edge += len(
                        cache.transitions[shifted].get((end, start), frozenset())
                        & pressure_reference
                    )

    return [
        len(incident) / role_denominator,
        len(partners) / role_denominator,
        sum(
            count
            for pair, count in cache.pair_counts.items()
            if (pair[0] in role and pair[1] in retained)
            or (pair[1] in role and pair[0] in retained)
        )
        / event_denominator,
        sum(
            count
            for pair, count in cache.pair_counts.items()
            if (pair[0] in role and pair[1] in outsider)
            or (pair[1] in role and pair[0] in outsider)
        )
        / event_denominator,
        repeated_excess / event_denominator,
        len(incident_indices & cache.vertex_indices) / event_denominator,
        len(incident_indices & cache.edge_indices) / event_denominator,
        len(incident_indices & cache.early_indices) / event_denominator,
        len(incident_indices & cache.late_indices) / event_denominator,
        statistics.fmean(normalized_times) if normalized_times else 0.0,
        statistics.pstdev(normalized_times) if len(normalized_times) > 1 else 0.0,
        len(incident_indices & cache.articulation_indices)
        / max(1, len(cache.articulation_indices)),
        len(incident_indices & cache.low_degree_indices)
        / max(1, len(cache.low_degree_indices)),
        lag_same_cell / pressure_denominator,
        same_time_adjacent / pressure_denominator,
        lag_opposite_edge / pressure_denominator,
    ]


def pair_motif_feature_vector(
    state: dict[str, Any], challenger_agents: Iterable[int], anchor_agents: Iterable[int]
) -> list[float]:
    """Construct the frozen 18D pre-action candidate/anchor motif vector."""

    return _pair_motif_feature_vector_from_cache(
        _pair_motif_cache(state), challenger_agents, anchor_agents
    )


def _pair_motif_feature_vector_from_cache(
    cache: PairMotifCache,
    challenger_agents: Iterable[int],
    anchor_agents: Iterable[int],
) -> list[float]:

    challenger = set(map(int, challenger_agents))
    anchor = set(map(int, anchor_agents))
    if not challenger or not anchor:
        raise ValueError("pair-motif candidates must be non-empty")
    known = set(cache.known_agents)
    if not challenger <= known or not anchor <= known:
        raise ValueError("pair-motif candidate references an unknown agent")
    analysis = cache.analysis
    retained = challenger & anchor
    entered = challenger - anchor
    displaced = anchor - challenger
    outsider = known - (challenger | anchor)
    pressure_reference = known - (entered | displaced)
    entered_values = _role_metrics(
        entered,
        retained=retained,
        outsider=outsider,
        pressure_reference=pressure_reference,
        cache=cache,
    )
    displaced_values = _role_metrics(
        displaced,
        retained=retained,
        outsider=outsider,
        pressure_reference=pressure_reference,
        cache=cache,
    )
    internal_challenger = {
        _event_identity(event)
        for event in analysis.events
        if int(event.left) in challenger and int(event.right) in challenger
    }
    internal_anchor = {
        _event_identity(event)
        for event in analysis.events
        if int(event.left) in anchor and int(event.right) in anchor
    }
    challenger_occupancy = {
        (time_index, _position(cache.paths, agent, time_index))
        for agent in challenger
        for time_index in range(cache.horizon_raw + 1)
    }
    anchor_occupancy = {
        (time_index, _position(cache.paths, agent, time_index))
        for agent in anchor
        for time_index in range(cache.horizon_raw + 1)
    }
    values = [
        *(left - right for left, right in zip(entered_values, displaced_values)),
        _set_jaccard(internal_challenger, internal_anchor),
        _set_jaccard(challenger_occupancy, anchor_occupancy),
    ]
    if len(values) != len(MOTIF_FEATURE_NAMES) or any(
        not math.isfinite(float(value)) for value in values
    ):
        raise ValueError("pair-motif feature vector is invalid")
    return list(map(float, values))


def _target_states(
    config: dict[str, Any],
    *,
    project_root: Path,
    inputs: dict[str, Path],
    state_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    selections = _unique(
        (
            row
            for row in _read_jsonl(inputs["state_selection"])
            if str(row["state_id"]) in state_ids
        ),
        "state_id",
    )
    if set(key[0] for key in selections) != state_ids:
        raise ValueError("pair-motif occurrence coverage changed")
    selection_report = _read_json(inputs["state_selection_report"])
    if selection_report.get("passed") is not True:
        raise ValueError("registered target-state selection did not pass")
    manifest_indices: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}
    for name in (
        "source_v4_official_manifest",
        "source_v4_v2_manifest",
        "da2_v2_official_manifest",
        "da2_v2_v2_manifest",
    ):
        manifest_indices[name] = _unique(
            (row for row in _read_jsonl(inputs[name]) if str(row.get("status")) == "ok"),
            "episode_id",
        )
    states: dict[str, dict[str, Any]] = {}
    trace_cache: dict[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for key, selection in sorted(selections.items()):
        state_id = key[0]
        namespace = str(selection["source_namespace"])
        manifest_name = _manifest_input_name(namespace, str(selection["source_manifest_policy"]))
        episode_id = str(selection["episode_id"])
        manifest = manifest_indices[manifest_name].get((episode_id,))
        if manifest is None:
            raise ValueError(f"pair-motif source episode missing: {episode_id}")
        root = (project_root / str(config["source_roots"][namespace])).resolve()
        cache_key = (namespace, episode_id)
        if cache_key not in trace_cache:
            trace_path = contained_file(root, manifest["trace_file"], field="pair-motif source trace")
            if sha256_file(trace_path) != str(manifest["trace_sha256"]):
                raise ValueError(f"pair-motif source trace changed: {episode_id}")
            events = read_trace_events(trace_path)
            if not events or events[0].get("event") != "initial":
                raise ValueError("pair-motif source trace initial event changed")
            trace_cache[cache_key] = (_initial_state(root, trace_path, events[0]), events)
        initial, events = trace_cache[cache_key]
        target = int(selection["decision_index"])
        transitions = (event for event in events if event.get("event") == "transition")
        state = state_before_decision(
            initial,
            transitions,
            decision_index=target,
            expected_fingerprint=str(selection["before_fingerprint"]),
        )
        if int(state["num_of_colliding_pairs"]) != int(selection["before_conflicts"]):
            raise ValueError(f"pair-motif target conflict count changed: {state_id}")
        states[state_id] = state
    return states, {key[0]: value for key, value in selections.items()}


def _validate_prior_sequential_evidence(inputs: dict[str, Path]) -> None:
    prior_config = _read_json(inputs["prior_temporal_config"])
    prior_report = _read_json(inputs["prior_temporal_report"])
    if (
        prior_config.get("schema")
        != "lns2.stride.temporal_anchor_incremental_readiness_config.v1"
        or prior_report.get("schema")
        != "lns2.stride.temporal_anchor_incremental_readiness_report.v1"
        or prior_report.get("offline_passed") is not False
        or not math.isclose(
            float(prior_report["fair_comparison_static_baseline"]["history_candidate_metrics"]["roc_auc"]),
            0.6399471379337135,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(prior_report["temporal_model"]["history_candidate_metrics"]["roc_auc"]),
            0.6535202508356857,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("prior failed sequential SafeSlot evidence changed")


def build_ablation_rows(
    config: dict[str, Any], *, project_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_config(config, project_root=project_root)
    inputs = _registered(project_root, config)
    _validate_prior_sequential_evidence(inputs)
    safeslot_config = _read_json(inputs["safeslot_config"])
    validate_safeslot_gate_config(safeslot_config, project_root=project_root)
    base_rows, base_audit = build_safeslot_gate_rows(
        safeslot_config, project_root=project_root
    )
    agents = _candidate_agent_sets(safeslot_config, base_rows, project_root=project_root)
    state_ids = {str(row["state_id"]) for row in base_rows}
    states, selections = _target_states(
        config, project_root=project_root, inputs=inputs, state_ids=state_ids
    )
    base_by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for base in base_rows:
        base_by_state[str(base["state_id"])].append(base)
    motif_by_pair: dict[tuple[str, str, str], list[float]] = {}
    rows: list[dict[str, Any]] = []
    static_grids: dict[str, StaticGridAnalysis] = {}
    for state_id in sorted(base_by_state):
        map_id = str(base_by_state[state_id][0]["map_id"])
        static_grid = static_grids.get(map_id)
        if static_grid is None:
            static_grid = analyze_static_grid(states[state_id])
            static_grids[map_id] = static_grid
        cache = _pair_motif_cache(states[state_id], static_grid=static_grid)
        selection = selections[state_id]
        for base in base_by_state[state_id]:
            anchor_id = str(base["anchor_candidate_id"])
            challenger_id = str(base["challenger_candidate_id"])
            motif = _pair_motif_feature_vector_from_cache(
                cache,
                agents[(state_id, challenger_id)],
                agents[(state_id, anchor_id)],
            )
            motif_by_pair[(state_id, anchor_id, challenger_id)] = motif
            rows.append(
                {
                    **base,
                    "schema": ROW_SCHEMA,
                    "decision_index": int(selection["decision_index"]),
                    "before_fingerprint": str(selection["before_fingerprint"]),
                    "static_feature_values": list(map(float, base["feature_values"])),
                    "motif_feature_values": motif,
                    "feature_values": [*map(float, base["feature_values"]), *motif],
                }
            )
        del cache
    cohort = dict(config["cohort"])
    if (
        len(rows) != int(cohort["candidate_count"])
        or len(states) != int(cohort["state_count"])
        or len({str(row["map_id"]) for row in rows}) != int(cohort["map_count"])
        or sum(bool(row["label"]) for row in rows)
        != int(cohort["positive_candidate_count"])
        or sum(not bool(row["label"]) for row in rows)
        != int(cohort["negative_candidate_count"])
        or any(len(row["static_feature_values"]) != len(STATIC_FEATURE_NAMES) for row in rows)
        or any(len(row["feature_values"]) != len(FEATURE_NAMES) for row in rows)
    ):
        raise ValueError("pair-motif ablation row audit failed")
    return rows, {
        **base_audit,
        "feature_dimension": len(FEATURE_NAMES),
        "motif_dimension": len(MOTIF_FEATURE_NAMES),
        "motif_feature_names": list(MOTIF_FEATURE_NAMES),
        "state_feature_hashes": {
            state_id: json_fingerprint(
                {
                    challenger_id: motif_by_pair[(state_id, anchor_id, challenger_id)]
                    for candidate_state, anchor_id, challenger_id in sorted(motif_by_pair)
                    if candidate_state == state_id
                }
            )
            for state_id in sorted(states)
        },
        "target_and_future_fields_used": False,
        "solver_native_or_pp_executed": False,
    }


def _classification(rows: list[dict[str, Any]], probabilities: list[float]) -> dict[str, float]:
    metrics = candidate_classification_metrics(rows, probabilities)
    return {
        "roc_auc": float(metrics["roc_auc"]),
        "average_precision": float(metrics["average_precision"]),
    }


def _fixed_fold_fit(
    *,
    rows: list[dict[str, Any]],
    values: Any,
    labels: Any,
    weights: Any,
    fold: dict[str, Any],
    parameters: dict[str, Any],
) -> tuple[int, list[int], list[float], dict[str, Any]]:
    train_maps = set(map(str, fold["train_maps"]))
    validation_maps = set(map(str, fold["validation_maps"]))
    train_indices = [
        index for index, row in enumerate(rows) if str(row["map_id"]) in train_maps
    ]
    validation_indices = [
        index for index, row in enumerate(rows) if str(row["map_id"]) in validation_maps
    ]
    if train_maps & validation_maps:
        raise RuntimeError("pair-motif whole-map fold leakage")
    estimator = _fit_model(values, labels, weights, train_indices, parameters)
    probabilities = list(
        map(float, estimator.predict_proba(values[validation_indices])[:, 1])
    )
    fold_rows = [rows[index] for index in validation_indices]
    return (
        int(fold["fold"]),
        validation_indices,
        probabilities,
        {
            "outer_fold": int(fold["fold"]),
            "train_maps": sorted(train_maps),
            "validation_maps": sorted(validation_maps),
            "candidate_count": len(validation_indices),
            "metrics": _classification(fold_rows, probabilities),
        },
    )


def _fixed_oof(
    rows: list[dict[str, Any]],
    *,
    feature_key: str,
    folds: list[dict[str, Any]],
    parameters: dict[str, Any],
    workers: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    import numpy as np

    values = np.asarray([row[feature_key] for row in rows], dtype=np.float32)
    labels = np.asarray([bool(row["label"]) for row in rows], dtype=np.int8)
    weights = np.asarray([float(row["sample_weight"]) for row in rows], dtype=np.float64)
    probabilities: dict[int, float] = {}
    diagnostics: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _fixed_fold_fit,
                rows=rows,
                values=values,
                labels=labels,
                weights=weights,
                fold=fold,
                parameters=parameters,
            )
            for fold in folds
        ]
        for future in concurrent.futures.as_completed(futures):
            fold_id, indices, predicted, diagnostic = future.result()
            if any(index in probabilities for index in indices):
                raise RuntimeError("pair-motif OOF row predicted more than once")
            probabilities.update(dict(zip(indices, predicted)))
            diagnostic["outer_fold"] = fold_id
            diagnostics.append(diagnostic)
    if set(probabilities) != set(range(len(rows))):
        raise RuntimeError("pair-motif OOF coverage changed")
    return (
        [probabilities[index] for index in range(len(rows))],
        sorted(diagnostics, key=lambda row: int(row["outer_fold"])),
    )


def plan(*, config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = _read_json(config_path)
    validate_config(config, project_root=project_root)
    return {
        "schema": CONFIG_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": config["scientific_status"],
        "cohort": dict(config["cohort"]),
        "feature_dimensions": {"baseline": 193, "motif": 18, "augmented": 211},
        "comparison": "same_587_rows_same_weights_same_4_whole_map_folds_fixed_parameter_index_0",
        "model_fit_count": 8,
        "configured_workers": int(config["model"]["workers"]),
        "old_oof_identity_reproduction_required": True,
        "metrics": ["roc_auc", "average_precision"],
        "new_ablation_threshold_policy_or_export": False,
        "legacy_identity_reproduction_replays_registered_static_calibration": True,
        "legacy_calibration_not_interpreted_or_exported": True,
        "solver_or_controller_invoked": False,
        "claim_boundary": dict(config["claim_boundary"]),
        "output": str(config["outputs"]["ablation"]),
    }


def run_ablation(
    *, config_path: str | Path, output: str | Path, dry_run: bool = False
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = _read_json(config_path)
    validate_config(config, project_root=project_root)
    planned = plan(config_path=config_path)
    if dry_run:
        return {**planned, "dry_run": True}
    expected_output = (project_root / str(config["outputs"]["ablation"])).resolve()
    output_path = Path(output).resolve()
    if output_path != expected_output:
        raise ValueError(f"pair-motif output must be the registered fresh root: {expected_output}")
    started = time.monotonic()
    producer = producer_identity(
        project_root=project_root,
        source_files=PRODUCER_FILES,
        native_required=False,
        package_names=("numpy", "scikit-learn"),
    )
    identity = {
        "schema": CONFIG_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "producer_identity": producer,
    }
    prepare_run_output(output_path, resume=False, identity=identity)
    rows, row_audit = build_ablation_rows(config, project_root=project_root)
    if time.monotonic() - started > int(config["analysis_wall_fuse_seconds"]):
        raise TimeoutError("pair-motif row construction exceeded registered wall fuse")
    inputs = _registered(project_root, config)
    safeslot_config = _read_json(inputs["safeslot_config"])
    identity_probabilities, identity_diagnostics = _static_oof(rows, safeslot_config)
    reproduction_error, folds_equal = _reproduction_error(
        rows,
        identity_probabilities,
        _read_jsonl(inputs["safeslot_oof_predictions"]),
        identity_diagnostics,
    )
    maximum_error = float(
        config["identity_reproduction"]["maximum_probability_absolute_error"]
    )
    if reproduction_error > maximum_error or not folds_equal:
        raise RuntimeError("registered SafeSlot OOF identity reproduction failed")
    all_maps = {str(row["map_id"]) for row in rows}
    folds = _map_folds(rows, all_maps, 4)
    expected_folds = [sorted(map(str, fold)) for fold in config["model"]["outer_map_folds"]]
    if [sorted(map(str, fold["validation_maps"])) for fold in folds] != expected_folds:
        raise RuntimeError("pair-motif frozen outer folds changed")
    parameters = _parameters(safeslot_config, 0)
    if parameters != dict(config["model"]["parameters"]):
        raise RuntimeError("pair-motif fixed parameter index 0 changed")
    workers = int(config["model"]["workers"])
    baseline_probabilities, baseline_folds = _fixed_oof(
        rows,
        feature_key="static_feature_values",
        folds=folds,
        parameters=parameters,
        workers=workers,
    )
    augmented_probabilities, augmented_folds = _fixed_oof(
        rows,
        feature_key="feature_values",
        folds=folds,
        parameters=parameters,
        workers=workers,
    )
    if time.monotonic() - started > int(config["analysis_wall_fuse_seconds"]):
        raise TimeoutError("pair-motif ablation exceeded registered wall fuse")
    baseline_metrics = _classification(rows, baseline_probabilities)
    augmented_metrics = _classification(rows, augmented_probabilities)
    auc_gain = augmented_metrics["roc_auc"] - baseline_metrics["roc_auc"]
    ap_gain = augmented_metrics["average_precision"] - baseline_metrics["average_precision"]
    fold_rows: list[dict[str, Any]] = []
    for baseline, augmented in zip(baseline_folds, augmented_folds):
        if (
            int(baseline["outer_fold"]) != int(augmented["outer_fold"])
            or baseline["validation_maps"] != augmented["validation_maps"]
            or int(baseline["candidate_count"]) != int(augmented["candidate_count"])
        ):
            raise RuntimeError("pair-motif A/B fold parity changed")
        fold_rows.append(
            {
                "outer_fold": int(baseline["outer_fold"]),
                "validation_maps": list(baseline["validation_maps"]),
                "candidate_count": int(baseline["candidate_count"]),
                "baseline": dict(baseline["metrics"]),
                "augmented": dict(augmented["metrics"]),
                "roc_auc_gain": float(
                    augmented["metrics"]["roc_auc"] - baseline["metrics"]["roc_auc"]
                ),
                "average_precision_gain": float(
                    augmented["metrics"]["average_precision"]
                    - baseline["metrics"]["average_precision"]
                ),
            }
        )
    gates = dict(config["offline_acceptance"])
    positive_fold_count = sum(float(row["roc_auc_gain"]) > 0.0 for row in fold_rows)
    checks = {
        "overall_roc_auc_gain": auc_gain >= float(gates["minimum_overall_roc_auc_gain"]),
        "overall_average_precision_gain": ap_gain
        >= float(gates["minimum_overall_average_precision_gain"]),
        "each_fold_augmented_roc_auc": all(
            float(row["augmented"]["roc_auc"])
            >= float(gates["minimum_each_fold_augmented_roc_auc"])
            for row in fold_rows
        ),
        "each_fold_roc_auc_gain": all(
            float(row["roc_auc_gain"])
            >= float(gates["minimum_each_fold_roc_auc_gain"])
            for row in fold_rows
        ),
        "positive_roc_auc_gain_fold_count": positive_fold_count
        >= int(gates["minimum_positive_gain_fold_count"]),
    }
    offline_passed = all(checks.values())
    prediction_rows = [
        {
            "schema": PREDICTION_SCHEMA,
            "state_id": str(row["state_id"]),
            "map_id": str(row["map_id"]),
            "anchor_candidate_id": str(row["anchor_candidate_id"]),
            "challenger_candidate_id": str(row["challenger_candidate_id"]),
            "label": bool(row["label"]),
            "sample_weight": float(row["sample_weight"]),
            "identity_static_probability": float(identity),
            "fixed_baseline_probability": float(baseline),
            "fixed_augmented_probability": float(augmented),
        }
        for row, identity, baseline, augmented in zip(
            rows,
            identity_probabilities,
            baseline_probabilities,
            augmented_probabilities,
        )
    ]
    feature_rows = [
        {
            "schema": ROW_SCHEMA,
            "state_id": str(row["state_id"]),
            "map_id": str(row["map_id"]),
            "anchor_candidate_id": str(row["anchor_candidate_id"]),
            "challenger_candidate_id": str(row["challenger_candidate_id"]),
            "decision_index": int(row["decision_index"]),
            "before_fingerprint": str(row["before_fingerprint"]),
            "motif_feature_values": list(map(float, row["motif_feature_values"])),
        }
        for row in rows
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": config["scientific_status"],
        "config_sha256": sha256_file(config_path),
        "producer_identity": producer,
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "row_audit": row_audit,
        "old_oof_identity_reproduction": {
            "maximum_absolute_error": reproduction_error,
            "folds_equal": folds_equal,
            "passed": reproduction_error <= maximum_error and folds_equal,
            "legacy_identity_reproduction_replays_registered_static_calibration": True,
            "legacy_calibration_not_interpreted_or_exported": True,
        },
        "fair_ablation": {
            "same_rows_weights_folds": True,
            "parameter_index": 0,
            "parameters": parameters,
            "configured_workers": workers,
            "baseline": baseline_metrics,
            "augmented": augmented_metrics,
            "roc_auc_gain": auc_gain,
            "average_precision_gain": ap_gain,
            "folds": fold_rows,
        },
        "offline_checks": checks,
        "offline_passed": offline_passed,
        "prior_sequential_evidence": dict(config["prior_sequential_evidence"]),
        "claim_boundary": dict(config["claim_boundary"]),
        "solver_native_or_pp_executed": False,
        "analysis_wall_seconds": time.monotonic() - started,
        "next_decision": (
            "register_fresh_map_paired_label_collection"
            if offline_passed
            else "permanently_stop_pair_motif_representation_route"
        ),
    }
    _write_jsonl(output_path / "motif_features.jsonl", feature_rows)
    _write_jsonl(output_path / "oof_predictions.jsonl", prediction_rows)
    _write_json(output_path / "pair_motif_sequential_ablation_report.json", report)
    return report


__all__ = [
    "FEATURE_NAMES",
    "MOTIF_FEATURE_NAMES",
    "build_ablation_rows",
    "pair_motif_feature_vector",
    "plan",
    "run_ablation",
    "validate_config",
]
