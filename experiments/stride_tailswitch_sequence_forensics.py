from __future__ import annotations

import collections
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import contained_file, registered_input, sha256_file
from experiments.closed_loop_trace_storage import (
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
)
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    state_fingerprint,
)


CONFIG_SCHEMA = "lns2.stride.tailswitch_sequence_forensics_registration.v1"
REPORT_SCHEMA = "lns2.stride.tailswitch_sequence_forensics_report.v1"
STATUS_SCHEMA = "lns2.stride.tailswitch_sequence_forensics_status.v1"
POLICIES = (
    "v2-then-v2",
    "struct-then-v2",
    "v2-then-struct",
    "struct-then-struct",
)
CONTINUATION_PAIRS = {
    "continuation_after_v2": ("v2-then-v2", "v2-then-struct"),
    "continuation_after_struct": ("struct-then-v2", "struct-then-struct"),
}


def _average(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _median(values: Iterable[float]) -> float:
    rows = list(values)
    return float(statistics.median(rows)) if rows else 0.0


def _stats(rows: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    values = [float(row[field]) for row in rows]
    return {
        "count": len(values),
        "mean": _average(values),
        "median": _median(values),
        "minimum": min(values) if values else 0.0,
        "maximum": max(values) if values else 0.0,
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _largest_component(edges: set[tuple[int, int]]) -> int:
    adjacency: dict[int, set[int]] = collections.defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[int] = set()
    largest = 0
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        largest = max(largest, size)
    return largest


def _conflict_band(conflicts: int) -> str:
    if conflicts <= 5:
        return "low"
    if conflicts <= 20:
        return "moderate"
    return "high"


def transition_metrics(
    before_edges: set[tuple[int, int]],
    after_edges: set[tuple[int, int]],
    selected_agents: set[int],
    previous_agents: list[set[int]],
    *,
    selected_candidate_id: str,
    previous_candidate_ids: list[str],
) -> dict[str, float | bool]:
    incident = {agent for edge in before_edges for agent in edge}
    touched = {edge for edge in before_edges if selected_agents & set(edge)}
    full = {edge for edge in before_edges if set(edge) <= selected_agents}
    recent = set().union(*previous_agents[-3:]) if previous_agents else set()
    before_largest = _largest_component(before_edges)
    after_largest = _largest_component(after_edges)
    return {
        "adjacent_agent_jaccard": (
            _jaccard(selected_agents, previous_agents[-1])
            if previous_agents
            else 0.0
        ),
        "recent_three_reuse_fraction": (
            len(selected_agents & recent) / len(selected_agents)
            if selected_agents
            else 0.0
        ),
        "exact_candidate_repeat": selected_candidate_id
        in previous_candidate_ids[-3:],
        "conflict_touch_coverage": (
            len(touched) / len(before_edges) if before_edges else 0.0
        ),
        "conflict_full_coverage": (
            len(full) / len(before_edges) if before_edges else 0.0
        ),
        "selected_conflict_agent_fraction": (
            len(selected_agents & incident) / len(selected_agents)
            if selected_agents
            else 0.0
        ),
        "unresolved_edge_fraction": (
            len(before_edges & after_edges) / len(before_edges)
            if before_edges
            else 0.0
        ),
        "new_edge_fraction": (
            len(after_edges - before_edges) / len(after_edges)
            if after_edges
            else 0.0
        ),
        "largest_component_persistence": (
            after_largest / before_largest if before_largest else 0.0
        ),
    }


def _selected_candidate(
    controller: dict[str, Any], action: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    selected_id = str(controller.get("selected_candidate_id") or "")
    pool = list(controller.get("candidate_pool") or [])
    matches = [row for row in pool if str(row.get("candidate_id")) == selected_id]
    if len(matches) != 1:
        raise ValueError(f"selected candidate coverage changed: {selected_id}")
    candidate = dict(matches[0])
    action_agents = list(map(int, action.get("agents") or ()))
    if action_agents != list(map(int, candidate.get("agents") or ())):
        raise ValueError(f"selected candidate agents changed: {selected_id}")
    families = list(map(str, candidate.get("selection_families") or ()))
    return candidate, families


def original_pool_anchor(candidate_pool: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    original = [
        dict(value)
        for value in candidate_pool
        if not any(
            str(family).startswith("structpool-")
            for family in value.get("selection_families") or ()
        )
    ]
    if not original:
        return None
    return sorted(
        original,
        key=lambda value: (
            -round(float(value.get("score", 0.0)), 12),
            str(value["candidate_id"]),
        ),
    )[0]


def _episode_rows(
    collection: Path,
    manifest: dict[str, Any],
    *,
    state_id: str,
    policy: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    trace_path = contained_file(
        collection, manifest.get("trace_file"), field="TailSwitch trace_file"
    )
    if sha256_file(trace_path) != str(manifest.get("trace_sha256")):
        raise ValueError(f"trace SHA-256 changed: {state_id}/{policy}")
    events = read_trace_events(trace_path)
    if len(events) < 2 or events[0].get("event") != "initial":
        raise ValueError(f"invalid trace event sequence: {state_id}/{policy}")
    initial = events[0]
    state = read_state_blob(
        resolve_state_blob(trace_path, str(initial["state_blob"]), collection)
    )
    state.update(dict(initial.get("state_extras") or {}))
    previous_agents: list[set[int]] = []
    previous_candidate_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    for event in events[1:-1]:
        if event.get("event") != "transition":
            raise ValueError(f"unexpected trace event: {state_id}/{policy}")
        if state_fingerprint(state) != str(event.get("before_fingerprint")):
            raise ValueError(f"before fingerprint changed: {state_id}/{policy}")
        action = dict(event.get("action") or {})
        controller = dict(event.get("controller") or {})
        candidate, families = _selected_candidate(controller, action)
        selected_id = str(candidate["candidate_id"])
        agents = set(map(int, action["agents"]))
        before_edges = {
            tuple(map(int, edge)) for edge in state.get("conflict_edges") or ()
        }
        after = apply_state_delta(state, dict(event["state_delta"]))
        after.update(apply_extras_delta(state, dict(event["state_extras_delta"])))
        if state_fingerprint(after) != str(event.get("after_fingerprint")):
            raise ValueError(f"after fingerprint changed: {state_id}/{policy}")
        after_edges = {
            tuple(map(int, edge)) for edge in after.get("conflict_edges") or ()
        }
        metrics = dict(event.get("metrics") or {})
        before_conflicts = int(metrics.get("conflicts_before", -1))
        after_conflicts = int(metrics.get("conflicts_after", -1))
        if before_conflicts != len(before_edges) or after_conflicts != len(after_edges):
            raise ValueError(f"conflict-edge count changed: {state_id}/{policy}")
        structural = any(value.startswith("structpool-") for value in families)
        pool = [dict(value) for value in controller.get("candidate_pool") or ()]
        original_anchor = original_pool_anchor(pool)
        original_anchor_agents = (
            set(map(int, original_anchor.get("agents") or ()))
            if original_anchor is not None
            else set()
        )
        original_anchor_touched = {
            edge for edge in before_edges if original_anchor_agents & set(edge)
        }
        row = {
            **metadata,
            "state_id": state_id,
            "policy": policy,
            "decision_index": int(event["decision_index"]),
            "forced_first_action": controller.get("forced_first_action") is True,
            "selected_candidate_id": selected_id,
            "selection_families": families,
            "selected_kind": "structural" if structural else "base",
            "selected_size": len(agents),
            "original_pool_anchor_candidate_id": (
                str(original_anchor["candidate_id"])
                if original_anchor is not None
                else None
            ),
            "differs_from_original_pool_anchor": bool(
                original_anchor is not None
                and selected_id != str(original_anchor["candidate_id"])
            ),
            "selected_vs_original_anchor_jaccard": (
                _jaccard(agents, original_anchor_agents)
                if original_anchor is not None
                else 0.0
            ),
            "selected_minus_original_anchor_score": (
                float(candidate.get("score", 0.0))
                - float(original_anchor.get("score", 0.0))
                if original_anchor is not None
                else 0.0
            ),
            "conflict_touch_coverage_minus_original_anchor": (
                len({edge for edge in before_edges if agents & set(edge)})
                / len(before_edges)
                - len(original_anchor_touched) / len(before_edges)
                if before_edges and original_anchor is not None
                else 0.0
            ),
            "selected_feature_out_of_range_fraction": float(
                candidate.get("feature_out_of_range_fraction", 0.0)
            ),
            "original_anchor_feature_out_of_range_fraction": (
                float(original_anchor.get("feature_out_of_range_fraction", 0.0))
                if original_anchor is not None
                else None
            ),
            "score_margin": float(controller.get("score_margin", 0.0)),
            "conflict_band": _conflict_band(before_conflicts),
            "conflicts_before": before_conflicts,
            "conflicts_after": after_conflicts,
            "normalized_conflict_progress": (
                (before_conflicts - after_conflicts) / max(before_conflicts, 1)
            ),
            **transition_metrics(
                before_edges,
                after_edges,
                agents,
                previous_agents,
                selected_candidate_id=selected_id,
                previous_candidate_ids=previous_candidate_ids,
            ),
        }
        rows.append(row)
        previous_agents.append(agents)
        previous_candidate_ids.append(selected_id)
        state = after
    return rows


SEQUENCE_FIELDS = (
    "adjacent_agent_jaccard",
    "recent_three_reuse_fraction",
    "conflict_touch_coverage",
    "conflict_full_coverage",
    "selected_conflict_agent_fraction",
    "unresolved_edge_fraction",
    "new_edge_fraction",
    "normalized_conflict_progress",
    "largest_component_persistence",
    "selected_feature_out_of_range_fraction",
)


def _episode_summary(rows: list[dict[str, Any]], early_window: int) -> dict[str, Any]:
    continuation = [row for row in rows if int(row["decision_index"]) >= 1]
    early = continuation[:early_window]
    structural = [row for row in continuation if row["selected_kind"] == "structural"]
    max_run = 0
    current = 0
    for row in continuation:
        current = current + 1 if row["selected_kind"] == "structural" else 0
        max_run = max(max_run, current)
    result: dict[str, Any] = {
        "transition_count": len(rows),
        "continuation_count": len(continuation),
        "early_window_observed_count": len(early),
        "structural_selection_count": len(structural),
        "structural_selection_rate": (
            len(structural) / len(continuation) if continuation else 0.0
        ),
        "original_pool_anchor_disagreement_count": sum(
            bool(row["differs_from_original_pool_anchor"]) for row in continuation
        ),
        "exact_candidate_repeat_rate": _average(
            float(bool(row["exact_candidate_repeat"])) for row in continuation
        ),
        "maximum_structural_run_length": max_run,
        "first_post_first_structural_decision": (
            int(structural[0]["decision_index"]) if structural else None
        ),
    }
    for field in SEQUENCE_FIELDS:
        result[f"mean_{field}"] = _average(float(row[field]) for row in continuation)
        result[f"early_mean_{field}"] = _average(float(row[field]) for row in early)
        result[f"structural_mean_{field}"] = _average(
            float(row[field]) for row in structural
        )
    return result


def _kind_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"transition_count": len(rows)}
    for field in SEQUENCE_FIELDS:
        result[field] = _stats(rows, field)
    result["exact_candidate_repeat_rate"] = _average(
        float(bool(row["exact_candidate_repeat"])) for row in rows
    )
    result["original_pool_anchor_disagreement_rate"] = _average(
        float(bool(row["differs_from_original_pool_anchor"])) for row in rows
    )
    for field in (
        "selected_vs_original_anchor_jaccard",
        "selected_minus_original_anchor_score",
        "conflict_touch_coverage_minus_original_anchor",
    ):
        result[field] = _stats(rows, field)
    return result


def _candidate_targeting(
    transitions: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in transitions:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        structural = [row for row in rows if row["selected_kind"] == "structural"]
        base = [row for row in rows if row["selected_kind"] == "base"]
        if not structural or not base:
            continue
        deltas = {
            field: _average(float(row[field]) for row in structural)
            - _average(float(row[field]) for row in base)
            for field in (
                "conflict_touch_coverage",
                "normalized_conflict_progress",
                "unresolved_edge_fraction",
            )
        }
        output.append(
            {
                **dict(zip(group_fields, key)),
                "structural_count": len(structural),
                "base_count": len(base),
                "structural_minus_base": deltas,
                "all_registered_directions": (
                    deltas["conflict_touch_coverage"] < 0.0
                    and deltas["normalized_conflict_progress"] < 0.0
                    and deltas["unresolved_edge_fraction"] > 0.0
                ),
            }
        )
    return output


def _pair_rows(
    report_states: list[dict[str, Any]],
    episode_summaries: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    fields = (
        "structural_selection_count",
        "structural_selection_rate",
        "exact_candidate_repeat_rate",
        "maximum_structural_run_length",
        "mean_recent_three_reuse_fraction",
        "mean_unresolved_edge_fraction",
        "mean_conflict_touch_coverage",
        "mean_normalized_conflict_progress",
    )
    for state in report_states:
        state_id = str(state["state_id"])
        for name, (reference_policy, treatment_policy) in CONTINUATION_PAIRS.items():
            reference = episode_summaries[(state_id, reference_policy)]
            treatment = episode_summaries[(state_id, treatment_policy)]
            contrast = dict(state["contrasts"][f"struct_{name}"])
            output.append(
                {
                    "state_id": state_id,
                    "contrast": name,
                    "map_id": str(state["map_id"]),
                    "task_id": str(state["task_id"]),
                    "solver_seed": int(state["solver_seed"]),
                    "challenger": str(state["challenger"]),
                    "classification": str(contrast["classification"]),
                    "normalized_auc_delta": float(
                        contrast["normalized_auc_delta"]
                    ),
                    "final_conflict_delta": int(contrast["final_conflict_delta"]),
                    "treatment_structural_selection_count": int(
                        treatment["structural_selection_count"]
                    ),
                    "metric_deltas": {
                        field: float(treatment[field]) - float(reference[field])
                        for field in fields
                    },
                }
            )
    return output


def _selection_memory_report(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    adverse = [row for row in pair_rows if row["classification"] == "adverse"]
    other = [row for row in pair_rows if row["classification"] != "adverse"]
    fields = {
        "recent_three_reuse_fraction": "mean_recent_three_reuse_fraction",
        "exact_candidate_repeat_rate": "exact_candidate_repeat_rate",
        "unresolved_edge_fraction": "mean_unresolved_edge_fraction",
    }
    directions = {}
    for name, field in fields.items():
        adverse_mean = _average(float(row["metric_deltas"][field]) for row in adverse)
        other_mean = _average(float(row["metric_deltas"][field]) for row in other)
        directions[name] = {
            "adverse_mean_treatment_minus_reference": adverse_mean,
            "non_adverse_mean_treatment_minus_reference": other_mean,
            "adverse_minus_non_adverse": adverse_mean - other_mean,
            "registered_direction": adverse_mean > other_mean,
        }
    supported_count = sum(row["registered_direction"] for row in directions.values())
    return {
        "adverse_pair_count": len(adverse),
        "non_adverse_pair_count": len(other),
        "directions": directions,
        "registered_direction_count": supported_count,
        "supported": supported_count >= 2,
        "adverse_with_at_least_two_post_first_structural_selections": sum(
            int(row["treatment_structural_selection_count"]) >= 2 for row in adverse
        ),
    }


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_existing_trajectory_sequence_mechanism_diagnostic"
        or config.get("experiment_id")
        != "stride-tailswitch-sequence-forensics-v1"
        or config.get("pre_registration_parent_commit")
        != "0b2f22699437a6908331cc1d13a4dc8a6d997f45"
    ):
        raise ValueError("TailSwitch sequence-forensics registration changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    if tuple(config["cohort"]["policies"]) != POLICIES:
        raise ValueError("TailSwitch sequence-forensics policies changed")
    if dict(config["claim_boundary"]) != {
        "existing_trajectory_diagnostic_only": True,
        "model_training_allowed": False,
        "new_solver_runs_allowed": False,
        "ttf_improvement_claim": False,
        "generalization_claim": False,
        "per_transition_causal_claim": False,
        "default_controller_replacement_allowed": False,
        "no_result_based_exclusion": True,
    }:
        raise ValueError("TailSwitch sequence-forensics claim boundary changed")
    return path, root, config, inputs


def analyze_sequence_forensics(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, root, config, inputs = load_registration(config_path)
    tailswitch_report = _read_json(inputs["tailswitch_report"])
    tailswitch_status = _read_json(inputs["tailswitch_status"])
    selection = _read_jsonl(inputs["state_selection"])
    if (
        tailswitch_report.get("integrity_passed") is not True
        or tailswitch_report.get("causal_conclusion") != "inconclusive"
        or tailswitch_status.get("complete") is not True
        or len(selection) != 66
        or len(tailswitch_report.get("states") or ()) != 66
    ):
        raise ValueError("completed TailSwitch evidence contract changed")
    report_by_state = {
        str(row["state_id"]): row for row in tailswitch_report["states"]
    }
    selection_by_state = {str(row["state_id"]): row for row in selection}
    if set(report_by_state) != set(selection_by_state):
        raise ValueError("TailSwitch state coverage changed")
    tailswitch_root = inputs["tailswitch_report"].parent
    transitions: list[dict[str, Any]] = []
    episode_summaries: dict[tuple[str, str], dict[str, Any]] = {}
    manifest_hashes: dict[str, str] = {}
    trace_hashes: dict[str, str] = {}
    early_window = int(config["transition_scope"]["early_continuation_window"])
    for state_id in sorted(report_by_state):
        metadata = {
            "map_id": str(report_by_state[state_id]["map_id"]),
            "task_id": str(report_by_state[state_id]["task_id"]),
            "solver_seed": int(report_by_state[state_id]["solver_seed"]),
            "challenger": str(report_by_state[state_id]["challenger"]),
            "tail_category": str(report_by_state[state_id]["tail_category"]),
        }
        state_dir = tailswitch_root / "states" / _fingerprint(
            {"state_id": state_id}
        )[:20]
        for policy in POLICIES:
            collection = state_dir / policy
            manifest_path = collection / "realized_dynamic_manifest.jsonl"
            manifest_rows = _read_jsonl(manifest_path)
            if len(manifest_rows) != 1 or manifest_rows[0].get("status") != "ok":
                raise ValueError(f"TailSwitch manifest changed: {state_id}/{policy}")
            manifest = manifest_rows[0]
            key = f"{state_id}/{policy}"
            manifest_hashes[key] = sha256_file(manifest_path)
            trace_path = contained_file(
                collection, manifest["trace_file"], field="TailSwitch trace_file"
            )
            trace_hashes[key] = sha256_file(trace_path)
            rows = _episode_rows(
                collection,
                manifest,
                state_id=state_id,
                policy=policy,
                metadata=metadata,
            )
            transitions.extend(rows)
            episode_summaries[(state_id, policy)] = {
                **metadata,
                "state_id": state_id,
                "policy": policy,
                **_episode_summary(rows, early_window),
            }
    continuation = [row for row in transitions if int(row["decision_index"]) >= 1]
    augmented = [row for row in continuation if str(row["policy"]).endswith("struct")]
    by_kind = {
        kind: _kind_summary([row for row in augmented if row["selected_kind"] == kind])
        for kind in ("base", "structural")
    }
    by_challenger = _candidate_targeting(augmented, ("challenger",))
    by_map = _candidate_targeting(augmented, ("map_id",))
    by_stratum = _candidate_targeting(
        augmented, ("map_id", "challenger", "conflict_band")
    )
    targeting = {
        "by_selected_kind": by_kind,
        "by_challenger": by_challenger,
        "by_map": by_map,
        "by_map_challenger_conflict_band": by_stratum,
        "supporting_challenger_count": sum(
            bool(row["all_registered_directions"]) for row in by_challenger
        ),
        "supporting_map_count": sum(
            bool(row["all_registered_directions"]) for row in by_map
        ),
    }
    targeting["supported"] = (
        int(targeting["supporting_challenger_count"]) >= 2
        and int(targeting["supporting_map_count"]) >= 2
    )
    pairs = _pair_rows(list(tailswitch_report["states"]), episode_summaries)
    memory = _selection_memory_report(pairs)
    first_reports = [
        tailswitch_report["registered_contrast_reports"][name]
        for name in (
            "first_action_under_v2_continuation",
            "first_action_under_struct_continuation",
        )
    ]
    continuation_reports = [
        tailswitch_report["registered_contrast_reports"][name]
        for name in (
            "struct_continuation_after_v2",
            "struct_continuation_after_struct",
        )
    ]
    first_not_adverse = all(
        float(row["mean_normalized_auc_delta"]) < 0.0 for row in first_reports
    )
    continuation_adverse = all(
        float(row["mean_normalized_auc_delta"]) > 0.0
        for row in continuation_reports
    )
    if memory["supported"] and first_not_adverse and continuation_adverse:
        interpretation = "selection_memory_and_missing_exit_association"
    elif targeting["supported"]:
        interpretation = "candidate_targeting_association"
    elif first_not_adverse and continuation_adverse:
        interpretation = "continuation_policy_exit_problem"
    else:
        interpretation = "inconclusive"
    episode_rows = [episode_summaries[key] for key in sorted(episode_summaries)]
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": "stride-tailswitch-sequence-forensics-v1",
        "integrity_passed": (
            len(report_by_state) == 66
            and len(episode_rows) == 264
            and len({(row["state_id"], row["policy"]) for row in episode_rows})
            == 264
        ),
        "state_count": len(report_by_state),
        "episode_count": len(episode_rows),
        "transition_count": len(transitions),
        "continuation_transition_count": len(continuation),
        "augmented_continuation_transition_count": len(augmented),
        "structural_continuation_transition_count": sum(
            row["selected_kind"] == "structural" for row in augmented
        ),
        "candidate_targeting": targeting,
        "selection_memory": memory,
        "policy_level_exit_evidence": {
            "first_structural_action_not_adverse_on_average": first_not_adverse,
            "continued_augmented_selection_adverse_on_average": continuation_adverse,
            "adverse_pair_count": memory["adverse_pair_count"],
            "adverse_with_at_least_two_post_first_structural_selections": memory[
                "adverse_with_at_least_two_post_first_structural_selections"
            ],
            "prevalence_gate": "not_preregistered_descriptive_only",
        },
        "primary_interpretation": interpretation,
        "episode_summaries": episode_rows,
        "paired_sequence_comparisons": pairs,
        "secondary_diagnostic_notice": (
            "feature-range and score fields are descriptive and were not used "
            "in the preregistered interpretation rules"
        ),
        "inputs": {
            "registration_sha256": sha256_file(path),
            "tailswitch_report_sha256": sha256_file(inputs["tailswitch_report"]),
            "tailswitch_status_sha256": sha256_file(inputs["tailswitch_status"]),
            "state_selection_sha256": sha256_file(inputs["state_selection"]),
            "manifest_sha256": manifest_hashes,
            "trace_sha256": trace_hashes,
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "sequence_forensics_report.json"
    _write_json(report_path, report)
    _write_json(
        output / "sequence_forensics_status.json",
        {
            "schema": STATUS_SCHEMA,
            "complete": True,
            "integrity_passed": report["integrity_passed"],
            "state_count": report["state_count"],
            "episode_count": report["episode_count"],
            "report_sha256": sha256_file(report_path),
        },
    )
    return report


__all__ = [
    "analyze_sequence_forensics",
    "load_registration",
    "original_pool_anchor",
    "transition_metrics",
]
