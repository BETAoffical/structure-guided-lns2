from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from experiments._common import _native_filesystem_path, sha256_file
from experiments.closed_loop_trace_storage import read_trace_events
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.stride_slotpool_structpool_ttf import (
    _expected_keys,
    load_slotpool_structpool_ttf_config,
)


CONFIG_SCHEMA = "lns2.stride.safeslot_first_divergence_config.v1"
REPORT_SCHEMA = "lns2.stride.safeslot_first_divergence_report.v1"
EXPERIMENT_ID = "stride-safeslot-first-divergence-v1"
CHALLENGERS = ("v2-plus-structpool", "v2-plus-slotpool")
REPORT_FILENAME = "safeslot_first_divergence_report.json"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    path = (root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"registered SafeSlot divergence input changed: {path}")
    return path


def load_safeslot_first_divergence_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], set[tuple[str, str, int]]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "retrospective_first_divergence_diagnostic_no_training"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("parent_commit")
        != "f33be7f03837c9f1139e3d5d6cd5f6745b0638b5"
        or tuple(map(str, config.get("controllers") or ()))
        != ("v2-full", *CHALLENGERS)
    ):
        raise ValueError("SafeSlot first-divergence identity changed")
    if dict(config.get("cohort") or {}) != {
        "source_config": "configs/stride_slotpool_structpool_ttf_diagnostic_v1.json",
        "paired_key_count": 29,
        "known_tail_exclusion_count": 1,
        "short_window_repairs": 8,
    }:
        raise ValueError("SafeSlot first-divergence cohort changed")
    if dict(config.get("integrity") or {}) != {
        "initial_fingerprint_and_conflicts_must_match": True,
        "identical_prefix_candidate_and_pp_seed_must_match": True,
        "identical_prefix_after_fingerprint_must_match": True,
        "challenger_anchor_must_equal_v2_action_at_first_divergence": True,
        "first_divergent_action_must_be_structural": True,
        "all_29_keys_required": True,
    }:
        raise ValueError("SafeSlot first-divergence integrity contract changed")
    if dict(config.get("claim_boundary") or {}) != {
        "retrospective_existing_outcomes": True,
        "model_training_allowed": False,
        "threshold_selection_allowed": False,
        "formal_ttf_claim": False,
        "cross_run_v2_timing_comparison_allowed": False,
        "conflict_and_repair_trajectory_diagnostic_allowed": True,
        "known_tail_remains_excluded": True,
        "next_step_only_freezes_safeslot_label_and_interface_design": True,
    }:
        raise ValueError("SafeSlot first-divergence claim boundary changed")
    expected_inputs = {
        "cohort_config",
        "v2_baseline_config",
        "v2_baseline_report",
        "v2_manifest",
        "structpool_manifest",
        "slotpool_manifest",
        "slotpool_structpool_report",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("SafeSlot first-divergence input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    source_path, _source_root, source = load_slotpool_structpool_ttf_config(
        inputs["cohort_config"]
    )
    if source_path != inputs["cohort_config"]:
        raise ValueError("SafeSlot cohort registration resolved unexpectedly")
    expected = _expected_keys(source)
    if len(expected) != 29:
        raise ValueError("SafeSlot first-divergence requires exactly 29 keys")
    if _read_json(inputs["slotpool_structpool_report"]).get("integrity_passed") is not True:
        raise ValueError("SlotPool versus StructPool source integrity did not pass")
    v2_report = _read_json(inputs["v2_baseline_report"])
    if v2_report.get("integrity_passed") is not True:
        raise ValueError("registered V2 source integrity did not pass")
    return path, root, config, expected


def _manifest_index(
    path: Path,
    expected: set[tuple[str, str, int]],
    *,
    allow_superset: bool,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    by_task_seed = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in _read_jsonl(path)
    }
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    expected_task_seeds = {(task, seed) for _group, task, seed in expected}
    if not allow_superset and set(by_task_seed) != expected_task_seeds:
        raise ValueError(f"manifest coverage changed: {path}")
    if expected_task_seeds - set(by_task_seed):
        raise ValueError(f"manifest is missing SafeSlot keys: {path}")
    for group, task, seed in expected:
        row = by_task_seed[(task, seed)]
        if row.get("status") != "ok" or row.get("error") is not None:
            raise ValueError(f"non-ok SafeSlot source row: {task} seed {seed}")
        result[(group, task, seed)] = row
    return result


def _jaccard(left: list[int], right: list[int]) -> float:
    left_set = set(map(int, left))
    right_set = set(map(int, right))
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _trace_summary(collection: Path, row: dict[str, Any]) -> dict[str, Any]:
    trace_path = collection / str(row["trace_file"])
    filesystem_path = _native_filesystem_path(trace_path.resolve())
    if sha256_file(filesystem_path) != str(row["trace_sha256"]):
        raise ValueError(f"trace hash changed: {trace_path}")
    events = read_trace_events(filesystem_path)
    transitions = [event for event in events if event.get("event") == "transition"]
    summary = dict(row["summary"])
    trajectory = list(map(int, summary["conflict_trajectory"]))
    if len(trajectory) != len(transitions) + 1:
        raise ValueError(f"trace trajectory length changed: {trace_path}")
    if int(summary["repair_iterations"]) != len(transitions):
        raise ValueError(f"trace repair count changed: {trace_path}")
    decisions: list[dict[str, Any]] = []
    for index, event in enumerate(transitions):
        controller = dict(event.get("controller") or {})
        proposal = dict(controller.get("proposal") or {})
        pool = list(controller.get("candidate_pool") or ())
        selected_id = str(controller.get("selected_candidate_id") or "")
        selected = next(
            (
                candidate
                for candidate in pool
                if str(candidate.get("candidate_id")) == selected_id
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"selected candidate missing from trace pool: {trace_path}")
        base = [
            candidate
            for candidate in pool
            if not bool(candidate.get("structpool_family_groups"))
        ]
        if not base:
            raise ValueError(f"candidate pool has no V2 base candidate: {trace_path}")
        scored_base = [
            candidate for candidate in base if candidate.get("score") is not None
        ]
        mixed_pool_top_base_id = None
        if scored_base:
            mixed_pool_top_base_id = str(
                sorted(
                    scored_base,
                    key=lambda candidate: (
                        -round(float(candidate["score"]), 12),
                        str(candidate["candidate_id"]),
                    ),
                )[0]["candidate_id"]
            )
        anchor_id = proposal.get("slotpool_v2_anchor_candidate_id")
        anchor = None
        if anchor_id is not None:
            anchor = next(
                (
                    candidate
                    for candidate in pool
                    if str(candidate.get("candidate_id")) == str(anchor_id)
                ),
                None,
            )
            if anchor is None:
                raise ValueError(f"logged SlotPool anchor is absent: {trace_path}")
        before = int(dict(event.get("metrics") or {}).get("conflicts_before", -1))
        after = int(dict(event.get("metrics") or {}).get("conflicts_after", -1))
        if before != trajectory[index] or after != trajectory[index + 1]:
            raise ValueError(f"trace conflict metrics changed: {trace_path}")
        decisions.append(
            {
                "decision_index": index,
                "before_conflicts": before,
                "after_conflicts": after,
                "conflict_reduction": before - after,
                "before_fingerprint": str(event.get("before_fingerprint") or ""),
                "after_fingerprint": str(event.get("after_fingerprint") or ""),
                "pp_random_seed": int(
                    dict(event.get("action") or {}).get("pp_random_seed", -1)
                ),
                "selected_candidate_id": selected_id,
                "selected_score": (
                    float(selected["score"])
                    if selected.get("score") is not None
                    else None
                ),
                "selected_size": int(selected.get("actual_size", 0)),
                "selected_agents": list(map(int, selected.get("agents") or ())),
                "selected_families": list(selected.get("selection_families") or ()),
                "selected_structural": bool(
                    selected.get("structpool_family_groups")
                ),
                "selected_structural_groups": list(
                    selected.get("structpool_family_groups") or ()
                ),
                "base_candidate_ids": [
                    str(candidate["candidate_id"]) for candidate in base
                ],
                "mixed_pool_top_base_candidate_id": mixed_pool_top_base_id,
                "v2_anchor_candidate_id": (
                    str(anchor_id) if anchor_id is not None else None
                ),
            }
        )
    return {
        "initial_conflicts": int(summary["initial_conflicts"]),
        "initial_fingerprint": str(summary["initial_fingerprint"]),
        "success": bool(summary["success"]),
        "repair_iterations": int(summary["repair_iterations"]),
        "raw_wall_ttf_seconds": float(summary["wall_time_to_feasible"]),
        "conflict_trajectory": trajectory,
        "decisions": decisions,
        "trace_sha256": str(row["trace_sha256"]),
    }


def _short_window(
    decisions: list[dict[str, Any]], start: int, length: int
) -> dict[str, Any]:
    selected = decisions[start : start + length]
    if not selected:
        return {
            "observed_repairs": 0,
            "conflict_reduction": 0,
            "strict_progress_repairs": 0,
            "maximum_no_progress_streak": 0,
            "end_conflicts": 0,
        }
    streak = 0
    maximum_streak = 0
    strict_progress = 0
    for decision in selected:
        if int(decision["after_conflicts"]) < int(decision["before_conflicts"]):
            strict_progress += 1
            streak = 0
        else:
            streak += 1
            maximum_streak = max(maximum_streak, streak)
    return {
        "observed_repairs": len(selected),
        "conflict_reduction": int(selected[0]["before_conflicts"])
        - int(selected[-1]["after_conflicts"]),
        "strict_progress_repairs": strict_progress,
        "maximum_no_progress_streak": maximum_streak,
        "end_conflicts": int(selected[-1]["after_conflicts"]),
    }


def first_divergence(
    v2: dict[str, Any], challenger: dict[str, Any], *, short_window: int
) -> dict[str, Any] | None:
    if (
        v2["initial_fingerprint"] != challenger["initial_fingerprint"]
        or v2["initial_conflicts"] != challenger["initial_conflicts"]
    ):
        raise ValueError("initial state differs before first-divergence audit")
    v2_decisions = list(v2["decisions"])
    challenger_decisions = list(challenger["decisions"])
    shared = min(len(v2_decisions), len(challenger_decisions))
    for index in range(shared):
        left = v2_decisions[index]
        right = challenger_decisions[index]
        if (
            left["before_fingerprint"] != right["before_fingerprint"]
            or left["before_conflicts"] != right["before_conflicts"]
        ):
            raise ValueError("state diverged before the first different action")
        if left["selected_candidate_id"] == right["selected_candidate_id"]:
            if (
                left["pp_random_seed"] != right["pp_random_seed"]
                or left["after_fingerprint"] != right["after_fingerprint"]
                or left["after_conflicts"] != right["after_conflicts"]
            ):
                raise ValueError("identical prefix action did not replay identically")
            continue
        if not bool(right["selected_structural"]):
            raise ValueError("first challenger divergence is not structural")
        if left["selected_candidate_id"] not in set(right["base_candidate_ids"]):
            raise ValueError("V2 action is absent from the challenger base pool")
        if (
            right["v2_anchor_candidate_id"] is not None
            and right["v2_anchor_candidate_id"] != left["selected_candidate_id"]
        ):
            raise ValueError(
                "challenger V2 anchor differs from the V2 action: "
                f"{right['v2_anchor_candidate_id']} != "
                f"{left['selected_candidate_id']}"
            )
        v2_window = _short_window(v2_decisions, index, short_window)
        challenger_window = _short_window(
            challenger_decisions, index, short_window
        )
        immediate_advantage = int(right["conflict_reduction"]) - int(
            left["conflict_reduction"]
        )
        window_advantage = int(challenger_window["conflict_reduction"]) - int(
            v2_window["conflict_reduction"]
        )
        terminal_round_delta = int(challenger["repair_iterations"]) - int(
            v2["repair_iterations"]
        )
        return {
            "decision_index": index,
            "before_fingerprint": str(left["before_fingerprint"]),
            "before_conflicts": int(left["before_conflicts"]),
            "v2_action": {
                key: left[key]
                for key in (
                    "selected_candidate_id",
                    "selected_size",
                    "selected_families",
                    "selected_score",
                    "pp_random_seed",
                    "after_conflicts",
                    "conflict_reduction",
                )
            },
            "challenger_action": {
                key: right[key]
                for key in (
                    "selected_candidate_id",
                    "selected_size",
                    "selected_families",
                    "selected_structural_groups",
                    "selected_score",
                    "pp_random_seed",
                    "after_conflicts",
                    "conflict_reduction",
                )
            },
            "immediate_conflict_reduction_advantage": immediate_advantage,
            "short_window_repairs": short_window,
            "v2_short_window": v2_window,
            "challenger_short_window": challenger_window,
            "short_window_conflict_reduction_advantage": window_advantage,
            "v2_terminal_repair_iterations": int(v2["repair_iterations"]),
            "challenger_terminal_repair_iterations": int(
                challenger["repair_iterations"]
            ),
            "terminal_repair_iterations_delta": terminal_round_delta,
            "selected_anchor_jaccard": _jaccard(
                list(right["selected_agents"]), list(left["selected_agents"])
            ),
            "anchor_source": (
                "logged_slotpool_base_only_v2"
                if right["v2_anchor_candidate_id"] is not None
                else "matched_current_runtime_v2_trace"
            ),
            "mixed_pool_top_base_matches_v2": (
                right["mixed_pool_top_base_candidate_id"]
                == left["selected_candidate_id"]
            ),
            "immediate_better_but_terminal_worse": (
                immediate_advantage > 0 and terminal_round_delta > 0
            ),
            "immediate_not_worse_but_terminal_worse": (
                immediate_advantage >= 0 and terminal_round_delta > 0
            ),
            "short_window_better_but_terminal_worse": (
                window_advantage > 0 and terminal_round_delta > 0
            ),
        }
    if len(v2_decisions) != len(challenger_decisions):
        raise ValueError("one trace ended before any different action was recorded")
    if v2["conflict_trajectory"] != challenger["conflict_trajectory"]:
        raise ValueError("no-divergence traces have different conflict trajectories")
    return None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    divergences = [row for row in rows if row["divergence"] is not None]
    relations = Counter()
    family_counts: Counter[str] = Counter()
    for row in divergences:
        divergence = dict(row["divergence"])
        immediate = int(divergence["immediate_conflict_reduction_advantage"])
        terminal = int(divergence["terminal_repair_iterations_delta"])
        relations[f"immediate_{'better' if immediate > 0 else 'worse' if immediate < 0 else 'tie'}"] += 1
        relations[f"terminal_{'better' if terminal < 0 else 'worse' if terminal > 0 else 'tie'}"] += 1
        for family in divergence["challenger_action"]["selected_structural_groups"]:
            family_counts[str(family)] += 1
    terminal_deltas = [
        int(row["divergence"]["terminal_repair_iterations_delta"])
        for row in divergences
    ]
    return {
        "paired_key_count": len(rows),
        "first_divergence_count": len(divergences),
        "no_divergence_count": len(rows) - len(divergences),
        **dict(sorted(relations.items())),
        "immediate_better_but_terminal_worse_count": sum(
            bool(row["divergence"]["immediate_better_but_terminal_worse"])
            for row in divergences
        ),
        "immediate_not_worse_but_terminal_worse_count": sum(
            bool(row["divergence"]["immediate_not_worse_but_terminal_worse"])
            for row in divergences
        ),
        "short_window_better_but_terminal_worse_count": sum(
            bool(row["divergence"]["short_window_better_but_terminal_worse"])
            for row in divergences
        ),
        "mixed_pool_top_base_differs_from_v2_count": sum(
            not bool(row["divergence"]["mixed_pool_top_base_matches_v2"])
            for row in divergences
        ),
        "mean_first_divergence_decision": (
            fmean(int(row["divergence"]["decision_index"]) for row in divergences)
            if divergences
            else None
        ),
        "mean_terminal_repair_iterations_delta": (
            fmean(terminal_deltas) if terminal_deltas else 0.0
        ),
        "selected_structural_group_counts": dict(sorted(family_counts.items())),
    }


def analyze_safeslot_first_divergence(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, root, config, expected = load_safeslot_first_divergence_config(
        config_path
    )
    inputs = {
        name: (root / str(specification["path"])).resolve()
        for name, specification in dict(config["inputs"]).items()
    }
    manifests = {
        "v2-full": inputs["v2_manifest"],
        "v2-plus-structpool": inputs["structpool_manifest"],
        "v2-plus-slotpool": inputs["slotpool_manifest"],
    }
    indices = {
        controller: _manifest_index(
            manifest,
            expected,
            allow_superset=controller == "v2-full",
        )
        for controller, manifest in manifests.items()
    }
    collections = {
        controller: manifest.parent for controller, manifest in manifests.items()
    }
    short_window = int(config["cohort"]["short_window_repairs"])
    comparisons: dict[str, list[dict[str, Any]]] = {
        controller: [] for controller in CHALLENGERS
    }
    errors: list[str] = []
    initial_fingerprints: dict[tuple[str, str, int], str] = {}
    for key in sorted(expected):
        group, task, seed = key
        try:
            traces = {
                controller: _trace_summary(collections[controller], indices[controller][key])
                for controller in ("v2-full", *CHALLENGERS)
            }
            fingerprints = {
                str(trace["initial_fingerprint"]) for trace in traces.values()
            }
            conflicts = {int(trace["initial_conflicts"]) for trace in traces.values()}
            if len(fingerprints) != 1 or len(conflicts) != 1:
                raise ValueError("three-controller initial state mismatch")
            initial_fingerprints[key] = next(iter(fingerprints))
            for controller in CHALLENGERS:
                divergence = first_divergence(
                    traces["v2-full"],
                    traces[controller],
                    short_window=short_window,
                )
                comparisons[controller].append(
                    {
                        "group_id": group,
                        "task_id": task,
                        "solver_seed": seed,
                        "initial_conflicts": int(traces["v2-full"]["initial_conflicts"]),
                        "v2_trace_sha256": traces["v2-full"]["trace_sha256"],
                        "challenger_trace_sha256": traces[controller]["trace_sha256"],
                        "v2_repair_iterations": int(
                            traces["v2-full"]["repair_iterations"]
                        ),
                        "challenger_repair_iterations": int(
                            traces[controller]["repair_iterations"]
                        ),
                        "v2_raw_wall_ttf_seconds_provenance_only": float(
                            traces["v2-full"]["raw_wall_ttf_seconds"]
                        ),
                        "challenger_raw_wall_ttf_seconds_provenance_only": float(
                            traces[controller]["raw_wall_ttf_seconds"]
                        ),
                        "divergence": divergence,
                    }
                )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{group}/{task}/seed-{seed}: {error}")
    summaries = {
        controller: _summary(rows) for controller, rows in comparisons.items()
    }
    disagreement_count = sum(
        int(summary["immediate_not_worse_but_terminal_worse_count"])
        for summary in summaries.values()
    )
    mixed_pool_coupling_count = sum(
        int(summary["mixed_pool_top_base_differs_from_v2_count"])
        for summary in summaries.values()
    )
    integrity_passed = (
        not errors
        and len(initial_fingerprints) == 29
        and all(len(rows) == 29 for rows in comparisons.values())
    )
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": (
            "retrospective_first_divergence_complete_no_training_or_ttf_claim"
        ),
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "integrity_passed": integrity_passed,
        "errors": errors,
        "paired_key_count": len(initial_fingerprints),
        "comparison_count": sum(len(rows) for rows in comparisons.values()),
        "short_window_repairs": short_window,
        "controller_summaries": summaries,
        "comparisons": comparisons,
        "design_findings": {
            "immediate_only_target_is_insufficient": disagreement_count > 0,
            "immediate_not_worse_but_terminal_worse_count": disagreement_count,
            "independent_base_only_v2_anchor_required": (
                mixed_pool_coupling_count > 0
            ),
            "mixed_pool_base_order_change_count": mixed_pool_coupling_count,
            "safe_anchor_relative_gate_required": True,
            "post_state_residual_structure_required": disagreement_count > 0,
            "runtime_short_hazard_monitor_required": disagreement_count > 0,
            "short_horizon_future_as_primary_training_label_allowed": False,
            "uncertain_action_must_abstain_to_v2": True,
            "rollback_is_separate_runtime_safety_not_a_training_label": True,
            "cost_to_go_or_remaining_round_target_allowed": False,
        },
        "claim_boundary": dict(config["claim_boundary"]),
        "input_manifest_sha256": {
            controller: sha256_file(manifest)
            for controller, manifest in manifests.items()
        },
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / REPORT_FILENAME, report)
    return report


__all__ = [
    "CHALLENGERS",
    "REPORT_FILENAME",
    "analyze_safeslot_first_divergence",
    "first_divergence",
    "load_safeslot_first_divergence_config",
]
