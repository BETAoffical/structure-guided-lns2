from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json


CONFIG_SCHEMA = "lns2.stride.guardpool_registration.v1"
AUDIT_SCHEMA = "lns2.stride.guardpool_threshold_audit.v1"


def _registered(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"registered GuardPool input is missing: {path}")
    observed = sha256_file(path)
    if observed != str(specification["sha256"]):
        raise ValueError(
            f"registered GuardPool input changed: {path}: "
            f"expected {specification['sha256']}, got {observed}"
        )
    return path


def validate_guardpool_registration(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("controller_id") != "stride-guardpool-v1"
        or config.get("candidate_pool_id") != "stride-slotpool-v1"
        or config.get("scientific_status")
        != "preregistered_runtime_design_after_slotpool_fresh_pass"
    ):
        raise ValueError("GuardPool registration identity changed")
    audit = dict(config.get("stall_threshold_audit") or {})
    if (
        tuple(map(int, audit.get("candidate_limits") or ())) != (5, 8, 12)
        or audit.get("false_trigger_unit")
        != "first_threshold_crossing_per_episode_divided_by_all_v2_repair_decisions"
        or float(audit.get("maximum_false_trigger_rate", -1.0)) != 0.01
        or int(audit.get("expected_episode_count", -1)) != 29
        or int(audit.get("expected_decision_count", -1)) != 402
        or dict(audit.get("expected_trigger_counts") or {})
        != {"5": 5, "8": 3, "12": 1}
        or int(audit.get("selected_no_progress_limit", -1)) != 8
        or audit.get("known_regression_result_read") is not False
    ):
        raise ValueError("GuardPool stall-threshold contract changed")
    runtime = dict(config.get("slotpool_runtime") or {})
    if (
        tuple(map(int, runtime.get("allowed_sizes") or ())) != (8, 16, 24, 32)
        or int(runtime.get("maximum_challengers", -1)) != 6
        or runtime.get("ranking")
        != "frozen_pairwise_mean_win_probability_borda"
        or runtime.get("base_pool") != "frozen_v2_exact_base_candidates"
        or runtime.get("base_anchor") != "frozen_v2_winner_over_base_pool"
        or runtime.get("final_ranker") != "frozen_v2_over_base_plus_slotpool"
        or runtime.get("additional_jaccard_filter") is not False
    ):
        raise ValueError("GuardPool SlotPool runtime contract changed")
    guard = dict(config.get("stall_guard") or {})
    if guard != {
        "no_progress_limit": 8,
        "recovery_controller": "v2-full",
        "release_condition": "strict_conflict_decrease",
        "tabu_scope": "last_structural_candidate_under_conflict_signature",
        "wall_time_condition": None,
    }:
        raise ValueError("GuardPool stall guard changed")
    regression = dict(config.get("known_maze_regression") or {})
    if regression != {
        "task_id": "maze-128-128-1__derived_opposite_exchange__task_seed_0233__agents_0100",
        "solver_seed": 3,
        "initial_conflicts": 66,
        "v2_repair_iterations": 15,
        "maximum_guardpool_repair_iterations": 30,
        "treatments": [
            "v2-full",
            "v2-plus-structpool",
            "v2-plus-slotpool",
            "stride-guardpool-v1",
        ],
        "slotpool_without_guard_is_ablation_only": True,
        "used_for_parameter_selection": False,
        "run_only_after_runtime_semantics_tests": True,
    }:
        raise ValueError("GuardPool known-Maze regression contract changed")
    boundary = dict(config.get("claim_boundary") or {})
    if any(
        bool(boundary.get(name))
        for name in (
            "slotpool_model_retraining",
            "v2_ranker_changed",
            "pp_or_sipps_changed",
            "time_limit_guard",
            "formal_ttf_claim_before_paired_evaluation",
            "default_replacement_allowed",
        )
    ):
        raise ValueError("GuardPool claim boundary changed")
    if project_root is not None:
        inputs = dict(config.get("inputs") or {})
        for name in ("slotpool_fresh_report", "slotpool_model", "frozen_v2_manifest"):
            _registered(project_root.resolve(), dict(inputs[name]))
        for specification in inputs["historical_v2_manifests"]:
            _registered(project_root.resolve(), dict(specification))


def audit_guardpool_threshold(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_guardpool_registration(config, project_root=project_root)
    confirmation = _read_json(
        _registered(project_root, dict(config["inputs"]["slotpool_fresh_report"]))
    )
    if (
        confirmation.get("runtime_integration_allowed") is not True
        or confirmation.get("model_retrained") is not False
        or confirmation.get("development_map_overlap") != []
    ):
        raise ValueError("GuardPool requires the passed frozen fresh-map confirmation")
    excluded = dict(config["stall_threshold_audit"]["exclude_known_regression_key"])
    rows = []
    for specification in config["inputs"]["historical_v2_manifests"]:
        rows.extend(_read_jsonl(_registered(project_root, dict(specification))))
    rows = [
        row
        for row in rows
        if not (
            str(row["task_id"]) == str(excluded["task_id"])
            and int(row["solver_seed"]) == int(excluded["solver_seed"])
        )
    ]
    if any(
        row.get("status") != "ok"
        or row.get("summary", {}).get("controller_mode") != "v2-full"
        for row in rows
    ):
        raise ValueError("GuardPool historical V2 audit contains invalid episodes")
    decision_count = sum(
        len(row["summary"]["conflict_trajectory"]) - 1 for row in rows
    )
    trigger_counts = collections.Counter()
    episode_maxima = []
    for row in rows:
        trajectory = list(map(int, row["summary"]["conflict_trajectory"]))
        streak = 0
        maximum = 0
        for before, after in zip(trajectory, trajectory[1:]):
            streak = 0 if after < before else streak + 1
            maximum = max(maximum, streak)
        episode_maxima.append(maximum)
        for limit in config["stall_threshold_audit"]["candidate_limits"]:
            trigger_counts[int(limit)] += int(maximum >= int(limit))
    rates = {
        str(limit): trigger_counts[int(limit)] / max(1, decision_count)
        for limit in config["stall_threshold_audit"]["candidate_limits"]
    }
    expected = dict(config["stall_threshold_audit"])
    checks = {
        "episode_count": len(rows) == int(expected["expected_episode_count"]),
        "decision_count": decision_count == int(expected["expected_decision_count"]),
        "trigger_counts": {
            str(limit): trigger_counts[int(limit)]
            for limit in expected["candidate_limits"]
        }
        == dict(expected["expected_trigger_counts"]),
        "selected_is_lowest_passing_limit": next(
            (
                int(limit)
                for limit in expected["candidate_limits"]
                if rates[str(limit)] <= float(expected["maximum_false_trigger_rate"])
            ),
            None,
        )
        == int(expected["selected_no_progress_limit"]),
        "known_regression_excluded": not any(
            str(row["task_id"]) == str(excluded["task_id"])
            and int(row["solver_seed"]) == int(excluded["solver_seed"])
            for row in rows
        ),
    }
    report = {
        "schema": AUDIT_SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "episode_count": len(rows),
        "decision_count": decision_count,
        "trigger_counts": {
            str(limit): trigger_counts[int(limit)]
            for limit in expected["candidate_limits"]
        },
        "false_trigger_rates": rates,
        "selected_no_progress_limit": int(expected["selected_no_progress_limit"]),
        "maximum_episode_streak": max(episode_maxima, default=0),
        "known_regression_result_read": False,
        "formal_ttf_claim": False,
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "guardpool_threshold_audit.json", report)
    return report


__all__ = ["audit_guardpool_threshold", "validate_guardpool_registration"]
