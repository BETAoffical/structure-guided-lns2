from __future__ import annotations

import collections
import statistics
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_repairability_collection import _source_target_state
from experiments.online_feature_engine import TopologyAnalysisCache
from experiments.stride_structpool_size_ablation import (
    _best,
    _forbidden_hits,
    state_artifact_tree_sha256,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.topology_candidates import generate_scalepool_candidates


CONFIG_SCHEMA = "lns2.stride.scalepool_registration.v1"
REPORT_SCHEMA = "lns2.stride.scalepool_offline_evaluation.v1"
STATE_SCHEMA = "lns2.stride.scalepool_offline_state.v1"


def validate_scalepool_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected ScalePool registration schema")
    if config.get("implementation_id") != "stride-scalepool-v1":
        raise ValueError("unexpected ScalePool implementation id")
    policy = dict(config.get("size_policy") or {})
    if policy.get("allowed_sizes") != [8, 16, 24, 32]:
        raise ValueError("ScalePool requires the four preregistered sizes")
    if int(policy.get("maximum_candidates", 0)) != 6:
        raise ValueError("ScalePool requires six maximum candidates")
    if float(policy.get("jaccard_limit", -1.0)) != 0.8:
        raise ValueError("ScalePool candidate Jaccard limit drifted")
    if float(policy.get("anchor_jaccard_limit", -1.0)) != 0.9:
        raise ValueError("ScalePool anchor Jaccard limit drifted")
    if policy.get("exact_duplicate_action") != "merge_provenance":
        raise ValueError("ScalePool duplicate behavior drifted")
    if policy.get("fallback_generation") != "generate_next_size_only_after_rejection":
        raise ValueError("ScalePool lazy fallback behavior drifted")
    gates = dict(config.get("offline_acceptance") or {})
    expected_gates = {
        "global_best_retention_minimum": 0.9,
        "mean_normalized_regret_maximum": 0.02,
        "maximum_map_or_topology_group_mean_regret": 0.05,
        "first_fixed_half_best_retention_minimum": 0.85,
        "second_fixed_half_best_retention_minimum": 0.85,
        "raw_candidate_count_must_be_lower_than_full_grid": True,
    }
    if gates != expected_gates:
        raise ValueError("ScalePool offline acceptance gates drifted")
    boundary = dict(config.get("claim_boundary") or {})
    if any(
        bool(boundary.get(name))
        for name in (
            "runtime_integration_before_acceptance",
            "formal_ttf_claim",
            "future_trajectory_read",
            "cost_to_go_read",
            "known_maze_result_used_for_parameters",
        )
    ):
        raise ValueError("ScalePool claim boundary drifted")
    if project_root is not None:
        grid = dict(config["inputs"]["stage2_grid"])
        grid_root = (project_root / str(grid["path"])).resolve()
        if sha256_file(grid_root / "grid_manifest.jsonl") != str(
            grid["manifest_sha256"]
        ):
            raise ValueError("ScalePool registered Stage 2 grid manifest changed")
        if state_artifact_tree_sha256(grid_root / "states") != str(
            grid["state_artifact_tree_sha256"]
        ):
            raise ValueError("ScalePool registered Stage 2 grid tree changed")


def _mean_by(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(float(row["normalized_regret"]))
    return {
        key: statistics.fmean(values) for key, values in sorted(grouped.items())
    }


def summarize_scalepool_acceptance(
    *,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    full_raw_candidate_count: int,
    scalepool_raw_candidate_count: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("ScalePool acceptance requires state rows")
    map_regret = _mean_by(rows, "map_id")
    topology_regret = _mean_by(rows, "layout_mode")
    gates = dict(config["offline_acceptance"])
    metrics = {
        "global_best_retention": statistics.fmean(
            bool(row["global_best_retained"]) for row in rows
        ),
        "mean_normalized_regret": statistics.fmean(
            float(row["normalized_regret"]) for row in rows
        ),
        "maximum_map_mean_regret": max(map_regret.values()),
        "maximum_topology_group_mean_regret": max(topology_regret.values()),
        "first_fixed_half_best_retention": statistics.fmean(
            bool(row["first_fixed_half_best_retained"]) for row in rows
        ),
        "second_fixed_half_best_retention": statistics.fmean(
            bool(row["second_fixed_half_best_retained"]) for row in rows
        ),
        "full_raw_candidate_count": int(full_raw_candidate_count),
        "scalepool_raw_candidate_count": int(scalepool_raw_candidate_count),
        "raw_candidate_reduction": int(full_raw_candidate_count)
        - int(scalepool_raw_candidate_count),
        "map_mean_regret": map_regret,
        "topology_group_mean_regret": topology_regret,
    }
    checks = {
        "global_best_retention": metrics["global_best_retention"]
        >= float(gates["global_best_retention_minimum"]),
        "mean_normalized_regret": metrics["mean_normalized_regret"]
        <= float(gates["mean_normalized_regret_maximum"]),
        "map_group_regret": metrics["maximum_map_mean_regret"]
        <= float(gates["maximum_map_or_topology_group_mean_regret"]),
        "topology_group_regret": metrics["maximum_topology_group_mean_regret"]
        <= float(gates["maximum_map_or_topology_group_mean_regret"]),
        "first_fixed_half_best_retention": metrics[
            "first_fixed_half_best_retention"
        ]
        >= float(gates["first_fixed_half_best_retention_minimum"]),
        "second_fixed_half_best_retention": metrics[
            "second_fixed_half_best_retention"
        ]
        >= float(gates["second_fixed_half_best_retention_minimum"]),
        "raw_candidate_count_reduced": int(scalepool_raw_candidate_count)
        < int(full_raw_candidate_count),
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def evaluate_scalepool(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_scalepool_config(config, project_root=project_root)
    grid_root = (project_root / config["inputs"]["stage2_grid"]["path"]).resolve()
    labels_root = (
        project_root / config["inputs"]["stage2_labels"]["path"]
    ).resolve()
    grid_report = _read_json(grid_root / "grid_report.json")
    label_report = _read_json(labels_root / "collection_report.json")
    if grid_report.get("passed") is not True:
        raise ValueError("ScalePool evaluation requires a complete Stage 2 grid")
    if label_report.get("passed") is not True:
        raise ValueError("ScalePool evaluation requires complete Stage 2 labels")
    if label_report.get("label_matrix_validation", {}).get("passed") is not True:
        raise ValueError("ScalePool evaluation requires strict label matrix validation")
    manifests = _read_jsonl(grid_root / "grid_manifest.jsonl")
    aggregates = _read_jsonl(labels_root / "candidate_aggregates.jsonl")
    aggregate_by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in aggregates:
        aggregate_by_state[str(row["state_id"])].append(row)

    state_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    full_raw_count = 0
    scale_raw_count = 0
    for manifest in manifests:
        grid_state = _read_json(Path(str(manifest["state_file"])))
        decision = dict(grid_state["decision"])
        state, _source_manifest, _trace = _source_target_state(decision)
        if (
            state_fingerprint(state) != str(grid_state["before_fingerprint"])
            or repair_structure_fingerprint(state)
            != str(grid_state["before_repair_fingerprint"])
        ):
            raise RuntimeError("ScalePool restored state differs from Stage 2 grid")
        cache = TopologyAnalysisCache(state, backend="native")
        if cache.analysis is None:
            raise RuntimeError("ScalePool native topology analysis is missing")
        generated = generate_scalepool_candidates(
            state,
            cache.analysis,
            v2_anchor_agents=grid_state["v2_base_anchor"]["agents"],
            allowed_sizes=config["size_policy"]["allowed_sizes"],
            maximum_candidates=int(config["size_policy"]["maximum_candidates"]),
            maximum_jaccard_similarity=float(
                config["size_policy"]["jaccard_limit"]
            ),
            maximum_anchor_jaccard_similarity=float(
                config["size_policy"]["anchor_jaccard_limit"]
            ),
        )
        state_id = str(grid_state["state_id"])
        full = aggregate_by_state[state_id]
        by_id = {str(row["candidate_id"]): row for row in full}
        selected_ids = [str(row["candidate_id"]) for row in generated.candidates]
        if not selected_ids or any(candidate_id not in by_id for candidate_id in selected_ids):
            raise RuntimeError("ScalePool generated candidate is absent from full grid")
        full_grid_agents = {
            str(row["candidate_id"]): tuple(map(int, row["agents"]))
            for row in grid_state["candidates"]
        }
        for candidate in generated.candidates:
            candidate_id = str(candidate["candidate_id"])
            if tuple(map(int, candidate["agents"])) != full_grid_agents[candidate_id]:
                raise RuntimeError("ScalePool candidate agents differ from full grid")
        selected = [by_id[candidate_id] for candidate_id in selected_ids]
        best = _best(full)
        selected_best = _best(selected)
        first_best = _best(full, key="first_fixed_half_mean")
        second_best = _best(full, key="second_fixed_half_mean")
        state_rows.append(
            {
                "schema": STATE_SCHEMA,
                "state_id": state_id,
                "map_id": str(decision["map_id"]),
                "layout_mode": str(decision["layout_mode"]),
                "agent_band": str(decision["agent_band"]),
                "conflict_band": str(decision["conflict_band"]),
                "full_candidate_count": len(full),
                "scalepool_candidate_count": len(selected),
                "full_raw_candidate_count": int(grid_state["raw_draft_count"]),
                "scalepool_raw_candidate_count": generated.raw_candidate_count,
                "selected_candidate_ids": selected_ids,
                "global_best_candidate_id": str(best["candidate_id"]),
                "scalepool_best_candidate_id": str(selected_best["candidate_id"]),
                "global_best_retained": str(best["candidate_id"]) in selected_ids,
                "first_fixed_half_best_retained": str(first_best["candidate_id"])
                in selected_ids,
                "second_fixed_half_best_retained": str(second_best["candidate_id"])
                in selected_ids,
                "normalized_regret": float(best["seed_mean"])
                - float(selected_best["seed_mean"]),
            }
        )
        for attempt in generated.attempts:
            attempt_rows.append({"state_id": state_id, **attempt})
        full_raw_count += int(grid_state["raw_draft_count"])
        scale_raw_count += generated.raw_candidate_count

    if _forbidden_hits([state_rows, attempt_rows]):
        raise RuntimeError("forbidden future or runtime field entered ScalePool evaluation")
    acceptance = summarize_scalepool_acceptance(
        rows=state_rows,
        config=config,
        full_raw_candidate_count=full_raw_count,
        scalepool_raw_candidate_count=scale_raw_count,
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    states_path = output / "state_evaluation.jsonl"
    attempts_path = output / "candidate_attempts.jsonl"
    _write_jsonl(states_path, state_rows)
    _write_jsonl(attempts_path, attempt_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": (
            "offline_acceptance_passed" if acceptance["passed"]
            else "offline_acceptance_failed"
        ),
        "implementation_id": "stride-scalepool-v1",
        "state_count": len(state_rows),
        "map_count": len({row["map_id"] for row in state_rows}),
        "known_maze_long_tail_included": False,
        "candidate_generation_outcome_blind": True,
        "formal_ttf_claim": False,
        "acceptance": acceptance,
        "runtime_integration_allowed": bool(acceptance["passed"]),
        "producer": producer_identity(
            project_root=project_root,
            source_files=(
                "experiments/stride_scalepool_evaluation.py",
                "lns2_selector/runtime/topology_candidates.py",
                "src/online_features.cpp",
                "src/python_bindings.cpp",
            ),
            native_required=True,
            package_names=("numpy",),
        ),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "grid_manifest_sha256": sha256_file(grid_root / "grid_manifest.jsonl"),
            "label_report_sha256": sha256_file(labels_root / "collection_report.json"),
            "candidate_aggregates_sha256": sha256_file(
                labels_root / "candidate_aggregates.jsonl"
            ),
        },
        "artifacts": {
            "state_evaluation_sha256": sha256_file(states_path),
            "candidate_attempts_sha256": sha256_file(attempts_path),
        },
    }
    _write_json(output / "scalepool_evaluation_report.json", report)
    return report
