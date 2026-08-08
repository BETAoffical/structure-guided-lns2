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
    _family_variant,
    _forbidden_hits,
    state_artifact_tree_sha256,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.topology_candidates import generate_scalepool_candidates


CONFIG_SCHEMA = "lns2.stride.scalepool_registration.v1"
REPORT_SCHEMA = "lns2.stride.scalepool_offline_evaluation.v1"
STATE_SCHEMA = "lns2.stride.scalepool_offline_state.v1"
LABEL_AUDIT_SCHEMA = "lns2.stride.structpool_size_label_audit.v1"


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


def _validate_external_label_audit(
    *,
    label_report: dict[str, Any],
    audit_report: dict[str, Any],
    observed_artifacts: dict[str, str],
) -> None:
    if audit_report.get("schema") != LABEL_AUDIT_SCHEMA:
        raise ValueError("ScalePool label audit schema changed")
    if audit_report.get("passed") is not True:
        raise ValueError("ScalePool label audit did not pass")
    if audit_report.get("source_artifacts_modified") is not False:
        raise ValueError("ScalePool label audit modified source artifacts")
    if audit_report.get("matrix_validation", {}).get("passed") is not True:
        raise ValueError("ScalePool label audit lacks strict matrix validation")
    if audit_report.get("source_run_fingerprint") != label_report.get(
        "run_fingerprint"
    ):
        raise ValueError("ScalePool label audit run fingerprint differs")
    for field in ("state_count", "candidate_count", "trial_count"):
        source_field = (
            "completed_state_count" if field == "state_count" else field
        )
        if int(audit_report.get(field, -1)) != int(label_report.get(source_field, -2)):
            raise ValueError(f"ScalePool label audit {field} differs")
    audited_artifacts = dict(audit_report.get("source_artifacts") or {})
    registered_artifacts = dict(label_report.get("artifacts") or {})
    if audited_artifacts != observed_artifacts:
        raise ValueError("ScalePool label audit source artifact hashes differ")
    if any(registered_artifacts.get(name) != value for name, value in observed_artifacts.items()):
        raise ValueError("ScalePool label report artifact hashes differ")


def evaluate_scalepool(
    *,
    config_path: str | Path,
    output: str | Path,
    label_audit: str | Path | None = None,
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
    observed_label_artifacts = {
        "repair_trials_sha256": sha256_file(labels_root / "repair_trials.jsonl"),
        "candidate_aggregates_sha256": sha256_file(
            labels_root / "candidate_aggregates.jsonl"
        ),
        "state_manifest_sha256": sha256_file(labels_root / "state_manifest.jsonl"),
        "state_artifact_tree_sha256": state_artifact_tree_sha256(
            labels_root / "states"
        ),
    }
    label_audit_path: Path | None = None
    if label_report.get("label_matrix_validation", {}).get("passed") is not True:
        if label_audit is None:
            raise ValueError(
                "ScalePool evaluation requires strict label matrix validation"
            )
        label_audit_path = Path(label_audit).resolve()
        _validate_external_label_audit(
            label_report=label_report,
            audit_report=_read_json(label_audit_path),
            observed_artifacts=observed_label_artifacts,
        )
    manifests = _read_jsonl(grid_root / "grid_manifest.jsonl")
    aggregates = _read_jsonl(labels_root / "candidate_aggregates.jsonl")
    aggregate_by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in aggregates:
        aggregate_by_state[str(row["state_id"])].append(row)

    state_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    support_alignment_rows: list[dict[str, Any]] = []
    rejection_reason_counts: collections.Counter[str] = collections.Counter()
    miss_attribution: collections.Counter[str] = collections.Counter()
    selected_size_counts: dict[str, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    global_best_size_counts: dict[str, collections.Counter[int]] = (
        collections.defaultdict(collections.Counter)
    )
    uniform_size_rows: dict[int, list[dict[str, Any]]] = {
        size: [] for size in config["size_policy"]["allowed_sizes"]
    }
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
        selected_by_variant: dict[str, set[int]] = collections.defaultdict(set)
        attempts_by_variant: dict[str, list[dict[str, Any]]] = collections.defaultdict(
            list
        )
        for attempt in generated.attempts:
            variant = str(attempt["family_variant"])
            attempts_by_variant[variant].append(attempt)
            if attempt["decision"] == "selected":
                size = int(attempt["attempted_size"])
                selected_by_variant[variant].add(size)
                selected_size_counts[variant][size] += 1
            else:
                rejection_reason_counts[str(attempt["rejection_reason"])] += 1

        best_variant_sizes = {
            (variant, size)
            for family in best["selection_families"]
            for variant, size, _group in [_family_variant(str(family))]
        }
        for variant, size in sorted(best_variant_sizes):
            global_best_size_counts[variant][size] += 1
            variant_attempts = attempts_by_variant.get(variant, [])
            support_alignment_rows.append(
                {
                    "state_id": state_id,
                    "map_id": str(decision["map_id"]),
                    "layout_mode": str(decision["layout_mode"]),
                    "family_variant": variant,
                    "support_count": (
                        int(variant_attempts[0]["support_count"])
                        if variant_attempts
                        else None
                    ),
                    "global_best_nominal_size": size,
                    "selected_nominal_sizes": sorted(selected_by_variant.get(variant, set())),
                    "global_best_retained": str(best["candidate_id"]) in selected_ids,
                }
            )
        if str(best["candidate_id"]) not in selected_ids:
            if any(
                size in selected_by_variant.get(variant, set())
                for variant, size in best_variant_sizes
            ):
                miss_attribution["same_nominal_size_different_agent_set"] += 1
            elif any(variant in selected_by_variant for variant, _size in best_variant_sizes):
                miss_attribution["support_nearest_size_mismatch"] += 1
            else:
                matching_rejections = [
                    str(attempt["rejection_reason"])
                    for variant, size in best_variant_sizes
                    for attempt in attempts_by_variant.get(variant, [])
                    if int(attempt["attempted_size"]) == size
                    and attempt["decision"] == "rejected"
                ]
                if matching_rejections:
                    miss_attribution[
                        "best_size_rejected:" + sorted(matching_rejections)[0]
                    ] += 1
                else:
                    miss_attribution["best_family_or_size_not_generated"] += 1

        for nominal_size, diagnostic_rows in uniform_size_rows.items():
            uniform_ids = {
                str(row["candidate_id"])
                for row in full
                if any(
                    _family_variant(str(family))[1] == nominal_size
                    for family in row["selection_families"]
                )
            }
            uniform_candidates = [by_id[candidate_id] for candidate_id in uniform_ids]
            if not uniform_candidates:
                continue
            uniform_best = _best(uniform_candidates)
            diagnostic_rows.append(
                {
                    "state_id": state_id,
                    "candidate_count": len(uniform_candidates),
                    "global_best_retained": str(best["candidate_id"]) in uniform_ids,
                    "normalized_regret": float(best["seed_mean"])
                    - float(uniform_best["seed_mean"]),
                }
            )
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
    uniform_size_diagnostics = {
        str(size): {
            "state_count": len(rows),
            "mean_candidate_count": statistics.fmean(
                int(row["candidate_count"]) for row in rows
            ),
            "global_best_retention": statistics.fmean(
                bool(row["global_best_retained"]) for row in rows
            ),
            "mean_normalized_regret": statistics.fmean(
                float(row["normalized_regret"]) for row in rows
            ),
            "maximum_normalized_regret": max(
                float(row["normalized_regret"]) for row in rows
            ),
        }
        for size, rows in sorted(uniform_size_rows.items())
        if rows
    }
    failure_diagnostics = {
        "miss_attribution": dict(sorted(miss_attribution.items())),
        "attempt_rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "selected_size_counts_by_variant": {
            variant: {str(size): count for size, count in sorted(counts.items())}
            for variant, counts in sorted(selected_size_counts.items())
        },
        "global_best_size_counts_by_variant": {
            variant: {str(size): count for size, count in sorted(counts.items())}
            for variant, counts in sorted(global_best_size_counts.items())
        },
        "uniform_size_pool_diagnostics": uniform_size_diagnostics,
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    states_path = output / "state_evaluation.jsonl"
    attempts_path = output / "candidate_attempts.jsonl"
    support_alignment_path = output / "support_size_alignment.jsonl"
    _write_jsonl(states_path, state_rows)
    _write_jsonl(attempts_path, attempt_rows)
    _write_jsonl(support_alignment_path, support_alignment_rows)
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
        "failure_diagnostics": failure_diagnostics,
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
            "label_audit_sha256": (
                sha256_file(label_audit_path) if label_audit_path is not None else None
            ),
            "candidate_aggregates_sha256": sha256_file(
                labels_root / "candidate_aggregates.jsonl"
            ),
        },
        "artifacts": {
            "state_evaluation_sha256": sha256_file(states_path),
            "candidate_attempts_sha256": sha256_file(attempts_path),
            "support_size_alignment_sha256": sha256_file(support_alignment_path),
        },
    }
    _write_json(output / "scalepool_evaluation_report.json", report)
    return report
