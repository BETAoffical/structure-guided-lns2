from __future__ import annotations

import datetime as dt
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.state_analysis import summarize_initial_state_complexity


STRIDE_RESEARCH_LINE = "stride-lns"
STRIDE_CONTROL_CONTROLLER_ID = "stride-control-v1"
STRIDE_QUALITY_CONTROLLER_ID = "stride-quality-v1"
STRIDE_LABEL_SCHEMA = "lns2.stride.quality_label.v1"
STRIDE_STAGE1_CONFIG_SCHEMA = "lns2.stride.stage1_config.v1"
STRIDE_STAGE1_REPORT_SCHEMA = "lns2.stride.stage1_audit.v1"
STRIDE_TRIAL_SCHEMA = "lns2.stride.repair_trial.v1"
STRIDE_CANDIDATE_SCHEMA = "lns2.stride.candidate_aggregate.v1"
FROZEN_FEATURE_SCHEMA_ID = "lns2.realized_features.v2"
FROZEN_FEATURE_DIMENSION = 124

REQUIRED_POST_STRUCTURE_FIELDS = frozenset(
    {
        "post_largest_component_ratio",
        "post_conflict_edge_density",
        "post_event_density",
        "post_degree_concentration",
    }
)


def validate_post_structure_metrics(value: Any) -> dict[str, float]:
    """Return canonical finite, nonnegative STRIDE post-structure metrics."""

    if not isinstance(value, dict) or set(value) != REQUIRED_POST_STRUCTURE_FIELDS:
        raise ValueError("STRIDE requires the exact post-structure metric schema")
    metrics: dict[str, float] = {}
    for name in REQUIRED_POST_STRUCTURE_FIELDS:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("post-structure metrics must be numeric")
        metric = float(raw)
        if not math.isfinite(metric) or metric < 0.0:
            raise ValueError("post-structure metrics must be finite and nonnegative")
        metrics[name] = metric
    return metrics


def post_structure_metrics(state: dict[str, Any]) -> dict[str, float]:
    """Compute STRIDE post-repair difficulty metrics normalized by all agents."""

    summary = summarize_initial_state_complexity(state)
    agent_count = int(summary["agent_count"])
    edges = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in state.get("conflict_edges", [])
    }
    degree: Counter[int] = Counter()
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    return {
        "post_largest_component_ratio": float(
            summary["largest_conflict_component_ratio"]
        ),
        "post_conflict_edge_density": len(edges) / agent_count,
        "post_event_density": int(summary["conflict_event_count"]) / agent_count,
        "post_degree_concentration": max(degree.values(), default=0) / agent_count,
    }


def aggregate_stride_candidate(
    *, before_conflicts: int, outcomes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate exactly four paired PP outcomes for one candidate."""

    if type(before_conflicts) is not int or before_conflicts <= 0:
        raise ValueError("STRIDE before_conflicts must be a positive integer")
    if len(outcomes) != 4:
        raise ValueError("STRIDE Pilot requires exactly four paired PP outcomes")
    seeds: set[int] = set()
    reductions: list[float] = []
    structures: list[dict[str, float]] = []
    feasible_count = 0
    progress_count = 0
    for outcome in outcomes:
        if type(outcome.get("pp_seed")) is not int:
            raise ValueError("each STRIDE outcome requires an integer pp_seed")
        seed = int(outcome["pp_seed"])
        if seed in seeds:
            raise ValueError("paired PP outcomes must use four distinct seeds")
        seeds.add(seed)
        if type(outcome.get("feasible")) is not bool:
            raise ValueError("each STRIDE outcome requires a strict feasible boolean")
        if type(outcome.get("conflicts_after")) is not int:
            raise ValueError("each STRIDE outcome requires integer conflicts_after")
        conflicts_after = int(outcome["conflicts_after"])
        if conflicts_after < 0:
            raise ValueError("conflicts_after must be nonnegative")
        reduction = float(before_conflicts - conflicts_after)
        reductions.append(reduction)
        feasible_count += int(outcome["feasible"])
        progress_count += int(conflicts_after < before_conflicts)
        structures.append(
            validate_post_structure_metrics(outcome.get("post_structure"))
        )

    two_worst = sorted(reductions)[:2]
    return {
        "trial_count": 4,
        "pp_seeds": sorted(seeds),
        "feasible_rate": feasible_count / 4.0,
        "progress_rate": progress_count / 4.0,
        "mean_conflicts_after": statistics.fmean(
            before_conflicts - reduction for reduction in reductions
        ),
        "mean_conflict_reduction": statistics.fmean(reductions),
        "robust_reduction": statistics.fmean(two_worst),
        "mean_post_structure": {
            name: statistics.fmean(item[name] for item in structures)
            for name in sorted(REQUIRED_POST_STRUCTURE_FIELDS)
        },
    }


def assign_structure_scores(candidates: list[dict[str, Any]]) -> None:
    """Attach the mean within-state percentile of the four post metrics."""

    if not candidates:
        raise ValueError("cannot score an empty candidate pool")
    count = len(candidates)
    validated = [
        validate_post_structure_metrics(candidate.get("mean_post_structure"))
        for candidate in candidates
    ]
    for candidate_index, candidate in enumerate(candidates):
        percentiles: list[float] = []
        for name in sorted(REQUIRED_POST_STRUCTURE_FIELDS):
            current = validated[candidate_index][name]
            if count == 1:
                percentile = 0.0
            else:
                lower = sum(
                    other[name] < current for other in validated
                )
                equal_other = sum(
                    other[name] == current for other in validated
                ) - 1
                percentile = (lower + 0.5 * equal_other) / (count - 1)
            percentiles.append(percentile)
        candidate["structural_score"] = statistics.fmean(percentiles)


def stride_dominates(
    left: dict[str, Any], right: dict[str, Any], *, before_conflicts: int
) -> bool:
    """Return whether left quality-dominates right under the STRIDE V1 label."""

    epsilon = 1e-12
    reduction_tolerance = max(1.0, 0.02 * float(before_conflicts))
    left_feasible = float(left["feasible_rate"])
    right_feasible = float(right["feasible_rate"])
    left_progress = float(left["progress_rate"])
    right_progress = float(right["progress_rate"])
    left_reduction = float(left["robust_reduction"])
    right_reduction = float(right["robust_reduction"])
    left_structure = float(left["structural_score"])
    right_structure = float(right["structural_score"])
    noninferior = (
        left_feasible + epsilon >= right_feasible
        and left_progress + epsilon >= right_progress
        and left_reduction + reduction_tolerance + epsilon >= right_reduction
        and left_structure <= right_structure + epsilon
    )
    strict = (
        left_feasible > right_feasible + epsilon
        or left_progress > right_progress + epsilon
        or left_reduction > right_reduction + reduction_tolerance + epsilon
        or left_structure + 0.05 <= right_structure + epsilon
    )
    return noninferior and strict


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_stride_labels(*, trials_path: Path, output: Path) -> dict[str, Any]:
    """Build state-balanced STRIDE dominance pairs from paired repair trials."""

    state_metadata: dict[str, dict[str, Any]] = {}
    candidate_rows: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    candidate_features: dict[tuple[str, str], dict[str, float]] = {}
    seen_trials: set[tuple[str, str, int]] = set()
    for row in _read_jsonl(trials_path):
        if row.get("schema") != STRIDE_TRIAL_SCHEMA:
            raise ValueError(
                f"expected trial schema {STRIDE_TRIAL_SCHEMA}, found {row.get('schema')}"
            )
        state_id = str(row["state_id"])
        candidate_id = str(row["candidate_id"])
        pp_seed = row.get("pp_seed")
        if type(pp_seed) is not int:
            raise ValueError("trial pp_seed must be an integer")
        trial_key = (state_id, candidate_id, int(pp_seed))
        if trial_key in seen_trials:
            raise ValueError(f"duplicate STRIDE trial: {trial_key}")
        seen_trials.add(trial_key)
        metadata = {
            "map_id": str(row["map_id"]),
            "split": str(row["split"]),
            "source_policy": str(row["source_policy"]),
            "decision_stage": str(row["decision_stage"]),
            "before_conflicts": int(row["before_conflicts"]),
            "agent_count": int(row["agent_count"]),
        }
        if state_id in state_metadata and state_metadata[state_id] != metadata:
            raise ValueError(f"inconsistent state metadata: {state_id}")
        state_metadata[state_id] = metadata
        features = row.get("features")
        if not isinstance(features, dict) or len(features) != FROZEN_FEATURE_DIMENSION:
            raise ValueError(
                f"STRIDE trials require exactly {FROZEN_FEATURE_DIMENSION} features"
            )
        normalized_features = {str(name): float(value) for name, value in features.items()}
        key = (state_id, candidate_id)
        if key in candidate_features and candidate_features[key] != normalized_features:
            raise ValueError(f"candidate features changed across PP seeds: {key}")
        candidate_features[key] = normalized_features
        candidate_rows[key].append(row)

    candidates_by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for (state_id, candidate_id), rows in sorted(candidate_rows.items()):
        metadata = state_metadata[state_id]
        aggregate = aggregate_stride_candidate(
            before_conflicts=int(metadata["before_conflicts"]),
            outcomes=rows,
        )
        aggregate.update(
            {
                "schema": STRIDE_CANDIDATE_SCHEMA,
                "label_schema": STRIDE_LABEL_SCHEMA,
                "state_id": state_id,
                "candidate_id": candidate_id,
                "features": candidate_features[(state_id, candidate_id)],
                **metadata,
            }
        )
        candidates_by_state[state_id].append(aggregate)

    candidate_aggregates: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    pair_counts: dict[str, int] = {}
    for state_id, candidates in sorted(candidates_by_state.items()):
        candidates.sort(key=lambda item: str(item["candidate_id"]))
        assign_structure_scores(candidates)
        candidate_aggregates.extend(candidates)
        winners: list[tuple[dict[str, Any], dict[str, Any]]] = []
        before_conflicts = int(state_metadata[state_id]["before_conflicts"])
        for left, right in combinations(candidates, 2):
            if stride_dominates(left, right, before_conflicts=before_conflicts):
                winners.append((left, right))
            elif stride_dominates(right, left, before_conflicts=before_conflicts):
                winners.append((right, left))
        pair_counts[state_id] = len(winners)
        if not winners:
            continue
        weight = 1.0 / (2.0 * len(winners))
        for winner, loser in winners:
            shared = {
                "schema": STRIDE_LABEL_SCHEMA,
                "state_id": state_id,
                "map_id": state_metadata[state_id]["map_id"],
                "split": state_metadata[state_id]["split"],
                "sample_weight": weight,
            }
            pairs.append(
                {
                    **shared,
                    "left_candidate_id": winner["candidate_id"],
                    "right_candidate_id": loser["candidate_id"],
                    "label": 1,
                }
            )
            pairs.append(
                {
                    **shared,
                    "left_candidate_id": loser["candidate_id"],
                    "right_candidate_id": winner["candidate_id"],
                    "label": 0,
                }
            )

    state_count = len(candidates_by_state)
    covered_states = sum(count >= 5 for count in pair_counts.values())
    coverage_rate = covered_states / state_count if state_count else 0.0
    summary = {
        "schema": "lns2.stride.label_build_summary.v1",
        "label_schema": STRIDE_LABEL_SCHEMA,
        "trial_sha256": sha256_file(trials_path),
        "state_count": state_count,
        "map_count": len({item["map_id"] for item in state_metadata.values()}),
        "candidate_count": len(candidate_aggregates),
        "dominance_pair_count": len(pairs) // 2,
        "oriented_training_row_count": len(pairs),
        "states_with_at_least_five_pairs": covered_states,
        "coverage_rate": coverage_rate,
        "coverage_gate": 0.8,
        "coverage_gate_passed": state_count > 0 and coverage_rate >= 0.8,
        "runtime_used_in_label": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "candidate_aggregates.jsonl", candidate_aggregates)
    _write_jsonl(output / "dominance_pairs.jsonl", pairs)
    (output / "label_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield payload


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _audit_bundle(root: Path, specification: dict[str, Any]) -> dict[str, Any]:
    bundle_root = _resolve(root, str(specification["path"]))
    manifest_path = bundle_root / "controller_manifest.json"
    result: dict[str, Any] = {
        "controller_id": str(specification["controller_id"]),
        "path": str(bundle_root),
        "exists": manifest_path.is_file(),
        "errors": [],
    }
    if not manifest_path.is_file():
        result["errors"].append("controller_manifest.json is missing")
        result["valid"] = False
        return result

    manifest = _read_json(manifest_path)
    expected_id = result["controller_id"]
    actual_id = str(
        manifest.get("controller_id", manifest.get("default_controller", ""))
    )
    feature_schema_id = str(manifest.get("feature_schema_id", ""))
    feature_dimensions = manifest.get("feature_dimensions", {})
    realized_dimension = (
        feature_dimensions.get("realized_dynamic")
        if isinstance(feature_dimensions, dict)
        else None
    )
    if actual_id != expected_id:
        result["errors"].append(
            f"controller id mismatch: expected {expected_id}, found {actual_id}"
        )
    if feature_schema_id != FROZEN_FEATURE_SCHEMA_ID:
        result["errors"].append(
            "feature schema mismatch: "
            f"expected {FROZEN_FEATURE_SCHEMA_ID}, found {feature_schema_id}"
        )
    if realized_dimension != FROZEN_FEATURE_DIMENSION:
        result["errors"].append(
            "realized feature dimension mismatch: "
            f"expected {FROZEN_FEATURE_DIMENSION}, found {realized_dimension}"
        )
    result.update(
        {
            "manifest_sha256": sha256_file(manifest_path),
            "feature_schema_id": feature_schema_id,
            "feature_schema_sha256": manifest.get("feature_schema_sha256"),
            "realized_feature_dimension": realized_dimension,
            "valid": not result["errors"],
        }
    )
    return result


def _post_structure_available(payload: dict[str, Any]) -> bool:
    candidates = [payload]
    for key in ("post_state", "post_structure", "outcome"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    return any(REQUIRED_POST_STRUCTURE_FIELDS.issubset(item) for item in candidates)


def _audit_ranking_index(path: Path) -> dict[str, Any]:
    states: set[str] = set()
    maps: set[str] = set()
    splits: Counter[str] = Counter()
    agents: Counter[int] = Counter()
    trial_counts: Counter[int] = Counter()
    feature_dimensions: Counter[int] = Counter()
    row_count = 0
    rows_with_post_structure = 0
    rows_with_individual_trials = 0

    for row in _read_jsonl(path):
        row_count += 1
        states.add(str(row.get("state_id", "")))
        maps.add(str(row.get("map_id", "")))
        splits[str(row.get("split", ""))] += 1
        try:
            agents[int(row.get("agent_count", 0))] += 1
        except (TypeError, ValueError):
            agents[0] += 1
        try:
            trial_counts[int(row.get("trial_count", 0))] += 1
        except (TypeError, ValueError):
            trial_counts[0] += 1
        features = row.get("features", {})
        realized = features.get("realized_dynamic", {}) if isinstance(features, dict) else {}
        if isinstance(realized, dict):
            feature_dimensions[len(realized)] += 1
        if _post_structure_available(row):
            rows_with_post_structure += 1
        if isinstance(row.get("trials"), list):
            rows_with_individual_trials += 1

    return {
        "row_count": row_count,
        "state_count": len(states - {""}),
        "map_count": len(maps - {""}),
        "map_ids": sorted(maps - {""}),
        "splits": dict(sorted(splits.items())),
        "agent_counts": {str(key): value for key, value in sorted(agents.items())},
        "trial_counts": {
            str(key): value for key, value in sorted(trial_counts.items())
        },
        "feature_dimensions": {
            str(key): value for key, value in sorted(feature_dimensions.items())
        },
        "has_individual_paired_trials": (
            row_count > 0 and rows_with_individual_trials == row_count
        ),
        "has_post_state_structure": (
            row_count > 0 and rows_with_post_structure == row_count
        ),
    }


def _audit_sequence_trials(path: Path) -> dict[str, Any]:
    states: set[str] = set()
    maps: set[str] = set()
    splits: Counter[str] = Counter()
    trials_per_candidate: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    pp_seeds_per_candidate: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    row_count = 0
    rows_with_post_structure = 0
    executed_step_count = 0

    for row in _read_jsonl(path):
        row_count += 1
        state_id = str(row.get("state_id", ""))
        states.add(state_id)
        maps.add(str(row.get("map_id", "")))
        splits[str(row.get("split", ""))] += 1
        try:
            trial_index = int(row.get("trial_index", -1))
        except (TypeError, ValueError):
            trial_index = -1
        steps = row.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict) or not step.get("executed", False):
                continue
            executed_step_count += 1
            candidate_id = str(step.get("candidate_id", ""))
            key = (state_id, candidate_id)
            trials_per_candidate[key].add(trial_index)
            action = step.get("action", {})
            if isinstance(action, dict) and action.get("pp_random_seed") is not None:
                try:
                    pp_seeds_per_candidate[key].add(int(action["pp_random_seed"]))
                except (TypeError, ValueError):
                    pass
            if _post_structure_available(step):
                rows_with_post_structure += 1

    seed_multiplicities = Counter(len(value) for value in pp_seeds_per_candidate.values())
    return {
        "row_count": row_count,
        "state_count": len(states - {""}),
        "map_count": len(maps - {""}),
        "map_ids": sorted(maps - {""}),
        "splits": dict(sorted(splits.items())),
        "executed_step_count": executed_step_count,
        "candidate_count": len(trials_per_candidate),
        "pp_seed_multiplicity": {
            str(key): value for key, value in sorted(seed_multiplicities.items())
        },
        "has_individual_paired_trials": bool(pp_seeds_per_candidate)
        and all(len(value) >= 4 for value in pp_seeds_per_candidate.values()),
        "has_post_state_structure": executed_step_count > 0
        and rows_with_post_structure == executed_step_count,
    }


def _audit_source(root: Path, specification: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(root, str(specification["path"]))
    kind = str(specification["kind"])
    result: dict[str, Any] = {
        "name": str(specification["name"]),
        "role": str(specification.get("role", "historical")),
        "kind": kind,
        "path": str(path),
        "exists": path.is_file(),
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append("source file is missing")
        result["valid"] = False
        result["reusable_for_stride_labels"] = False
        return result
    try:
        if kind == "ranking_index":
            details = _audit_ranking_index(path)
        elif kind == "sequence_trials":
            details = _audit_sequence_trials(path)
        else:
            raise ValueError(f"unsupported source kind: {kind}")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result["errors"].append(str(error))
        result["valid"] = False
        result["reusable_for_stride_labels"] = False
        return result
    result.update(details)
    result["sha256"] = sha256_file(path)
    result["valid"] = True
    result["reusable_for_stride_labels"] = bool(
        details["has_individual_paired_trials"]
        and details["has_post_state_structure"]
    )
    result["reuse_blockers"] = [
        name
        for condition, name in (
            (
                details["has_individual_paired_trials"],
                "missing complete per-candidate paired PP outcomes",
            ),
            (
                details["has_post_state_structure"],
                "missing normalized post-repair structure metrics",
            ),
        )
        if not condition
    ]
    return result


def _map_leakage(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_role: defaultdict[str, set[str]] = defaultdict(set)
    for source in sources:
        by_role[str(source["role"])].update(source.get("map_ids", []))
    findings: list[dict[str, Any]] = []
    roles = sorted(by_role)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            if "historical" in {left, right}:
                continue
            overlap = sorted(by_role[left] & by_role[right])
            if overlap:
                findings.append({"left_role": left, "right_role": right, "maps": overlap})
    return findings


def run_stage1_audit(
    *, config_path: Path, output: Path, project_root: Path
) -> dict[str, Any]:
    config = _read_json(config_path)
    if config.get("schema") != STRIDE_STAGE1_CONFIG_SCHEMA:
        raise ValueError(
            f"expected config schema {STRIDE_STAGE1_CONFIG_SCHEMA}, "
            f"found {config.get('schema')}"
        )
    if config.get("research_line") != STRIDE_RESEARCH_LINE:
        raise ValueError(f"research_line must be {STRIDE_RESEARCH_LINE}")

    bundles = [
        _audit_bundle(project_root, item)
        for item in config.get("frozen_bundles", [])
    ]
    sources = [
        _audit_source(project_root, item)
        for item in config.get("historical_sources", [])
    ]
    leakage = _map_leakage(sources)
    bundle_ids = {item["controller_id"] for item in bundles if item.get("valid")}
    required_bundle_ids = {"v2-full", "mixed-full-v2"}
    ready = (
        required_bundle_ids.issubset(bundle_ids)
        and all(item.get("valid", False) for item in bundles)
        and all(item.get("valid", False) for item in sources)
        and not leakage
    )
    report: dict[str, Any] = {
        "schema": STRIDE_STAGE1_REPORT_SCHEMA,
        "schema_version": 1,
        "research_line": STRIDE_RESEARCH_LINE,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reserved_names": {
            "control_controller": STRIDE_CONTROL_CONTROLLER_ID,
            "quality_controller": STRIDE_QUALITY_CONTROLLER_ID,
            "label_schema": STRIDE_LABEL_SCHEMA,
        },
        "frozen_contract": {
            "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
            "feature_dimension": FROZEN_FEATURE_DIMENSION,
            "initializer_changes_allowed": False,
            "candidate_pool_changes_allowed": False,
            "pp_changes_allowed": False,
        },
        "frozen_bundles": bundles,
        "historical_sources": sources,
        "map_leakage": leakage,
        "requires_fresh_stride_collection": not any(
            item.get("reusable_for_stride_labels", False) for item in sources
        ),
        "decision": (
            "ready_for_stride_pilot" if ready else "stage1_audit_failed"
        ),
        "stage1_passed": ready,
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "stage1_audit_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
