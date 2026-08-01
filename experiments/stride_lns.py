from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STRIDE_RESEARCH_LINE = "stride-lns"
STRIDE_CONTROL_CONTROLLER_ID = "stride-control-v1"
STRIDE_QUALITY_CONTROLLER_ID = "stride-quality-v1"
STRIDE_LABEL_SCHEMA = "lns2.stride.quality_label.v1"
STRIDE_STAGE1_CONFIG_SCHEMA = "lns2.stride.stage1_config.v1"
STRIDE_STAGE1_REPORT_SCHEMA = "lns2.stride.stage1_audit.v1"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            "manifest_sha256": _sha256(manifest_path),
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
    result["sha256"] = _sha256(path)
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
