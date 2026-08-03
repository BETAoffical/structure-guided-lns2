from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import online_candidate_rows
from experiments.compact_controller_model import load_controller_bundle
from experiments.controller_performance_benchmark import _trace_samples
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES, canonicalize_features
from experiments.online_feature_engine import OnlineFeatureEngine, static_grid_for_state
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from lns2_selector.runtime.online_selection import score_online_candidates


CONFIG_SCHEMA = "lns2.stride.feature_cache_audit_config.v1"
REPORT_SCHEMA = "lns2.stride.feature_cache_audit.v1"
REGISTERED_PATHS = (
    "native-dense-model-projection",
    "native-dense-full-124",
    "python-incremental-full-124",
    "python-reference-full-124",
)


def validate_feature_cache_audit_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected STRIDE feature-cache audit config")
    if (
        config.get("scientific_status") != "action_preserving_runtime_audit"
        or config.get("controller_id") != "v2-full"
        or int(config.get("maximum_decisions_per_episode", -1)) != 3
        or int(config.get("repeats", -1)) != 3
        or float(config.get("floating_tolerance", -1.0)) != 1e-12
        or tuple(map(str, config.get("registered_paths") or ()))
        != REGISTERED_PATHS
        or config.get("selection_rule")
        != "fastest_action_equivalent_path_with_exact_feature_equivalence"
        or bool(config.get("candidate_actions_may_change"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("STRIDE feature-cache audit contract changed")


def _project_root(path: str, root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _feature_dict(row: dict[str, Any]) -> dict[str, float]:
    if "feature_names" in row:
        names = tuple(map(str, row["feature_names"]))
        values = tuple(map(float, row["feature_values"]))
        if len(names) != len(values):
            raise ValueError("dense feature row has mismatched names and values")
        return dict(zip(names, values))
    return canonicalize_features(
        row["features"]["realized_dynamic"], "realized_dynamic"
    )


def _engine_sequence(
    samples: list[dict[str, Any]],
    *,
    backend: str,
    dense: bool,
    required_names: tuple[str, ...],
) -> list[list[dict[str, Any]]]:
    engine = OnlineFeatureEngine(
        samples[0]["state"],
        backend=backend,
        shadow_validation=False,
        required_features={"realized_dynamic": required_names},
        dense_output=dense,
    )
    result = []
    for index, sample in enumerate(samples):
        if index:
            engine.prepare(
                sample["state"], changed_agents=sample["changed_agents"] or []
            )
        rows, _ = engine.realized_rows(
            sample["candidates"], state_hash=f"stride-feature-audit-{index}"
        )
        result.append(rows)
    return result


def _reference_sequence(
    samples: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    static = static_grid_for_state(samples[0]["state"])
    return [
        online_candidate_rows(
            sample["state"], sample["candidates"], static_grid=static
        )
        for sample in samples
    ]


def _compare(
    reference: list[list[dict[str, Any]]],
    candidate: list[list[dict[str, Any]]],
    *,
    path_id: str,
    names: tuple[str, ...],
    model: Any,
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("feature-cache audit sequence lengths differ")
    maximum_delta = 0.0
    action_mismatches = 0
    ranking_mismatches = 0
    for expected_rows, actual_rows in zip(reference, candidate):
        if len(expected_rows) != len(actual_rows):
            raise ValueError("feature-cache audit candidate counts differ")
        normalized_reference = []
        for expected, actual in zip(expected_rows, actual_rows):
            expected_values = canonicalize_features(
                expected["features"]["realized_dynamic"], "realized_dynamic"
            )
            actual_values = _feature_dict(actual)
            if set(actual_values) != set(names):
                missing = sorted(set(names) - set(actual_values))
                extra = sorted(set(actual_values) - set(names))
                raise ValueError(
                    "feature-cache audit path returned wrong features: "
                    f"path={path_id}, missing={missing}, extra={extra}"
                )
            maximum_delta = max(
                maximum_delta,
                *(abs(expected_values[name] - actual_values[name]) for name in names),
            )
            normalized_reference.append(
                {
                    "candidate_id": expected["candidate_id"],
                    "candidate_key": expected["candidate_key"],
                    "features": {
                        "realized_dynamic": {
                            name: expected_values[name] for name in names
                        }
                    },
                }
            )
        reference_index, reference_scores, _ = score_online_candidates(
            normalized_reference, model
        )
        actual_index, actual_scores, _ = score_online_candidates(actual_rows, model)
        action_mismatches += int(reference_index != actual_index)
        reference_order = sorted(
            range(len(reference_scores)),
            key=lambda index: (-float(reference_scores[index]), index),
        )
        actual_order = sorted(
            range(len(actual_scores)),
            key=lambda index: (-float(actual_scores[index]), index),
        )
        ranking_mismatches += int(reference_order != actual_order)
    return {
        "maximum_feature_delta": maximum_delta,
        "action_mismatch_count": action_mismatches,
        "ranking_mismatch_count": ranking_mismatches,
    }


def run_feature_cache_audit(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_feature_cache_audit_config(config)
    source_report = _project_root(str(config["source_report"]), project_root)
    if sha256_file(source_report) != str(config["source_report_sha256"]):
        raise ValueError("feature-cache audit source report SHA256 mismatch")
    report_source = _read_json(source_report)
    if report_source.get("passed") is not True:
        raise ValueError("feature-cache audit requires the passed raw-TTF report")
    source = _project_root(str(config["source_collection"]), project_root)
    manifests = [
        row
        for row in _read_jsonl(source / "realized_dynamic_manifest.jsonl")
        if str(row.get("status")) in {"ok", "resumed"}
    ]
    if len(manifests) != 18:
        raise ValueError("feature-cache audit requires all 18 stall episodes")
    model_bundle = load_controller_bundle(
        _project_root(str(config["model_bundle"]), project_root)
    )
    if str(model_bundle.manifest.get("default_controller")) != "v2-full":
        raise ValueError("feature-cache audit model is not frozen v2-full")
    model = model_bundle.main_models["realized_dynamic"]
    model_names = tuple(map(str, model.base_feature_names))
    full_names = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
    samples_by_episode = {
        str(manifest["episode_id"]): _trace_samples(
            source,
            manifest,
            int(config["maximum_decisions_per_episode"]),
        )
        for manifest in manifests
    }
    if any(not samples for samples in samples_by_episode.values()):
        raise ValueError("feature-cache audit found an episode without decisions")

    def execute(path_id: str) -> list[list[dict[str, Any]]]:
        combined = []
        for samples in samples_by_episode.values():
            if path_id == "native-dense-model-projection":
                rows = _engine_sequence(
                    samples,
                    backend="native",
                    dense=True,
                    required_names=model_names,
                )
            elif path_id == "native-dense-full-124":
                rows = _engine_sequence(
                    samples,
                    backend="native",
                    dense=True,
                    required_names=full_names,
                )
            elif path_id == "python-incremental-full-124":
                rows = _engine_sequence(
                    samples,
                    backend="python",
                    dense=False,
                    required_names=full_names,
                )
            elif path_id == "python-reference-full-124":
                rows = _reference_sequence(samples)
            else:
                raise ValueError(f"unknown feature-cache path: {path_id}")
            combined.extend(rows)
        return combined

    reference = execute("python-reference-full-124")
    results: dict[str, Any] = {}
    for path_id in REGISTERED_PATHS:
        rows = reference if path_id == "python-reference-full-124" else execute(path_id)
        names = model_names if path_id == "native-dense-model-projection" else full_names
        comparison = _compare(
            reference, rows, path_id=path_id, names=names, model=model
        )
        execute(path_id)
        timings = []
        for _ in range(int(config["repeats"])):
            started = time.perf_counter()
            execute(path_id)
            timings.append(time.perf_counter() - started)
        results[path_id] = {
            **comparison,
            "feature_dimension": len(names),
            "median_seconds": statistics.median(timings),
            "minimum_seconds": min(timings),
            "maximum_seconds": max(timings),
        }
    tolerance = float(config["floating_tolerance"])
    eligible = [
        path_id
        for path_id, row in results.items()
        if float(row["maximum_feature_delta"]) <= tolerance
        and int(row["action_mismatch_count"]) == 0
        and int(row["ranking_mismatch_count"]) == 0
    ]
    if not eligible:
        raise ValueError("feature-cache audit has no action-equivalent path")
    selected = min(
        eligible,
        key=lambda path_id: (float(results[path_id]["median_seconds"]), path_id),
    )
    raw_summary = dict(report_source["controller_summaries"])[
        "stride-boundary-stall-guard-v1"
    ]
    raw_ttf = float(raw_summary["mean_raw_wall_time_to_feasible"])
    feature_seconds = float(raw_summary["mean_feature_seconds"])
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": config["scientific_status"],
        "config_sha256": sha256_file(config_path),
        "source_report_sha256": sha256_file(source_report),
        "source_manifest_sha256": sha256_file(
            source / "realized_dynamic_manifest.jsonl"
        ),
        "episode_count": len(manifests),
        "decision_sample_count": sum(map(len, samples_by_episode.values())),
        "model_feature_dimension": len(model_names),
        "canonical_feature_dimension": len(full_names),
        "paths": results,
        "selected_path": selected,
        "all_paths_action_equivalent": all(
            int(row["action_mismatch_count"]) == 0 for row in results.values()
        ),
        "all_paths_ranking_equivalent": all(
            int(row["ranking_mismatch_count"]) == 0 for row in results.values()
        ),
        "deployment_feature_share_of_raw_ttf": (
            feature_seconds / raw_ttf if raw_ttf else math.inf
        ),
        "candidate_actions_changed": False,
        "formal_speed_claim": False,
        "passed": selected == "native-dense-model-projection",
        "next_decision": (
            "retain_native_dense_model_projection"
            if selected == "native-dense-model-projection"
            else "review_feature_path_before_collection"
        ),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "feature_cache_audit_report.json", report)
    return report


__all__ = ["run_feature_cache_audit", "validate_feature_cache_audit_config"]
