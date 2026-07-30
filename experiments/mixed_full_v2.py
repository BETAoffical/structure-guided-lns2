from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import score_online_candidates
from experiments.compact_controller_model import (
    export_controller_bundle,
    load_controller_bundle,
)
from experiments.context_audit import PairwiseModel
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.v2_factorial_audit import (
    MODEL_PARAMETERS,
    _aggregate_high_load,
    _fit_pairwise,
    _group_states,
    _legacy_candidates,
    _legacy_candidates_any_split,
    _full_v2_specs,
    build_pair_rows,
)


MIXED_FULL_EXPORT_SCHEMA = "lns2.mixed_full_v2_export.v1"


def _portable_payload(model: PairwiseModel, source_sha256: str) -> dict[str, Any]:
    estimator = model.estimator
    if list(map(int, estimator.classes_)) != [0, 1]:
        raise ValueError("portable exporter supports only binary pairwise models")
    trees = []
    for stage in estimator._predictors:
        if len(stage) != 1:
            raise ValueError("portable exporter supports one binary tree per stage")
        nodes = []
        for node in stage[0].nodes:
            if bool(node["is_categorical"]):
                raise ValueError("portable exporter does not support categorical nodes")
            nodes.append(
                {
                    "value": float(node["value"]),
                    "feature_idx": int(node["feature_idx"]),
                    "num_threshold": float(node["num_threshold"]),
                    "missing_go_to_left": bool(node["missing_go_to_left"]),
                    "left": int(node["left"]),
                    "right": int(node["right"]),
                    "is_leaf": bool(node["is_leaf"]),
                }
            )
        trees.append(nodes)
    return {
        "schema": "lns2.portable_pairwise_hist_gbdt.v1",
        "schema_version": 1,
        "profile": model.profile,
        "source_model_sha256": source_sha256,
        "feature_names": list(model.feature_names),
        "baseline": float(estimator._baseline_prediction[0, 0]),
        "trees": trees,
    }


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    partial.replace(path)


def _feature_ranges(
    candidates: list[dict[str, Any]], feature_names: list[str]
) -> dict[str, list[float]]:
    values = {name: [] for name in feature_names}
    for row in candidates:
        features = dict(row["features"])
        for name in feature_names:
            values[name].append(float(features.get(name, 0.0)))
    return {name: [min(rows), max(rows)] for name, rows in values.items()}


def _training_candidates(
    legacy_training_index: Path, high_load_collection: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregate_rows = _read_jsonl(legacy_training_index)
    legacy = _legacy_candidates(aggregate_rows, "policy_train")
    if len({str(row["state_id"]) for row in aggregate_rows}) > len(
        {str(row["state_id"]) for row in legacy}
    ):
        legacy = _legacy_candidates_any_split(aggregate_rows)
    high = _aggregate_high_load(
        _read_jsonl(high_load_collection / "feature_index.jsonl"),
        _read_jsonl(high_load_collection / "trial_manifest.jsonl"),
    )
    high_train = [row for row in high if str(row["split"]) == "policy_train"]
    if not high_train:
        raise ValueError("high-load collection has no policy_train candidates")
    candidates = [*legacy, *high_train]
    return candidates, {
        "legacy_state_count": len(_group_states(legacy)),
        "high_load_state_count": len(_group_states(high_train)),
        "combined_state_count": len(_group_states(candidates)),
        "combined_candidate_count": len(candidates),
    }


def export_mixed_full_v2(
    *,
    legacy_training_index: str | Path,
    high_load_collection: str | Path,
    source_portable_bundle: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    legacy_path = Path(legacy_training_index).resolve()
    high_root = Path(high_load_collection).resolve()
    source_root = Path(source_portable_bundle).resolve()
    output_root = Path(output).resolve()
    candidates, coverage = _training_candidates(legacy_path, high_root)
    specs = _full_v2_specs()
    pair_rows, pair_diagnostics = build_pair_rows(candidates, specs)
    estimator = _fit_pairwise(pair_rows)
    feature_names = list(PROFILE_FEATURE_NAMES["realized_dynamic"])
    sklearn_model = PairwiseModel(
        profile="realized_dynamic",
        feature_names=feature_names,
        estimator=estimator,
    )

    sklearn_path = output_root / "sklearn" / "mixed_full_v2.pkl"
    _atomic_pickle(sklearn_path, sklearn_model)
    sklearn_sha = sha256_file(sklearn_path)

    source_bundle_root = output_root / "source_portable"
    source_manifest = _read_json(source_root / "portable_manifest.json")
    source_models = {
        str(row["profile"]): dict(row) for row in source_manifest["models"]
    }
    proposal_source = source_root / str(source_models["proposal_dynamic"]["file"])
    proposal_destination = source_bundle_root / "pairwise__proposal_dynamic.json"
    proposal_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(proposal_source, proposal_destination)
    realized_destination = source_bundle_root / "pairwise__realized_dynamic.json"
    _write_json(realized_destination, _portable_payload(sklearn_model, sklearn_sha))

    proposal_ranges = dict(source_manifest["feature_ranges"])["proposal_dynamic"]
    realized_ranges = _feature_ranges(candidates, feature_names)
    portable_manifest = {
        "schema": "lns2.portable_pairwise_bundle.v2",
        "schema_version": 2,
        "models": [
            {
                "profile": "proposal_dynamic",
                "file": proposal_destination.relative_to(source_bundle_root).as_posix(),
                "sha256": sha256_file(proposal_destination),
                "source_model_sha256": str(
                    source_models["proposal_dynamic"]["source_model_sha256"]
                ),
                "feature_count": int(source_models["proposal_dynamic"]["feature_count"]),
                "tree_count": int(source_models["proposal_dynamic"]["tree_count"]),
            },
            {
                "profile": "realized_dynamic",
                "file": realized_destination.relative_to(source_bundle_root).as_posix(),
                "sha256": sha256_file(realized_destination),
                "source_model_sha256": sklearn_sha,
                "feature_count": len(feature_names),
                "tree_count": len(estimator._predictors),
            },
        ],
        "feature_ranges": {
            "proposal_dynamic": proposal_ranges,
            "realized_dynamic": realized_ranges,
        },
        "development_index_sha256": sha256_file(legacy_path),
        "development_state_count": coverage["combined_state_count"],
        "development_candidate_count": coverage["combined_candidate_count"],
        "model_parameters": MODEL_PARAMETERS,
        "confirmation_labels_seen": False,
    }
    _write_json(source_bundle_root / "portable_manifest.json", portable_manifest)

    export_controller_bundle(
        source_bundle_root,
        output_root,
        promotion_report={
            "exact_acceleration_passed": True,
            "feature_performance_passed": True,
            "scientific_promotion_passed": False,
            "note": "Runtime parity only; end-to-end promotion remains pending.",
        },
    )
    compact = load_controller_bundle(output_root)
    compact_model = compact.main_models["realized_dynamic"]
    mismatches = []
    maximum_score_delta = 0.0
    for state in _group_states(candidates):
        rows = [
            {
                "candidate_id": row["candidate_id"],
                "candidate_key": row["candidate_key"],
                "features": {"realized_dynamic": row["features"]},
            }
            for row in state
        ]
        reference_index, reference_scores, _ = score_online_candidates(
            rows, sklearn_model
        )
        compact_index, compact_scores, _ = score_online_candidates(rows, compact_model)
        maximum_score_delta = max(
            maximum_score_delta,
            *(abs(left - right) for left, right in zip(reference_scores, compact_scores)),
        )
        if reference_index != compact_index:
            mismatches.append(str(state[0]["state_id"]))
    if mismatches or maximum_score_delta > 1e-10:
        raise ValueError(
            "mixed-full portable equivalence failed: "
            f"mismatches={len(mismatches)}, max_delta={maximum_score_delta}"
        )

    report = {
        "schema": MIXED_FULL_EXPORT_SCHEMA,
        "controller_id": "mixed-full-v2",
        "scientific_status": "unconfirmed_end_to_end",
        "input_dimension": len(specs),
        "base_feature_count": len(feature_names),
        "model_parameters": MODEL_PARAMETERS,
        "coverage": coverage,
        "pair_diagnostics": pair_diagnostics,
        "inputs": {
            "legacy_training_sha256": sha256_file(legacy_path),
            "high_feature_index_sha256": sha256_file(
                high_root / "feature_index.jsonl"
            ),
            "high_trial_manifest_sha256": sha256_file(
                high_root / "trial_manifest.jsonl"
            ),
            "source_portable_manifest_sha256": sha256_file(
                source_root / "portable_manifest.json"
            ),
        },
        "artifacts": {
            "sklearn_model": sklearn_path.relative_to(output_root).as_posix(),
            "sklearn_model_sha256": sklearn_sha,
            "source_portable_manifest_sha256": sha256_file(
                source_bundle_root / "portable_manifest.json"
            ),
            "base_controller_manifest_sha256": sha256_file(
                output_root / "controller_manifest.json"
            ),
        },
        "equivalence": {
            "state_count": coverage["combined_state_count"],
            "selection_mismatch_count": len(mismatches),
            "maximum_score_delta": maximum_score_delta,
            "passed": not mismatches and maximum_score_delta <= 1e-10,
        },
    }
    _write_json(output_root / "mixed_full_export_report.json", report)
    manifest = _read_json(output_root / "controller_manifest.json")
    manifest["controller_id"] = "mixed-full-v2"
    manifest["mixed_full_export_report"] = {
        "file": "mixed_full_export_report.json",
        "sha256": sha256_file(output_root / "mixed_full_export_report.json"),
    }
    _write_json(output_root / "controller_manifest.json", manifest)
    return report


__all__ = ["MIXED_FULL_EXPORT_SCHEMA", "export_mixed_full_v2"]
