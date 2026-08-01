from __future__ import annotations

import collections
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from lns2_selector.runtime.online_selection import (
    online_candidate_rows,
    score_online_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEARNED_POLICIES = ("proposal_dynamic", "realized_dynamic")


@dataclass
class FrozenPolicyBundle:
    models: dict[str, Any]
    ranges: dict[str, dict[str, tuple[float, float]]]
    manifest: dict[str, Any]


@dataclass
class PortablePairwiseModel:
    profile: str
    feature_names: list[str]
    baseline: float
    trees: list[list[dict[str, Any]]]
    native_predictor: Any | None = None

    def predict_positive(self, vectors: list[list[float]]) -> list[float]:
        if self.native_predictor is not None:
            return list(map(float, self.native_predictor.predict_positive(vectors)))
        probabilities = []
        for vector in vectors:
            raw = self.baseline
            for nodes in self.trees:
                index = 0
                while not bool(nodes[index]["is_leaf"]):
                    node = nodes[index]
                    value = float(vector[int(node["feature_idx"])])
                    go_left = (
                        math.isnan(value) and bool(node["missing_go_to_left"])
                    ) or (
                        not math.isnan(value)
                        and value <= float(node["num_threshold"])
                    )
                    index = int(node["left"] if go_left else node["right"])
                raw += float(nodes[index]["value"])
            if raw >= 0.0:
                probabilities.append(1.0 / (1.0 + math.exp(-raw)))
            else:
                exponential = math.exp(raw)
                probabilities.append(exponential / (1.0 + exponential))
        return probabilities


def _project_path(value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def export_portable_policy_bundle(
    frozen_root: str | Path,
    registration: dict[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    root = Path(frozen_root).resolve()
    freeze_manifest = _read_json(root / "freeze_manifest.json")
    if bool(freeze_manifest.get("confirmation_labels_seen", True)):
        raise ValueError("frozen model manifest has seen confirmation labels")
    model_rows = {str(row["profile"]): row for row in freeze_manifest["models"]}
    expected = {
        str(name): str(value).lower()
        for name, value in dict(registration["model_sha256"]).items()
    }
    output_root = Path(output).resolve()
    exported = []
    feature_names_by_profile: dict[str, list[str]] = {}
    for profile in LEARNED_POLICIES:
        source = root / str(model_rows[profile]["model_file"])
        source_sha = sha256_file(source)
        if source_sha != expected[profile]:
            raise ValueError(f"frozen model SHA256 mismatch: {profile}")
        with source.open("rb") as stream:
            model = pickle.load(stream)
        estimator = model.estimator
        if list(map(int, estimator.classes_)) != [0, 1]:
            raise ValueError("portable exporter supports only binary pairwise models")
        trees = []
        for stage in estimator._predictors:
            if len(stage) != 1:
                raise ValueError("portable exporter supports one binary tree per stage")
            predictor = stage[0]
            nodes = []
            for node in predictor.nodes:
                if bool(node["is_categorical"]):
                    raise ValueError(
                        "portable exporter does not support categorical tree nodes"
                    )
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
        payload = {
            "schema": "lns2.portable_pairwise_hist_gbdt.v1",
            "schema_version": 1,
            "profile": profile,
            "source_model_sha256": source_sha,
            "feature_names": list(model.feature_names),
            "baseline": float(estimator._baseline_prediction[0, 0]),
            "trees": trees,
        }
        path = output_root / f"pairwise__{profile}.json"
        _write_json(path, payload)
        feature_names_by_profile[profile] = list(model.feature_names)
        exported.append(
            {
                "profile": profile,
                "file": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
                "source_model_sha256": source_sha,
                "feature_count": len(model.feature_names),
                "tree_count": len(trees),
            }
        )
    index_path = _project_path(registration["development_index"])
    expected_index = str(registration["development_index_sha256"]).lower()
    if sha256_file(index_path) != expected_index:
        raise ValueError("development ranking index SHA256 mismatch")
    development_rows = _read_jsonl(index_path)
    feature_ranges = {}
    for profile, feature_names in feature_names_by_profile.items():
        values: dict[str, list[float]] = {name: [] for name in feature_names}
        for row in development_rows:
            features = dict(row["features"][profile])
            for name in feature_names:
                values[name].append(float(features.get(name, 0.0)))
        feature_ranges[profile] = {
            name: [min(numbers), max(numbers)] for name, numbers in values.items()
        }
    manifest = {
        "schema": "lns2.portable_pairwise_bundle.v2",
        "schema_version": 2,
        "models": exported,
        "feature_ranges": feature_ranges,
        "development_index_sha256": expected_index,
        "development_state_count": len(
            {str(row["state_id"]) for row in development_rows}
        ),
        "development_candidate_count": len(development_rows),
        "model_parameters": freeze_manifest.get("model_parameters", {}),
        "confirmation_labels_seen": False,
    }
    _write_json(output_root / "portable_manifest.json", manifest)
    return manifest


def _load_portable_models(
    portable_root: Path,
    expected_portable: dict[str, str],
    expected_source: dict[str, str],
) -> dict[str, PortablePairwiseModel]:
    manifest = _read_json(portable_root / "portable_manifest.json")
    if bool(manifest.get("confirmation_labels_seen", True)):
        raise ValueError("portable model manifest has seen confirmation labels")
    rows = {str(row["profile"]): row for row in manifest["models"]}
    models = {}
    for profile in LEARNED_POLICIES:
        row = rows[profile]
        path = portable_root / str(row["file"])
        digest = sha256_file(path)
        if digest != expected_portable[profile] or digest != str(row["sha256"]):
            raise ValueError(f"portable model SHA256 mismatch: {profile}")
        payload = _read_json(path)
        if (
            str(payload["profile"]) != profile
            or str(payload["source_model_sha256"]) != expected_source[profile]
        ):
            raise ValueError(f"portable model provenance mismatch: {profile}")
        native_predictor = None
        try:
            import lns2_env

            predictor_type = getattr(lns2_env, "PortableTreeEnsemble", None)
            if predictor_type is not None:
                native_predictor = predictor_type(
                    float(payload["baseline"]), list(payload["trees"])
                )
        except ImportError:
            pass
        models[profile] = PortablePairwiseModel(
            profile=profile,
            feature_names=list(map(str, payload["feature_names"])),
            baseline=float(payload["baseline"]),
            trees=list(payload["trees"]),
            native_predictor=native_predictor,
        )
    return models


def _load_deployment_policy_bundle(
    deployment_root: Path, registration: dict[str, Any]
) -> FrozenPolicyBundle:
    manifest_path = deployment_root / "portable_manifest.json"
    expected_manifest_sha = str(
        registration.get("deployment_manifest_sha256", "")
    ).lower()
    if not expected_manifest_sha or sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("deployment manifest SHA256 mismatch")
    manifest = _read_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != 2:
        raise ValueError("deployment bundle must use portable schema version 2")
    if str(manifest.get("development_index_sha256", "")).lower() != str(
        registration["development_index_sha256"]
    ).lower():
        raise ValueError("deployment bundle development index provenance mismatch")
    expected_models = {
        str(name): str(value).lower()
        for name, value in dict(registration["model_sha256"]).items()
    }
    expected_portable = {
        str(name): str(value).lower()
        for name, value in dict(registration["portable_model_sha256"]).items()
    }
    models = _load_portable_models(
        deployment_root, expected_portable, expected_models
    )
    stored_ranges = dict(manifest.get("feature_ranges", {}))
    ranges: dict[str, dict[str, tuple[float, float]]] = {}
    for profile, model in models.items():
        profile_ranges = dict(stored_ranges.get(profile, {}))
        if set(profile_ranges) != set(model.feature_names):
            raise ValueError(f"deployment feature ranges are incomplete: {profile}")
        ranges[profile] = {
            name: (float(profile_ranges[name][0]), float(profile_ranges[name][1]))
            for name in model.feature_names
        }
    return FrozenPolicyBundle(models=models, ranges=ranges, manifest=manifest)


def verify_portable_policy_bundle(
    frozen_root: str | Path, registration: dict[str, Any]
) -> dict[str, Any]:
    native_registration = {
        key: value
        for key, value in registration.items()
        if key not in {"deployment_bundle", "portable_models", "portable_model_sha256"}
    }
    native = load_frozen_policy_bundle(frozen_root, native_registration)
    portable = load_frozen_policy_bundle(frozen_root, registration)
    index_path = _project_path(registration["development_index"])
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in _read_jsonl(index_path):
        grouped[str(row["state_id"])].append(row)
    profiles = {}
    for profile in LEARNED_POLICIES:
        mismatches = 0
        maximum_score_delta = 0.0
        pair_count = 0
        for candidates in grouped.values():
            native_index, native_scores, _ = score_online_candidates(
                candidates, native.models[profile]
            )
            portable_index, portable_scores, _ = score_online_candidates(
                candidates, portable.models[profile]
            )
            mismatches += native_index != portable_index
            maximum_score_delta = max(
                maximum_score_delta,
                max(
                    abs(first - second)
                    for first, second in zip(native_scores, portable_scores)
                ),
            )
            pair_count += len(candidates) * (len(candidates) - 1) // 2
        profiles[profile] = {
            "state_count": len(grouped),
            "pair_count": pair_count,
            "selection_mismatch_count": mismatches,
            "maximum_score_delta": maximum_score_delta,
            "passed": mismatches == 0 and maximum_score_delta <= 1e-12,
        }
    feature_parity = None
    source_value = registration.get("development_candidates")
    if source_value:
        source_path = _project_path(source_value)
        expected_source_sha = str(
            registration["development_candidates_sha256"]
        ).lower()
        if sha256_file(source_path) != expected_source_sha:
            raise ValueError("development candidate source SHA256 mismatch")
        indexed = {
            (str(row["state_id"]), str(row["candidate_id"])): row
            for row in _read_jsonl(index_path)
        }
        checked = 0
        mismatches = []
        seen = set()
        for source in _read_jsonl(source_path):
            state_id = str(source["state_id"])
            for computed in online_candidate_rows(source["state"], source["candidates"]):
                key = (state_id, str(computed["candidate_id"]))
                checked += 1
                seen.add(key)
                if key not in indexed or computed["features"] != indexed[key]["features"]:
                    mismatches.append(list(key))
        missing = sorted([list(key) for key in set(indexed) - seen])
        feature_parity = {
            "candidate_count": checked,
            "mismatch_count": len(mismatches),
            "missing_count": len(missing),
            "mismatches": mismatches,
            "missing": missing,
            "passed": not mismatches and not missing and checked == len(indexed),
        }
    passed = all(row["passed"] for row in profiles.values()) and (
        feature_parity is None or bool(feature_parity["passed"])
    )
    return {
        "schema": "lns2.portable_pairwise_equivalence.v1",
        "schema_version": 1,
        "passed": passed,
        "profiles": profiles,
        "online_feature_parity": feature_parity,
    }


def load_frozen_policy_bundle(
    frozen_root: str | Path, registration: dict[str, Any]
) -> FrozenPolicyBundle:
    deployment_value = registration.get("deployment_bundle")
    if deployment_value:
        return _load_deployment_policy_bundle(
            _project_path(deployment_value), registration
        )
    root = Path(frozen_root).resolve()
    manifest = _read_json(root / "freeze_manifest.json")
    if bool(manifest.get("confirmation_labels_seen", True)):
        raise ValueError("frozen model manifest has seen confirmation labels")
    registered_index = registration.get("development_index")
    index_path = (
        _project_path(registered_index)
        if registered_index
        else Path(str(manifest["development_index"])).resolve()
    )
    expected_index = str(registration["development_index_sha256"]).lower()
    if sha256_file(index_path) != expected_index:
        raise ValueError("development ranking index SHA256 mismatch")
    model_rows = {str(row["profile"]): row for row in manifest.get("models", [])}
    expected_models = {
        str(name): str(value).lower()
        for name, value in dict(registration["model_sha256"]).items()
    }
    if not set(LEARNED_POLICIES).issubset(model_rows):
        raise ValueError("frozen model set is incomplete")
    models: dict[str, Any] = {}
    for profile in LEARNED_POLICIES:
        row = model_rows[profile]
        path = root / str(row["model_file"])
        digest = sha256_file(path)
        if digest != expected_models[profile] or digest != str(
            row["model_sha256"]
        ).lower():
            raise ValueError(f"frozen model SHA256 mismatch: {profile}")
    portable_value = registration.get("portable_models")
    if portable_value:
        expected_portable = {
            str(name): str(value).lower()
            for name, value in dict(registration["portable_model_sha256"]).items()
        }
        models = _load_portable_models(
            _project_path(portable_value), expected_portable, expected_models
        )
    else:
        for profile in LEARNED_POLICIES:
            row = model_rows[profile]
            path = root / str(row["model_file"])
            with path.open("rb") as stream:
                model = pickle.load(stream)
            if str(model.profile) != profile:
                raise ValueError(f"frozen model profile mismatch: {profile}")
            models[profile] = model
    development_rows = _read_jsonl(index_path)
    ranges: dict[str, dict[str, tuple[float, float]]] = {}
    for profile, model in models.items():
        values: dict[str, list[float]] = {name: [] for name in model.feature_names}
        for row in development_rows:
            features = dict(row["features"][profile])
            for name in model.feature_names:
                values[name].append(float(features.get(name, 0.0)))
        ranges[profile] = {
            name: (min(numbers), max(numbers)) for name, numbers in values.items()
        }
    return FrozenPolicyBundle(models=models, ranges=ranges, manifest=manifest)


__all__ = [
    "FrozenPolicyBundle",
    "PortablePairwiseModel",
    "export_portable_policy_bundle",
    "load_frozen_policy_bundle",
    "verify_portable_policy_bundle",
]
