from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.feature_schema_v2 import (
    CONSTANT_REDUNDANT_FEATURES,
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    PROFILE_FEATURE_NAMES,
    canonical_feature,
)


COMPACT_MODEL_SCHEMA = "lns2.portable_pairwise_hist_gbdt.compact.v2"
CONTROLLER_BUNDLE_SCHEMA = "lns2.portable_controller_bundle.v3"
CONTROLLER_BUNDLE_VERSION = 3


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def portable_model_semantic_fingerprint(payload: dict[str, Any]) -> str:
    return _fingerprint(
        {
            "profile": payload["profile"],
            "feature_names": payload["feature_names"],
            "baseline": payload["baseline"],
            "trees": payload["trees"],
        }
    )


def _source_input(
    feature_index: int, feature_names: list[str]
) -> tuple[str, str, float] | tuple[str, None, float]:
    shared_names = [
        name for name in feature_names if name.startswith(("state.", "context."))
    ]
    if feature_index < len(feature_names):
        mode = "delta"
        name = feature_names[feature_index]
    else:
        shared_index = feature_index - len(feature_names)
        if shared_index < 0 or shared_index >= len(shared_names):
            raise ValueError(f"portable tree references invalid feature {feature_index}")
        mode = "shared"
        name = shared_names[shared_index]
    if name in CONSTANT_REDUNDANT_FEATURES:
        return "constant", None, float(CONSTANT_REDUNDANT_FEATURES[name])
    canonical, scale = canonical_feature(name)
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError(f"invalid canonical scale for {name}")
    return mode, canonical, scale


def compact_portable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    profile = str(payload.get("profile"))
    if profile not in PROFILE_FEATURE_NAMES:
        raise ValueError(f"feature-v2 does not support compact profile {profile}")
    feature_names = list(map(str, payload.get("feature_names", [])))
    if not feature_names:
        raise ValueError("portable model has no feature names")
    allowed = set(PROFILE_FEATURE_NAMES[profile])

    specs: set[tuple[str, str]] = set()
    for tree in payload.get("trees", []):
        for node in tree:
            if bool(node.get("is_leaf")):
                continue
            mode, name, _ = _source_input(int(node["feature_idx"]), feature_names)
            if mode == "constant":
                continue
            assert name is not None
            if name not in allowed:
                raise ValueError(f"compact model requires unregistered feature {name}")
            specs.add((mode, name))
    ordered_specs = sorted(specs, key=lambda item: (item[0] != "delta", item[1]))
    index_by_spec = {spec: index for index, spec in enumerate(ordered_specs)}

    compact_trees = []
    for source_tree in payload.get("trees", []):
        output_nodes: list[dict[str, Any]] = []

        def copy_node(source_index: int) -> int:
            source = dict(source_tree[source_index])
            if bool(source["is_leaf"]):
                target_index = len(output_nodes)
                output_nodes.append(
                    {
                        "value": float(source["value"]),
                        "feature_idx": 0,
                        "num_threshold": 0.0,
                        "missing_go_to_left": False,
                        "left": 0,
                        "right": 0,
                        "is_leaf": True,
                    }
                )
                return target_index
            mode, name, scale = _source_input(
                int(source["feature_idx"]), feature_names
            )
            if mode == "constant":
                constant = scale
                branch = (
                    int(source["left"])
                    if constant <= float(source["num_threshold"])
                    else int(source["right"])
                )
                return copy_node(branch)
            assert name is not None
            target_index = len(output_nodes)
            output_nodes.append({})
            left = copy_node(int(source["left"]))
            right = copy_node(int(source["right"]))
            output_nodes[target_index] = {
                "value": float(source["value"]),
                "feature_idx": index_by_spec[(mode, name)],
                "num_threshold": float(source["num_threshold"]) / scale,
                "missing_go_to_left": bool(source["missing_go_to_left"]),
                "left": left,
                "right": right,
                "is_leaf": False,
            }
            return target_index

        if source_tree:
            root = copy_node(0)
            if root != 0:
                raise RuntimeError("compacted portable tree root is not zero")
        compact_trees.append(output_nodes)

    input_features = [
        {"mode": mode, "name": name} for mode, name in ordered_specs
    ]
    base_features = sorted({name for _, name in ordered_specs})
    result = {
        "schema": COMPACT_MODEL_SCHEMA,
        "schema_version": 2,
        "profile": profile,
        "source_model_sha256": str(payload.get("source_model_sha256", "")),
        "source_semantic_fingerprint": portable_model_semantic_fingerprint(payload),
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "input_features": input_features,
        "input_dimension": len(input_features),
        "base_feature_names": base_features,
        "baseline": float(payload["baseline"]),
        "trees": compact_trees,
    }
    result["semantic_fingerprint"] = _fingerprint(
        {
            "profile": profile,
            "source_semantic_fingerprint": result["source_semantic_fingerprint"],
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        }
    )
    return result


@dataclass
class CompactPortablePairwiseModel:
    profile: str
    input_features: list[tuple[str, str]]
    base_feature_names: list[str]
    baseline: float
    trees: list[list[dict[str, Any]]]
    semantic_fingerprint: str
    native_predictor: Any | None = None
    _base_feature_index: dict[str, int] = field(init=False, repr=False)
    _base_feature_names_tuple: tuple[str, ...] = field(init=False, repr=False)
    _input_feature_indices: list[tuple[str, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.base_feature_names) != len(set(self.base_feature_names)):
            raise ValueError("compact model contains duplicate base features")
        self._base_feature_index = {
            name: index for index, name in enumerate(self.base_feature_names)
        }
        self._base_feature_names_tuple = tuple(self.base_feature_names)
        try:
            self._input_feature_indices = [
                (mode, self._base_feature_index[name])
                for mode, name in self.input_features
            ]
        except KeyError as error:
            raise ValueError(
                f"compact model input references an unknown base feature: {error.args[0]}"
            ) from error

    @property
    def feature_names(self) -> list[str]:
        return self.base_feature_names

    @property
    def inference_backend(self) -> str:
        return (
            "native-portable-tree"
            if self.native_predictor is not None
            else "python-portable-tree"
        )

    def pair_vector(self, left: dict[str, Any], right: dict[str, Any]) -> list[float]:
        left_values = self._row_values(left)
        right_values = self._row_values(right)
        result = []
        for mode, index in self._input_feature_indices:
            first = left_values[index]
            second = right_values[index]
            result.append(first - second if mode == "delta" else (first + second) / 2.0)
        return result

    def _row_values(
        self, row: dict[str, Any]
    ) -> list[float] | tuple[float, ...]:
        if "feature_values" in row:
            names = row.get("feature_names", ())
            if str(row.get("feature_profile")) != self.profile or (
                names
                if isinstance(names, tuple)
                else tuple(map(str, names))
            ) != self._base_feature_names_tuple:
                raise ValueError("dense feature row does not match the compact model")
            raw_values = row["feature_values"]
            dense = (
                raw_values
                if isinstance(raw_values, tuple)
                else tuple(map(float, raw_values))
            )
            if len(dense) != len(self.base_feature_names):
                raise ValueError("dense feature row has the wrong dimension")
            return dense
        features = dict(row["features"][self.profile])
        return [float(features.get(name, 0.0)) for name in self.base_feature_names]

    def pair_vectors(
        self, rows: list[dict[str, Any]]
    ) -> tuple[list[list[float]], list[list[float]], list[tuple[int, int]]]:
        values = [self._row_values(row) for row in rows]
        forward: list[list[float]] = []
        reverse: list[list[float]] = []
        pairs: list[tuple[int, int]] = []
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                left_values = values[left]
                right_values = values[right]
                forward.append(
                    [
                        left_values[index] - right_values[index]
                        if mode == "delta"
                        else (left_values[index] + right_values[index]) / 2.0
                        for mode, index in self._input_feature_indices
                    ]
                )
                reverse.append(
                    [
                        right_values[index] - left_values[index]
                        if mode == "delta"
                        else (right_values[index] + left_values[index]) / 2.0
                        for mode, index in self._input_feature_indices
                    ]
                )
                pairs.append((left, right))
        return forward, reverse, pairs

    def score_candidates(self, rows: list[dict[str, Any]]) -> list[float] | None:
        native = getattr(self.native_predictor, "score_pairwise_dense", None)
        if not callable(native):
            return None
        values = [self._row_values(row) for row in rows]
        modes = [0 if mode == "delta" else 1 for mode, _ in self._input_feature_indices]
        indices = [index for _, index in self._input_feature_indices]
        return list(map(float, native(values, modes, indices)))

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


def load_compact_model(payload: dict[str, Any]) -> CompactPortablePairwiseModel:
    if str(payload.get("schema")) != COMPACT_MODEL_SCHEMA:
        raise ValueError("unexpected compact portable model schema")
    if str(payload.get("feature_schema_id")) != FEATURE_SCHEMA_ID or str(
        payload.get("feature_schema_sha256")
    ) != FEATURE_SCHEMA_SHA256:
        raise ValueError("compact model feature schema mismatch")
    input_features = [
        (str(value["mode"]), str(value["name"]))
        for value in payload.get("input_features", [])
    ]
    if len(input_features) != int(payload.get("input_dimension", -1)):
        raise ValueError("compact model input dimension mismatch")
    if any(mode not in {"delta", "shared"} for mode, _ in input_features):
        raise ValueError("compact model contains an invalid input mode")
    return CompactPortablePairwiseModel(
        profile=str(payload["profile"]),
        input_features=input_features,
        base_feature_names=list(map(str, payload.get("base_feature_names", []))),
        baseline=float(payload["baseline"]),
        trees=list(payload["trees"]),
        semantic_fingerprint=str(payload["semantic_fingerprint"]),
    )


def _connect_native_predictor(
    model: CompactPortablePairwiseModel,
) -> CompactPortablePairwiseModel:
    try:
        import lns2_env
    except ImportError:
        return model
    predictor_type = getattr(lns2_env, "PortableTreeEnsemble", None)
    if predictor_type is None:
        return model
    model.native_predictor = predictor_type(model.baseline, model.trees)
    return model


def compact_runtime_model(model: Any) -> CompactPortablePairwiseModel:
    payload = {
        "schema": "lns2.portable_pairwise_hist_gbdt.v1",
        "schema_version": 1,
        "profile": str(model.profile),
        "source_model_sha256": "",
        "feature_names": list(map(str, model.feature_names)),
        "baseline": float(model.baseline),
        "trees": list(model.trees),
    }
    compact_payload = compact_portable_payload(payload)
    return _connect_native_predictor(load_compact_model(compact_payload))


@dataclass
class ControllerBundleV2:
    main_models: dict[str, CompactPortablePairwiseModel]
    main_ranges: dict[str, dict[str, tuple[float, float]]]
    pruner_model: CompactPortablePairwiseModel | None
    pruner_ranges: dict[str, tuple[float, float]]
    pruner_threshold: float | None
    manifest: dict[str, Any]
    promotion_report: dict[str, Any]


def export_controller_bundle(
    source_bundle: str | Path,
    output: str | Path,
    *,
    pruner_payload: dict[str, Any] | None = None,
    pruner_ranges: dict[str, tuple[float, float]] | None = None,
    pruner_threshold: float | None = None,
    pruner_metadata: dict[str, Any] | None = None,
    promotion_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_root = Path(source_bundle).resolve()
    output_root = Path(output).resolve()
    source_manifest_path = source_root / "portable_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    source_rows = {
        str(row["profile"]): dict(row) for row in source_manifest.get("models", [])
    }
    main_rows: dict[str, Any] = {}
    main_ranges: dict[str, dict[str, list[float]]] = {}
    stored_ranges = dict(source_manifest.get("feature_ranges", {}))
    for profile in sorted(PROFILE_FEATURE_NAMES):
        if profile not in source_rows:
            raise ValueError(f"source bundle is missing {profile}")
        source_path = source_root / str(source_rows[profile]["file"])
        source_payload = _read_json(source_path)
        compact_payload = compact_portable_payload(source_payload)
        destination = output_root / f"main__{profile}.json"
        _write_json(destination, compact_payload)
        used_ids = [
            PROFILE_FEATURE_NAMES[profile].index(name)
            for name in compact_payload["base_feature_names"]
        ]
        main_rows[profile] = {
            "file": destination.relative_to(output_root).as_posix(),
            "sha256": _file_sha256(destination),
            "pairwise_input_dimension": int(compact_payload["input_dimension"]),
            "used_feature_names": list(compact_payload["base_feature_names"]),
            "used_feature_ids": used_ids,
            "source_semantic_fingerprint": compact_payload[
                "source_semantic_fingerprint"
            ],
            "semantic_fingerprint": compact_payload["semantic_fingerprint"],
        }
        profile_ranges = dict(stored_ranges.get(profile, {}))
        main_ranges[profile] = {
            name: [float(profile_ranges[name][0]), float(profile_ranges[name][1])]
            for name in compact_payload["base_feature_names"]
            if name in profile_ranges
        }
        if set(main_ranges[profile]) != set(compact_payload["base_feature_names"]):
            missing = sorted(
                set(compact_payload["base_feature_names"]) - set(main_ranges[profile])
            )
            raise ValueError(f"source bundle lacks ranges for {profile}: {missing}")

    pruner_row = None
    if pruner_payload is not None:
        compact_pruner = (
            pruner_payload
            if str(pruner_payload.get("schema")) == COMPACT_MODEL_SCHEMA
            else compact_portable_payload(pruner_payload)
        )
        if str(compact_pruner.get("profile")) != "proposal_dynamic":
            raise ValueError("the candidate pruner must use proposal_dynamic features")
        if pruner_threshold is None or not 0.5 <= float(pruner_threshold) <= 1.0:
            raise ValueError("a promoted pruner requires a threshold in [0.5, 1.0]")
        ranges = dict(pruner_ranges or {})
        expected = set(PROFILE_FEATURE_NAMES["proposal_dynamic"])
        if set(ranges) != expected:
            raise ValueError("pruner ranges must cover the complete feature-v2 proposal schema")
        destination = output_root / "proposal_pruner_v2.json"
        _write_json(destination, compact_pruner)
        pruner_row = {
            "schema": "lns2.proposal_pruner.v2",
            "file": destination.relative_to(output_root).as_posix(),
            "sha256": _file_sha256(destination),
            "threshold": float(pruner_threshold),
            "pairwise_input_dimension": int(compact_pruner["input_dimension"]),
            "used_feature_names": list(compact_pruner["base_feature_names"]),
            "feature_ranges": {
                name: [float(ranges[name][0]), float(ranges[name][1])]
                for name in sorted(ranges)
            },
            **dict(pruner_metadata or {}),
        }

    report = dict(promotion_report or {})
    exact_passed = bool(report.get("exact_acceleration_passed", False))
    performance_passed = bool(report.get("feature_performance_passed", False))
    pruning_passed = bool(report.get("pruning_promotion_passed", False))
    if pruning_passed and pruner_row is None:
        raise ValueError("promotion report passes pruning but no pruner was supplied")
    default_controller = (
        "v2-cascade"
        if pruning_passed and pruner_row is not None
        else "v2-full" if exact_passed and performance_passed else "v1-full"
    )
    report.update(
        {
            "schema": "lns2.controller_promotion_report.v2",
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "default_controller": default_controller,
        }
    )
    report_path = output_root / "promotion_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema": CONTROLLER_BUNDLE_SCHEMA,
        "schema_version": CONTROLLER_BUNDLE_VERSION,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_dimensions": {
            profile: len(names) for profile, names in PROFILE_FEATURE_NAMES.items()
        },
        "main_rankers": main_rows,
        "main_ranges": main_ranges,
        "main_ranker_semantic_fingerprint": main_rows["realized_dynamic"][
            "source_semantic_fingerprint"
        ],
        "pruner": pruner_row,
        "fallback_rules": {
            "missing_or_unexpected_family": "full_pool",
            "invalid_family_cardinality": "full_pool",
            "non_finite_feature": "full_pool",
            "unsupported_actual_size": "full_pool",
            "maximum_outside_feature_fraction": 0.10,
            "schema_mismatch": "reject_bundle",
        },
        "default_controller": default_controller,
        "source_bundle": {
            "manifest_sha256": _file_sha256(source_manifest_path),
            "confirmation_labels_seen": bool(
                source_manifest.get("confirmation_labels_seen", True)
            ),
        },
        "promotion_report": {
            "file": report_path.relative_to(output_root).as_posix(),
            "sha256": _file_sha256(report_path),
        },
        "storage_format_dependency": None,
    }
    _write_json(output_root / "controller_manifest.json", manifest)
    return manifest


def load_controller_bundle(path: str | Path) -> ControllerBundleV2:
    root = Path(path).resolve()
    manifest = _read_json(root / "controller_manifest.json")
    if str(manifest.get("schema")) != CONTROLLER_BUNDLE_SCHEMA:
        raise ValueError("unexpected controller bundle schema")
    if int(manifest.get("schema_version", -1)) != CONTROLLER_BUNDLE_VERSION:
        raise ValueError("unsupported controller bundle version")
    if str(manifest.get("feature_schema_id")) != FEATURE_SCHEMA_ID or str(
        manifest.get("feature_schema_sha256")
    ) != FEATURE_SCHEMA_SHA256:
        raise ValueError("controller bundle feature schema mismatch")
    models = {}
    for profile, row_value in dict(manifest.get("main_rankers", {})).items():
        row = dict(row_value)
        model_path = root / str(row["file"])
        if _file_sha256(model_path) != str(row["sha256"]):
            raise ValueError(f"controller main model SHA256 mismatch: {profile}")
        model = _connect_native_predictor(load_compact_model(_read_json(model_path)))
        if model.profile != profile:
            raise ValueError(f"controller main model profile mismatch: {profile}")
        models[profile] = model
    if set(models) != set(PROFILE_FEATURE_NAMES):
        raise ValueError("controller bundle has an incomplete main model set")
    ranges = {
        str(profile): {
            str(name): (float(bounds[0]), float(bounds[1]))
            for name, bounds in dict(profile_ranges).items()
        }
        for profile, profile_ranges in dict(manifest.get("main_ranges", {})).items()
    }
    report_row = dict(manifest.get("promotion_report", {}))
    report_path = root / str(report_row["file"])
    if _file_sha256(report_path) != str(report_row["sha256"]):
        raise ValueError("controller promotion report SHA256 mismatch")
    report = _read_json(report_path)
    pruner_model = None
    pruner_ranges: dict[str, tuple[float, float]] = {}
    pruner_threshold = None
    if manifest.get("pruner") is not None:
        row = dict(manifest["pruner"])
        model_path = root / str(row["file"])
        if _file_sha256(model_path) != str(row["sha256"]):
            raise ValueError("controller pruner SHA256 mismatch")
        pruner_model = _connect_native_predictor(
            load_compact_model(_read_json(model_path))
        )
        pruner_threshold = float(row["threshold"])
        pruner_ranges = {
            str(name): (float(bounds[0]), float(bounds[1]))
            for name, bounds in dict(row["feature_ranges"]).items()
        }
    return ControllerBundleV2(
        main_models=models,
        main_ranges=ranges,
        pruner_model=pruner_model,
        pruner_ranges=pruner_ranges,
        pruner_threshold=pruner_threshold,
        manifest=manifest,
        promotion_report=report,
    )


def update_controller_promotion_evidence(
    controller_bundle: str | Path,
    *,
    performance_benchmark: str | Path | None = None,
    quick_status: str | Path | None = None,
    formal_status: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(controller_bundle).resolve()
    loaded = load_controller_bundle(root)
    manifest = dict(loaded.manifest)
    report_row = dict(manifest["promotion_report"])
    report_path = root / str(report_row["file"])
    report = dict(loaded.promotion_report)

    if performance_benchmark is not None:
        benchmark_path = Path(performance_benchmark).resolve()
        benchmark = _read_json(benchmark_path)
        gate = dict(
            benchmark.get("performance_gate") or benchmark.get("python_gate") or {}
        )
        maximum_delta = max(
            (
                float(row.get("maximum_feature_delta", float("inf")))
                for row in benchmark.get("cases", [])
            ),
            default=float("inf"),
        )
        performance_passed = (
            bool(gate.get("passed"))
            and maximum_delta <= 1e-12
            and not bool(benchmark.get("native_backend_required"))
        )
        try:
            benchmark_reference = benchmark_path.relative_to(root).as_posix()
        except ValueError:
            benchmark_reference = str(benchmark_path)
        report["feature_performance_passed"] = performance_passed
        report["performance_benchmark"] = {
            "file": benchmark_reference,
            "sha256": _file_sha256(benchmark_path),
            "feature_backend": benchmark.get("feature_backend"),
            "maximum_feature_delta": maximum_delta,
            "overall_feature_time_reduction": benchmark.get("overall", {}).get(
                "feature_time_reduction"
            ),
            "overall_feature_speedup": benchmark.get("overall", {}).get(
                "speedup"
            ),
            "estimated_controller_time_reduction": benchmark.get(
                "overall", {}
            ).get("estimated_controller_time_reduction"),
            "estimated_controller_speedup": benchmark.get("overall", {}).get(
                "estimated_controller_speedup"
            ),
            "estimated_end_to_end_time_reduction": benchmark.get(
                "overall", {}
            ).get("estimated_end_to_end_time_reduction"),
            "estimated_end_to_end_speedup": benchmark.get("overall", {}).get(
                "estimated_end_to_end_speedup"
            ),
            "end_to_end_measurement": benchmark.get("overall", {}).get(
                "end_to_end_measurement"
            ),
            "performance_gate": gate,
            "maze600_feature_time_reduction": next(
                (
                    row.get("feature_time_reduction")
                    for row in benchmark.get("cases", [])
                    if str(row.get("case")) == "maze600"
                ),
                None,
            ),
            "passed": performance_passed,
        }

    for name, value in (("quick", quick_status), ("formal", formal_status)):
        if value is None:
            continue
        status_path = Path(value).resolve()
        status = _read_json(status_path)
        report[f"{name}_run"] = {
            "file": str(status_path),
            "sha256": _file_sha256(status_path),
            "status": status.get("status"),
            "controller": status.get("controller"),
            "valid_trace_count": status.get("valid_trace_count"),
            "conclusion": status.get("conclusion"),
            "passed": str(status.get("status")) == "complete",
        }

    pruner_present = manifest.get("pruner") is not None
    quick_passed = bool(report.get("quick_run", {}).get("passed"))
    formal_passed = bool(report.get("formal_run", {}).get("passed"))
    report["pruning_promotion_passed"] = bool(
        pruner_present
        and report.get("pruner_offline_validation_passed")
        and quick_passed
        and formal_passed
    )
    report["default_controller"] = (
        "v2-cascade"
        if report["pruning_promotion_passed"]
        else "v2-full"
        if report.get("exact_acceleration_passed")
        and report.get("feature_performance_passed")
        else "v1-full"
    )
    _write_json(report_path, report)
    manifest["default_controller"] = report["default_controller"]
    manifest["promotion_report"] = {
        "file": report_path.relative_to(root).as_posix(),
        "sha256": _file_sha256(report_path),
    }
    _write_json(root / "controller_manifest.json", manifest)
    return {"manifest": manifest, "promotion_report": report}


__all__ = [
    "COMPACT_MODEL_SCHEMA",
    "CONTROLLER_BUNDLE_SCHEMA",
    "CONTROLLER_BUNDLE_VERSION",
    "CompactPortablePairwiseModel",
    "ControllerBundleV2",
    "compact_runtime_model",
    "compact_portable_payload",
    "export_controller_bundle",
    "load_controller_bundle",
    "update_controller_promotion_evidence",
    "load_compact_model",
    "portable_model_semantic_fingerprint",
]
