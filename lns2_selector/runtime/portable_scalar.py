from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Iterable

from lns2_selector.runtime.fingerprints import semantic_fingerprint


PORTABLE_SCALAR_MODEL_SCHEMA = "lns2.portable_scalar_hist_gbdt.v1"


def _row_features(row: dict[str, Any], profile: str) -> dict[str, float]:
    if "feature_values" in row:
        if str(row.get("feature_profile")) != profile:
            raise ValueError("dense row profile does not match portable model")
        names = tuple(map(str, row.get("feature_names", ())))
        values = tuple(map(float, row.get("feature_values", ())))
        if len(names) != len(values) or len(names) != len(set(names)):
            raise ValueError("invalid dense feature row")
        return dict(zip(names, values))
    return {
        str(name): float(value)
        for name, value in dict(row["features"][profile]).items()
    }


@dataclass
class PortableScalarModel:
    name: str
    profile: str
    feature_names: list[str]
    baseline: float
    trees: list[list[dict[str, Any]]]
    transform: str
    semantic_fingerprint: str
    input_precision: str = "float64"
    native_predictor: Any | None = None
    declared_feature_count: int | None = None

    @property
    def source_feature_count(self) -> int:
        return int(self.declared_feature_count or len(self.feature_names))

    @property
    def inference_backend(self) -> str:
        return (
            "native-portable-tree"
            if self.native_predictor is not None
            else "python-portable-tree"
        )

    def _vectors(self, rows: Iterable[dict[str, Any]]) -> list[list[float]]:
        vectors = []
        for row in rows:
            features = _row_features(row, self.profile)
            missing = set(self.feature_names) - set(features)
            if missing:
                raise ValueError(
                    f"portable model row is missing features: {sorted(missing)}"
                )
            vector = [float(features[name]) for name in self.feature_names]
            if self.input_precision == "float32":
                vector = [
                    struct.unpack("!f", struct.pack("!f", value))[0]
                    for value in vector
                ]
            if any(not math.isfinite(value) for value in vector):
                raise ValueError("portable model received non-finite features")
            vectors.append(vector)
        return vectors

    def _python_raw(self, vectors: list[list[float]]) -> list[float]:
        outputs = []
        for vector in vectors:
            raw = float(self.baseline)
            for nodes in self.trees:
                index = 0
                while not bool(nodes[index]["is_leaf"]):
                    node = nodes[index]
                    value = vector[int(node["feature_idx"])]
                    go_left = (
                        math.isnan(value) and bool(node["missing_go_to_left"])
                    ) or (
                        not math.isnan(value)
                        and value <= float(node["num_threshold"])
                    )
                    index = int(node["left"] if go_left else node["right"])
                raw += float(nodes[index]["value"])
            outputs.append(raw)
        return outputs

    def predict(self, rows: list[dict[str, Any]]) -> list[float]:
        vectors = self._vectors(rows)
        if self.native_predictor is not None:
            method = (
                self.native_predictor.predict_positive
                if self.transform == "sigmoid"
                else self.native_predictor.predict_raw
            )
            return list(map(float, method(vectors)))
        raw = self._python_raw(vectors)
        if self.transform == "identity":
            return raw
        values = []
        for value in raw:
            if value >= 0.0:
                values.append(1.0 / (1.0 + math.exp(-value)))
            else:
                exponential = math.exp(value)
                values.append(exponential / (1.0 + exponential))
        return values


def _connect_native(model: PortableScalarModel) -> PortableScalarModel:
    try:
        import lns2_env
    except ImportError:
        return model
    predictor = getattr(lns2_env, "PortableTreeEnsemble", None)
    if predictor is None:
        return model
    model.native_predictor = predictor(model.baseline, model.trees)
    return model


def compact_portable_scalar_model(
    model: PortableScalarModel,
) -> PortableScalarModel:
    """Project a fitted scalar ensemble onto features used by split nodes."""

    used_indices = sorted(
        {
            int(node["feature_idx"])
            for tree in model.trees
            for node in tree
            if not bool(node["is_leaf"])
        }
    )
    if any(index < 0 or index >= len(model.feature_names) for index in used_indices):
        raise ValueError("portable scalar model references an invalid feature index")
    if used_indices == list(range(len(model.feature_names))):
        return _connect_native(model)
    compact_index = {
        source_index: target_index
        for target_index, source_index in enumerate(used_indices)
    }
    compact_trees = []
    for source_tree in model.trees:
        output_tree = []
        for source_node in source_tree:
            node = dict(source_node)
            node["feature_idx"] = (
                0
                if bool(node["is_leaf"])
                else compact_index[int(node["feature_idx"])]
            )
            output_tree.append(node)
        compact_trees.append(output_tree)
    return _connect_native(
        PortableScalarModel(
            name=model.name,
            profile=model.profile,
            feature_names=[model.feature_names[index] for index in used_indices],
            baseline=model.baseline,
            trees=compact_trees,
            transform=model.transform,
            semantic_fingerprint=model.semantic_fingerprint,
            input_precision=model.input_precision,
            declared_feature_count=model.source_feature_count,
        )
    )


def load_portable_scalar_model(
    payload: dict[str, Any], *, compact_features: bool = False
) -> PortableScalarModel:
    if str(payload.get("schema")) != PORTABLE_SCALAR_MODEL_SCHEMA:
        raise ValueError("unexpected portable scalar model schema")
    transform = str(payload.get("transform"))
    if transform not in {"identity", "sigmoid"}:
        raise ValueError("portable scalar model has an invalid transform")
    input_precision = str(payload.get("input_precision", "float64"))
    if input_precision not in {"float32", "float64"}:
        raise ValueError("portable scalar model has invalid input precision")
    names = list(map(str, payload.get("feature_names", ())))
    if not names or len(names) != len(set(names)):
        raise ValueError("portable scalar model has invalid feature names")
    fingerprint_payload = {
        "name": payload.get("name"),
        "profile": payload.get("profile"),
        "feature_names": names,
        "baseline": payload.get("baseline"),
        "trees": payload.get("trees"),
        "transform": transform,
    }
    if "input_precision" in payload:
        fingerprint_payload["input_precision"] = input_precision
    expected = semantic_fingerprint(fingerprint_payload)
    if str(payload.get("semantic_fingerprint")) != expected:
        raise ValueError("portable scalar model semantic fingerprint mismatch")
    model = PortableScalarModel(
        name=str(payload["name"]),
        profile=str(payload["profile"]),
        feature_names=names,
        baseline=float(payload["baseline"]),
        trees=list(payload["trees"]),
        transform=transform,
        semantic_fingerprint=expected,
        input_precision=input_precision,
        declared_feature_count=len(names),
    )
    return (
        compact_portable_scalar_model(model)
        if compact_features
        else _connect_native(model)
    )
