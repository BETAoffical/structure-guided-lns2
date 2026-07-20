from __future__ import annotations

import collections
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments._common import (
    mean as _mean,
    relative_improvement as _relative_improvement,
)
from experiments.closed_loop_confirmation import online_candidate_rows
from experiments.repair_collection import (
    SCHEMA_VERSION,
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from research.studies.sequential.repair_order_probe import repair_order_for_policy
from research.studies.sequential.sequential_credit_audit import _quantile, _sha256


SCHEMA = "lns2.contextual_repair_order_audit.v1"
INDEX_SCHEMA = "lns2.contextual_repair_order_index.v1"
POLICIES = (
    "agent_id_ascending",
    "conflict_degree_descending",
    "delay_descending",
    "path_length_descending",
)
FORBIDDEN_FEATURE_PARTS = (
    "outcome",
    "after",
    "runtime",
    "generated",
    "expanded",
    "conflict_auc",
    "final_conflicts",
    "feasible",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported contextual repair-order audit config")
    if tuple(map(str, config.get("policies", []))) != POLICIES:
        raise ValueError("repair-order policies differ from the registration")
    expected = {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": 20260714,
    }
    if dict(config.get("model_parameters", {})) != expected:
        raise ValueError("contextual repair-order model parameters changed")
    if int(config.get("bootstrap_samples", 0)) != 5000:
        raise ValueError("formal audit requires 5,000 map bootstrap samples")
    if int(config.get("permutation_samples", 0)) != 500:
        raise ValueError("formal audit requires 500 context permutations")
    if int(config.get("horizon", 0)) != 4:
        raise ValueError("contextual repair-order audit requires Horizon 4")


def _registered_inputs(config: dict[str, Any]) -> dict[str, Any]:
    source = dict(config["source"])
    root = _resolve(source["root"])
    files = {
        "selected_states": root / str(source["selected_states"]),
        "condition_index": root / str(source["condition_index"]),
        "trial_manifest": root / str(source["trial_manifest"]),
        "report": root / str(source["report"]),
        "run_config": root / str(source["run_config"]),
    }
    hashes = {}
    for name, path in files.items():
        if not path.is_file():
            raise ValueError(f"registered input is missing: {path}")
        hashes[name] = _sha256(path)
        expected = str(source[f"{name}_sha256"]).lower()
        if hashes[name] != expected:
            raise ValueError(f"registered input SHA256 mismatch: {name}")
    return {
        "root": str(root),
        "paths": {name: str(path) for name, path in files.items()},
        "sha256": hashes,
    }


def _ordered_features(
    state: dict[str, Any], agents: list[int], policy: str
) -> dict[str, float]:
    order = repair_order_for_policy(state, agents, policy)
    by_id = {int(row["id"]): row for row in state["agents"]}
    selected = set(order)
    size = len(order)
    if size == 0:
        raise ValueError("repair-order candidate cannot be empty")
    result = {f"order.rule.{name}": float(name == policy) for name in POLICIES}
    measurements = {
        "conflict_degree": [float(by_id[agent]["conflict_degree"]) for agent in order],
        "delay": [float(by_id[agent]["delay"]) for agent in order],
        "path_length": [float(len(by_id[agent]["path"])) for agent in order],
        "path_cost": [float(by_id[agent]["path_cost"]) for agent in order],
    }
    middle = max(1, (size + 1) // 2)
    weights = [(size - index) / size for index in range(size)]
    for name, values in measurements.items():
        first = values[:middle]
        second = values[middle:]
        result[f"order.{name}.first"] = values[0]
        result[f"order.{name}.last"] = values[-1]
        result[f"order.{name}.weighted_mean"] = sum(
            value * weight for value, weight in zip(values, weights)
        ) / sum(weights)
        result[f"order.{name}.half_delta"] = _mean(first) - _mean(second or first)
    positions = {agent: index for index, agent in enumerate(order)}
    internal_distances = [
        abs(positions[int(left)] - positions[int(right)]) / max(1, size - 1)
        for left, right in state["conflict_edges"]
        if int(left) in selected and int(right) in selected
    ]
    boundary_agents = {
        int(left)
        for left, right in state["conflict_edges"]
        if int(left) in selected and int(right) not in selected
    } | {
        int(right)
        for left, right in state["conflict_edges"]
        if int(right) in selected and int(left) not in selected
    }
    result["order.internal_edge_distance.mean"] = _mean(internal_distances)
    result["order.internal_edge_distance.max"] = max(internal_distances, default=0.0)
    result["order.boundary_agents.first_half_fraction"] = _mean(
        positions[agent] < middle for agent in boundary_agents
    )
    result["order.selected_size"] = float(size)
    return result


def build_index(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_config(config)
    registered = _registered_inputs(config)
    selected = _read_jsonl(Path(registered["paths"]["selected_states"]))
    outcomes = _read_jsonl(Path(registered["paths"]["condition_index"]))
    outcome_by_key = {
        (str(row["state_id"]), str(row["candidate_id"])): row for row in outcomes
    }
    if len(selected) != 24 or len(outcomes) != 144:
        raise ValueError("registered audit requires 24 states and 144 decisions")
    rows = []
    seen = set()
    feature_names: set[str] = set()
    for source in selected:
        state_id = str(source["state_id"])
        state = dict(source["state"])
        initial_conflicts = int(state["num_of_colliding_pairs"])
        if initial_conflicts <= 0:
            raise ValueError("repair-order audit states must have conflicts")
        candidates = list(source["probe_candidates"])
        realized = {
            str(row["candidate_id"]): dict(row["features"]["realized_dynamic"])
            for row in online_candidate_rows(state, candidates)
        }
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            key = (state_id, candidate_id)
            if key not in outcome_by_key:
                raise ValueError(f"missing deterministic outcome: {key}")
            indexed = outcome_by_key[key]
            for policy in POLICIES:
                outcome = dict(indexed["deterministic_h4"][policy])
                features = {
                    **{
                        f"realized.{name}": float(value)
                        for name, value in realized[candidate_id].items()
                        if not any(part in name.lower() for part in FORBIDDEN_FEATURE_PARTS)
                    },
                    **_ordered_features(state, list(map(int, candidate["agents"])), policy),
                }
                lowered = " ".join(features).lower()
                if any(part in lowered for part in FORBIDDEN_FEATURE_PARTS):
                    raise ValueError("contextual repair-order feature leakage detected")
                feature_names.update(features)
                row_id = (state_id, candidate_id, policy)
                if row_id in seen:
                    raise ValueError("duplicate contextual repair-order row")
                seen.add(row_id)
                rows.append(
                    {
                        "schema": INDEX_SCHEMA,
                        "schema_version": SCHEMA_VERSION,
                        "row_id": _fingerprint(row_id),
                        "state_id": state_id,
                        "decision_id": f"{state_id}::{candidate_id}",
                        "candidate_id": candidate_id,
                        "map_id": str(source["map_id"]),
                        "task_id": str(source["task_id"]),
                        "policy": policy,
                        "initial_conflicts": initial_conflicts,
                        "features": features,
                        "target": {
                            "normalized_h4_auc": float(outcome["conflict_auc"])
                            / max(1.0, 4.0 * initial_conflicts),
                            "conflict_auc": float(outcome["conflict_auc"]),
                            "feasible_rate": float(outcome["feasible_rate"]),
                            "final_conflicts": float(outcome["final_conflicts"]),
                        },
                    }
                )
    integrity = {
        "state_count": len({row["state_id"] for row in rows}),
        "decision_count": len({row["decision_id"] for row in rows}),
        "row_count": len(rows),
        "map_count": len({row["map_id"] for row in rows}),
        "feature_count": len(feature_names),
        "forbidden_feature_count": 0,
        "passed": len(rows) == 576 and len(seen) == 576,
        "registered_inputs": registered,
    }
    return rows, integrity


@dataclass
class PortableRegressor:
    feature_names: list[str]
    baseline: float
    trees: list[list[dict[str, Any]]]

    def predict(self, vectors: list[list[float]]) -> list[float]:
        predictions = []
        for vector in vectors:
            value = self.baseline
            for nodes in self.trees:
                index = 0
                while not bool(nodes[index]["is_leaf"]):
                    node = nodes[index]
                    feature = float(vector[int(node["feature_idx"])])
                    go_left = (
                        math.isnan(feature) and bool(node["missing_go_to_left"])
                    ) or (
                        not math.isnan(feature)
                        and feature <= float(node["num_threshold"])
                    )
                    index = int(node["left"] if go_left else node["right"])
                value += float(nodes[index]["value"])
            predictions.append(value)
        return predictions


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({name for row in rows for name in row["features"]})


def _vectors(rows: list[dict[str, Any]], names: list[str]):
    import numpy as np

    return np.asarray(
        [[float(row["features"].get(name, 0.0)) for name in names] for row in rows],
        dtype=float,
    )


def _fit(rows: list[dict[str, Any]], parameters: dict[str, Any]):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    names = _feature_names(rows)
    counts = collections.Counter(str(row["state_id"]) for row in rows)
    weights = np.asarray([1.0 / counts[str(row["state_id"])] for row in rows], dtype=float)
    estimator = HistGradientBoostingRegressor(**parameters)
    estimator.fit(
        _vectors(rows, names),
        np.asarray([float(row["target"]["normalized_h4_auc"]) for row in rows]),
        sample_weight=weights,
    )
    state_weights = collections.defaultdict(float)
    for row, weight in zip(rows, weights):
        state_weights[str(row["state_id"])] += float(weight)
    if not all(math.isclose(value, 1.0, abs_tol=1e-12) for value in state_weights.values()):
        raise ValueError("repair-order training does not give each state total weight one")
    return estimator, names


def _portable_payload(estimator: Any, names: list[str], source_sha: str) -> dict[str, Any]:
    trees = []
    for stage in estimator._predictors:
        if len(stage) != 1:
            raise ValueError("portable repair-order regressor expects one tree per stage")
        nodes = []
        for node in stage[0].nodes:
            if bool(node["is_categorical"]):
                raise ValueError("portable repair-order regressor does not support categorical splits")
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
        "schema": "lns2.portable_contextual_repair_order_gbdt.v1",
        "schema_version": SCHEMA_VERSION,
        "source_model_sha256": source_sha,
        "feature_names": names,
        "baseline": float(estimator._baseline_prediction[0, 0]),
        "trees": trees,
    }


def fit_frozen_model(
    rows: list[dict[str, Any]], config: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    import pickle

    report_path = output_root / "contextual_repair_order_audit.json"
    if not report_path.is_file() or not bool(_read_json(report_path).get("passed")):
        raise ValueError("portable model export requires a passing formal Stage 1 report")
    estimator, names = _fit(rows, dict(config["model_parameters"]))
    model_path = output_root / "frozen" / "contextual_repair_order.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as stream:
        pickle.dump({"feature_names": names, "estimator": estimator}, stream)
    payload = _portable_payload(estimator, names, _sha256(model_path))
    portable_path = output_root / "frozen" / "contextual_repair_order.json"
    _write_json(portable_path, payload)
    portable = PortableRegressor(names, float(payload["baseline"]), list(payload["trees"]))
    vectors = _vectors(rows, names)
    native = list(map(float, estimator.predict(vectors)))
    reproduced = portable.predict(vectors.tolist())
    maximum_delta = max(abs(left - right) for left, right in zip(native, reproduced))
    manifest = {
        "schema": "lns2.contextual_repair_order_freeze.v1",
        "schema_version": SCHEMA_VERSION,
        "model_sha256": _sha256(model_path),
        "portable_sha256": _sha256(portable_path),
        "index_sha256": _sha256(output_root / "contextual_repair_order_index.jsonl"),
        "audit_report_sha256": _sha256(report_path),
        "feature_count": len(names),
        "maximum_portable_prediction_delta": maximum_delta,
        "portable_parity": maximum_delta <= 1e-12,
        "independent_confirmation_labels_seen": False,
    }
    if not manifest["portable_parity"]:
        raise ValueError("portable repair-order model does not reproduce sklearn")
    _write_json(output_root / "frozen" / "freeze_manifest.json", manifest)
    return manifest


def _group_decisions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["decision_id"])].append(row)
    for decision, values in grouped.items():
        if {str(row["policy"]) for row in values} != set(POLICIES):
            raise ValueError(f"decision lacks the four registered policies: {decision}")
    return grouped


def _best_policy(rows: list[dict[str, Any]]) -> str:
    values = collections.defaultdict(list)
    for row in rows:
        values[str(row["policy"])].append(float(row["target"]["normalized_h4_auc"]))
    return min(POLICIES, key=lambda policy: (_mean(values[policy]), POLICIES.index(policy)))


def _predict_records(
    train: list[dict[str, Any]], test: list[dict[str, Any]], parameters: dict[str, Any]
) -> list[dict[str, Any]]:
    estimator, names = _fit(train, parameters)
    fixed_policy = _best_policy(train)
    grouped = _group_decisions(test)
    records = []
    for decision_id, values in sorted(grouped.items()):
        by_policy = {str(row["policy"]): row for row in values}
        predictions = estimator.predict(_vectors(values, names))
        predicted = min(
            range(len(values)),
            key=lambda index: (round(float(predictions[index]), 12), POLICIES.index(str(values[index]["policy"]))),
        )
        model = values[predicted]
        fixed = by_policy[fixed_policy]
        oracle = min(
            values,
            key=lambda row: (float(row["target"]["normalized_h4_auc"]), POLICIES.index(str(row["policy"]))),
        )
        records.append(
            {
                "decision_id": decision_id,
                "state_id": str(model["state_id"]),
                "candidate_id": str(model["candidate_id"]),
                "map_id": str(model["map_id"]),
                "model_policy": str(model["policy"]),
                "fixed_policy": fixed_policy,
                "oracle_policy": str(oracle["policy"]),
                "model_prediction": float(predictions[predicted]),
                "model": model["target"],
                "fixed": fixed["target"],
                "oracle": oracle["target"],
                "uniform": {
                    name: _mean(row["target"][name] for row in values)
                    for name in ("normalized_h4_auc", "conflict_auc", "feasible_rate", "final_conflicts")
                },
            }
        )
    return records


def _map_bootstrap(records: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    by_map: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for row in records:
        fixed = float(row["fixed"]["normalized_h4_auc"])
        model = float(row["model"]["normalized_h4_auc"])
        by_map[str(row["map_id"])].append((fixed, model))
    means = {
        name: _relative_improvement(
            _mean(fixed for fixed, _ in values),
            _mean(model for _, model in values),
        )
        for name, values in by_map.items()
    }
    rng = random.Random(seed)
    maps = sorted(by_map)
    estimates = []
    for _ in range(samples):
        sampled = [rng.choice(maps) for _ in maps]
        fixed = [value for name in sampled for value, _ in by_map[name]]
        model = [value for name in sampled for _, value in by_map[name]]
        estimates.append(_relative_improvement(_mean(fixed), _mean(model)))
    return {
        "unit": "map_id",
        "samples": samples,
        "mean": _relative_improvement(
            _mean(value for values in by_map.values() for value, _ in values),
            _mean(value for values in by_map.values() for _, value in values),
        ),
        "ci95": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
        "by_map": dict(sorted(means.items())),
    }


def _registered_gates(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    thresholds = dict(config["thresholds"])
    fixed_final = float(summary["fixed_final_conflicts"])
    final_degradation = (
        (float(summary["model_final_conflicts"]) - fixed_final) / fixed_final
        if fixed_final > 0.0
        else (0.0 if float(summary["model_final_conflicts"]) == 0.0 else float("inf"))
    )
    return {
        "auc_improvement": float(summary["model_vs_fixed_auc_improvement"])
        >= float(thresholds["minimum_auc_improvement"]),
        "bootstrap": float(summary["bootstrap"]["ci95"][0])
        >= float(thresholds["bootstrap_lower_bound"]),
        "map_consistency": int(summary["maps_no_worse"])
        >= int(thresholds["minimum_maps_no_worse"]),
        "near_oracle": float(summary["near_oracle_share_gain"])
        >= float(thresholds["minimum_near_oracle_gain"]),
        "no_action_collapse": float(summary["maximum_policy_share"])
        <= float(thresholds["maximum_policy_share"]),
        "context_permutation": float(summary["context_permutation"]["real_percentile"])
        >= float(thresholds["minimum_permutation_percentile"]),
        "feasibility": float(summary["model_feasible_rate"])
        >= float(summary["fixed_feasible_rate"])
        - float(thresholds["maximum_feasibility_drop"]),
        "final_conflicts": final_degradation
        <= float(thresholds["maximum_final_conflict_degradation"]),
    }


def _summarize(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    model_auc = _mean(row["model"]["normalized_h4_auc"] for row in records)
    fixed_auc = _mean(row["fixed"]["normalized_h4_auc"] for row in records)
    oracle_auc = _mean(row["oracle"]["normalized_h4_auc"] for row in records)
    uniform_auc = _mean(row["uniform"]["normalized_h4_auc"] for row in records)
    tolerance = float(config["thresholds"]["near_oracle_normalized_regret"])
    model_near = _mean(
        float(row["model"]["normalized_h4_auc"]) - float(row["oracle"]["normalized_h4_auc"]) <= tolerance
        for row in records
    )
    fixed_near = _mean(
        float(row["fixed"]["normalized_h4_auc"]) - float(row["oracle"]["normalized_h4_auc"]) <= tolerance
        for row in records
    )
    map_pairs = collections.defaultdict(lambda: [[], []])
    for row in records:
        map_pairs[str(row["map_id"])][0].append(float(row["model"]["normalized_h4_auc"]))
        map_pairs[str(row["map_id"])][1].append(float(row["fixed"]["normalized_h4_auc"]))
    policy_counts = collections.Counter(str(row["model_policy"]) for row in records)
    return {
        "decision_count": len(records),
        "model_normalized_auc": model_auc,
        "fixed_normalized_auc": fixed_auc,
        "uniform_normalized_auc": uniform_auc,
        "oracle_normalized_auc": oracle_auc,
        "model_vs_fixed_auc_improvement": _relative_improvement(fixed_auc, model_auc),
        "model_near_oracle_share": model_near,
        "fixed_near_oracle_share": fixed_near,
        "near_oracle_share_gain": model_near - fixed_near,
        "model_feasible_rate": _mean(row["model"]["feasible_rate"] for row in records),
        "fixed_feasible_rate": _mean(row["fixed"]["feasible_rate"] for row in records),
        "model_final_conflicts": _mean(row["model"]["final_conflicts"] for row in records),
        "fixed_final_conflicts": _mean(row["fixed"]["final_conflicts"] for row in records),
        "maps_no_worse": sum(_mean(values[0]) <= _mean(values[1]) for values in map_pairs.values()),
        "map_count": len(map_pairs),
        "selected_policy_counts": dict(sorted(policy_counts.items())),
        "maximum_policy_share": max(policy_counts.values(), default=0) / max(1, len(records)),
        "bootstrap": _map_bootstrap(records, int(config["bootstrap_samples"]), int(config["bootstrap_seed"])),
    }


def _permuted_rows(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    grouped = _group_decisions(rows)
    decisions = sorted(grouped)
    sources = list(decisions)
    rng.shuffle(sources)
    result = []
    for target_id, source_id in zip(decisions, sources):
        target = {str(row["policy"]): row for row in grouped[target_id]}
        source = {str(row["policy"]): row for row in grouped[source_id]}
        for policy in POLICIES:
            result.append({**target[policy], "features": dict(source[policy]["features"])})
    return result


def cross_validate(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    maps = sorted({str(row["map_id"]) for row in rows})
    if len(maps) != 12:
        raise ValueError("formal contextual repair-order audit requires 12 maps")
    records = []
    for map_id in maps:
        train = [row for row in rows if str(row["map_id"]) != map_id]
        test = [row for row in rows if str(row["map_id"]) == map_id]
        records.extend(_predict_records(train, test, dict(config["model_parameters"])))
    summary = _summarize(records, config)
    rng = random.Random(int(config["permutation_seed"]))
    permutation_improvements = []
    for _ in range(int(config["permutation_samples"])):
        permutation_records = []
        for map_id in maps:
            train = [row for row in rows if str(row["map_id"]) != map_id]
            test = [row for row in rows if str(row["map_id"]) == map_id]
            permutation_records.extend(
                _predict_records(_permuted_rows(train, rng), test, dict(config["model_parameters"]))
            )
        permuted_model = _mean(
            row["model"]["normalized_h4_auc"] for row in permutation_records
        )
        permuted_fixed = _mean(
            row["fixed"]["normalized_h4_auc"] for row in permutation_records
        )
        permutation_improvements.append(
            _relative_improvement(permuted_fixed, permuted_model)
        )
    real = float(summary["model_vs_fixed_auc_improvement"])
    summary["context_permutation"] = {
        "samples": len(permutation_improvements),
        "real_improvement": real,
        "mean_permuted_improvement": _mean(permutation_improvements),
        "real_percentile": _mean(real > value for value in permutation_improvements),
        "quantiles": {
            "p05": _quantile(permutation_improvements, 0.05),
            "p50": _quantile(permutation_improvements, 0.50),
            "p95": _quantile(permutation_improvements, 0.95),
        },
    }
    gates = _registered_gates(summary, config)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal": True,
        "metrics": summary,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "fit_and_confirm" if all(gates.values()) else "stop_before_independent_confirmation",
        "predictions": records,
    }


def run_audit(
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
) -> dict[str, Any]:
    config_path = _resolve(config_path)
    output_root = _resolve(output)
    config = _read_json(config_path)
    _validate_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    rows, integrity = build_index(config)
    if not integrity["passed"]:
        raise ValueError("contextual repair-order index integrity failed")
    _write_jsonl(output_root / "contextual_repair_order_index.jsonl", rows)
    _write_json(output_root / "index_integrity.json", integrity)
    if phase == "index":
        return integrity
    if phase == "fit":
        return fit_frozen_model(rows, config, output_root)
    if phase == "confirm":
        report_path = output_root / "contextual_repair_order_audit.json"
        if not report_path.is_file() or not bool(_read_json(report_path).get("passed")):
            raise ValueError("independent confirmation is gated by a passing Stage 1 audit")
        raise NotImplementedError("independent confirmation collection is enabled only after model freeze")
    report = cross_validate(rows, config)
    predictions = report.pop("predictions")
    _write_jsonl(output_root / "lomo_predictions.jsonl", predictions)
    report["configuration_sha256"] = _sha256(config_path)
    report["index_sha256"] = _sha256(output_root / "contextual_repair_order_index.jsonl")
    _write_json(output_root / "contextual_repair_order_audit.json", report)
    markdown = [
        "# Contextual Repair-Order Audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        f"- LOMO decisions: {report['metrics']['decision_count']}",
        f"- Model vs fold-fixed normalized H4 AUC improvement: {report['metrics']['model_vs_fixed_auc_improvement']:.2%}",
        f"- Maps no worse: {report['metrics']['maps_no_worse']}/{report['metrics']['map_count']}",
        f"- Near-oracle share gain: {report['metrics']['near_oracle_share_gain']:.2%}",
        f"- Context permutation percentile: {report['metrics']['context_permutation']['real_percentile']:.2%}",
        "",
        "Validation, Test/OOD and independent-confirmation labels were not used.",
    ]
    (output_root / "contextual_repair_order_audit.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if phase == "all" and bool(report["passed"]):
        report["freeze_manifest"] = fit_frozen_model(rows, config, output_root)
    return report


__all__ = [
    "POLICIES",
    "PortableRegressor",
    "build_index",
    "cross_validate",
    "fit_frozen_model",
    "run_audit",
]
