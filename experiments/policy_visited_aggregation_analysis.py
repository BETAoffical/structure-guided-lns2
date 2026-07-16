from __future__ import annotations

import collections
import itertools
import math
import pickle
import random
import statistics
from pathlib import Path
from typing import Any

from experiments._common import (
    feature_names as _feature_names,
    mean as _mean,
    quantile as _quantile,
    relative_improvement as _relative_improvement,
)
from experiments.closed_loop_confirmation import (
    PortablePairwiseModel,
    _closed_loop_episode_worker,
    _sha256,
    controller_implementation_fingerprint,
    load_frozen_policy_bundle,
    score_online_candidates,
    validate_closed_loop_trace,
)
from experiments.closed_loop_confirmation_analysis import (
    compare_policies,
    summarize_policy,
)
from experiments.context_audit import PairwiseModel, _pair_vector
from experiments.policy_visited_aggregation import (
    ALLOWED_PROFILES,
    POLICY_VISITED_SCHEMA,
    build_policy_visited_index,
)
from experiments.realized_neighborhood_ranking_audit import (
    _grouped,
    _selection_record,
    effectiveness_dominates,
    pairwise_accuracy,
    summarize_records,
)
from experiments.repair_collection import (
    SCHEMA_VERSION,
    _CollectionRunLock,
    _dataset_fingerprint,
    _fingerprint,
    _load_dataset_rows,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)


ANALYSIS_SCHEMA = "lns2.policy_visited_analysis.v1"
V2_VALIDATION_SCHEMA = "lns2.policy_visited_v2_validation.v1"


def _validate_analysis_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported policy-visited analysis config")
    if tuple(map(str, config.get("feature_profiles", []))) != ALLOWED_PROFILES:
        raise ValueError("analysis may train only proposal_dynamic and realized_dynamic")
    parameters = dict(config.get("model_parameters", {}))
    expected = {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": 20260714,
    }
    if parameters != expected:
        raise ValueError("pairwise model parameters differ from the registered learner")
    if int(config.get("bootstrap_samples", 0)) != 5000:
        raise ValueError("analysis requires 5,000 map bootstrap samples")
    if "inverse_layout_weighting_sensitivity" in config and not isinstance(
        config["inverse_layout_weighting_sensitivity"], bool
    ):
        raise ValueError("inverse layout weighting sensitivity must be boolean")
    if "study_role" in config and str(config["study_role"]) != "development":
        raise ValueError("policy-visited aggregate training study_role must be development")


def train_equal_state_pairwise_model(
    rows: list[dict[str, Any]],
    profile: str,
    model_parameters: dict[str, Any],
    *,
    state_weight_multipliers: dict[str, float] | None = None,
) -> tuple[PairwiseModel, dict[str, Any]]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    names = _feature_names(rows, profile)
    grouped = _grouped(rows)
    examples: list[list[float]] = []
    labels: list[int] = []
    weights: list[float] = []
    state_pair_counts: dict[str, int] = {}
    for state_id, candidates in sorted(grouped.items()):
        pairs = []
        for left, right in itertools.combinations(candidates, 2):
            if effectiveness_dominates(left["outcome"], right["outcome"]):
                pairs.append((left, right, 1))
            elif effectiveness_dominates(right["outcome"], left["outcome"]):
                pairs.append((left, right, 0))
        if not pairs:
            continue
        state_pair_counts[state_id] = len(pairs)
        multiplier = float((state_weight_multipliers or {}).get(state_id, 1.0))
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(f"invalid state weight multiplier for {state_id}")
        state_example_weight = multiplier / (2.0 * len(pairs))
        for left, right, label in pairs:
            examples.append(_pair_vector(left, right, profile, names))
            labels.append(label)
            weights.append(state_example_weight)
            examples.append(_pair_vector(right, left, profile, names))
            labels.append(1 - label)
            weights.append(state_example_weight)
    if not examples or not state_pair_counts:
        raise ValueError("no policy-visited dominance pairs are available")
    estimator = HistGradientBoostingClassifier(**model_parameters)
    estimator.fit(
        np.asarray(examples, dtype=float),
        np.asarray(labels, dtype=int),
        sample_weight=np.asarray(weights, dtype=float),
    )
    state_weights = collections.defaultdict(float)
    offset = 0
    for state_id, pair_count in sorted(state_pair_counts.items()):
        count = 2 * pair_count
        state_weights[state_id] = sum(weights[offset : offset + count])
        offset += count
    return PairwiseModel(profile, names, estimator), {
        "state_count": len(state_pair_counts),
        "dominance_pair_count": sum(state_pair_counts.values()),
        "example_count": len(examples),
        "state_weight_min": min(state_weights.values()),
        "state_weight_max": max(state_weights.values()),
        "equal_state_weight": all(
            math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for value in state_weights.values()
        ),
        "state_weighting": (
            "custom" if state_weight_multipliers is not None else "equal_state"
        ),
    }


def inverse_layout_state_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    state_layout = {}
    for row in rows:
        state_id = str(row["state_id"])
        layout = str(row.get("layout_mode", "unknown"))
        existing = state_layout.setdefault(state_id, layout)
        if existing != layout:
            raise ValueError(f"state {state_id} has multiple layout labels")
    counts = collections.Counter(state_layout.values())
    if not counts:
        raise ValueError("cannot calculate layout weights without states")
    total = len(state_layout)
    layout_count = len(counts)
    return {
        state_id: total / (layout_count * counts[layout])
        for state_id, layout in state_layout.items()
    }


def _portable_payload(model: PairwiseModel, source_sha: str) -> dict[str, Any]:
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
        "schema_version": SCHEMA_VERSION,
        "profile": model.profile,
        "source_model_sha256": source_sha,
        "feature_names": list(model.feature_names),
        "baseline": float(estimator._baseline_prediction[0, 0]),
        "trees": trees,
    }


def _portable_model(payload: dict[str, Any]) -> PortablePairwiseModel:
    return PortablePairwiseModel(
        profile=str(payload["profile"]),
        feature_names=list(map(str, payload["feature_names"])),
        baseline=float(payload["baseline"]),
        trees=list(payload["trees"]),
    )


def export_aggregated_portable_bundle(
    models: dict[str, PairwiseModel],
    train_rows: list[dict[str, Any]],
    model_paths: dict[str, Path],
    train_index_path: Path,
    output: str | Path,
    training_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_root = Path(output).resolve()
    exported = []
    portable_models = {}
    ranges = {}
    maximum_delta = 0.0
    mismatch_count = 0
    grouped = _grouped(train_rows)
    for profile in ALLOWED_PROFILES:
        model = models[profile]
        source_sha = _sha256(model_paths[profile])
        payload = _portable_payload(model, source_sha)
        path = output_root / f"pairwise__{profile}.json"
        _write_json(path, payload)
        portable = _portable_model(payload)
        portable_models[profile] = portable
        exported.append(
            {
                "profile": profile,
                "file": path.relative_to(output_root).as_posix(),
                "sha256": _sha256(path),
                "source_model_sha256": source_sha,
                "feature_count": len(model.feature_names),
                "tree_count": len(payload["trees"]),
            }
        )
        values = {name: [] for name in model.feature_names}
        for row in train_rows:
            features = dict(row["features"][profile])
            for name in model.feature_names:
                values[name].append(float(features.get(name, 0.0)))
        ranges[profile] = {
            name: [min(numbers), max(numbers)] for name, numbers in values.items()
        }
        for candidates in grouped.values():
            native_index, native_scores, _ = score_online_candidates(candidates, model)
            portable_index, portable_scores, _ = score_online_candidates(
                candidates, portable
            )
            mismatch_count += native_index != portable_index
            maximum_delta = max(
                maximum_delta,
                max(
                    abs(left - right)
                    for left, right in zip(native_scores, portable_scores)
                ),
            )
    equivalence = {
        "schema": "lns2.policy_visited_portable_equivalence.v1",
        "schema_version": SCHEMA_VERSION,
        "state_count": len(grouped),
        "selection_mismatch_count": mismatch_count,
        "maximum_score_delta": maximum_delta,
        "passed": mismatch_count == 0 and maximum_delta <= 1e-12,
    }
    if not equivalence["passed"]:
        raise ValueError("aggregated portable model differs from sklearn inference")
    manifest = {
        "schema": "lns2.portable_pairwise_bundle.v2",
        "schema_version": 2,
        "models": exported,
        "feature_ranges": ranges,
        "development_index_sha256": _sha256(train_index_path),
        "development_state_count": len(grouped),
        "development_candidate_count": len(train_rows),
        "model_parameters": training_metadata["model_parameters"],
        "training_provenance": training_metadata,
        "confirmation_labels_seen": False,
    }
    _write_json(output_root / "portable_manifest.json", manifest)
    _write_json(output_root / "equivalence_report.json", equivalence)
    return manifest, equivalence


def _native_v1_bundle(collection_config: dict[str, Any], project_root: Path):
    registration = {
        key: value
        for key, value in dict(collection_config["model_registration"]).items()
        if key
        not in {
            "deployment_bundle",
            "deployment_manifest_sha256",
            "portable_models",
            "portable_model_sha256",
        }
    }
    frozen_root = Path(str(collection_config["frozen_models"]))
    if not frozen_root.is_absolute():
        frozen_root = project_root / frozen_root
    return load_frozen_policy_bundle(frozen_root, registration)


def run_policy_visited_training(
    collection: str | Path,
    config_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    collection_root = Path(collection).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    _validate_analysis_config(config)
    run_config = _read_json(collection_root / "run_config.json")
    if str(run_config.get("schema")) != POLICY_VISITED_SCHEMA:
        raise ValueError("training source is not a policy-visited collection")
    collection_role = str(
        run_config.get("configuration", {}).get("study_role", "legacy_confirmation")
    )
    analysis_role = str(config.get("study_role", collection_role))
    if collection_role != analysis_role:
        raise ValueError("analysis and collection study roles differ")
    qualification = _read_json(collection_root / "qualification_report.json")
    if not bool(qualification.get("passed")):
        raise ValueError("training source qualification did not pass")
    if str(qualification.get("study_role", collection_role)) != collection_role:
        raise ValueError("qualification and collection study roles differ")
    index, integrity = build_policy_visited_index(collection_root)
    if not integrity["passed"] or integrity["forbidden_split_rows"]:
        raise ValueError("policy-visited index integrity failed")
    new_train = [row for row in index if str(row["split"]) == "policy_train"]
    validation = [
        row for row in index if str(row["split"]) == "policy_validation"
    ]
    if len(new_train) + len(validation) != len(index):
        raise ValueError("policy-visited index contains an unknown split")
    development_path = Path(str(config["development_index"]))
    if not development_path.is_absolute():
        development_path = project_root / development_path
    development_path = development_path.resolve()
    if _sha256(development_path) != str(config["development_index_sha256"]):
        raise ValueError("historical development index SHA256 mismatch")
    old = _read_jsonl(development_path)
    if {str(row["state_id"]) for row in old} & {
        str(row["state_id"]) for row in index
    }:
        raise ValueError("historical and policy-visited states overlap")
    train_rows = old + new_train
    output_root.mkdir(parents=True, exist_ok=True)
    train_index_path = output_root / "aggregate_train_index.jsonl"
    validation_index_path = output_root / "validation_index.jsonl"
    complete_index_path = output_root / "policy_visited_index.jsonl"
    _write_jsonl(train_index_path, train_rows)
    _write_jsonl(validation_index_path, validation)
    _write_jsonl(complete_index_path, index)
    models = {}
    model_paths = {}
    diagnostics = {}
    for profile in ALLOWED_PROFILES:
        model, diagnostic = train_equal_state_pairwise_model(
            train_rows, profile, dict(config["model_parameters"])
        )
        path = output_root / "models" / f"pairwise__{profile}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(model, stream)
        models[profile] = model
        model_paths[profile] = path
        diagnostics[profile] = {**diagnostic, "model_sha256": _sha256(path)}
    sensitivity_diagnostics = {}
    sensitivity_paths = {}
    if bool(config.get("inverse_layout_weighting_sensitivity", False)):
        layout_weights = inverse_layout_state_weights(train_rows)
        for profile in ALLOWED_PROFILES:
            model, diagnostic = train_equal_state_pairwise_model(
                train_rows,
                profile,
                dict(config["model_parameters"]),
                state_weight_multipliers=layout_weights,
            )
            path = (
                output_root
                / "sensitivity"
                / f"pairwise_inverse_layout__{profile}.pkl"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                pickle.dump(model, stream)
            sensitivity_paths[profile] = path
            sensitivity_diagnostics[profile] = {
                **diagnostic,
                "model_sha256": _sha256(path),
            }
    training_metadata = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_run_fingerprint": str(run_config["run_fingerprint"]),
        "historical_index_sha256": _sha256(development_path),
        "policy_visited_index_sha256": _sha256(complete_index_path),
        "aggregate_train_index_sha256": _sha256(train_index_path),
        "historical_state_count": len({str(row["state_id"]) for row in old}),
        "new_train_state_count": len({str(row["state_id"]) for row in new_train}),
        "validation_state_count": len({str(row["state_id"]) for row in validation}),
        "model_parameters": dict(config["model_parameters"]),
        "validation_labels_used_for_training": False,
        "study_role": collection_role,
        "qualification_report_sha256": _sha256(
            collection_root / "qualification_report.json"
        ),
        "primary_state_weighting": "equal_state",
        "inverse_layout_weighting_is_sensitivity_only": bool(
            sensitivity_paths
        ),
    }
    portable_root = output_root / "portable"
    manifest, equivalence = export_aggregated_portable_bundle(
        models,
        train_rows,
        model_paths,
        train_index_path,
        portable_root,
        training_metadata,
    )
    freeze_manifest = {
        **training_metadata,
        "models": [
            {
                "profile": profile,
                "model_file": model_paths[profile].relative_to(output_root).as_posix(),
                "model_sha256": _sha256(model_paths[profile]),
            }
            for profile in ALLOWED_PROFILES
        ],
        "confirmation_labels_seen": False,
    }
    _write_json(output_root / "freeze_manifest.json", freeze_manifest)

    def registered_path(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(project_root).as_posix()
        except ValueError:
            return str(resolved)

    registration = {
        "deployment_bundle": registered_path(portable_root),
        "deployment_manifest_sha256": _sha256(
            portable_root / "portable_manifest.json"
        ),
        "development_index": registered_path(train_index_path),
        "development_index_sha256": _sha256(train_index_path),
        "portable_model_sha256": {
            row["profile"]: row["sha256"] for row in manifest["models"]
        },
        "model_sha256": {
            profile: _sha256(model_paths[profile]) for profile in ALLOWED_PROFILES
        },
    }
    _write_json(output_root / "model_registration.json", registration)
    report = {
        **training_metadata,
        "integrity": integrity,
        "train_candidate_count": len(train_rows),
        "validation_candidate_count": len(validation),
        "model_diagnostics": diagnostics,
        "sensitivity_model_diagnostics": sensitivity_diagnostics,
        "sensitivity_models": {
            profile: {
                "model_file": path.relative_to(output_root).as_posix(),
                "model_sha256": _sha256(path),
                "eligible_for_deployment": False,
            }
            for profile, path in sorted(sensitivity_paths.items())
        },
        "portable_equivalence": equivalence,
        "registration_sha256": _sha256(output_root / "model_registration.json"),
    }
    _write_json(output_root / "training_report.json", report)
    return report


def _evaluate_any_model(
    rows: list[dict[str, Any]], model: Any, selector: str
) -> dict[str, dict[str, Any]]:
    records = {}
    for state_id, candidates in _grouped(rows).items():
        selected_index, _, _ = score_online_candidates(candidates, model)
        records[state_id] = _selection_record(
            candidates[selected_index], candidates, selector
        )
    return records


def _map_bootstrap_offline(
    v1: dict[str, dict[str, Any]],
    v2: dict[str, dict[str, Any]],
    samples: int,
) -> dict[str, Any]:
    if set(v1) != set(v2):
        raise ValueError("offline model records do not share validation states")
    by_map: dict[str, list[str]] = collections.defaultdict(list)
    for state_id, record in v1.items():
        if str(record["map_id"]) != str(v2[state_id]["map_id"]):
            raise ValueError("offline record map mismatch")
        by_map[str(record["map_id"])].append(state_id)
    maps = sorted(by_map)
    rng = random.Random(20270317)
    top1 = []
    regret = []
    for _ in range(samples):
        selected_maps = [rng.choice(maps) for _ in maps]
        states = [state for map_id in selected_maps for state in by_map[map_id]]
        top1.append(
            _mean(v2[state]["pareto_hit"] for state in states)
            - _mean(v1[state]["pareto_hit"] for state in states)
        )
        regret.append(
            _relative_improvement(
                _mean(v1[state]["conflict_regret"] for state in states),
                _mean(v2[state]["conflict_regret"] for state in states),
            )
        )
    return {
        "map_count": len(maps),
        "samples": samples,
        "top1_delta_95_ci": [_quantile(top1, 0.025), _quantile(top1, 0.975)],
        "conflict_regret_improvement_95_ci": [
            _quantile(regret, 0.025),
            _quantile(regret, 0.975),
        ],
    }


def _oracle_size_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    multiple = 0
    for candidates in _grouped(rows).values():
        sizes = {
            int(row["actual_size"])
            for row in candidates
            if bool(row["labels"]["effectiveness_pareto"])
        }
        multiple += len(sizes) > 1
    return {
        "state_count": len(_grouped(rows)),
        "multiple_sizes_supported_states": multiple,
        "multiple_sizes_supported": multiple > 0,
    }


def offline_acceptance(
    v1_summary: dict[str, Any],
    v2_summary: dict[str, Any],
    bootstrap: dict[str, Any],
    oracle: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    top1_delta = float(v2_summary["pareto_top1_hit_rate"]) - float(
        v1_summary["pareto_top1_hit_rate"]
    )
    regret_improvement = _relative_improvement(
        float(v1_summary["mean_conflict_regret"]),
        float(v2_summary["mean_conflict_regret"]),
    )
    top1_qualifies = top1_delta >= float(thresholds["minimum_top1_improvement"])
    regret_qualifies = regret_improvement >= float(
        thresholds["minimum_conflict_regret_improvement"]
    )
    other_not_degraded = (
        (not top1_qualifies or regret_improvement >= -float(
            thresholds["maximum_conflict_regret_degradation"]
        ))
        and (
            not regret_qualifies
            or top1_delta >= -float(thresholds["maximum_top1_degradation"])
        )
    )
    bootstrap_not_degraded = (
        top1_qualifies and bootstrap["top1_delta_95_ci"][1] >= 0.0
    ) or (
        regret_qualifies
        and bootstrap["conflict_regret_improvement_95_ci"][1] >= 0.0
    )
    no_collapse = (
        not oracle["multiple_sizes_supported"]
        or float(v2_summary["maximum_size_share"])
        <= float(thresholds["maximum_single_size_share"])
    )
    gates = {
        "top1_or_conflict_regret_improves": top1_qualifies or regret_qualifies,
        "other_metric_not_degraded": other_not_degraded,
        "map_bootstrap_not_degraded": bootstrap_not_degraded,
        "no_unsupported_size_collapse": no_collapse,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "top1_delta": top1_delta,
        "conflict_regret_improvement": regret_improvement,
        "top1_qualifies": top1_qualifies,
        "conflict_regret_qualifies": regret_qualifies,
    }


def _feature_shift(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    ranges: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    result = {}
    for profile in ALLOWED_PROFILES:
        profile_ranges = ranges[profile]
        fractions = []
        feature_counts = collections.Counter()
        for row in new_rows:
            features = dict(row["features"][profile])
            outside = []
            for name, bounds in profile_ranges.items():
                value = float(features.get(name, 0.0))
                if value < float(bounds[0]) or value > float(bounds[1]):
                    outside.append(name)
                    feature_counts[name] += 1
            fractions.append(len(outside) / len(profile_ranges))
        result[profile] = {
            "candidate_count": len(new_rows),
            "outside_fraction": {
                "mean": _mean(fractions),
                "median": statistics.median(fractions) if fractions else 0.0,
                "p90": _quantile(fractions, 0.9),
                "max": max(fractions, default=0.0),
            },
            "most_common_outside_features": feature_counts.most_common(20),
        }
    result["historical_state_count"] = len({str(row["state_id"]) for row in old_rows})
    result["policy_visited_state_count"] = len(
        {str(row["state_id"]) for row in new_rows}
    )
    return result


def _state_domain_auc(
    old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    states = []
    for label, rows in ((0, old_rows), (1, new_rows)):
        for candidates in _grouped(rows).values():
            row = candidates[0]
            features = {
                name: float(value)
                for name, value in row["features"]["realized_dynamic"].items()
                if name.startswith("state.")
            }
            states.append((features, label, str(row["map_id"])))
    names = sorted({name for features, _, _ in states for name in features})
    matrix = np.asarray(
        [[features.get(name, 0.0) for name in names] for features, _, _ in states],
        dtype=float,
    )
    labels = np.asarray([label for _, label, _ in states], dtype=int)
    groups = np.asarray([group for _, _, group in states])
    folds = min(3, len(set(groups)))
    aucs = []
    if folds >= 2:
        for train, validation in GroupKFold(n_splits=folds).split(
            matrix, labels, groups
        ):
            if len(set(labels[train])) < 2 or len(set(labels[validation])) < 2:
                continue
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=100,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                l2_regularization=0.1,
                random_state=20260714,
            )
            model.fit(matrix[train], labels[train])
            probabilities = model.predict_proba(matrix[validation])[:, 1]
            aucs.append(float(roc_auc_score(labels[validation], probabilities)))
    return {
        "state_count": len(states),
        "feature_count": len(names),
        "valid_fold_count": len(aucs),
        "fold_auc": aucs,
        "mean_auc": _mean(aucs),
        "interpretation": "diagnostic_only; 0.5 means indistinguishable and 1.0 means separable",
    }


def run_offline_policy_visited_analysis(
    collection: str | Path,
    training: str | Path,
    config_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    collection_root = Path(collection).resolve()
    training_root = Path(training).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    _validate_analysis_config(config)
    collection_run = _read_json(collection_root / "run_config.json")
    training_report = _read_json(training_root / "training_report.json")
    if str(training_report["source_run_fingerprint"]) != str(
        collection_run["run_fingerprint"]
    ):
        raise ValueError("training output belongs to another collection")
    validation = _read_jsonl(training_root / "validation_index.jsonl")
    train_rows = _read_jsonl(training_root / "aggregate_train_index.jsonl")
    development_path = Path(str(config["development_index"]))
    if not development_path.is_absolute():
        development_path = project_root / development_path
    old_rows = _read_jsonl(development_path.resolve())
    collection_config = dict(collection_run["configuration"])
    v1_bundle = _native_v1_bundle(collection_config, project_root)
    v2_models = {}
    sensitivity_models = {}
    for profile in ALLOWED_PROFILES:
        with (training_root / "models" / f"pairwise__{profile}.pkl").open("rb") as stream:
            v2_models[profile] = pickle.load(stream)
        sensitivity_path = (
            training_root
            / "sensitivity"
            / f"pairwise_inverse_layout__{profile}.pkl"
        )
        if sensitivity_path.is_file():
            with sensitivity_path.open("rb") as stream:
                sensitivity_models[profile] = pickle.load(stream)
    records = {}
    summaries = {}
    pairwise = {}
    for profile in ALLOWED_PROFILES:
        v1_name = f"v1_{profile}"
        v2_name = f"v2_{profile}"
        records[v1_name] = _evaluate_any_model(
            validation, v1_bundle.models[profile], v1_name
        )
        records[v2_name] = _evaluate_any_model(validation, v2_models[profile], v2_name)
        pairwise[v1_name] = pairwise_accuracy(validation, v1_bundle.models[profile])
        pairwise[v2_name] = pairwise_accuracy(validation, v2_models[profile])
        summaries[v1_name] = summarize_records(records[v1_name], pairwise[v1_name])
        summaries[v2_name] = summarize_records(records[v2_name], pairwise[v2_name])
        if profile in sensitivity_models:
            sensitivity_name = f"sensitivity_inverse_layout_{profile}"
            records[sensitivity_name] = _evaluate_any_model(
                validation, sensitivity_models[profile], sensitivity_name
            )
            pairwise[sensitivity_name] = pairwise_accuracy(
                validation, sensitivity_models[profile]
            )
            summaries[sensitivity_name] = summarize_records(
                records[sensitivity_name], pairwise[sensitivity_name]
            )
    bootstrap = _map_bootstrap_offline(
        records["v1_realized_dynamic"],
        records["v2_realized_dynamic"],
        int(config["bootstrap_samples"]),
    )
    oracle = _oracle_size_support(validation)
    acceptance = offline_acceptance(
        summaries["v1_realized_dynamic"],
        summaries["v2_realized_dynamic"],
        bootstrap,
        oracle,
        dict(config["thresholds"]),
    )
    deployment = _read_json(
        project_root
        / str(collection_config["model_registration"]["deployment_bundle"])
        / "portable_manifest.json"
    )
    new_train = [
        row for row in train_rows if str(row.get("split")) == "policy_train"
    ]
    shift = {
        "feature_ranges": _feature_shift(
            old_rows, new_train, dict(deployment["feature_ranges"])
        ),
        "state_domain_classifier": _state_domain_auc(old_rows, new_train),
        "stages": dict(
            sorted(collections.Counter(row.get("stage", "historical") for row in new_train).items())
        ),
        "layouts": dict(
            sorted(collections.Counter(row.get("layout_mode", "unknown") for row in new_train).items())
        ),
        "tasks": dict(
            sorted(collections.Counter(row.get("task_variant", "unknown") for row in new_train).items())
        ),
        "agent_counts": dict(
            sorted(collections.Counter(str(row.get("agent_count")) for row in new_train).items())
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for name, values in records.items():
        _write_jsonl(
            output_root / f"offline_predictions__{name}.jsonl",
            [values[key] for key in sorted(values)],
        )
    report = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_run_fingerprint": str(collection_run["run_fingerprint"]),
        "training_report_sha256": _sha256(training_root / "training_report.json"),
        "validation_state_count": len({str(row["state_id"]) for row in validation}),
        "validation_candidate_count": len(validation),
        "summaries": summaries,
        "bootstrap": bootstrap,
        "oracle_size_support": oracle,
        "distribution_shift": shift,
        "offline_acceptance": acceptance,
        "sensitivity_models_control_acceptance": False,
        "validation_labels_used_for_training": False,
        "static_context_used": False,
    }
    _write_json(output_root / "offline_report.json", report)
    return report


def run_v2_closed_loop_validation(
    dataset: str | Path,
    collection: str | Path,
    training: str | Path,
    output: str | Path,
    *,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(dataset).resolve()
    collection_root = Path(collection).resolve()
    training_root = Path(training).resolve()
    output_root = Path(output).resolve()
    collection_run = _read_json(collection_root / "run_config.json")
    config = dict(collection_run["configuration"])
    registration = _read_json(training_root / "model_registration.json")
    training_report = _read_json(training_root / "training_report.json")
    if bool(training_report.get("validation_labels_used_for_training")):
        raise ValueError("v2 training manifest reports validation leakage")
    rows = _load_dataset_rows(dataset_root, ["policy_validation"])
    dataset_fp = _dataset_fingerprint(dataset_root)
    if dataset_fp != str(collection_run["dataset_fingerprint"]):
        raise ValueError("v2 validation dataset fingerprint mismatch")
    model_manifest = _read_json(training_root / "portable" / "portable_manifest.json")
    implementation = controller_implementation_fingerprint(project_root)
    run_fp = _fingerprint(
        {
            "dataset_fingerprint": dataset_fp,
            "source_collection": collection_run["run_fingerprint"],
            "training_report": _sha256(training_root / "training_report.json"),
            "model_registration": registration,
            "model_manifest": model_manifest,
            "implementation": implementation,
        }
    )
    run_config = {
        "schema": V2_VALIDATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_fp,
        "source_collection": str(collection_root),
        "source_run_fingerprint": str(collection_run["run_fingerprint"]),
        "training": str(training_root),
        "training_report_sha256": _sha256(training_root / "training_report.json"),
        "model_registration": registration,
        "run_fingerprint": run_fp,
        "implementation": implementation,
    }
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fp:
            raise ValueError("v2 validation output contains another run")
        if not resume:
            raise ValueError("v2 validation output exists; pass resume")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, run_config)
    jobs = [
        {
            "row": row,
            "policy": "realized_dynamic",
            "solver_seed": seed,
            "dataset_root": str(dataset_root),
            "environment": config["environment"],
            "proposal": config["proposal"],
            "max_decisions": int(config["max_decisions"]),
            "metric_iteration_budget": int(config["metric_iteration_budget"]),
            "wall_time_budget_seconds": float(config["wall_time_budget_seconds"]),
            "frozen_models": str(training_root),
            "model_registration": registration,
            "output_root": str(output_root),
            "run_fingerprint": run_fp,
            "resume": resume,
        }
        for row in rows
        for seed in map(int, config["solver_seeds"])
    ]
    manifest_path = output_root / "realized_dynamic_manifest.jsonl"
    existing = _read_jsonl(manifest_path) if resume and manifest_path.is_file() else []
    manifest_by_episode = {str(row["episode_id"]): row for row in existing}

    def record(result: dict[str, Any]) -> None:
        manifest_by_episode[str(result["episode_id"])] = result
        _write_jsonl(
            manifest_path,
            [manifest_by_episode[key] for key in sorted(manifest_by_episode)],
        )

    with _CollectionRunLock(output_root, run_fp, "policy-visited-v2-validation"):
        results = _run_jobs(
            _closed_loop_episode_worker,
            jobs,
            workers,
            phase="policy-visited-v2-validation",
            output_root=output_root,
            run_fingerprint=run_fp,
            timeout_seconds=float(config["episode_process_timeout_seconds"]),
            on_result=record,
        )
    _write_jsonl(
        manifest_path, sorted(results, key=lambda row: str(row["episode_id"]))
    )
    summary = {
        "schema": V2_VALIDATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fp,
        "episode_count": len(results),
        "success_count": sum(
            bool(row.get("summary", {}).get("success")) for row in results
        ),
        "error_count": sum(
            str(row.get("status")) not in {"ok", "resumed"} for row in results
        ),
    }
    _write_json(output_root / "collection_summary.json", summary)
    return summary


def _fingerprint_integrity(
    manifests: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    indexed: dict[tuple[str, int], dict[str, str]] = collections.defaultdict(dict)
    for policy, rows in manifests.items():
        for row in rows:
            if str(row.get("status")) not in {"ok", "resumed"}:
                continue
            indexed[(str(row["task_id"]), int(row["solver_seed"]))][policy] = str(
                row["summary"]["initial_fingerprint"]
            )
    mismatches = []
    for key, values in sorted(indexed.items()):
        if len(values) != len(manifests) or len(set(values.values())) != 1:
            mismatches.append({"task_seed": list(key), "fingerprints": values})
    return {"passed": not mismatches, "paired_count": len(indexed), "mismatches": mismatches}


def closed_loop_v2_acceptance(
    adaptive_summary: dict[str, Any],
    v1_summary: dict[str, Any],
    v2_summary: dict[str, Any],
    v2_vs_adaptive: dict[str, Any],
    v2_vs_v1: dict[str, Any],
    integrity: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    adaptive_auc = v2_vs_adaptive["metrics"]["fixed_budget_conflict_auc"]
    v1_auc = v2_vs_v1["metrics"]["fixed_budget_conflict_auc"]
    gates = {
        "success_not_below_adaptive": int(v2_summary["success_count"])
        >= int(adaptive_summary["success_count"]),
        "success_not_below_v1": int(v2_summary["success_count"])
        >= int(v1_summary["success_count"]),
        "auc_improves_over_adaptive": float(adaptive_auc["relative_improvement"])
        >= float(thresholds["minimum_closed_loop_auc_improvement_over_adaptive"]),
        "auc_within_v1_bound": float(v1_auc["relative_improvement"])
        >= -float(thresholds["maximum_closed_loop_auc_degradation_from_v1"]),
        "minimum_maps_no_worse_than_v1": int(v1_auc["maps_no_worse"])
        >= int(thresholds["minimum_validation_maps_no_worse_than_v1"]),
        "fingerprints_match": bool(integrity["passed"]),
        "all_episodes_valid": sum(
            int(value["error_count"])
            for value in (adaptive_summary, v1_summary, v2_summary)
        )
        == 0,
        "no_invalid_actions": int(v2_summary["invalid_action_count"]) == 0,
        "no_fingerprint_mismatch": int(v2_summary["fingerprint_mismatch_count"])
        == 0,
    }
    return {"passed": all(gates.values()), "gates": gates}


def run_final_policy_visited_analysis(
    collection: str | Path,
    training: str | Path,
    offline: str | Path,
    v2_validation: str | Path,
    config_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    collection_root = Path(collection).resolve()
    training_root = Path(training).resolve()
    offline_root = Path(offline).resolve()
    v2_root = Path(v2_validation).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    _validate_analysis_config(config)
    collection_run = _read_json(collection_root / "run_config.json")
    v2_run = _read_json(v2_root / "run_config.json")
    if str(v2_run["source_run_fingerprint"]) != str(collection_run["run_fingerprint"]):
        raise ValueError("v2 validation belongs to another source collection")
    offline_report = _read_json(offline_root / "offline_report.json")
    source = _read_jsonl(collection_root / "source_manifest.jsonl")
    adaptive = [
        row
        for row in source
        if str(row["split"]) == "policy_validation"
        and str(row["policy"]) == "official_adaptive"
    ]
    v1 = [
        row
        for row in source
        if str(row["split"]) == "policy_validation"
        and str(row["policy"]) == "realized_dynamic"
    ]
    v2 = _read_jsonl(v2_root / "realized_dynamic_manifest.jsonl")
    collection_config = dict(collection_run["configuration"])
    budget = int(collection_config["metric_iteration_budget"])
    for rows, root, run_fp in (
        (adaptive, collection_root, str(collection_run["run_fingerprint"])),
        (v1, collection_root, str(collection_run["run_fingerprint"])),
        (v2, v2_root, str(v2_run["run_fingerprint"])),
    ):
        for row in rows:
            if str(row.get("status")) not in {"ok", "resumed"}:
                continue
            validate_closed_loop_trace(
                root / str(row["trace_file"]),
                run_fp,
                expected_episode_id=str(row["episode_id"]),
                expected_policy="official_adaptive"
                if rows is adaptive
                else "realized_dynamic",
                expected_solver_seed=int(row["solver_seed"]),
                metric_iteration_budget=budget,
            )
    summaries = {
        "official_adaptive": summarize_policy(adaptive),
        "realized_dynamic_v1": summarize_policy(v1),
        "realized_dynamic_v2": summarize_policy(v2),
    }
    bootstrap_samples = int(config["bootstrap_samples"])
    v2_vs_adaptive = compare_policies(adaptive, v2, bootstrap_samples, budget)
    v2_vs_v1 = compare_policies(v1, v2, bootstrap_samples, budget)
    v1_vs_adaptive = compare_policies(adaptive, v1, bootstrap_samples, budget)
    integrity = _fingerprint_integrity(
        {
            "official_adaptive": adaptive,
            "realized_dynamic_v1": v1,
            "realized_dynamic_v2": v2,
        }
    )
    closed_loop = closed_loop_v2_acceptance(
        summaries["official_adaptive"],
        summaries["realized_dynamic_v1"],
        summaries["realized_dynamic_v2"],
        v2_vs_adaptive,
        v2_vs_v1,
        integrity,
        dict(config["thresholds"]),
    )
    offline_pass = bool(offline_report["offline_acceptance"]["passed"])
    v1_robust = (
        summaries["realized_dynamic_v1"]["success_count"]
        >= summaries["official_adaptive"]["success_count"]
        and v1_vs_adaptive["metrics"]["fixed_budget_conflict_auc"][
            "relative_improvement"
        ]
        >= float(
            config["thresholds"][
                "minimum_closed_loop_auc_improvement_over_adaptive"
            ]
        )
    )
    if offline_pass and closed_loop["passed"]:
        decision = "use_v2_as_rl_warm_start"
    elif v1_robust:
        decision = "retain_v1_warm_start_and_use_policy_visited_replay"
    else:
        decision = "keep_rl_paused_and_redesign_candidate_control"
    report = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "offline_acceptance": offline_report["offline_acceptance"],
        "closed_loop_acceptance": closed_loop,
        "v1_remains_robust": v1_robust,
        "policy_summaries": summaries,
        "comparisons": {
            "v2_vs_adaptive": v2_vs_adaptive,
            "v2_vs_v1": v2_vs_v1,
            "v1_vs_adaptive": v1_vs_adaptive,
        },
        "initial_fingerprint_integrity": integrity,
        "training_report_sha256": _sha256(training_root / "training_report.json"),
        "offline_report_sha256": _sha256(offline_root / "offline_report.json"),
        "static_context_used": False,
        "rl_trained": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "policy_visited_report.json", report)
    _write_json(
        output_root / "policy_visited_summary.json",
        {
            "decision": decision,
            "offline_passed": offline_pass,
            "closed_loop_passed": bool(closed_loop["passed"]),
            "v1_remains_robust": v1_robust,
        },
    )
    lines = [
        "# InitLNS policy-visited aggregation",
        "",
        f"Decision: `{decision}`",
        "",
        "## Offline",
        "",
        f"- V1 realized top-1: {offline_report['summaries']['v1_realized_dynamic']['pareto_top1_hit_rate']:.3f}",
        f"- V2 realized top-1: {offline_report['summaries']['v2_realized_dynamic']['pareto_top1_hit_rate']:.3f}",
        f"- V1 conflict regret: {offline_report['summaries']['v1_realized_dynamic']['mean_conflict_regret']:.3f}",
        f"- V2 conflict regret: {offline_report['summaries']['v2_realized_dynamic']['mean_conflict_regret']:.3f}",
        f"- Registered offline gate: {'PASS' if offline_pass else 'FAIL'}",
        "",
        "## Closed loop",
        "",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"- `{name}`: success {summary['success_count']}/{summary['valid_count']}, "
            f"fixed AUC {summary['mean_fixed_budget_conflict_auc']:.3f}"
        )
    lines.extend(
        [
            f"- Registered closed-loop gate: {'PASS' if closed_loop['passed'] else 'FAIL'}",
            "",
            "Static context was excluded and no RL policy was trained.",
            "",
        ]
    )
    (output_root / "policy_visited_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = [
    "closed_loop_v2_acceptance",
    "export_aggregated_portable_bundle",
    "inverse_layout_state_weights",
    "offline_acceptance",
    "run_final_policy_visited_analysis",
    "run_offline_policy_visited_analysis",
    "run_policy_visited_training",
    "run_v2_closed_loop_validation",
    "train_equal_state_pairwise_model",
]
