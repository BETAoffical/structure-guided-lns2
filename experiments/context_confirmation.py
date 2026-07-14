from __future__ import annotations

import collections
import hashlib
import json
import pickle
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.context_audit import (
    FEATURE_PROFILES,
    MODEL_SEED,
    PairwiseModel,
    _dataset_root,
    _feature_names,
    _pair_vector,
    _safe_ratio,
    _validate_split_isolation,
    _vector,
    build_index,
)


SECONDARY_AUDIT_SCHEMA_VERSION = 1
OBJECTIVE_MODES = ("primary", "runtime_sensitivity")
LEARNERS = ("pairwise", "pareto_member")
PRIMARY_LEARNER = "pareto_member"


def _mean(values: Iterable[float | int]) -> float:
    numbers = [float(value) for value in values]
    return statistics.fmean(numbers) if numbers else 0.0


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _categoricalize_sizes(index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in index:
        row = dict(source)
        row["features"] = {}
        size = int(row["candidate_action"]["neighborhood_size"])
        for profile in FEATURE_PROFILES:
            features = dict(source["features"][profile])
            features.pop("action.neighborhood_size", None)
            features[f"action.neighborhood_size={size}"] = 1.0
            row["features"][profile] = features
        result.append(row)
    return result


def _objective_values(value: dict[str, Any], mode: str) -> tuple[float, ...]:
    if mode not in OBJECTIVE_MODES:
        raise ValueError(f"unknown objective mode: {mode}")
    values = (
        -float(value.get("solved_rate", float(bool(value["solved"])))),
        float(value["conflicts_after"]),
        float(value["conflict_auc"]),
        float(value["generated"]),
    )
    if mode == "runtime_sensitivity":
        return values + (float(value["branch_runtime"]),)
    return values


def _dominates(left: dict[str, Any], right: dict[str, Any], mode: str) -> bool:
    left_values = _objective_values(left, mode)
    right_values = _objective_values(right, mode)
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _relabel(index: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for source in index:
        row = dict(source)
        grouped[str(row["state_id"])].append(row)
    result: list[dict[str, Any]] = []
    for candidates in grouped.values():
        for candidate in candidates:
            candidate["pareto"] = not any(
                other is not candidate
                and _dominates(other["outcome"], candidate["outcome"], mode)
                for other in candidates
            )
            candidate["objective_mode"] = mode
            result.append(candidate)
    result.sort(key=lambda row: (str(row["state_id"]), str(row["candidate_key"])))
    return result


def _pairwise_examples(
    rows: list[dict[str, Any]], profile: str, names: list[str], mode: str
) -> tuple[list[list[float]], list[int]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    examples: list[list[float]] = []
    labels: list[int] = []
    for candidates in grouped.values():
        candidates.sort(key=lambda row: str(row["candidate_key"]))
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                if _dominates(left["outcome"], right["outcome"], mode):
                    label = 1
                elif _dominates(right["outcome"], left["outcome"], mode):
                    label = 0
                else:
                    continue
                examples.append(_pair_vector(left, right, profile, names))
                labels.append(label)
                examples.append(_pair_vector(right, left, profile, names))
                labels.append(1 - label)
    if not examples:
        raise ValueError("no Pareto-dominance pairs are available for training")
    return examples, labels


@dataclass
class ParetoMembershipModel:
    profile: str
    feature_names: list[str]
    estimator: Any

    def scores(self, rows: list[dict[str, Any]]) -> list[float]:
        import numpy as np

        matrix = np.asarray(
            [_vector(row, self.profile, self.feature_names) for row in rows],
            dtype=float,
        )
        probabilities = self.estimator.predict_proba(matrix)
        positive = list(self.estimator.classes_).index(1)
        return [float(value) for value in probabilities[:, positive]]


def _train_ranker(
    rows: list[dict[str, Any]], profile: str, learner: str, mode: str
) -> PairwiseModel | ParetoMembershipModel:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    names = _feature_names(rows, profile)
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=MODEL_SEED,
    )
    if learner == "pairwise":
        examples, labels = _pairwise_examples(rows, profile, names, mode)
        estimator.fit(
            np.asarray(examples, dtype=float), np.asarray(labels, dtype=int)
        )
        return PairwiseModel(profile, names, estimator)
    if learner != "pareto_member":
        raise ValueError(f"unknown learner: {learner}")

    labels = [int(bool(row["pareto"])) for row in rows]
    if len(set(labels)) != 2:
        raise ValueError("Pareto membership training requires both classes")
    state_counts = collections.Counter(str(row["state_id"]) for row in rows)
    class_counts = collections.Counter(labels)
    weights = [
        (1.0 / state_counts[str(row["state_id"])])
        * (len(rows) / (2.0 * class_counts[label]))
        for row, label in zip(rows, labels)
    ]
    matrix = np.asarray(
        [_vector(row, profile, names) for row in rows], dtype=float
    )
    estimator.fit(
        matrix,
        np.asarray(labels, dtype=int),
        sample_weight=np.asarray(weights, dtype=float),
    )
    return ParetoMembershipModel(profile, names, estimator)


def _grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    for candidates in grouped.values():
        candidates.sort(key=lambda row: str(row["candidate_key"]))
    return grouped


def _evaluate_ranker(
    rows: list[dict[str, Any]],
    model: PairwiseModel | ParetoMembershipModel,
) -> dict[str, dict[str, Any]]:
    grouped = _grouped(rows)
    direct_scores: dict[str, float] = {}
    if isinstance(model, ParetoMembershipModel):
        direct_scores = {
            f"{row['state_id']}|{row['candidate_key']}": score
            for row, score in zip(rows, model.scores(rows))
        }

    records: dict[str, dict[str, Any]] = {}
    for state_id, candidates in sorted(grouped.items()):
        if isinstance(model, ParetoMembershipModel):
            selected_index = min(
                range(len(candidates)),
                key=lambda index: (
                    -direct_scores[
                        f"{state_id}|{candidates[index]['candidate_key']}"
                    ],
                    str(candidates[index]["candidate_key"]),
                ),
            )
        else:
            selected_index = model.select(candidates)
        selected = candidates[selected_index]
        minimum_auc = min(float(row["outcome"]["conflict_auc"]) for row in candidates)
        minimum_conflicts = min(
            float(row["outcome"]["conflicts_after"]) for row in candidates
        )
        selected_auc = float(selected["outcome"]["conflict_auc"])
        selected_conflicts = float(selected["outcome"]["conflicts_after"])
        action = selected["candidate_action"]
        records[state_id] = {
            "map_id": str(selected["map_id"]),
            "task_id": str(selected["task_id"]),
            "stage": str(selected["stage"]),
            "pareto_hit": float(bool(selected["pareto"])),
            "auc_regret": _safe_ratio(
                selected_auc - minimum_auc, max(1.0, abs(minimum_auc))
            ),
            "conflict_regret": _safe_ratio(
                selected_conflicts - minimum_conflicts,
                max(1.0, abs(minimum_conflicts)),
            ),
            "selected_family": (
                f"{action['heuristic']}:{int(action['neighborhood_size'])}"
            ),
            "selected_size": int(action["neighborhood_size"]),
        }
    return records


def _summarize(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    families = collections.Counter(
        str(row["selected_family"]) for row in records.values()
    )
    sizes = collections.Counter(int(row["selected_size"]) for row in records.values())
    return {
        "state_count": len(records),
        "pareto_top1_hit_rate": _mean(row["pareto_hit"] for row in records.values()),
        "mean_auc_regret": _mean(row["auc_regret"] for row in records.values()),
        "mean_conflict_regret": _mean(
            row["conflict_regret"] for row in records.values()
        ),
        "selected_action_families": dict(sorted(families.items())),
        "selected_sizes": {str(key): value for key, value in sorted(sizes.items())},
        "maximum_size_share": _safe_ratio(max(sizes.values(), default=0), len(records)),
    }


def _random_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    state_values = []
    for candidates in _grouped(rows).values():
        minimum_auc = min(float(row["outcome"]["conflict_auc"]) for row in candidates)
        minimum_conflicts = min(
            float(row["outcome"]["conflicts_after"]) for row in candidates
        )
        state_values.append(
            {
                "pareto_hit": _mean(float(bool(row["pareto"])) for row in candidates),
                "auc_regret": _mean(
                    _safe_ratio(
                        float(row["outcome"]["conflict_auc"]) - minimum_auc,
                        max(1.0, abs(minimum_auc)),
                    )
                    for row in candidates
                ),
                "conflict_regret": _mean(
                    _safe_ratio(
                        float(row["outcome"]["conflicts_after"]) - minimum_conflicts,
                        max(1.0, abs(minimum_conflicts)),
                    )
                    for row in candidates
                ),
            }
        )
    return {
        "state_count": len(state_values),
        "pareto_top1_hit_rate": _mean(row["pareto_hit"] for row in state_values),
        "mean_auc_regret": _mean(row["auc_regret"] for row in state_values),
        "mean_conflict_regret": _mean(row["conflict_regret"] for row in state_values),
    }


def _partitions(
    rows: list[dict[str, Any]], evaluation_mode: str
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    if evaluation_mode == "confirmation":
        _validate_split_isolation(rows)
        train = [row for row in rows if row["split"] == "train"]
        validation = [row for row in rows if row["split"] == "validation"]
        if not train or not validation:
            raise ValueError("confirmation requires Train and Validation rows")
        return [
            (
                train,
                validation,
                {
                    "fold": 0,
                    "train_maps": sorted({str(row["map_id"]) for row in train}),
                    "validation_maps": sorted(
                        {str(row["map_id"]) for row in validation}
                    ),
                },
            )
        ]
    if evaluation_mode != "development":
        raise ValueError("evaluation_mode must be development or confirmation")

    from sklearn.model_selection import GroupKFold

    states = sorted(_grouped(rows).values(), key=lambda value: str(value[0]["state_id"]))
    state_ids = [str(value[0]["state_id"]) for value in states]
    groups = [str(value[0]["map_id"]) for value in states]
    map_count = len(set(groups))
    if map_count < 3:
        raise ValueError("development audit requires at least three maps")
    splitter = GroupKFold(n_splits=min(3, map_count))
    result = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(state_ids, groups=groups)
    ):
        train_states = {state_ids[index] for index in train_indices}
        test_states = {state_ids[index] for index in test_indices}
        train_rows = [row for row in rows if str(row["state_id"]) in train_states]
        test_rows = [row for row in rows if str(row["state_id"]) in test_states]
        result.append(
            (
                train_rows,
                test_rows,
                {
                    "fold": fold,
                    "train_maps": sorted({str(row["map_id"]) for row in train_rows}),
                    "validation_maps": sorted(
                        {str(row["map_id"]) for row in test_rows}
                    ),
                },
            )
        )
    return result


def _context_bundle(row: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in row["features"]["full_context"].items()
        if key.startswith("context.")
    }


def _permuted_context_rows(
    rows: list[dict[str, Any]], permutation_index: int, fold: int
) -> list[dict[str, Any]]:
    task_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_rows.setdefault(str(row["task_id"]), row)
    tasks = sorted(task_rows)
    donors = list(tasks)
    rng = random.Random(MODEL_SEED + permutation_index * 1009 + fold * 9176)
    rng.shuffle(donors)
    bundles = {task: _context_bundle(task_rows[task]) for task in tasks}
    donor_for = dict(zip(tasks, donors))
    result = []
    for source in rows:
        row = dict(source)
        row["features"] = dict(source["features"])
        features = {
            key: value
            for key, value in source["features"]["full_context"].items()
            if not key.startswith("context.")
        }
        features.update(bundles[donor_for[str(source["task_id"])]])
        row["features"]["full_context"] = features
        result.append(row)
    return result


def _comparison(
    dynamic: dict[str, dict[str, Any]], full: dict[str, dict[str, Any]]
) -> dict[str, float]:
    dynamic_summary = _summarize(dynamic)
    full_summary = _summarize(full)
    dynamic_auc = float(dynamic_summary["mean_auc_regret"])
    return {
        "full_context_minus_dynamic_hit_rate": (
            float(full_summary["pareto_top1_hit_rate"])
            - float(dynamic_summary["pareto_top1_hit_rate"])
        ),
        "relative_auc_regret_reduction": _safe_ratio(
            dynamic_auc - float(full_summary["mean_auc_regret"]),
            max(1e-12, dynamic_auc),
        ),
    }


def _map_bootstrap(
    dynamic: dict[str, dict[str, Any]],
    full: dict[str, dict[str, Any]],
    samples: int = 2000,
) -> dict[str, list[float]]:
    by_map: dict[str, list[str]] = collections.defaultdict(list)
    for state_id, row in dynamic.items():
        if state_id in full:
            by_map[str(row["map_id"])].append(state_id)
    map_hit = {
        map_id: _mean(
            full[state_id]["pareto_hit"] - dynamic[state_id]["pareto_hit"]
            for state_id in state_ids
        )
        for map_id, state_ids in by_map.items()
    }
    map_auc = {
        map_id: _mean(
            dynamic[state_id]["auc_regret"] - full[state_id]["auc_regret"]
            for state_id in state_ids
        )
        for map_id, state_ids in by_map.items()
    }
    maps = sorted(by_map)
    rng = random.Random(MODEL_SEED ^ 0xB00757)
    hit_values = []
    auc_values = []
    for _ in range(samples):
        sample = [rng.choice(maps) for _ in maps]
        hit_values.append(_mean(map_hit[map_id] for map_id in sample))
        auc_values.append(_mean(map_auc[map_id] for map_id in sample))

    def interval(values: list[float]) -> list[float]:
        values.sort()
        return [
            values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))],
        ]

    return {
        "hit_gain_95_ci": interval(hit_values),
        "auc_improvement_95_ci": interval(auc_values),
    }


def _percentile(real: float, null_values: list[float]) -> float:
    return _safe_ratio(sum(real > value for value in null_values), len(null_values))


def _categorical_value(row: dict[str, Any], prefix: str) -> str:
    marker = prefix + "="
    for name, value in row["features"]["full_context"].items():
        if name.startswith(marker) and float(value) > 0.5:
            return name[len(marker) :]
    return "unknown"


def _oracle_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _grouped(rows)
    sizes = (4, 8, 16)
    size_coverage = {
        str(size): _safe_ratio(
            sum(
                any(
                    bool(row["pareto"])
                    and int(row["candidate_action"]["neighborhood_size"]) == size
                    for row in candidates
                )
                for candidates in grouped.values()
            ),
            len(grouped),
        )
        for size in sizes
    }
    supported_sizes = [
        size for size, coverage in size_coverage.items() if coverage >= 0.10
    ]
    dimensions: dict[str, dict[str, Any]] = {}
    labels = {
        "layout_mode": lambda row: _categorical_value(row, "context.layout_mode"),
        "task_variant": lambda row: _categorical_value(row, "context.task_variant"),
        "agent_count": lambda row: str(
            int(row["features"]["full_context"].get("context.agent_count", 0))
        ),
        "stage": lambda row: str(row["stage"]),
    }
    for dimension, label in labels.items():
        by_value: dict[str, list[list[dict[str, Any]]]] = collections.defaultdict(list)
        for candidates in grouped.values():
            by_value[label(candidates[0])].append(candidates)
        values: dict[str, Any] = {}
        family_rates: dict[str, list[float]] = collections.defaultdict(list)
        for value, states in sorted(by_value.items()):
            families = sorted(
                {
                    f"{row['candidate_action']['heuristic']}:"
                    f"{int(row['candidate_action']['neighborhood_size'])}"
                    for candidates in states
                    for row in candidates
                }
            )
            rates = {
                family: _mean(
                    any(
                        bool(row["pareto"])
                        and (
                            f"{row['candidate_action']['heuristic']}:"
                            f"{int(row['candidate_action']['neighborhood_size'])}"
                        )
                        == family
                        for row in candidates
                    )
                    for candidates in states
                )
                for family in families
            }
            best_rate = max(rates.values(), default=0.0)
            best = sorted(
                family for family, rate in rates.items() if abs(rate - best_rate) < 1e-12
            )
            values[value] = {
                "state_count": len(states),
                "pareto_family_rates": rates,
                "best_families": best,
            }
            for family, rate in rates.items():
                family_rates[family].append(rate)
        max_spread = max(
            (max(rates) - min(rates) for rates in family_rates.values() if len(rates) > 1),
            default=0.0,
        )
        best_sets = {
            tuple(value["best_families"]) for value in values.values()
        }
        dimensions[dimension] = {
            "values": values,
            "maximum_family_rate_spread": max_spread,
            "best_family_set_changes": len(best_sets) > 1,
        }
    static_qualified = any(
        dimensions[name]["maximum_family_rate_spread"] >= 0.10
        and dimensions[name]["best_family_set_changes"]
        for name in ("layout_mode", "task_variant", "agent_count")
    )
    return {
        "state_count": len(grouped),
        "size_state_coverage": size_coverage,
        "supported_sizes": supported_sizes,
        "multiple_sizes_supported": len(supported_sizes) >= 2,
        "dimensions": dimensions,
        "static_heterogeneity_qualified": static_qualified,
    }


def _run_objective(
    rows: list[dict[str, Any]],
    mode: str,
    evaluation_mode: str,
    permutations: int,
) -> dict[str, Any]:
    labeled = _relabel(rows, mode)
    records: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        learner: {profile: {} for profile in FEATURE_PROFILES}
        for learner in LEARNERS
    }
    permutation_records = [dict() for _ in range(permutations)]
    folds = []
    for train_rows, validation_rows, fold in _partitions(labeled, evaluation_mode):
        folds.append(fold)
        for learner in LEARNERS:
            for profile in FEATURE_PROFILES:
                model = _train_ranker(train_rows, profile, learner, mode)
                selected = _evaluate_ranker(validation_rows, model)
                records[learner][profile].update(selected)
                if (
                    mode == "primary"
                    and learner == PRIMARY_LEARNER
                    and profile == "full_context"
                ):
                    for permutation_index in range(permutations):
                        permuted = _permuted_context_rows(
                            validation_rows,
                            permutation_index,
                            int(fold["fold"]),
                        )
                        permutation_records[permutation_index].update(
                            _evaluate_ranker(permuted, model)
                        )

    profiles: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for learner in LEARNERS:
        profiles[learner] = {
            profile: _summarize(records[learner][profile])
            for profile in FEATURE_PROFILES
        }
        comparison = _comparison(
            records[learner]["dynamic"], records[learner]["full_context"]
        )
        comparison.update(
            _map_bootstrap(
                records[learner]["dynamic"], records[learner]["full_context"]
            )
        )
        comparisons[learner] = comparison

    permutation_report: dict[str, Any] = {
        "count": permutations,
        "learner": PRIMARY_LEARNER,
    }
    if permutations:
        dynamic = records[PRIMARY_LEARNER]["dynamic"]
        null_comparisons = [
            _comparison(dynamic, permuted) for permuted in permutation_records
        ]
        real = comparisons[PRIMARY_LEARNER]
        hit_null = [
            value["full_context_minus_dynamic_hit_rate"]
            for value in null_comparisons
        ]
        auc_null = [value["relative_auc_regret_reduction"] for value in null_comparisons]
        permutation_report.update(
            {
                "real_hit_gain": real["full_context_minus_dynamic_hit_rate"],
                "real_auc_regret_reduction": real["relative_auc_regret_reduction"],
                "hit_gain_percentile": _percentile(
                    real["full_context_minus_dynamic_hit_rate"], hit_null
                ),
                "auc_reduction_percentile": _percentile(
                    real["relative_auc_regret_reduction"], auc_null
                ),
                "null_hit_gain_range": [min(hit_null), max(hit_null)],
                "null_auc_reduction_range": [min(auc_null), max(auc_null)],
            }
        )

    return {
        "objective_mode": mode,
        "folds": folds,
        "random_baseline": _random_baseline(labeled),
        "profiles": profiles,
        "comparisons": comparisons,
        "permutation": permutation_report,
        "oracle": _oracle_diagnostics(labeled),
        "records": records,
    }


def _development_acceptance(primary: dict[str, Any]) -> dict[str, Any]:
    learner = primary["profiles"][PRIMARY_LEARNER]
    random_baseline = primary["random_baseline"]
    action_seed = learner["action_seed"]
    dynamic = learner["dynamic"]
    full = learner["full_context"]
    permutation = primary["permutation"]
    oracle = primary["oracle"]
    action_auc_reduction = _safe_ratio(
        random_baseline["mean_auc_regret"] - action_seed["mean_auc_regret"],
        max(1e-12, random_baseline["mean_auc_regret"]),
    )
    dynamic_auc_reduction = _safe_ratio(
        action_seed["mean_auc_regret"] - dynamic["mean_auc_regret"],
        max(1e-12, action_seed["mean_auc_regret"]),
    )
    dynamic_hit_gain = (
        dynamic["pareto_top1_hit_rate"] - action_seed["pareto_top1_hit_rate"]
    )
    dynamic_rankable = (
        dynamic_hit_gain >= 0.02 and dynamic_auc_reduction >= -0.05
    ) or (
        dynamic_auc_reduction >= 0.03 and dynamic_hit_gain >= -0.03
    )
    collapsed = bool(oracle["multiple_sizes_supported"]) and (
        float(full["maximum_size_share"]) > 0.90
    )
    gates = [
        {
            "name": "labels_are_learnable",
            "actual": {
                "top1_gain_over_random": (
                    action_seed["pareto_top1_hit_rate"]
                    - random_baseline["pareto_top1_hit_rate"]
                ),
                "auc_regret_reduction": action_auc_reduction,
            },
            "requirement": "top1 gain >= 0.03 and AUC regret reduction >= 0.05",
            "passed": (
                action_seed["pareto_top1_hit_rate"]
                - random_baseline["pareto_top1_hit_rate"]
                >= 0.03
                and action_auc_reduction >= 0.05
            ),
        },
        {
            "name": "dynamic_state_is_rankable",
            "actual": {
                "top1_gain_over_action_seed": dynamic_hit_gain,
                "auc_regret_reduction": dynamic_auc_reduction,
            },
            "requirement": (
                "top1 gain >= 0.02 without >5% AUC degradation, or "
                "AUC reduction >= 0.03 without >3 point top1 degradation"
            ),
            "passed": dynamic_rankable,
        },
        {
            "name": "static_oracle_heterogeneity",
            "actual": oracle["static_heterogeneity_qualified"],
            "requirement": "a static dimension changes best families with spread >= 0.10",
            "passed": bool(oracle["static_heterogeneity_qualified"]),
        },
        {
            "name": "real_context_beats_permutations",
            "actual": {
                "hit_gain_percentile": permutation.get("hit_gain_percentile", 0.0),
                "auc_reduction_percentile": permutation.get(
                    "auc_reduction_percentile", 0.0
                ),
            },
            "requirement": "both percentiles >= 0.95 over 500 task-level permutations",
            "passed": (
                permutation.get("count") == 500
                and permutation.get("hit_gain_percentile", 0.0) >= 0.95
                and permutation.get("auc_reduction_percentile", 0.0) >= 0.95
            ),
        },
        {
            "name": "no_unsupported_size_collapse",
            "actual": {
                "oracle_supported_sizes": oracle["supported_sizes"],
                "full_context_maximum_size_share": full["maximum_size_share"],
            },
            "requirement": (
                "maximum selected-size share <= 0.90 when oracle supports "
                "multiple sizes"
            ),
            "passed": not collapsed,
        },
    ]
    return {"passed": all(bool(gate["passed"]) for gate in gates), "gates": gates}


def _confirmation_acceptance(primary: dict[str, Any]) -> dict[str, Any]:
    comparison = primary["comparisons"][PRIMARY_LEARNER]
    full = primary["profiles"][PRIMARY_LEARNER]["full_context"]
    oracle = primary["oracle"]
    permutation = primary["permutation"]
    collapsed = bool(oracle["multiple_sizes_supported"]) and (
        float(full["maximum_size_share"]) > 0.90
    )
    bootstrap_passed = (
        comparison["hit_gain_95_ci"][1] >= 0.0
        and comparison["auc_improvement_95_ci"][1] >= 0.0
    )
    gates = [
        {
            "name": "pareto_hit_gain",
            "actual": comparison["full_context_minus_dynamic_hit_rate"],
            "requirement": ">= 0.05",
            "passed": comparison["full_context_minus_dynamic_hit_rate"] >= 0.05,
        },
        {
            "name": "relative_auc_regret_reduction",
            "actual": comparison["relative_auc_regret_reduction"],
            "requirement": ">= 0.05",
            "passed": comparison["relative_auc_regret_reduction"] >= 0.05,
        },
        {
            "name": "map_grouped_bootstrap_no_significant_degradation",
            "actual": {
                "hit_gain_95_ci": comparison["hit_gain_95_ci"],
                "auc_improvement_95_ci": comparison["auc_improvement_95_ci"],
            },
            "requirement": "both upper bounds >= 0",
            "passed": bootstrap_passed,
        },
        {
            "name": "real_context_beats_permutations",
            "actual": {
                "hit_gain_percentile": permutation.get("hit_gain_percentile", 0.0),
                "auc_reduction_percentile": permutation.get(
                    "auc_reduction_percentile", 0.0
                ),
            },
            "requirement": "both percentiles >= 0.95 over 500 task-level permutations",
            "passed": (
                permutation.get("count") == 500
                and permutation.get("hit_gain_percentile", 0.0) >= 0.95
                and permutation.get("auc_reduction_percentile", 0.0) >= 0.95
            ),
        },
        {
            "name": "no_unsupported_size_collapse",
            "actual": full["maximum_size_share"],
            "requirement": "<= 0.90 when oracle supports multiple sizes",
            "passed": not collapsed,
        },
    ]
    return {"passed": all(bool(gate["passed"]) for gate in gates), "gates": gates}


def _strip_records(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "records"}


def _dataset_map_seeds(root: Path) -> set[int]:
    seeds: set[int] = set()
    for manifest in root.glob("*/manifest.jsonl"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seeds.add(int(json.loads(line)["map_seed"]))
    return seeds


def render_markdown(report: dict[str, Any]) -> str:
    primary = report["objectives"]["primary"]
    lines = [
        "# InitLNS Context Secondary Audit",
        "",
        f"Mode: `{report['evaluation_mode']}`",
        f"Acceptance: **{'PASS' if report['acceptance']['passed'] else 'FAIL'}**",
        "",
        "## Data",
        "",
        f"- Candidate rows: {report['counts']['candidate_rows']}",
        f"- States: {report['counts']['states']}",
        f"- Maps: {report['counts']['maps']}",
        f"- Tasks: {report['counts']['tasks']}",
        "",
        "## Primary Objective",
        "",
        "| Learner | Profile | Top-1 | AUC regret | Conflict regret | Max size share |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for learner in LEARNERS:
        for profile in FEATURE_PROFILES:
            value = primary["profiles"][learner][profile]
            lines.append(
                f"| {learner} | {profile} | {value['pareto_top1_hit_rate']:.4f} | "
                f"{value['mean_auc_regret']:.4f} | "
                f"{value['mean_conflict_regret']:.4f} | "
                f"{value['maximum_size_share']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | Requirement | Result |",
            "| --- | --- | --- |",
        ]
    )
    for gate in report["acceptance"]["gates"]:
        lines.append(
            f"| {gate['name']} | {gate['requirement']} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    permutation = primary["permutation"]
    lines.extend(
        [
            "",
            "## Context Permutation",
            "",
            f"- Permutations: {permutation['count']}",
            f"- Hit-gain percentile: {permutation.get('hit_gain_percentile', 0.0):.4f}",
            f"- AUC-reduction percentile: {permutation.get('auc_reduction_percentile', 0.0):.4f}",
            "",
            "## Oracle Size Coverage",
            "",
        ]
    )
    for size, coverage in primary["oracle"]["size_state_coverage"].items():
        lines.append(f"- Size {size}: {coverage:.4f} of states")
    return "\n".join(lines) + "\n"


def run_secondary_audit(
    collection: str | Path,
    output: str | Path,
    dataset: str | Path | None = None,
    evaluation_mode: str = "development",
    permutations: int = 500,
    reference_dataset: str | Path | None = None,
) -> dict[str, Any]:
    if permutations < 0:
        raise ValueError("permutations must be non-negative")
    output_root = Path(output).resolve()
    collection_root = Path(collection).resolve()
    resolved_dataset = _dataset_root(collection_root, dataset)
    index = _categoricalize_sizes(
        build_index(collection_root, dataset=resolved_dataset)
    )
    unexpected = sorted(
        {str(row["split"]) for row in index} - {"train", "validation"}
    )
    if unexpected:
        raise ValueError(f"secondary audit contains forbidden label splits: {unexpected}")
    if evaluation_mode == "development":
        for row in index:
            row["source_split"] = row["split"]
            row["split"] = "development"
    elif evaluation_mode == "confirmation":
        _validate_split_isolation(index)
    else:
        raise ValueError("evaluation_mode must be development or confirmation")

    seed_overlap: list[int] = []
    if reference_dataset is not None:
        if resolved_dataset is None:
            raise ValueError("reference seed check requires a dataset")
        reference_root = Path(reference_dataset).resolve()
        seed_overlap = sorted(
            _dataset_map_seeds(resolved_dataset) & _dataset_map_seeds(reference_root)
        )
        if seed_overlap:
            raise ValueError(f"confirmation map seeds overlap reference data: {seed_overlap}")

    objective_results = {
        mode: _run_objective(
            index,
            mode,
            evaluation_mode,
            permutations if mode == "primary" else 0,
        )
        for mode in OBJECTIVE_MODES
    }
    primary = objective_results["primary"]
    acceptance = (
        _development_acceptance(primary)
        if evaluation_mode == "development"
        else _confirmation_acceptance(primary)
    )
    digest = hashlib.sha256(
        "\n".join(
            f"{row['state_id']}|{row['candidate_key']}|{row.get('source_split', row['split'])}"
            for row in index
        ).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": SECONDARY_AUDIT_SCHEMA_VERSION,
        "evaluation_mode": evaluation_mode,
        "model_seed": MODEL_SEED,
        "primary_learner": PRIMARY_LEARNER,
        "collection": str(collection_root),
        "dataset": str(resolved_dataset) if resolved_dataset else None,
        "reference_dataset": (
            str(Path(reference_dataset).resolve()) if reference_dataset else None
        ),
        "reference_map_seed_overlap": seed_overlap,
        "index_sha256": digest,
        "counts": {
            "candidate_rows": len(index),
            "states": len({str(row["state_id"]) for row in index}),
            "maps": len({str(row["map_id"]) for row in index}),
            "tasks": len({str(row["task_id"]) for row in index}),
        },
        "pre_registration": {
            "neighborhood_size_encoding": "categorical",
            "learners": list(LEARNERS),
            "primary_learner": PRIMARY_LEARNER,
            "objective_modes": {
                "primary": [
                    "solved_rate",
                    "conflicts_after",
                    "conflict_auc",
                    "generated_nodes",
                ],
                "runtime_sensitivity": [
                    "solved_rate",
                    "conflicts_after",
                    "conflict_auc",
                    "generated_nodes",
                    "branch_runtime",
                ],
            },
            "permutations": permutations,
            "permutation_unit": "task_id",
            "bootstrap_unit": "map_id",
        },
        "objectives": {
            mode: _strip_records(value) for mode, value in objective_results.items()
        },
        "acceptance": acceptance,
    }
    _write_jsonl(
        output_root / "secondary_candidate_index.jsonl",
        _relabel(index, "primary"),
    )
    _write_json(output_root / "secondary_audit.json", report)
    (output_root / "secondary_audit.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    models_root = output_root / "models"
    labeled = _relabel(index, "primary")
    final_train = (
        labeled
        if evaluation_mode == "development"
        else [row for row in labeled if row["split"] == "train"]
    )
    for learner in LEARNERS:
        for profile in FEATURE_PROFILES:
            model = _train_ranker(final_train, profile, learner, "primary")
            path = models_root / f"{learner}__{profile}.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                pickle.dump(model, stream)
    return report
