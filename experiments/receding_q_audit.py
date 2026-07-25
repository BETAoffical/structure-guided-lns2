"""Offline learnability audit for a receding-horizon actual-candidate value model.

The audit deliberately does not claim end-to-end controller performance.  It
reuses registered S3 rollouts, groups them by the *actual* first neighborhood,
removes all future-template inputs, and asks whether map-grouped models can
predict fixed three-repair outcomes well enough to justify a small dedicated
receding-Q label collection.
"""

from __future__ import annotations

import collections
import csv
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_jsonl, sha256_file, write_json
from experiments.feature_schema_v3 import V3_FEATURE_NAMES
from experiments.v3_s3 import S3_TEMPORAL_FEATURE_NAMES


RECEDING_Q_AUDIT_SCHEMA = "lns2.receding_q_learnability_audit.v1"
HORIZON = 3
TARGETS = (
    "feasible_rate",
    "final_conflict_ratio",
    "normalized_step_auc",
    "no_progress_rate",
    "log_total_seconds",
)
ACTUAL_CANDIDATE_FEATURE_NAMES = (
    *V3_FEATURE_NAMES,
    *S3_TEMPORAL_FEATURE_NAMES,
)


def _atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write an empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({name for row in materialized for name in row})
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return statistics.fmean(materialized) if materialized else 0.0


def _std(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return statistics.pstdev(materialized) if len(materialized) > 1 else 0.0


def fixed_horizon_metrics(
    trial: dict[str, Any], *, horizon: int = HORIZON
) -> dict[str, float]:
    """Return quality and time labels padded to a common repair horizon."""

    if horizon <= 0:
        raise ValueError("receding-Q horizon must be positive")
    trajectory = list(map(int, trial.get("conflict_trajectory", ())))
    steps = sorted(
        (dict(value) for value in trial.get("steps", ())),
        key=lambda value: int(value["step"]),
    )
    if trajectory:
        initial_conflicts = int(trajectory[0])
    elif steps:
        initial_conflicts = int(steps[0]["conflicts_before"])
    else:
        raise ValueError("receding-Q trial has no conflict trajectory")
    if initial_conflicts <= 0:
        raise ValueError("receding-Q source state must contain conflicts")

    conflicts = [initial_conflicts]
    total_seconds = 0.0
    feasible = False
    for step in steps:
        executed = bool(step.get("executed", True))
        if not executed:
            total_seconds += max(0.0, float(step.get("selection_seconds", 0.0)))
            break
        if int(step["conflicts_before"]) != conflicts[-1]:
            raise ValueError("receding-Q trial conflict trajectory is discontinuous")
        total_seconds += max(0.0, float(step["total_seconds"]))
        conflicts_after = int(step["conflicts_after"])
        if conflicts_after < 0:
            raise ValueError("receding-Q trial contains negative conflicts")
        conflicts.append(conflicts_after)
        outcome = str(step.get("repair_outcome", ""))
        feasible = feasible or outcome == "feasible" or conflicts_after == 0
        if feasible or outcome in {"hard_failure", "accepted_noop"}:
            break
        if len(conflicts) > horizon:
            break

    conflicts = conflicts[: horizon + 1]
    while len(conflicts) < horizon + 1:
        conflicts.append(conflicts[-1])
    normalized_auc = math.fsum(
        (left + right) / 2.0
        for left, right in zip(conflicts[:-1], conflicts[1:])
    ) / (float(horizon) * float(initial_conflicts))
    final_conflicts = int(conflicts[-1])
    return {
        "initial_conflicts": float(initial_conflicts),
        "final_conflicts": float(final_conflicts),
        "final_conflict_ratio": float(final_conflicts) / float(initial_conflicts),
        "normalized_step_auc": float(normalized_auc),
        "feasible": float(feasible),
        "no_progress": float(min(conflicts) >= initial_conflicts),
        "total_seconds": float(total_seconds),
        "log_total_seconds": math.log1p(max(0.0, float(total_seconds))),
    }


def _candidate_feature_vector(
    feature: dict[str, Any],
    candidate_feature_names: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    names = tuple(map(str, feature["feature_names"]))
    values = tuple(map(float, feature["feature_values"]))
    if len(names) != len(values) or len(names) != len(set(names)):
        raise ValueError("receding-Q source feature row is malformed")
    source = dict(zip(names, values))
    missing = set(candidate_feature_names) - set(source)
    if missing:
        raise ValueError(
            f"receding-Q source lacks canonical candidate features: {sorted(missing)}"
        )
    return candidate_feature_names, tuple(
        float(source[name]) for name in candidate_feature_names
    )


def _v2_first_candidate_index(
    baseline_rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in baseline_rows:
        if str(row.get("controller")) != "v2-full":
            continue
        steps = sorted(
            (dict(value) for value in row.get("steps", ())),
            key=lambda value: int(value["step"]),
        )
        executed = [step for step in steps if bool(step.get("executed", True))]
        if not executed:
            raise ValueError("v2 baseline contains no executable first action")
        key = (str(row["split"]), str(row["state_id"]))
        grouped[key].add(str(executed[0]["candidate_id"]))
    result = {}
    for key, candidate_ids in grouped.items():
        if len(candidate_ids) != 1:
            raise ValueError("v2 baseline first action changes across paired seeds")
        result[key] = next(iter(candidate_ids))
    return result


def build_actual_candidate_rows(
    feature_rows: Iterable[dict[str, Any]],
    trial_rows: Iterable[dict[str, Any]],
    baseline_rows: Iterable[dict[str, Any]],
    *,
    candidate_feature_names: tuple[str, ...] = ACTUAL_CANDIDATE_FEATURE_NAMES,
) -> list[dict[str, Any]]:
    """Aggregate S3 sequences by their actual first-step neighborhood."""

    materialized_features = [dict(row) for row in feature_rows]
    features = {
        (str(row["state_id"]), str(row["sequence_id"])): dict(row)
        for row in materialized_features
    }
    if len(features) != len(materialized_features):
        raise ValueError("receding-Q features contain duplicate sequence keys")

    trials_by_sequence: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    trial_keys = set()
    for raw in trial_rows:
        row = dict(raw)
        key = (str(row["state_id"]), str(row["sequence_id"]))
        trial_key = (*key, int(row["trial_index"]))
        if trial_key in trial_keys:
            raise ValueError("receding-Q trials contain duplicate paired seeds")
        trial_keys.add(trial_key)
        trials_by_sequence[key].append(row)
    if set(features) != set(trials_by_sequence):
        raise ValueError("receding-Q features and trials differ in sequence coverage")

    v2_candidates = _v2_first_candidate_index(baseline_rows)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for sequence_key, feature in sorted(features.items()):
        trials = trials_by_sequence[sequence_key]
        candidate_ids = set()
        agent_sets = set()
        observations = []
        for trial in trials:
            steps = sorted(
                (dict(value) for value in trial.get("steps", ())),
                key=lambda value: int(value["step"]),
            )
            executed = [step for step in steps if bool(step.get("executed", True))]
            if not executed:
                raise ValueError("receding-Q sequence has no executable first action")
            first = executed[0]
            candidate_ids.add(str(first["candidate_id"]))
            agent_sets.add(tuple(sorted(map(int, first["agents"]))))
            observations.append(
                {
                    **fixed_horizon_metrics(trial),
                    "sequence_id": str(trial["sequence_id"]),
                    "trial_index": int(trial["trial_index"]),
                }
            )
        if len(candidate_ids) != 1 or len(agent_sets) != 1:
            raise ValueError(
                "receding-Q first actual neighborhood changes across paired seeds"
            )
        candidate_id = next(iter(candidate_ids))
        agents = next(iter(agent_sets))
        split = str(feature["split"])
        state_id = str(feature["state_id"])
        key = (split, state_id, candidate_id)
        names, values = _candidate_feature_vector(
            feature, tuple(candidate_feature_names)
        )
        templates = list(feature.get("templates", ()))
        if not templates:
            raise ValueError("receding-Q feature row has no first template")
        first_template = str(dict(templates[0])["template_key"])
        record = grouped.setdefault(
            key,
            {
                "split": split,
                "state_id": state_id,
                "map_id": str(feature["map_id"]),
                "layout_mode": str(feature["layout_mode"]),
                "agent_count": int(feature["agent_count"]),
                "source_stratum": str(feature.get("source_stratum", "unknown")),
                "candidate_id": candidate_id,
                "agents": agents,
                "feature_names": names,
                "feature_values": values,
                "first_templates": set(),
                "observations": [],
            },
        )
        if record["feature_names"] != names:
            raise ValueError("actual candidate feature names are inconsistent")
        if any(
            abs(float(left) - float(right)) > 1e-12
            for left, right in zip(record["feature_values"], values)
        ):
            raise ValueError("actual candidate features depend on future templates")
        record["first_templates"].add(first_template)
        record["observations"].extend(observations)

    rows = []
    for (split, state_id, candidate_id), record in sorted(grouped.items()):
        observations = list(record["observations"])
        if len(observations) < 2:
            raise ValueError("receding-Q candidate lacks repeated outcomes")
        values = {
            "feasible_rate": _mean(row["feasible"] for row in observations),
            "final_conflict_ratio": _mean(
                row["final_conflict_ratio"] for row in observations
            ),
            "normalized_step_auc": _mean(
                row["normalized_step_auc"] for row in observations
            ),
            "no_progress_rate": _mean(row["no_progress"] for row in observations),
            "log_total_seconds": _mean(
                row["log_total_seconds"] for row in observations
            ),
            "mean_total_seconds": _mean(
                row["total_seconds"] for row in observations
            ),
        }
        standard_deviations = {
            f"{name}_std": _std(row[name] for row in observations)
            for name in (
                "final_conflict_ratio",
                "normalized_step_auc",
                "log_total_seconds",
            )
        }
        v2_candidate = v2_candidates.get((split, state_id))
        rows.append(
            {
                **{name: value for name, value in record.items() if name != "observations"},
                **values,
                **standard_deviations,
                "first_templates": tuple(sorted(record["first_templates"])),
                "observation_count": len(observations),
                "paired_seed_count": len(
                    {int(row["trial_index"]) for row in observations}
                ),
                "v2_selected": candidate_id == v2_candidate,
                "v2_candidate_available": v2_candidate is not None,
                "observations": observations,
            }
        )
    return rows


def _actual_order_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -round(float(row["feasible_rate"]), 12),
        round(float(row["final_conflict_ratio"]), 12),
        round(float(row["normalized_step_auc"]), 12),
        round(float(row["no_progress_rate"]), 12),
        round(float(row["log_total_seconds"]), 12),
        str(row["candidate_id"]),
    )


def _predicted_order_key(
    row: dict[str, Any], prediction: dict[str, float]
) -> tuple[Any, ...]:
    return (
        -round(min(1.0, max(0.0, float(prediction["feasible_rate"]))), 6),
        round(max(0.0, float(prediction["final_conflict_ratio"])), 6),
        round(max(0.0, float(prediction["normalized_step_auc"])), 6),
        round(min(1.0, max(0.0, float(prediction["no_progress_rate"]))), 6),
        round(max(0.0, float(prediction["log_total_seconds"])), 6),
        str(row["candidate_id"]),
    )


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("correlation inputs must be non-empty and paired")
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = math.fsum(
        (first - left_mean) * (second - right_mean)
        for first, second in zip(left, right)
    )
    left_scale = math.sqrt(
        math.fsum((value - left_mean) ** 2 for value in left)
    )
    right_scale = math.sqrt(
        math.fsum((value - right_mean) ** 2 for value in right)
    )
    if left_scale <= 0.0 or right_scale <= 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _state_groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["state_id"])].append(index)
    return [grouped[key] for key in sorted(grouped)]


def _fit_models(
    rows: list[dict[str, Any]], indices: list[int]
) -> dict[str, Any]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    matrix = np.asarray([rows[index]["feature_values"] for index in indices], dtype=float)
    models = {}
    for target in TARGETS:
        values = np.asarray([float(rows[index][target]) for index in indices])
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=17,
        )
        model.fit(matrix, values)
        models[target] = model
    return models


def _predict_models(
    models: dict[str, Any], rows: list[dict[str, Any]], indices: list[int]
) -> list[dict[str, float]]:
    import numpy as np

    matrix = np.asarray([rows[index]["feature_values"] for index in indices], dtype=float)
    result = [dict() for _ in indices]
    for target, model in models.items():
        for local, value in enumerate(model.predict(matrix)):
            result[local][target] = float(value)
    return result


def map_group_predictions(
    rows: list[dict[str, Any]], *, diagnostic_split: str = "policy_validation"
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, dict[str, float]],
    dict[int, dict[str, float]],
    list[dict[str, Any]],
]:
    """Create policy-train OOF and map-isolated diagnostic predictions."""

    from sklearn.model_selection import GroupKFold

    train_indices = [
        index for index, row in enumerate(rows) if row["split"] == "policy_train"
    ]
    diagnostic_indices = [
        index for index, row in enumerate(rows) if row["split"] == diagnostic_split
    ]
    if not train_indices or not diagnostic_indices:
        raise ValueError("receding-Q audit requires train and diagnostic rows")
    train_maps = {str(rows[index]["map_id"]) for index in train_indices}
    diagnostic_maps = {str(rows[index]["map_id"]) for index in diagnostic_indices}
    if train_maps & diagnostic_maps:
        raise ValueError("receding-Q train and diagnostic maps overlap")
    if len(train_maps) < 4:
        raise ValueError("receding-Q OOF requires at least four training maps")

    groups = [str(rows[index]["map_id"]) for index in train_indices]
    folds = GroupKFold(n_splits=4)
    oof: dict[int, dict[str, float]] = {}
    oof_baseline: dict[int, dict[str, float]] = {}
    fold_rows = []
    for fold_index, (local_train, local_test) in enumerate(
        folds.split(train_indices, groups=groups)
    ):
        fitting = [train_indices[int(value)] for value in local_train]
        held = [train_indices[int(value)] for value in local_test]
        models = _fit_models(rows, fitting)
        predicted = _predict_models(models, rows, held)
        means = {
            target: _mean(float(rows[index][target]) for index in fitting)
            for target in TARGETS
        }
        for index, values in zip(held, predicted):
            oof[index] = values
            oof_baseline[index] = dict(means)
        fold_rows.append(
            {
                "fold": fold_index,
                "training_map_count": len(
                    {str(rows[index]["map_id"]) for index in fitting}
                ),
                "held_map_count": len(
                    {str(rows[index]["map_id"]) for index in held}
                ),
                "training_candidate_count": len(fitting),
                "held_candidate_count": len(held),
            }
        )
    if set(oof) != set(train_indices):
        raise ValueError("receding-Q OOF predictions are incomplete")

    models = _fit_models(rows, train_indices)
    predicted = _predict_models(models, rows, diagnostic_indices)
    means = {
        target: _mean(float(rows[index][target]) for index in train_indices)
        for target in TARGETS
    }
    diagnostic = {
        index: values for index, values in zip(diagnostic_indices, predicted)
    }
    diagnostic_baseline = {index: dict(means) for index in diagnostic_indices}
    baseline = {**oof_baseline, **diagnostic_baseline}
    return oof, diagnostic, baseline, fold_rows


def _target_diagnostics(
    rows: list[dict[str, Any]],
    indices: list[int],
    predictions: dict[int, dict[str, float]],
    baseline_predictions: dict[int, dict[str, float]],
    *,
    split: str,
) -> list[dict[str, Any]]:
    diagnostics = []
    for target in TARGETS:
        actual = [float(rows[index][target]) for index in indices]
        predicted = [float(predictions[index][target]) for index in indices]
        baseline = [float(baseline_predictions[index][target]) for index in indices]
        mae = _mean(abs(left - right) for left, right in zip(actual, predicted))
        baseline_mae = _mean(
            abs(left - right) for left, right in zip(actual, baseline)
        )
        denominator = math.fsum((value - _mean(actual)) ** 2 for value in actual)
        r2 = (
            1.0
            - math.fsum(
                (left - right) ** 2 for left, right in zip(actual, predicted)
            )
            / denominator
            if denominator > 0.0
            else 0.0
        )
        diagnostics.append(
            {
                "split": split,
                "target": target,
                "candidate_count": len(indices),
                "mae": mae,
                "mean_baseline_mae": baseline_mae,
                "mae_improvement_fraction": (
                    (baseline_mae - mae) / baseline_mae
                    if baseline_mae > 0.0
                    else 0.0
                ),
                "r2": r2,
            }
        )
    return diagnostics


def _compare_actual(
    selected: dict[str, Any], baseline: dict[str, Any]
) -> str:
    selected_key = _actual_order_key(selected)[:-1]
    baseline_key = _actual_order_key(baseline)[:-1]
    if selected_key < baseline_key:
        return "win"
    if selected_key > baseline_key:
        return "loss"
    return "tie"


def selection_diagnostics(
    rows: list[dict[str, Any]],
    indices: list[int],
    predictions: dict[int, dict[str, float]],
    *,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_rows = []
    correlations = []
    exact = 0
    top3 = 0
    wins = 0
    losses = 0
    ties = 0
    for state_indices in _state_groups([rows[index] for index in indices]):
        # _state_groups above returns local positions for the sliced rows.
        local_rows = [rows[indices[position]] for position in state_indices]
        global_indices = [indices[position] for position in state_indices]
        actual_order = sorted(
            range(len(local_rows)), key=lambda value: _actual_order_key(local_rows[value])
        )
        predicted_order = sorted(
            range(len(local_rows)),
            key=lambda value: _predicted_order_key(
                local_rows[value], predictions[global_indices[value]]
            ),
        )
        actual_ranks = [0.0] * len(local_rows)
        predicted_ranks = [0.0] * len(local_rows)
        for rank, value in enumerate(actual_order):
            actual_ranks[value] = float(rank)
        for rank, value in enumerate(predicted_order):
            predicted_ranks[value] = float(rank)
        correlations.append(_correlation(actual_ranks, predicted_ranks))
        selected = local_rows[predicted_order[0]]
        oracle = local_rows[actual_order[0]]
        exact += int(selected["candidate_id"] == oracle["candidate_id"])
        top3 += int(predicted_order[0] in set(actual_order[:3]))
        v2_rows = [row for row in local_rows if bool(row["v2_selected"])]
        outcome = "missing_v2"
        if len(v2_rows) == 1:
            outcome = _compare_actual(selected, v2_rows[0])
            wins += int(outcome == "win")
            losses += int(outcome == "loss")
            ties += int(outcome == "tie")
        elif len(v2_rows) > 1:
            raise ValueError("state contains multiple v2-selected candidates")
        selected_rows.append(
            {
                "split": split,
                "state_id": str(selected["state_id"]),
                "map_id": str(selected["map_id"]),
                "layout_mode": str(selected["layout_mode"]),
                "agent_count": int(selected["agent_count"]),
                "candidate_count": len(local_rows),
                "selected_candidate_id": str(selected["candidate_id"]),
                "oracle_candidate_id": str(oracle["candidate_id"]),
                "v2_candidate_id": (
                    str(v2_rows[0]["candidate_id"]) if len(v2_rows) == 1 else ""
                ),
                "oracle_exact_match": selected["candidate_id"]
                == oracle["candidate_id"],
                "oracle_top3_hit": predicted_order[0] in set(actual_order[:3]),
                "rank_correlation": correlations[-1],
                "outcome_vs_v2": outcome,
                **{
                    f"selected_{name}": float(selected[name])
                    for name in (
                        "feasible_rate",
                        "final_conflict_ratio",
                        "normalized_step_auc",
                        "no_progress_rate",
                        "mean_total_seconds",
                    )
                },
                **{
                    f"v2_{name}": (
                        float(v2_rows[0][name]) if len(v2_rows) == 1 else math.nan
                    )
                    for name in (
                        "feasible_rate",
                        "final_conflict_ratio",
                        "normalized_step_auc",
                        "no_progress_rate",
                        "mean_total_seconds",
                    )
                },
            }
        )
    count = len(selected_rows)
    paired = wins + losses + ties
    return (
        {
            "split": split,
            "state_count": count,
            "mean_rank_correlation": _mean(correlations),
            "oracle_exact_match_rate": exact / count,
            "oracle_top3_hit_rate": top3 / count,
            "v2_paired_state_count": paired,
            "wins_vs_v2": wins,
            "losses_vs_v2": losses,
            "ties_vs_v2": ties,
            "net_wins_vs_v2": wins - losses,
            **{
                f"{name}_delta_vs_v2": _mean(
                    float(row[f"selected_{name}"]) - float(row[f"v2_{name}"])
                    for row in selected_rows
                    if math.isfinite(float(row[f"v2_{name}"]))
                )
                for name in (
                    "feasible_rate",
                    "final_conflict_ratio",
                    "normalized_step_auc",
                    "no_progress_rate",
                    "mean_total_seconds",
                )
            },
        },
        selected_rows,
    )


def seed_stability(rows: list[dict[str, Any]]) -> dict[str, float]:
    correlations = []
    exact = 0
    covered = 0
    for indices in _state_groups(rows):
        candidates = [rows[index] for index in indices]
        seed_values = sorted(
            set.intersection(
                *[
                    {int(value["trial_index"]) for value in row["observations"]}
                    for row in candidates
                ]
            )
        )
        if len(seed_values) < 2:
            continue
        orders = []
        for seed in seed_values[:2]:
            seed_rows = []
            for row in candidates:
                observations = [
                    value
                    for value in row["observations"]
                    if int(value["trial_index"]) == seed
                ]
                seed_rows.append(
                    {
                        **row,
                        "feasible_rate": _mean(
                            value["feasible"] for value in observations
                        ),
                        "final_conflict_ratio": _mean(
                            value["final_conflict_ratio"] for value in observations
                        ),
                        "normalized_step_auc": _mean(
                            value["normalized_step_auc"] for value in observations
                        ),
                        "no_progress_rate": _mean(
                            value["no_progress"] for value in observations
                        ),
                        "log_total_seconds": _mean(
                            value["log_total_seconds"] for value in observations
                        ),
                    }
                )
            orders.append(
                sorted(
                    range(len(seed_rows)),
                    key=lambda value: _actual_order_key(seed_rows[value]),
                )
            )
        ranks = []
        for order in orders:
            values = [0.0] * len(order)
            for rank, candidate in enumerate(order):
                values[candidate] = float(rank)
            ranks.append(values)
        correlations.append(_correlation(ranks[0], ranks[1]))
        exact += int(orders[0][0] == orders[1][0])
        covered += 1
    return {
        "state_count": covered,
        "mean_candidate_rank_correlation": _mean(correlations),
        "exact_oracle_winner_agreement_rate": exact / covered if covered else 0.0,
    }


def _map_rows(selection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in selection_rows:
        grouped[str(row["map_id"])].append(row)
    result = []
    for map_id, rows in sorted(grouped.items()):
        outcomes = collections.Counter(str(row["outcome_vs_v2"]) for row in rows)
        result.append(
            {
                "map_id": map_id,
                "layout_mode": str(rows[0]["layout_mode"]),
                "state_count": len(rows),
                "wins_vs_v2": outcomes["win"],
                "losses_vs_v2": outcomes["loss"],
                "ties_vs_v2": outcomes["tie"],
                "mean_rank_correlation": _mean(
                    float(row["rank_correlation"]) for row in rows
                ),
                "oracle_exact_match_rate": _mean(
                    float(row["oracle_exact_match"]) for row in rows
                ),
            }
        )
    return result


def _markdown(report: dict[str, Any]) -> str:
    diagnostic = dict(report["diagnostic"])
    selection = dict(diagnostic["selection"])
    stability = dict(diagnostic["paired_seed_stability"])
    checks = dict(report["checks"])
    lines = [
        "# Receding-Q Offline Learnability Audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "This is a diagnostic reuse of S3 labels. It is not an end-to-end "
        "controller result and does not promote a model.",
        "",
        "## Coverage",
        "",
        f"- States: {report['coverage']['state_count']}",
        f"- Actual candidates: {report['coverage']['candidate_count']}",
        f"- Train/diagnostic maps: {report['coverage']['training_map_count']}/"
        f"{report['coverage']['diagnostic_map_count']}",
        f"- Actual-candidate feature count: {report['coverage']['feature_count']}",
        "",
        "## Diagnostic-map result",
        "",
        f"- Mean candidate-rank correlation: "
        f"{selection['mean_rank_correlation']:.4f}",
        f"- Oracle exact/top-3 rate: "
        f"{selection['oracle_exact_match_rate']:.4f}/"
        f"{selection['oracle_top3_hit_rate']:.4f}",
        f"- Wins/losses/ties versus the v2 first candidate under the same "
        f"registered-tail label distribution: {selection['wins_vs_v2']}/"
        f"{selection['losses_vs_v2']}/{selection['ties_vs_v2']}",
        f"- Paired-seed candidate-rank correlation: "
        f"{stability['mean_candidate_rank_correlation']:.4f}",
        f"- Paired-seed exact winner agreement: "
        f"{stability['exact_oracle_winner_agreement_rate']:.4f}",
        "",
        "## Checks",
        "",
        *[f"- {name}: `{str(value).lower()}`" for name, value in checks.items()],
        "",
        "## Boundary",
        "",
        "- Future outcomes come from the registered S3 continuation distribution, "
        "not from a deployed receding-Q policy.",
        "- The diagnostic maps were already inspected in prior S3 audits and are "
        "not a locked promotion set.",
        "- Passing only justifies a small fresh-label pilot; failing says the "
        "existing S3 labels are insufficient.",
        "",
    ]
    return "\n".join(lines)


def audit_receding_q_learnability(
    *, source: str | Path, output: str | Path
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    collection_root = (
        source_root / "collection"
        if (source_root / "collection").is_dir()
        else source_root
    )
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("receding-Q audit output is non-empty")
    required = {
        "features": collection_root / "sequence_features.jsonl",
        "trials": collection_root / "sequence_trials.jsonl",
        "baselines": collection_root / "external_baselines.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"receding-Q audit source is incomplete: {missing}")

    feature_rows = read_jsonl(required["features"])
    trial_rows = read_jsonl(required["trials"])
    baseline_rows = read_jsonl(required["baselines"])
    rows = build_actual_candidate_rows(feature_rows, trial_rows, baseline_rows)
    feature_names = {tuple(row["feature_names"]) for row in rows}
    if len(feature_names) != 1:
        raise ValueError("receding-Q actual candidates use multiple feature schemas")

    train_indices = [
        index for index, row in enumerate(rows) if row["split"] == "policy_train"
    ]
    diagnostic_indices = [
        index
        for index, row in enumerate(rows)
        if row["split"] == "policy_validation"
    ]
    oof, diagnostic_predictions, baseline_predictions, fold_rows = (
        map_group_predictions(rows)
    )
    target_rows = [
        *_target_diagnostics(
            rows,
            train_indices,
            oof,
            baseline_predictions,
            split="policy_train_oof",
        ),
        *_target_diagnostics(
            rows,
            diagnostic_indices,
            diagnostic_predictions,
            baseline_predictions,
            split="policy_validation",
        ),
    ]
    train_selection, train_state_rows = selection_diagnostics(
        rows, train_indices, oof, split="policy_train_oof"
    )
    diagnostic_selection, diagnostic_state_rows = selection_diagnostics(
        rows,
        diagnostic_indices,
        diagnostic_predictions,
        split="policy_validation",
    )
    stability = seed_stability(rows)
    train_maps = {
        str(rows[index]["map_id"]) for index in train_indices
    }
    diagnostic_maps = {
        str(rows[index]["map_id"]) for index in diagnostic_indices
    }
    candidate_counts = [
        len(indices) for indices in _state_groups(rows)
    ]
    target_lookup = {
        (str(row["split"]), str(row["target"])): row for row in target_rows
    }
    oof_improved = sum(
        float(target_lookup[("policy_train_oof", target)]["mae_improvement_fraction"])
        >= 0.10
        for target in TARGETS
    )
    diagnostic_improved = sum(
        float(target_lookup[("policy_validation", target)]["mae_improvement_fraction"])
        >= 0.10
        for target in TARGETS
    )
    checks = {
        "source_coverage_complete": len(rows) > 0
        and min(candidate_counts) >= 16
        and all(int(row["paired_seed_count"]) >= 2 for row in rows),
        "train_and_diagnostic_maps_disjoint": not bool(train_maps & diagnostic_maps),
        "at_least_three_oof_targets_improve_mae_10pct": oof_improved >= 3,
        "at_least_three_diagnostic_targets_improve_mae_10pct": (
            diagnostic_improved >= 3
        ),
        "diagnostic_rank_correlation_at_least_0_20": float(
            diagnostic_selection["mean_rank_correlation"]
        )
        >= 0.20,
        "paired_seed_rank_correlation_at_least_0_50": float(
            stability["mean_candidate_rank_correlation"]
        )
        >= 0.50,
        "diagnostic_net_wins_vs_v2_nonnegative": int(
            diagnostic_selection["net_wins_vs_v2"]
        )
        >= 0,
        "diagnostic_feasible_rate_not_lower_by_1pp": float(
            diagnostic_selection["feasible_rate_delta_vs_v2"]
        )
        >= -0.01,
        "diagnostic_auc_not_worse_by_2pct_absolute": float(
            diagnostic_selection["normalized_step_auc_delta_vs_v2"]
        )
        <= 0.02,
    }
    coverage_passed = bool(checks["source_coverage_complete"]) and bool(
        checks["train_and_diagnostic_maps_disjoint"]
    )
    passed = coverage_passed and all(checks.values())
    decision = (
        "receding_q_offline_signal_supported_for_small_fresh_label_pilot"
        if passed
        else (
            "existing_s3_labels_insufficient_for_receding_q_promotion"
            if coverage_passed
            else "receding_q_audit_blocked_by_source_integrity"
        )
    )
    report = {
        "schema": RECEDING_Q_AUDIT_SCHEMA,
        "decision": decision,
        "diagnostic_only": True,
        "source": {
            name: {
                "file": path.name,
                "sha256": sha256_file(path),
            }
            for name, path in required.items()
        },
        "method": {
            "horizon": HORIZON,
            "action_identity": "actual first-step candidate_id and sorted agent set",
            "feature_projection": (
                "feature-v3 94 audited candidate features plus six deterministic "
                "history features; all future-template and removed feature-v2 "
                "inputs are excluded"
            ),
            "targets": list(TARGETS),
            "model": (
                "five fixed-capacity HistGradientBoostingRegressor heads; "
                "four-fold map-group OOF"
            ),
            "selection_order": (
                "maximize feasible rate; minimize final conflict ratio, "
                "normalized fixed-step AUC, no-progress rate, and log time"
            ),
        },
        "coverage": {
            "state_count": len(_state_groups(rows)),
            "candidate_count": len(rows),
            "training_state_count": len(
                {str(rows[index]["state_id"]) for index in train_indices}
            ),
            "diagnostic_state_count": len(
                {str(rows[index]["state_id"]) for index in diagnostic_indices}
            ),
            "training_map_count": len(train_maps),
            "diagnostic_map_count": len(diagnostic_maps),
            "minimum_candidates_per_state": min(candidate_counts),
            "maximum_candidates_per_state": max(candidate_counts),
            "feature_count": len(next(iter(feature_names))),
            "v2_first_candidate_coverage": _mean(
                float(row["v2_candidate_available"]) for row in rows
            ),
        },
        "diagnostic": {
            "targets": target_rows,
            "selection": diagnostic_selection,
            "training_oof_selection": train_selection,
            "paired_seed_stability": stability,
        },
        "checks": checks,
        "limitations": [
            "S3 future tails are registered templates, not a receding-Q policy.",
            "Several outcomes for one first action share the same first-step PP result.",
            "Policy-validation maps have already been inspected and are diagnostic.",
            "This audit measures local fixed-three-repair learnability, not full episodes.",
        ],
        "recommended_next_step": (
            "Collect a small map-isolated true receding-Q label pilot with actual "
            "candidate actions and on-policy replanning."
            if passed
            else "Do not train or promote a receding-Q controller from these S3 labels."
        ),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    candidate_output = []
    combined_predictions = {**oof, **diagnostic_predictions}
    for index, row in enumerate(rows):
        candidate_output.append(
            {
                **{
                    name: value
                    for name, value in row.items()
                    if name
                    not in {
                        "agents",
                        "feature_names",
                        "feature_values",
                        "observations",
                        "first_templates",
                    }
                },
                "agent_ids": " ".join(map(str, row["agents"])),
                "first_templates": " ".join(row["first_templates"]),
                **{
                    f"predicted_{target}": float(
                        combined_predictions[index][target]
                    )
                    for target in TARGETS
                },
            }
        )
    _atomic_write_csv(output_root / "actual_candidate_values.csv", candidate_output)
    _atomic_write_csv(output_root / "target_diagnostics.csv", target_rows)
    state_rows = [*train_state_rows, *diagnostic_state_rows]
    _atomic_write_csv(output_root / "state_selection_diagnostics.csv", state_rows)
    _atomic_write_csv(output_root / "map_diagnostics.csv", _map_rows(state_rows))
    _atomic_write_csv(output_root / "map_folds.csv", fold_rows)
    write_json(output_root / "receding_q_audit_report.json", report)
    (output_root / "receding_q_audit_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    write_json(
        output_root / "status.json",
        {
            "schema": RECEDING_Q_AUDIT_SCHEMA,
            "status": "complete",
            "decision": decision,
            "error_count": 0,
        },
    )
    return report


__all__ = [
    "ACTUAL_CANDIDATE_FEATURE_NAMES",
    "HORIZON",
    "RECEDING_Q_AUDIT_SCHEMA",
    "TARGETS",
    "audit_receding_q_learnability",
    "build_actual_candidate_rows",
    "fixed_horizon_metrics",
    "map_group_predictions",
    "seed_stability",
    "selection_diagnostics",
]
