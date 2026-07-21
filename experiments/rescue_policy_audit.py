from __future__ import annotations

import collections
import csv
import itertools
import math
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file
from experiments.repair_aware import load_repair_aware_bundle, repair_aware_order
from experiments.repair_collection import _fingerprint, _read_jsonl, _write_json


RESCUE_POLICY_AUDIT_SCHEMA = "lns2.rescue_policy_audit.v1"
AUDITED_SIZES = (4, 8, 16)
STATE_CHANGE_ASSUMPTIONS = (
    "replan-success-stop",
    "conflict-reduction-stop",
)


@dataclass(frozen=True)
class RescuePolicy:
    policy_id: str
    size_order: tuple[int, ...]
    reference_only: bool = False


def enumerate_rescue_policies() -> list[RescuePolicy]:
    policies = [RescuePolicy("adaptive", ())]
    for length in range(1, len(AUDITED_SIZES) + 1):
        for order in itertools.permutations(AUDITED_SIZES, length):
            policies.append(
                RescuePolicy(
                    ">".join(map(str, (*order, "adaptive"))),
                    tuple(order),
                )
            )
    if len(policies) != 16:
        raise AssertionError("the fixed 4/8/16 policy grid must contain 16 rules")
    return policies


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        raise ValueError(f"cannot write an empty CSV: {path.name}")
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
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_state_fingerprints(
    source: Path, expected_states: set[str]
) -> dict[str, dict[str, Any]]:
    state_root = source / "collection" / "states"
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(state_root.glob("*/*.json")):
        payload = dict(read_json(path))
        state = dict(payload.get("state", {}))
        state_id = str(state.get("state_id", ""))
        if not state_id or state_id in rows:
            raise ValueError("state files contain a missing or duplicate state_id")
        for name in ("before_fingerprint", "before_repair_fingerprint"):
            value = str(state.get(name, ""))
            if len(value) != 64:
                raise ValueError(f"state {state_id} is missing {name}")
        rows[state_id] = state
    if set(rows) != expected_states:
        missing = sorted(expected_states - set(rows))
        extra = sorted(set(rows) - expected_states)
        raise ValueError(
            f"state fingerprint coverage mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    return rows


def _state_change(
    outcome: dict[str, Any], assumption: str
) -> tuple[bool, str]:
    if "state_changed" in outcome:
        return bool(outcome["state_changed"]), "explicit-state-changed"
    before = outcome.get("before_fingerprint")
    after = outcome.get("after_fingerprint")
    if before is not None or after is not None:
        if not before or not after:
            raise ValueError("trial has an incomplete before/after fingerprint pair")
        return str(before) != str(after), "after-fingerprint"
    if assumption == "replan-success-stop":
        return bool(outcome.get("replan_success")), "inferred-replan-success"
    if assumption == "conflict-reduction-stop":
        return bool(outcome.get("feasible")) or int(
            outcome.get("conflict_reduction", 0)
        ) > 0, "inferred-conflict-reduction"
    raise ValueError(f"unsupported state-change assumption: {assumption}")


def _balanced_map_folds(
    state_metadata: dict[str, dict[str, Any]], count: int = 4
) -> list[dict[str, Any]]:
    map_layout: dict[str, str] = {}
    for row in state_metadata.values():
        if str(row["split"]) != "policy_train":
            continue
        map_id = str(row["map_id"])
        layout = str(row.get("layout_mode", "unknown"))
        previous = map_layout.setdefault(map_id, layout)
        if previous != layout:
            raise ValueError(f"map {map_id} has inconsistent layouts")
    if len(map_layout) < count:
        raise ValueError("rescue policy OOF requires at least four training maps")
    by_layout: dict[str, list[str]] = collections.defaultdict(list)
    for map_id, layout in map_layout.items():
        by_layout[layout].append(map_id)
    validation: list[list[str]] = [[] for _ in range(count)]
    for layout in sorted(by_layout):
        for index, map_id in enumerate(sorted(by_layout[layout])):
            validation[index % count].append(map_id)
    if any(not fold for fold in validation):
        raise ValueError("rescue policy OOF produced an empty map fold")
    all_maps = set(map_layout)
    return [
        {
            "fold": index,
            "train_maps": sorted(all_maps - set(maps)),
            "validation_maps": sorted(maps),
        }
        for index, maps in enumerate(validation)
    ]


def _index_collection(
    feature_rows: list[dict[str, Any]], trial_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    features: dict[tuple[str, str], dict[str, Any]] = {}
    state_metadata: dict[str, dict[str, Any]] = {}
    for row in feature_rows:
        state_id = str(row.get("state_id", ""))
        candidate_id = str(row.get("candidate_id", ""))
        if not state_id or not candidate_id:
            raise ValueError("feature row is missing state_id or candidate_id")
        key = (state_id, candidate_id)
        if key in features:
            raise ValueError(f"duplicate candidate feature row: {key}")
        split = str(row.get("split", ""))
        if split not in {"policy_train", "policy_validation"}:
            raise ValueError(f"unexpected data split: {split}")
        metadata = {
            "state_id": state_id,
            "split": split,
            "map_id": str(row.get("map_id", "")),
            "layout_mode": str(row.get("layout_mode", "unknown")),
            "agent_count": int(row.get("agent_count", 0)),
        }
        previous = state_metadata.setdefault(state_id, metadata)
        if previous != metadata:
            raise ValueError(f"state metadata is inconsistent: {state_id}")
        features[key] = row

    trials: dict[tuple[str, str, int], dict[str, Any]] = {}
    trials_by_arm: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trial_rows:
        if str(row.get("status")) not in {"ok", "resumed"} or not bool(
            row.get("complete")
        ):
            raise ValueError("audit source contains an incomplete trial")
        key2 = (str(row.get("state_id", "")), str(row.get("candidate_id", "")))
        if key2 not in features:
            raise ValueError(f"trial does not have a candidate feature row: {key2}")
        if str(row.get("split")) != str(features[key2]["split"]):
            raise ValueError(f"trial split differs from its feature row: {key2}")
        trial_index = int(row.get("trial_index", -1))
        key3 = (*key2, trial_index)
        if trial_index < 0 or key3 in trials:
            raise ValueError(f"duplicate or invalid paired trial: {key3}")
        trials[key3] = row
        trials_by_arm[key2].append(row)
    if set(trials_by_arm) != set(features):
        raise ValueError("candidate/trial coverage is incomplete")

    candidate_ids: dict[str, list[str]] = collections.defaultdict(list)
    for state_id, candidate_id in features:
        candidate_ids[state_id].append(candidate_id)
    trial_indices: dict[str, tuple[int, ...]] = {}
    for state_id, ids in candidate_ids.items():
        expected_indices: tuple[int, ...] | None = None
        seeds_by_trial: dict[int, int] = {}
        for candidate_id in ids:
            rows = sorted(
                trials_by_arm[(state_id, candidate_id)],
                key=lambda row: int(row["trial_index"]),
            )
            indices = tuple(int(row["trial_index"]) for row in rows)
            if expected_indices is None:
                expected_indices = indices
            elif indices != expected_indices:
                raise ValueError(f"paired trial coverage differs within state {state_id}")
            for row in rows:
                index = int(row["trial_index"])
                seed = int(row["random_seed"])
                previous = seeds_by_trial.setdefault(index, seed)
                if previous != seed:
                    raise ValueError(f"paired seed mismatch in state {state_id}")
        if not expected_indices:
            raise ValueError(f"state {state_id} has no trials")
        trial_indices[state_id] = expected_indices

    train_maps = {
        str(row["map_id"])
        for row in state_metadata.values()
        if str(row["split"]) == "policy_train"
    }
    validation_maps = {
        str(row["map_id"])
        for row in state_metadata.values()
        if str(row["split"]) == "policy_validation"
    }
    if not train_maps or not validation_maps:
        raise ValueError("audit requires policy_train and policy_validation states")
    if train_maps & validation_maps:
        raise ValueError("training and diagnostic maps overlap")
    return {
        "features": features,
        "trials": trials,
        "candidate_ids": dict(candidate_ids),
        "trial_indices": trial_indices,
        "state_metadata": state_metadata,
    }


def _highest_v2_candidate(
    rows: Iterable[dict[str, Any]], actual_size: int
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if str(row.get("route")) == "model"
        and int(row.get("actual_size", 0)) == actual_size
        and not bool(row.get("base_selected"))
    ]
    if not eligible:
        raise ValueError(f"state has no unused exact size-{actual_size} candidate")
    return min(
        eligible,
        key=lambda row: (
            -round(float(row.get("main_score", -math.inf)), 12),
            str(row["candidate_id"]),
        ),
    )


def _learned_reference_choice(
    rows: list[dict[str, Any]], bundle: Any
) -> str | None:
    eligible = [
        row
        for row in rows
        if str(row.get("route")) == "model"
        and int(row.get("actual_size", 0)) in AUDITED_SIZES
        and not bool(row.get("base_selected"))
    ]
    adaptive = [row for row in rows if str(row.get("route")) == "official_adaptive"]
    if len(adaptive) != 1:
        raise ValueError("each state must contain exactly one Adaptive arm")
    predictions = bundle.predict(eligible)
    ordered = repair_aware_order(
        eligible,
        predictions,
        [float(row.get("main_score", -math.inf)) for row in eligible],
    )
    if not ordered:
        return None
    selected = ordered[0]
    adaptive_prediction = bundle.predict(adaptive)
    efficiency = float(predictions["efficiency"][selected])
    adaptive_efficiency = float(adaptive_prediction["efficiency"][0])
    minimum = float(bundle.thresholds["minimum_predicted_efficiency"])
    margin = float(bundle.thresholds["adaptive_efficiency_margin"])
    if efficiency + 1e-12 < max(minimum, adaptive_efficiency * (1.0 + margin)):
        return None
    return str(eligible[selected]["candidate_id"])


def _simulate_sequence(
    *,
    metadata: dict[str, Any],
    trial_index: int,
    policy_id: str,
    candidate_sequence: list[str],
    adaptive_id: str,
    trials: dict[tuple[str, str, int], dict[str, Any]],
    assumption: str,
    reference_only: bool,
) -> dict[str, Any]:
    state_id = str(metadata["state_id"])
    attempts = list(candidate_sequence) + [adaptive_id]
    total_seconds = 0.0
    total_pp_seconds = 0.0
    total_generated = 0
    total_expanded = 0
    total_reopened = 0
    reductions = 0
    changed = False
    terminal_hard_failure = False
    evidence = ""
    attempted: list[str] = []
    terminal_candidate = adaptive_id
    for candidate_id in attempts:
        key = (state_id, candidate_id, trial_index)
        if key not in trials:
            raise ValueError(f"missing policy branch trial: {key}")
        trial = trials[key]
        outcome = dict(trial["outcome"])
        attempted.append(candidate_id)
        terminal_candidate = candidate_id
        total_seconds += max(0.0, float(outcome["repair_seconds"]))
        total_pp_seconds += max(0.0, float(outcome.get("pp_replan_seconds", 0.0)))
        total_generated += max(0, int(outcome.get("generated", 0)))
        total_expanded += max(0, int(outcome.get("expanded", 0)))
        total_reopened += max(0, int(outcome.get("reopened", 0)))
        reductions += max(0, int(outcome.get("conflict_reduction", 0)))
        terminal_hard_failure = bool(outcome.get("hard_failure"))
        changed, evidence = _state_change(outcome, assumption)
        if changed:
            break
    return {
        "schema": RESCUE_POLICY_AUDIT_SCHEMA,
        **metadata,
        "trial_index": int(trial_index),
        "random_seed": int(trials[(state_id, attempted[0], trial_index)]["random_seed"]),
        "policy_id": policy_id,
        "reference_only": bool(reference_only),
        "state_change_assumption": assumption,
        "state_change_evidence": evidence,
        "state_escaped": int(changed),
        "exhausted_without_change": int(not changed),
        "final_hard_failure": int(terminal_hard_failure),
        "conflict_reduction": reductions,
        "repair_seconds": total_seconds,
        "pp_replan_seconds": total_pp_seconds,
        "generated": total_generated,
        "expanded": total_expanded,
        "reopened": total_reopened,
        "attempt_count": len(attempted),
        "attempted_candidate_ids": "|".join(attempted),
        "terminal_candidate_id": terminal_candidate,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty policy result")
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(row)
    total_seconds = math.fsum(float(row["repair_seconds"]) for row in rows)
    total_reduction = math.fsum(float(row["conflict_reduction"]) for row in rows)
    return {
        "state_count": len(by_state),
        "trial_count": len(rows),
        "state_escape_rate": statistics.fmean(float(row["state_escaped"]) for row in rows),
        "expected_escape_state_count": math.fsum(
            statistics.fmean(float(row["state_escaped"]) for row in state_rows)
            for state_rows in by_state.values()
        ),
        "final_hard_failure_rate": statistics.fmean(
            float(row["final_hard_failure"]) for row in rows
        ),
        "expected_hard_failure_state_count": math.fsum(
            statistics.fmean(float(row["final_hard_failure"]) for row in state_rows)
            for state_rows in by_state.values()
        ),
        "unchanged_exhaustion_rate": statistics.fmean(
            float(row["exhausted_without_change"]) for row in rows
        ),
        "mean_conflict_reduction": statistics.fmean(
            float(row["conflict_reduction"]) for row in rows
        ),
        "mean_repair_seconds": statistics.fmean(
            float(row["repair_seconds"]) for row in rows
        ),
        "conflict_reduction_per_second": total_reduction / max(1e-12, total_seconds),
        "mean_attempt_count": statistics.fmean(float(row["attempt_count"]) for row in rows),
        "mean_generated": statistics.fmean(float(row["generated"]) for row in rows),
        "mean_expanded": statistics.fmean(float(row["expanded"]) for row in rows),
        "mean_reopened": statistics.fmean(float(row["reopened"]) for row in rows),
    }


def _ratio(left: float, right: float) -> float:
    if abs(right) <= 1e-12:
        return 1.0 if abs(left) <= 1e-12 else math.inf
    return left / right


def _comparison(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_escape_rate_delta": float(metrics["state_escape_rate"])
        - float(baseline["state_escape_rate"]),
        "final_hard_failure_rate_delta": float(metrics["final_hard_failure_rate"])
        - float(baseline["final_hard_failure_rate"]),
        "efficiency_ratio": _ratio(
            float(metrics["conflict_reduction_per_second"]),
            float(baseline["conflict_reduction_per_second"]),
        ),
        "mean_reduction_ratio": _ratio(
            float(metrics["mean_conflict_reduction"]),
            float(baseline["mean_conflict_reduction"]),
        ),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    selected = report.get("selected_policy_id") or "none"
    diagnostic = report["diagnostic_gate"]
    lines = [
        "# 4/8/16 offline rescue policy audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Selected fixed policy: `{selected}`",
        f"- Training states: {report['training_state_count']}",
        f"- Diagnostic states: {report['diagnostic_state_count']}",
        f"- Candidate fixed policies: {report['candidate_policy_count']}",
        "- Solver executions started by this audit: 0",
        "- Size 12 runtime status: excluded (exploratory data retained)",
        "",
        "## State-change evidence limitation",
        "",
        "The v1 pilot trials contain the before-state fingerprints but do not store",
        "an after-state fingerprint. The audit therefore evaluates every rule under",
        "both `replan-success-stop` and `conflict-reduction-stop`. Promotion requires",
        "all gates to pass under both interpretations; no exact state change is invented.",
        "",
        "## Decision gates",
        "",
        f"- OOF gate passed: {str(report['oof_gate_passed']).lower()}",
        f"- Diagnostic gate passed: {str(diagnostic['passed']).lower()}",
        f"- OOF-eligible fixed policies: {', '.join(report['oof_eligible_policy_ids']) or 'none'}",
    ]
    if report.get("selected_policy_metrics"):
        lines.extend(["", "## Selected policy metrics", ""])
        for row in report["selected_policy_metrics"]:
            lines.append(
                "- {assumption}: OOF efficiency ratio {eff:.3f}, escape delta "
                "{escape:+.3f}, hard-failure delta {failure:+.3f}, reduction ratio "
                "{reduction:.3f}, non-inferior folds {folds}/4.".format(
                    assumption=row["state_change_assumption"],
                    eff=row["efficiency_ratio"],
                    escape=row["state_escape_rate_delta"],
                    failure=row["final_hard_failure_rate_delta"],
                    reduction=row["mean_reduction_ratio"],
                    folds=row["noninferior_fold_count"],
                )
            )
    lines.extend(
        [
            "",
            "## Reference-only policies",
            "",
            "Size 12 and the learned repair-aware selector are reported for context but",
            "cannot be promoted by this audit. In particular, the learned selector's",
            "policy-train result is in-sample because that bundle was fitted on the same",
            "48 states; only its diagnostic result is informative about transfer.",
        ]
    )
    lines.extend(
        [
            "",
            "## Next action",
            "",
            (
                "Collect a fresh, small independent confirmation set before implementing "
                "`v2-rescue-lite`; do not use the exposed 12-state diagnostic split as a "
                "new locked validation set."
                if report["decision"] == "rescue_lite_candidate"
                else "Stop tuning rescue order on this pilot and prepare a separate v3 "
                "high-load main-ranker data plan. No v3 job was started."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def audit_rescue_policies(*, source: str | Path, output: str | Path) -> dict[str, Any]:
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    if source_root == output_root:
        raise ValueError("audit output must differ from its source")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("audit output already exists and is nonempty")

    collection_root = source_root / "collection"
    feature_path = collection_root / "feature_index.jsonl"
    trial_path = collection_root / "trial_manifest.jsonl"
    collection_report = dict(read_json(collection_root / "collection_report.json"))
    pilot_final = dict(read_json(source_root / "size12_pilot_final_report.json"))
    if not bool(collection_report.get("complete")) or int(
        collection_report.get("error_state_count", -1)
    ) != 0:
        raise ValueError("high-load pilot collection is incomplete")
    feature_rows = _read_jsonl(feature_path)
    trial_rows = _read_jsonl(trial_path)
    indexed = _index_collection(feature_rows, trial_rows)
    metadata = indexed["state_metadata"]
    expected_states = set(metadata)
    _validate_state_fingerprints(source_root, expected_states)
    actual_split_counts = dict(
        collections.Counter(str(row["split"]) for row in metadata.values())
    )
    expected_split_counts = {
        str(key): int(value)
        for key, value in dict(collection_report["state_count_by_split"]).items()
    }
    if actual_split_counts != expected_split_counts:
        raise ValueError("state split counts differ from the collection report")

    rows_by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for (state_id, _candidate_id), row in indexed["features"].items():
        rows_by_state[state_id].append(row)
    adaptive_by_state: dict[str, str] = {}
    for state_id, rows in rows_by_state.items():
        for size in AUDITED_SIZES:
            _highest_v2_candidate(rows, size)
        adaptive = [row for row in rows if str(row.get("route")) == "official_adaptive"]
        if len(adaptive) != 1:
            raise ValueError("each state must contain exactly one Adaptive arm")
        adaptive_by_state[state_id] = str(adaptive[0]["candidate_id"])

    policies = enumerate_rescue_policies()
    references = [
        RescuePolicy("reference_12>adaptive", (12,), True),
        RescuePolicy("reference_learned_repair_aware", (), True),
    ]
    controller_root = source_root / "controller"
    learned_bundle = load_repair_aware_bundle(controller_root)
    learned_by_state = {
        state_id: _learned_reference_choice(rows, learned_bundle)
        for state_id, rows in rows_by_state.items()
    }

    all_results: list[dict[str, Any]] = []
    for state_id in sorted(metadata):
        for policy in [*policies, *references]:
            if policy.policy_id == "reference_learned_repair_aware":
                candidate = learned_by_state[state_id]
                candidate_sequence = [] if candidate is None else [candidate]
            else:
                candidate_sequence = [
                    str(
                        _highest_v2_candidate(rows_by_state[state_id], size)[
                            "candidate_id"
                        ]
                    )
                    for size in policy.size_order
                ]
            for trial_index in indexed["trial_indices"][state_id]:
                for assumption in STATE_CHANGE_ASSUMPTIONS:
                    all_results.append(
                        _simulate_sequence(
                            metadata=metadata[state_id],
                            trial_index=trial_index,
                            policy_id=policy.policy_id,
                            candidate_sequence=candidate_sequence,
                            adaptive_id=adaptive_by_state[state_id],
                            trials=indexed["trials"],
                            assumption=assumption,
                            reference_only=policy.reference_only,
                        )
                    )

    grouped_results: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in all_results:
        grouped_results[
            (
                str(row["split"]),
                str(row["state_change_assumption"]),
                str(row["policy_id"]),
            )
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped_results.items()):
        split, assumption, policy_id = key
        metrics = _aggregate(rows)
        baseline = _aggregate(grouped_results[(split, assumption, "adaptive")])
        summary_rows.append(
            {
                "schema": RESCUE_POLICY_AUDIT_SCHEMA,
                "split": split,
                "state_change_assumption": assumption,
                "policy_id": policy_id,
                "reference_only": bool(rows[0]["reference_only"]),
                **metrics,
                **_comparison(metrics, baseline),
            }
        )
    summary_index = {
        (str(row["split"]), str(row["state_change_assumption"]), str(row["policy_id"])): row
        for row in summary_rows
    }

    folds = _balanced_map_folds(metadata)
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        validation_maps = set(map(str, fold["validation_maps"]))
        for assumption in STATE_CHANGE_ASSUMPTIONS:
            for policy in policies:
                rows = [
                    row
                    for row in grouped_results[
                        ("policy_train", assumption, policy.policy_id)
                    ]
                    if str(row["map_id"]) in validation_maps
                ]
                baseline_rows = [
                    row
                    for row in grouped_results[("policy_train", assumption, "adaptive")]
                    if str(row["map_id"]) in validation_maps
                ]
                metrics = _aggregate(rows)
                baseline = _aggregate(baseline_rows)
                comparison = _comparison(metrics, baseline)
                fold_rows.append(
                    {
                        "schema": RESCUE_POLICY_AUDIT_SCHEMA,
                        "fold": int(fold["fold"]),
                        "validation_maps": "|".join(map(str, fold["validation_maps"])),
                        "state_change_assumption": assumption,
                        "policy_id": policy.policy_id,
                        **metrics,
                        **comparison,
                        "efficiency_noninferior": bool(
                            comparison["efficiency_ratio"] + 1e-12 >= 1.0
                        ),
                    }
                )

    eligible_rows: list[dict[str, Any]] = []
    selected_metric_rows: list[dict[str, Any]] = []
    for policy in policies[1:]:
        assumption_rows = []
        passed = True
        for assumption in STATE_CHANGE_ASSUMPTIONS:
            row = summary_index[("policy_train", assumption, policy.policy_id)]
            noninferior_folds = sum(
                int(fold["efficiency_noninferior"])
                for fold in fold_rows
                if str(fold["policy_id"]) == policy.policy_id
                and str(fold["state_change_assumption"]) == assumption
            )
            gate = bool(
                float(row["state_escape_rate_delta"]) >= -1e-12
                and float(row["final_hard_failure_rate_delta"]) <= 1e-12
                and float(row["efficiency_ratio"]) + 1e-12 >= 1.10
                and float(row["mean_reduction_ratio"]) + 1e-12 >= 0.98
                and noninferior_folds >= 3
            )
            assumption_rows.append(
                {
                    "policy_id": policy.policy_id,
                    "state_change_assumption": assumption,
                    "noninferior_fold_count": noninferior_folds,
                    "oof_assumption_passed": gate,
                    **{
                        key: row[key]
                        for key in (
                            "state_escape_rate_delta",
                            "final_hard_failure_rate_delta",
                            "efficiency_ratio",
                            "mean_reduction_ratio",
                        )
                    },
                }
            )
            passed = passed and gate
        if passed:
            eligible_rows.append(
                {
                    "policy": policy,
                    "metrics": assumption_rows,
                    "minimum_efficiency_ratio": min(
                        float(row["efficiency_ratio"]) for row in assumption_rows
                    ),
                    "mean_attempt_count": statistics.fmean(
                        float(
                            summary_index[
                                ("policy_train", assumption, policy.policy_id)
                            ]["mean_attempt_count"]
                        )
                        for assumption in STATE_CHANGE_ASSUMPTIONS
                    ),
                }
            )
    selected: dict[str, Any] | None = None
    if eligible_rows:
        selected = min(
            eligible_rows,
            key=lambda row: (
                -round(float(row["minimum_efficiency_ratio"]), 12),
                round(float(row["mean_attempt_count"]), 12),
                str(row["policy"].policy_id),
            ),
        )
        selected_metric_rows = list(selected["metrics"])

    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_passed = selected is not None
    if selected is not None:
        policy_id = str(selected["policy"].policy_id)
        for assumption in STATE_CHANGE_ASSUMPTIONS:
            metrics = summary_index[("policy_validation", assumption, policy_id)]
            baseline = summary_index[("policy_validation", assumption, "adaptive")]
            escape_count_delta = float(metrics["expected_escape_state_count"]) - float(
                baseline["expected_escape_state_count"]
            )
            hard_failure_count_delta = float(
                metrics["expected_hard_failure_state_count"]
            ) - float(baseline["expected_hard_failure_state_count"])
            gate = bool(
                escape_count_delta >= -1.0 - 1e-12
                and hard_failure_count_delta <= 1.0 + 1e-12
                and float(metrics["conflict_reduction_per_second"]) + 1e-12
                >= float(baseline["conflict_reduction_per_second"])
                and float(metrics["mean_conflict_reduction"]) + 1e-12
                >= 0.90 * float(baseline["mean_conflict_reduction"])
            )
            diagnostic_rows.append(
                {
                    "schema": RESCUE_POLICY_AUDIT_SCHEMA,
                    "selected_policy_id": policy_id,
                    "state_change_assumption": assumption,
                    "expected_escape_state_count_delta": escape_count_delta,
                    "expected_hard_failure_state_count_delta": hard_failure_count_delta,
                    "efficiency_ratio": float(metrics["efficiency_ratio"]),
                    "mean_reduction_ratio": float(metrics["mean_reduction_ratio"]),
                    "passed": gate,
                }
            )
            diagnostic_passed = diagnostic_passed and gate
    else:
        diagnostic_rows.append(
            {
                "schema": RESCUE_POLICY_AUDIT_SCHEMA,
                "selected_policy_id": "",
                "state_change_assumption": "not-evaluated",
                "expected_escape_state_count_delta": "",
                "expected_hard_failure_state_count_delta": "",
                "efficiency_ratio": "",
                "mean_reduction_ratio": "",
                "passed": False,
            }
        )

    decision = (
        "rescue_lite_candidate"
        if selected is not None and diagnostic_passed
        else "proceed_to_v3"
    )
    report = {
        "schema": RESCUE_POLICY_AUDIT_SCHEMA,
        "source": str(source_root),
        "source_fingerprint": _fingerprint(
            {
                "feature_index_sha256": sha256_file(feature_path),
                "trial_manifest_sha256": sha256_file(trial_path),
                "controller_source_fingerprint": learned_bundle.manifest.get(
                    "source_fingerprint"
                ),
            }
        ),
        "training_state_count": actual_split_counts["policy_train"],
        "diagnostic_state_count": actual_split_counts["policy_validation"],
        "candidate_policy_count": len(policies),
        "reference_policy_ids": [policy.policy_id for policy in references],
        "reference_policy_interpretation": {
            "reference_12>adaptive": "exploratory_only_not_runtime_eligible",
            "reference_learned_repair_aware": (
                "training_split_is_in_sample_diagnostic_split_is_transfer_only"
            ),
        },
        "state_change_exactly_observed": any(
            "state_changed" in dict(row["outcome"])
            or (
                "before_fingerprint" in dict(row["outcome"])
                and "after_fingerprint" in dict(row["outcome"])
            )
            for row in trial_rows
        ),
        "state_change_assumptions": list(STATE_CHANGE_ASSUMPTIONS),
        "size12_runtime_status": "excluded_experimental_data_retained",
        "size12_empirical_efficiency_change_fraction": float(
            pilot_final["size12_vs_best_4_8_16"]["efficiency_change_fraction"]
        ),
        "size12_winner_state_count": int(
            pilot_final["pilot_gate"]["size12_selected_state_count"]
        ),
        "size12_deployment_promoted": bool(
            pilot_final.get("deployment_promoted", False)
        ),
        "oof_gate_passed": selected is not None,
        "oof_eligible_policy_ids": sorted(
            str(row["policy"].policy_id) for row in eligible_rows
        ),
        "selected_policy_id": None if selected is None else selected["policy"].policy_id,
        "selected_policy_metrics": selected_metric_rows,
        "diagnostic_gate": {
            "passed": diagnostic_passed,
            "rows": diagnostic_rows,
            "validation_status": "diagnostic_previously_exposed_not_locked",
        },
        "decision": decision,
        "long_jobs_started": False,
        "default_controller_changed": False,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(output_root / "rescue_policy_trials.csv", all_results)
    _atomic_write_csv(output_root / "rescue_policy_summary.csv", summary_rows)
    _atomic_write_csv(output_root / "map_fold_results.csv", fold_rows)
    _atomic_write_csv(output_root / "diagnostic_validation.csv", diagnostic_rows)
    _write_json(output_root / "rescue_policy_audit_report.json", report)
    (output_root / "rescue_policy_audit_report.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    return report


__all__ = [
    "AUDITED_SIZES",
    "RESCUE_POLICY_AUDIT_SCHEMA",
    "STATE_CHANGE_ASSUMPTIONS",
    "RescuePolicy",
    "_index_collection",
    "_simulate_sequence",
    "_validate_state_fingerprints",
    "audit_rescue_policies",
    "enumerate_rescue_policies",
]
