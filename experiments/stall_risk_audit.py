from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, read_json, sha256_file, write_json
from experiments.closed_loop_confirmation import (
    generate_online_candidates,
)
from experiments.compact_controller_model import load_controller_bundle
from experiments.lns2_bottleneck import validate_manifest_trace
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _read_jsonl,
    state_fingerprint,
)
from experiments.run_output_guard import prepare_run_output
from experiments.stall_guard import repair_structure_fingerprint
from experiments.stall_oracle import STALL_ORACLE_SCHEMA
from experiments.stalled_state_probe import (
    STALLED_STATE_PROBE_SCHEMA,
    _candidate_pool_signature,
)
from experiments.trace_replay import decision_rows, replay_prefix
from experiments.v3_controller import load_v3_controller_bundle


STALL_RISK_AUDIT_SCHEMA = "lns2.stall_risk_audit.v1"
STALL_RISK_AUDIT_VERSION = 1
LOCAL_AUC_FLOOR = 0.65
STABLE_ALTERNATIVE_RISK_MARGIN = 0.05
TOP_K = 3


def _finite_probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be a finite probability")
    return number


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and values[order[end]] == values[order[offset]]:
            end += 1
        average = (offset + 1 + end) / 2.0
        for index in order[offset:end]:
            ranks[index] = average
        offset = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def _binary_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("AUC labels and scores have different lengths")
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def evaluate_stall_risk_predictions(
    predictions: Iterable[dict[str, Any]],
    oracle_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare frozen candidate-level risk predictions with paired PP outcomes.

    This is deliberately a local diagnostic.  A positive result can justify a
    wider shadow audit, but can never promote a controller by itself.
    """

    prediction_rows = [dict(row) for row in predictions]
    if not prediction_rows:
        raise ValueError("stall-risk audit received no candidate predictions")
    by_candidate: dict[str, dict[str, Any]] = {}
    for row in prediction_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in by_candidate:
            raise ValueError("stall-risk predictions have missing or duplicate ids")
        normalized = dict(row)
        normalized["predicted_no_progress_probability"] = _finite_probability(
            row.get("predicted_no_progress_probability"),
            field="predicted no-progress probability",
        )
        by_candidate[candidate_id] = normalized

    branches = [
        dict(row)
        for row in oracle_report.get("branches", ())
        if str(row.get("branch_mode")) == "explicit_neighborhood"
    ]
    if not branches:
        raise ValueError("stall-risk Oracle has no explicit candidate branches")
    oracle_ids = {str(row.get("candidate_id") or "") for row in branches}
    if not all(oracle_ids) or oracle_ids != set(by_candidate):
        raise ValueError("stall-risk predictions do not cover the Oracle pool")

    merged: list[dict[str, Any]] = []
    labels: list[int] = []
    scores: list[float] = []
    for branch in branches:
        candidate_id = str(branch["candidate_id"])
        prediction = by_candidate[candidate_id]
        trial_count = int(branch["trial_count"])
        escape_count = int(branch["escape_count"])
        if trial_count <= 0 or not 0 <= escape_count <= trial_count:
            raise ValueError("stall-risk Oracle branch counts are invalid")
        no_progress_count = trial_count - escape_count
        no_progress_fraction = no_progress_count / trial_count
        risk = float(prediction["predicted_no_progress_probability"])
        labels.extend([1] * no_progress_count)
        labels.extend([0] * escape_count)
        scores.extend([risk] * trial_count)
        merged.append(
            {
                **prediction,
                "candidate_rank": int(branch["candidate_rank"]),
                "candidate_size": int(branch["candidate_size"]),
                "trial_count": trial_count,
                "escape_count": escape_count,
                "escape_fraction": float(branch["escape_fraction"]),
                "actual_no_progress_fraction": no_progress_fraction,
                "stable_escape": bool(branch["stable_escape"]),
                "stable_failure": bool(branch["stable_failure"]),
                "pp_order_sensitive": bool(branch["pp_order_sensitive"]),
            }
        )

    risk_order = sorted(
        merged,
        key=lambda row: (
            float(row["predicted_no_progress_probability"]),
            int(row["candidate_rank"]),
            str(row["candidate_id"]),
        ),
    )
    for rank, row in enumerate(risk_order, 1):
        row["predicted_risk_rank"] = rank

    selected = [row for row in merged if int(row["candidate_rank"]) == 1]
    if len(selected) != 1:
        raise ValueError("stall-risk Oracle must identify one frozen v2 winner")
    selected_row = selected[0]
    stable_alternatives = [
        row
        for row in merged
        if bool(row["stable_escape"]) and int(row["candidate_rank"]) != 1
    ]
    stable_alternatives.sort(
        key=lambda row: (
            float(row["predicted_no_progress_probability"]),
            int(row["candidate_rank"]),
        )
    )
    selected_risk = float(selected_row["predicted_no_progress_probability"])
    minimum_stable_risk = (
        float(stable_alternatives[0]["predicted_no_progress_probability"])
        if stable_alternatives
        else None
    )
    risk_margin = (
        selected_risk - minimum_stable_risk
        if minimum_stable_risk is not None
        else None
    )
    auc = _binary_auc(labels, scores)
    actual_fractions = [float(row["actual_no_progress_fraction"]) for row in merged]
    predicted_risks = [
        float(row["predicted_no_progress_probability"]) for row in merged
    ]
    spearman = _pearson(
        _average_ranks(predicted_risks),
        _average_ranks(actual_fractions),
    )
    brier = statistics.fmean(
        (
            float(row["predicted_no_progress_probability"])
            - float(row["actual_no_progress_fraction"])
        )
        ** 2
        for row in merged
    )

    checks = {
        "oracle_is_selector_failure": bool(oracle_report.get("selector_failure")),
        "v2_winner_is_stable_failure": bool(selected_row["stable_failure"]),
        "stable_alternative_exists": bool(stable_alternatives),
        "stable_alternative_risk_margin_0_05": bool(
            risk_margin is not None
            and risk_margin >= STABLE_ALTERNATIVE_RISK_MARGIN - 1e-12
        ),
        "top3_contains_stable_alternative": any(
            bool(row["stable_escape"]) for row in risk_order[:TOP_K]
        ),
        "trial_no_progress_auc_at_least_0_65": bool(
            auc is not None and auc >= LOCAL_AUC_FLOOR - 1e-12
        ),
    }
    local_signal_supported = all(checks.values())
    return {
        "schema": STALL_RISK_AUDIT_SCHEMA,
        "schema_version": STALL_RISK_AUDIT_VERSION,
        "decision": (
            "local_stall_risk_signal_supported"
            if local_signal_supported
            else "frozen_v3_stall_risk_signal_not_supported"
        ),
        "local_signal_supported": local_signal_supported,
        "deployment_promoted": False,
        "training_started": False,
        "candidate_count": len(merged),
        "trial_count": len(labels),
        "trial_no_progress_auc": auc,
        "candidate_no_progress_spearman": spearman,
        "candidate_brier_score": brier,
        "selected_v2": {
            "candidate_id": str(selected_row["candidate_id"]),
            "candidate_rank": int(selected_row["candidate_rank"]),
            "candidate_size": int(selected_row["candidate_size"]),
            "predicted_no_progress_probability": selected_risk,
            "predicted_risk_rank": int(selected_row["predicted_risk_rank"]),
            "escape_fraction": float(selected_row["escape_fraction"]),
        },
        "best_stable_alternative": (
            {
                "candidate_id": str(stable_alternatives[0]["candidate_id"]),
                "candidate_rank": int(stable_alternatives[0]["candidate_rank"]),
                "candidate_size": int(stable_alternatives[0]["candidate_size"]),
                "predicted_no_progress_probability": minimum_stable_risk,
                "predicted_risk_rank": int(
                    stable_alternatives[0]["predicted_risk_rank"]
                ),
                "escape_fraction": float(stable_alternatives[0]["escape_fraction"]),
            }
            if stable_alternatives
            else None
        ),
        "selected_vs_best_stable_risk_margin": risk_margin,
        "checks": checks,
        "candidates": sorted(merged, key=lambda row: int(row["candidate_rank"])),
        "evidence_level": "single-state exact paired Oracle diagnostic",
        "next_step": (
            "audit additional exact stalled and natural-recovery states in shadow"
            if local_signal_supported
            else "do not reuse the frozen v3 head; redesign stall-specific labels"
        ),
    }


def _portable_native_parity(
    bundle: Any, rows: list[dict[str, Any]]
) -> tuple[dict[str, list[float]], float]:
    native = bundle.predict(rows)
    saved = {
        name: model.native_predictor for name, model in bundle.models.items()
    }
    try:
        for model in bundle.models.values():
            model.native_predictor = None
        portable = bundle.predict(rows)
    finally:
        for name, predictor in saved.items():
            bundle.models[name].native_predictor = predictor
    maximum = max(
        abs(float(a) - float(b))
        for name in native
        for a, b in zip(native[name], portable[name])
    )
    return native, maximum


def _markdown(report: dict[str, Any]) -> str:
    selected = dict(report["selected_v2"])
    alternative = report.get("best_stable_alternative")
    lines = [
        "# Frozen v3 stall-risk audit",
        "",
        f"- Decision: `{report['decision']}`.",
        f"- Evidence: {report['evidence_level']}.",
        f"- Candidates / paired trials: {report['candidate_count']} / {report['trial_count']}.",
        f"- Trial no-progress AUC: {report['trial_no_progress_auc']:.6f}.",
        f"- Candidate Spearman: {report['candidate_no_progress_spearman']:.6f}.",
        f"- Native / Python maximum prediction delta: {report['portable_native_maximum_delta']:.3e}.",
        "",
        "## Frozen v2 winner",
        "",
        f"- Candidate: `{selected['candidate_id']}` (v2 rank {selected['candidate_rank']}, size {selected['candidate_size']}).",
        f"- Predicted no-progress probability: {selected['predicted_no_progress_probability']:.6f}.",
        f"- Actual paired escape fraction: {selected['escape_fraction']:.3f}.",
    ]
    if isinstance(alternative, dict):
        lines.extend(
            [
                "",
                "## Lowest-risk stable alternative",
                "",
                f"- Candidate: `{alternative['candidate_id']}` (v2 rank {alternative['candidate_rank']}, size {alternative['candidate_size']}).",
                f"- Predicted no-progress probability: {alternative['predicted_no_progress_probability']:.6f}.",
                f"- Actual paired escape fraction: {alternative['escape_fraction']:.3f}.",
                f"- Risk margin versus v2: {report['selected_vs_best_stable_risk_margin']:.6f}.",
            ]
        )
    lines.extend(
        [
            "",
            "This audit does not modify v2, train a model, or promote a controller.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_frozen_v3_stall_risk(
    probe: str | Path,
    oracle: str | Path,
    v3_controller: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    probe_root = Path(probe).resolve()
    oracle_path = Path(oracle).resolve()
    v3_root = Path(v3_controller).resolve()
    output_root = Path(output).resolve()
    probe_report_path = probe_root / "stalled_state_probe_report.json"
    probe_runner_path = probe_root / "runner_config.json"
    probe_report = dict(read_json(probe_report_path))
    probe_runner = dict(read_json(probe_runner_path))
    oracle_report = dict(read_json(oracle_path))
    if str(probe_report.get("schema")) != STALLED_STATE_PROBE_SCHEMA:
        raise ValueError("stall-risk probe schema is unsupported")
    if str(oracle_report.get("schema")) != STALL_ORACLE_SCHEMA:
        raise ValueError("stall-risk Oracle schema is unsupported")
    if str(oracle_report.get("source_probe_report_sha256")) != sha256_file(
        probe_report_path
    ):
        raise ValueError("stall-risk Oracle does not bind the selected probe")
    if str(oracle_report.get("source_probe_runner_sha256")) != sha256_file(
        probe_runner_path
    ):
        raise ValueError("stall-risk Oracle does not bind the probe identity")

    v3_bundle = load_v3_controller_bundle(v3_root)
    v3_manifest_path = v3_root / "v3_manifest.json"
    identity = {
        "schema": STALL_RISK_AUDIT_SCHEMA,
        "schema_version": STALL_RISK_AUDIT_VERSION,
        "probe_report_sha256": sha256_file(probe_report_path),
        "probe_runner_sha256": sha256_file(probe_runner_path),
        "oracle_report_sha256": sha256_file(oracle_path),
        "v3_manifest_sha256": sha256_file(v3_manifest_path),
        "candidate_pool_fingerprint": str(
            probe_report["candidate_pool_fingerprint"]
        ),
        "before_repair_fingerprint": str(
            probe_report["before_repair_fingerprint"]
        ),
    }
    runner = prepare_run_output(output_root, resume=resume, identity=identity)

    source_root = Path(str(probe_report["source_collection"])).resolve()
    source_run = dict(read_json(source_root / "run_config.json"))
    configuration = dict(source_run["configuration"])
    manifests = [
        dict(row)
        for row in _read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
        if str(row.get("task_id")) == str(probe_report["task_id"])
        and int(row.get("solver_seed", -1)) == int(probe_report["solver_seed"])
        and str(row.get("status")) in {"ok", "resumed"}
    ]
    if len(manifests) != 1:
        raise ValueError("stall-risk source episode is missing or ambiguous")
    manifest = manifests[0]
    validate_manifest_trace(
        source_root,
        manifest,
        run_fingerprint=str(source_run["run_fingerprint"]),
        expected_policy="realized_dynamic",
    )
    decisions, _events = decision_rows(source_root, manifest)
    targets = [
        dict(row)
        for row in decisions
        if int(row["decision_index"]) == int(probe_report["decision_index"])
    ]
    if len(targets) != 1:
        raise ValueError("stall-risk target decision is missing")
    target = targets[0]

    dataset_root = Path(str(source_run["dataset"])).resolve()
    dataset_rows = {
        str(row["task_id"]): row
        for row in _load_dataset_rows(dataset_root, [str(configuration["split"])])
    }
    task_id = str(probe_report["task_id"])
    if task_id not in dataset_rows:
        raise ValueError("stall-risk task is absent from the source dataset")
    job = {
        "dataset_root": str(dataset_root),
        "row": dataset_rows[task_id],
        "environment": dict(configuration["environment"]),
        "solver_seed": int(probe_report["solver_seed"]),
    }
    environment, state = replay_prefix(job, target["prefix_actions"])
    if state_fingerprint(state) != str(probe_report["before_fingerprint"]):
        raise RuntimeError("stall-risk full-state replay fingerprint mismatch")
    if repair_structure_fingerprint(state) != str(
        probe_report["before_repair_fingerprint"]
    ):
        raise RuntimeError("stall-risk repair-state replay fingerprint mismatch")

    controller_root = Path(str(configuration["controller_bundle"])).resolve()
    controller_bundle = load_controller_bundle(controller_root)
    source_controller_manifest = source_run.get("controller_bundle")
    if not isinstance(source_controller_manifest, dict) or _fingerprint(
        controller_bundle.manifest
    ) != _fingerprint(source_controller_manifest):
        raise ValueError("stall-risk source v2 bundle has changed")

    before_full = state_fingerprint(state)
    optimized = str(configuration.get("controller_runtime", "reference")) == "optimized"
    candidates, _proposal = generate_online_candidates(
        environment,
        state,
        task_id=task_id,
        solver_seed=int(probe_report["solver_seed"]),
        decision_index=int(probe_report["decision_index"]),
        proposal_config=dict(configuration["proposal"]),
        state_hash=before_full,
        verify_full_state=True,
        proposal_backend="optimized" if optimized else "reference",
        shadow_validation=False,
    )
    saved_pool = [dict(row) for row in probe_report["candidate_pool"]]
    saved_scores = {
        str(row["candidate_id"]): float(row["score"]) for row in saved_pool
    }
    if set(saved_scores) != {str(row["candidate_id"]) for row in candidates}:
        raise ValueError("stall-risk regenerated candidate ids differ from the probe")
    scores = [saved_scores[str(row["candidate_id"])] for row in candidates]
    signature = _candidate_pool_signature(candidates, scores)
    if _fingerprint(signature) != str(probe_report["candidate_pool_fingerprint"]):
        raise ValueError("stall-risk regenerated candidate pool differs from the probe")

    required_features = set(v3_bundle.required_feature_names)
    engine = OnlineFeatureEngine(
        state,
        backend=str(configuration.get("feature_backend", "auto")),
        shadow_validation=False,
        required_features={"realized_dynamic": required_features},
        dense_output=optimized,
    )
    feature_rows, _feature_metrics = engine.realized_rows(
        candidates, state_hash=before_full
    )
    predicted, parity_delta = _portable_native_parity(v3_bundle, feature_rows)
    candidate_predictions = []
    for index, candidate in enumerate(candidates):
        candidate_predictions.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "v2_score": float(scores[index]),
                "predicted_no_progress_probability": float(
                    predicted["no_progress_probability"][index]
                ),
                "predicted_effective_progress_probability": float(
                    predicted["effective_progress_probability"][index]
                ),
                "predicted_conflict_reduction": float(
                    predicted["conflict_reduction"][index]
                ),
                "predicted_repair_seconds": float(
                    predicted["repair_seconds"][index]
                ),
                "predicted_utility": float(predicted["utility"][index]),
            }
        )

    report = evaluate_stall_risk_predictions(candidate_predictions, oracle_report)
    report.update(
        {
            "run_fingerprint": str(runner["identity_fingerprint"]),
            "probe": str(probe_root),
            "oracle": str(oracle_path),
            "v3_controller": str(v3_root),
            "v3_manifest_sha256": sha256_file(v3_manifest_path),
            "v3_bundle_deployment_promoted": bool(
                v3_bundle.manifest.get("deployment_promoted", False)
            ),
            "v3_inference_backends": list(v3_bundle.inference_backends),
            "portable_native_maximum_delta": parity_delta,
            "candidate_pool_fingerprint": str(
                probe_report["candidate_pool_fingerprint"]
            ),
            "before_fingerprint": str(probe_report["before_fingerprint"]),
            "before_repair_fingerprint": str(
                probe_report["before_repair_fingerprint"]
            ),
            "task_id": task_id,
            "solver_seed": int(probe_report["solver_seed"]),
            "decision_index": int(probe_report["decision_index"]),
        }
    )
    if parity_delta > 1e-12:
        raise RuntimeError("stall-risk native and Python predictions differ")
    atomic_write_csv(output_root / "stall_risk_candidates.csv", report["candidates"])
    write_json(output_root / "stall_risk_audit_report.json", report)
    (output_root / "stall_risk_audit_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return report


__all__ = [
    "STALL_RISK_AUDIT_SCHEMA",
    "audit_frozen_v3_stall_risk",
    "evaluate_stall_risk_predictions",
]
