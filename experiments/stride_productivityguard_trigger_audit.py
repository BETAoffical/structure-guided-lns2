from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from experiments._common import contained_file, registered_input, sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.stride_loopguard_trigger_audit import (
    load_registration as load_loopguard_registration,
)
from experiments.stride_tailswitch_sequence_forensics import (
    CONTINUATION_PAIRS,
    _average,
    _episode_rows,
    load_registration as load_sequence_registration,
)


CONFIG_SCHEMA = "lns2.stride.productivityguard_trigger_audit_registration.v1"
REPORT_SCHEMA = "lns2.stride.productivityguard_trigger_audit_report.v1"
STATUS_SCHEMA = "lns2.stride.productivityguard_trigger_audit_status.v1"
EXPERIMENT_ID = "stride-productivityguard-trigger-audit-v1"
SCIENTIFIC_STATUS = (
    "preregistered_existing_trajectory_historical_productivity_trigger_audit"
)
PARENT_COMMIT = "ac07eccc8735cf87d3db434432ce79e111a9394f"


def productivity_trigger_events(
    rows: list[dict[str, Any]], variant: dict[str, Any]
) -> list[dict[str, float | int]]:
    events: list[dict[str, float | int]] = []
    for position, row in enumerate(rows):
        if int(row["decision_index"]) < 1 or position < 2:
            continue
        run = rows[position - 2 : position + 1]
        if any(value["selected_kind"] != "structural" for value in run):
            continue
        if len({str(value["selected_candidate_id"]) for value in run}) != 1:
            continue
        history = run[:2]
        start_conflicts = int(history[0]["conflicts_before"])
        current_conflicts = int(row["conflicts_before"])
        cumulative_progress = (
            (start_conflicts - current_conflicts) / max(start_conflicts, 1)
        )
        mean_unresolved = _average(
            float(value["unresolved_edge_fraction"]) for value in history
        )
        if cumulative_progress > float(
            variant["maximum_prior_cumulative_conflict_progress"]
        ):
            continue
        if "minimum_prior_mean_unresolved_edge_fraction" in variant and (
            mean_unresolved
            < float(variant["minimum_prior_mean_unresolved_edge_fraction"])
        ):
            continue
        events.append(
            {
                "decision_index": int(row["decision_index"]),
                "prior_cumulative_conflict_progress": cumulative_progress,
                "prior_mean_unresolved_edge_fraction": mean_unresolved,
            }
        )
    return events


def evaluate_variant(
    pair_rows: list[dict[str, Any]],
    variant: dict[str, Any],
    gates: dict[str, Any],
    *,
    early_window: int,
) -> dict[str, Any]:
    evaluated = []
    for row in pair_rows:
        events = productivity_trigger_events(list(row["transitions"]), variant)
        early = [
            event
            for event in events
            if int(event["decision_index"]) <= early_window
        ]
        first = early[0] if early else None
        evaluated.append(
            {
                **{key: value for key, value in row.items() if key != "transitions"},
                "early_triggered": bool(early),
                "first_early_trigger_decision": (
                    int(first["decision_index"]) if first else None
                ),
                "first_early_prior_cumulative_conflict_progress": (
                    float(first["prior_cumulative_conflict_progress"])
                    if first
                    else None
                ),
                "first_early_prior_mean_unresolved_edge_fraction": (
                    float(first["prior_mean_unresolved_edge_fraction"])
                    if first
                    else None
                ),
                "full_episode_triggered": bool(events),
                "first_full_episode_trigger_decision": (
                    int(events[0]["decision_index"]) if events else None
                ),
                "full_episode_trigger_count": len(events),
            }
        )
    by_classification = {}
    for classification in ("adverse", "beneficial", "neutral"):
        rows = [
            row for row in evaluated if row["classification"] == classification
        ]
        early_triggered = [row for row in rows if row["early_triggered"]]
        full_triggered = [row for row in rows if row["full_episode_triggered"]]
        early_decisions = [
            int(row["first_early_trigger_decision"])
            for row in early_triggered
        ]
        by_classification[classification] = {
            "pair_count": len(rows),
            "early_triggered_pair_count": len(early_triggered),
            "early_trigger_rate": (
                len(early_triggered) / len(rows) if rows else 0.0
            ),
            "mean_first_early_trigger_decision": _average(early_decisions),
            "median_first_early_trigger_decision": (
                float(statistics.median(early_decisions))
                if early_decisions
                else None
            ),
            "full_episode_triggered_pair_count": len(full_triggered),
            "full_episode_trigger_rate": (
                len(full_triggered) / len(rows) if rows else 0.0
            ),
        }
    adverse_triggered = [
        row
        for row in evaluated
        if row["classification"] == "adverse" and row["early_triggered"]
    ]
    gate_results = {
        "minimum_adverse_recall": (
            by_classification["adverse"]["early_trigger_rate"]
            >= float(gates["minimum_adverse_recall"])
        ),
        "maximum_beneficial_false_trigger_rate": (
            by_classification["beneficial"]["early_trigger_rate"]
            <= float(gates["maximum_beneficial_false_trigger_rate"])
        ),
        "maximum_neutral_false_trigger_rate": (
            by_classification["neutral"]["early_trigger_rate"]
            <= float(gates["maximum_neutral_false_trigger_rate"])
        ),
        "minimum_triggered_adverse_pair_count": (
            len(adverse_triggered)
            >= int(gates["minimum_triggered_adverse_pair_count"])
        ),
        "minimum_triggered_adverse_map_count": (
            len({row["map_id"] for row in adverse_triggered})
            >= int(gates["minimum_triggered_adverse_map_count"])
        ),
        "minimum_triggered_adverse_challenger_count": (
            len({row["challenger"] for row in adverse_triggered})
            >= int(gates["minimum_triggered_adverse_challenger_count"])
        ),
    }
    return {
        "variant": dict(variant),
        "by_classification": by_classification,
        "early_triggered_adverse_map_count": len(
            {row["map_id"] for row in adverse_triggered}
        ),
        "early_triggered_adverse_challenger_count": len(
            {row["challenger"] for row in adverse_triggered}
        ),
        "gate_results": gate_results,
        "all_registered_gates_passed": all(gate_results.values()),
        "pair_results": evaluated,
    }


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status") != SCIENTIFIC_STATUS
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit") != PARENT_COMMIT
    ):
        raise ValueError("ProductivityGuard trigger-audit registration changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    if dict(config["claim_boundary"]) != {
        "existing_trajectory_diagnostic_only": True,
        "model_training_allowed": False,
        "solver_modification_allowed": False,
        "new_solver_runs_allowed": False,
        "ttf_improvement_claim": False,
        "generalization_claim": False,
        "causal_claim": False,
        "default_controller_replacement_allowed": False,
        "no_result_based_exclusion": True,
    }:
        raise ValueError("ProductivityGuard trigger-audit claim boundary changed")
    return path, root, config, inputs


def analyze_productivityguard_trigger_audit(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, _, config, inputs = load_registration(config_path)
    loopguard_status = _read_json(inputs["loopguard_status"])
    loopguard_report = _read_json(inputs["loopguard_report"])
    if (
        loopguard_status.get("complete") is not True
        or loopguard_status.get("integrity_passed") is not True
        or loopguard_status.get("trigger_readiness_passed") is not False
        or loopguard_report.get("integrity_passed") is not True
        or loopguard_report.get("trigger_readiness_passed") is not False
        or loopguard_report.get("qualified_variant") is not None
        or dict(loopguard_report.get("classification_counts") or {})
        != {"adverse": 17, "beneficial": 11, "neutral": 104}
    ):
        raise ValueError("completed LoopGuard evidence contract changed")
    _, _, _, loopguard_inputs = load_loopguard_registration(
        inputs["loopguard_registration"]
    )
    _, _, _, sequence_inputs = load_sequence_registration(
        loopguard_inputs["sequence_registration"]
    )
    sequence_report = _read_json(loopguard_inputs["sequence_report"])
    tailswitch_report = _read_json(sequence_inputs["tailswitch_report"])
    if (
        sequence_report.get("integrity_passed") is not True
        or tailswitch_report.get("integrity_passed") is not True
        or len(tailswitch_report.get("states") or ()) != 66
    ):
        raise ValueError("upstream TailSwitch evidence contract changed")

    tailswitch_root = sequence_inputs["tailswitch_report"].parent
    expected_trace_hashes = dict(sequence_report["inputs"]["trace_sha256"])
    expected_manifest_hashes = dict(sequence_report["inputs"]["manifest_sha256"])
    pair_rows: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    trace_hashes: dict[str, str] = {}
    for state in tailswitch_report["states"]:
        state_id = str(state["state_id"])
        metadata = {
            "map_id": str(state["map_id"]),
            "task_id": str(state["task_id"]),
            "solver_seed": int(state["solver_seed"]),
            "challenger": str(state["challenger"]),
            "tail_category": str(state["tail_category"]),
        }
        state_dir = tailswitch_root / "states" / _fingerprint(
            {"state_id": state_id}
        )[:20]
        for contrast, (_, treatment_policy) in CONTINUATION_PAIRS.items():
            collection = state_dir / treatment_policy
            manifest_path = collection / "realized_dynamic_manifest.jsonl"
            manifest_rows = _read_jsonl(manifest_path)
            if len(manifest_rows) != 1 or manifest_rows[0].get("status") != "ok":
                raise ValueError(
                    f"TailSwitch manifest changed: {state_id}/{treatment_policy}"
                )
            manifest = manifest_rows[0]
            key = f"{state_id}/{treatment_policy}"
            manifest_hashes[key] = sha256_file(manifest_path)
            trace_path = contained_file(
                collection, manifest["trace_file"], field="TailSwitch trace_file"
            )
            trace_hashes[key] = sha256_file(trace_path)
            if (
                manifest_hashes[key] != expected_manifest_hashes.get(key)
                or trace_hashes[key] != expected_trace_hashes.get(key)
            ):
                raise ValueError(f"sequence input identity changed: {key}")
            transitions = _episode_rows(
                collection,
                manifest,
                state_id=state_id,
                policy=treatment_policy,
                metadata=metadata,
                include_guard_metrics=True,
            )
            outcome = dict(state["contrasts"][f"struct_{contrast}"])
            pair_rows.append(
                {
                    **metadata,
                    "state_id": state_id,
                    "contrast": contrast,
                    "treatment_policy": treatment_policy,
                    "classification": str(outcome["classification"]),
                    "normalized_auc_delta": float(outcome["normalized_auc_delta"]),
                    "final_conflict_delta": int(outcome["final_conflict_delta"]),
                    "transitions": transitions,
                }
            )

    gates = dict(config["registered_gates"])
    early_window = int(config["trigger_scope"]["early_continuation_window"])
    results = [
        evaluate_variant(pair_rows, dict(variant), gates, early_window=early_window)
        for variant in config["trigger_variants_in_frozen_priority_order"]
    ]
    qualified = next(
        (
            str(result["variant"]["id"])
            for result in results
            if result["all_registered_gates_passed"]
        ),
        None,
    )
    classification_counts = {
        value: sum(row["classification"] == value for row in pair_rows)
        for value in ("adverse", "beneficial", "neutral")
    }
    integrity_passed = (
        len(pair_rows) == 132
        and len(
            {(row["state_id"], row["contrast"]) for row in pair_rows}
        )
        == 132
        and classification_counts
        == {"adverse": 17, "beneficial": 11, "neutral": 104}
        and len(manifest_hashes) == 132
        and len(trace_hashes) == 132
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity_passed": integrity_passed,
        "state_count": 66,
        "continuation_contrast_count": len(pair_rows),
        "classification_counts": classification_counts,
        "trigger_uses_only_information_available_before_current_repair": True,
        "variant_results": results,
        "qualified_variant": qualified,
        "trigger_readiness_passed": qualified is not None,
        "interpretation": (
            "eligible_for_new_paired_runtime_guard_ablation_only"
            if qualified is not None
            else "historical_productivity_features_do_not_support_guard"
        ),
        "same_cohort_validation_notice": (
            "The TailSwitch outcomes evaluate trigger association on the same "
            "cohort. Passing does not establish causal benefit or generalization."
        ),
        "inputs": {
            "registration_sha256": sha256_file(path),
            "loopguard_registration_sha256": sha256_file(
                inputs["loopguard_registration"]
            ),
            "loopguard_status_sha256": sha256_file(inputs["loopguard_status"]),
            "loopguard_report_sha256": sha256_file(inputs["loopguard_report"]),
            "manifest_sha256": manifest_hashes,
            "trace_sha256": trace_hashes,
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "productivityguard_trigger_audit_report.json"
    _write_json(report_path, report)
    _write_json(
        output / "productivityguard_trigger_audit_status.json",
        {
            "schema": STATUS_SCHEMA,
            "complete": True,
            "integrity_passed": integrity_passed,
            "trigger_readiness_passed": report["trigger_readiness_passed"],
            "qualified_variant": qualified,
            "report_sha256": sha256_file(report_path),
        },
    )
    return report


__all__ = [
    "analyze_productivityguard_trigger_audit",
    "evaluate_variant",
    "load_registration",
    "productivity_trigger_events",
]
