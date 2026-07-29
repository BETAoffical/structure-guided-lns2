from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, sha256_file
from experiments.repair_collection import _read_json, _write_json
from experiments.run_output_guard import prepare_run_output


STALL_TRIGGER_POLICY_AUDIT_SCHEMA = "lns2.stall_trigger_policy_audit.v2"
DEFAULT_SAME_NEIGHBORHOOD_THRESHOLDS = tuple(range(3, 7))


def validate_source_threshold_coverage(
    requested: tuple[int, ...], source: tuple[int, ...]
) -> None:
    """Reject trigger thresholds whose outcomes the legacy sequence labels hide.

    Sequence qualification is anchored to the extraction threshold range.  A
    later audit above that range can incorrectly call long natural recoveries
    confirmed stalls, while an audit below the range can miss short runs that
    were never emitted.  Runtime shadow evidence has no such selection bias.
    """

    if not source or tuple(sorted(set(source))) != source:
        raise ValueError("stall sequence source thresholds are invalid")
    if not requested or requested[0] < source[0] or requested[-1] > source[-1]:
        raise ValueError(
            "requested trigger thresholds exceed the labelled sequence coverage; "
            "collect matching sequence thresholds or use runtime shadow audit"
        )


def same_neighborhood_trigger_index(
    history: Iterable[dict[str, Any]], threshold: int
) -> int | None:
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 2:
        raise ValueError("same-neighborhood threshold must be an integer >= 2")
    attempts: dict[str, set[str]] = {}
    for index, row in enumerate(history, 1):
        neighborhood = str(row.get("neighborhood_key") or "")
        attempt = str(row.get("attempt_key") or "")
        if not neighborhood or not attempt:
            raise ValueError("stall history has a missing neighborhood or attempt key")
        attempts.setdefault(neighborhood, set()).add(attempt)
        if len(attempts[neighborhood]) >= threshold:
            return index
    return None


def wilson_upper(false_count: int, sample_count: int) -> float | None:
    if sample_count <= 0:
        return None
    if false_count < 0 or false_count > sample_count:
        raise ValueError("Wilson counts are invalid")
    z = 1.959963984540054
    proportion = false_count / sample_count
    denominator = 1.0 + z * z / sample_count
    center = proportion + z * z / (2.0 * sample_count)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / sample_count
        + z * z / (4.0 * sample_count * sample_count)
    )
    return (center + spread) / denominator


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError as error:
                raise ValueError(
                    f"invalid JSONL row {line_number}: {path}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(payload)
    return rows


def _oracle_classifications(batch_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for job_root in sorted((batch_root / "jobs").iterdir()):
        probe_path = job_root / "stalled_state_probe_report.json"
        audit_path = job_root / "audit" / "stall_oracle_report.json"
        if not probe_path.is_file() or not audit_path.is_file():
            raise ValueError(f"Oracle job is incomplete: {job_root.name}")
        probe = _read_json(probe_path)
        audit = _read_json(audit_path)
        fingerprint = str(probe.get("before_repair_fingerprint") or "")
        classification = str(audit.get("classification") or "")
        if not fingerprint or not classification or fingerprint in result:
            raise ValueError("Oracle classifications are missing or duplicated")
        result[fingerprint] = classification
    return result


def _zero_failure_sample_requirement(limit: float) -> int:
    sample_count = 1
    while float(wilson_upper(0, sample_count)) > limit:
        sample_count += 1
    return sample_count


def audit_stall_trigger_policies(
    extension: str | Path,
    oracle_batch: str | Path,
    output: str | Path,
    *,
    thresholds: tuple[int, ...] = DEFAULT_SAME_NEIGHBORHOOD_THRESHOLDS,
    maximum_false_trigger_rate: float = 0.01,
    resume: bool = False,
) -> dict[str, Any]:
    extension_root = Path(extension).resolve()
    batch_root = Path(oracle_batch).resolve()
    output_root = Path(output).resolve()
    sequence_path = extension_root / "stall_sequences.jsonl"
    extension_runner_path = extension_root / "runner_config.json"
    batch_runner_path = batch_root / "runner_config.json"
    if (
        not sequence_path.is_file()
        or not extension_runner_path.is_file()
        or not batch_runner_path.is_file()
    ):
        raise ValueError("stall trigger audit source artifact is missing")
    if (
        not thresholds
        or tuple(sorted(set(thresholds))) != thresholds
        or thresholds[0] < 2
    ):
        raise ValueError("stall trigger thresholds must be unique ascending integers >= 2")
    extension_runner = _read_json(extension_runner_path)
    extension_identity = extension_runner.get("identity")
    if not isinstance(extension_identity, dict):
        raise ValueError("stall sequence runner identity is missing")
    raw_source_thresholds = extension_identity.get("thresholds")
    if not isinstance(raw_source_thresholds, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_source_thresholds
    ):
        raise ValueError("stall sequence source thresholds are invalid")
    source_thresholds = tuple(raw_source_thresholds)
    validate_source_threshold_coverage(thresholds, source_thresholds)
    identity = {
        "schema": STALL_TRIGGER_POLICY_AUDIT_SCHEMA,
        "sequence_sha256": sha256_file(sequence_path),
        "extension_runner_sha256": sha256_file(extension_runner_path),
        "oracle_batch_runner_sha256": sha256_file(batch_runner_path),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "thresholds": list(thresholds),
        "source_sequence_thresholds": list(source_thresholds),
        "maximum_false_trigger_rate": maximum_false_trigger_rate,
    }
    prepare_run_output(output_root, resume=resume, identity=identity)
    classifications = _oracle_classifications(batch_root)
    sequences = [
        row
        for row in _read_jsonl(sequence_path)
        if bool(row.get("new_independent_state"))
    ]
    labelled: list[dict[str, Any]] = []
    for sequence in sequences:
        sequence_class = str(sequence.get("qualification_class"))
        fingerprint = str(sequence.get("before_repair_fingerprint") or "")
        oracle_class = classifications.get(fingerprint)
        if sequence_class == "natural_recovery":
            label = "natural_recovery_control"
        elif sequence_class == "confirmed_long_stall" and oracle_class == "selector_failure":
            label = "selector_failure"
        elif sequence_class == "confirmed_long_stall":
            label = "confirmed_stall_not_selector_failure"
        else:
            continue
        labelled.append({**sequence, "audit_label": label, "oracle_class": oracle_class})
    controls = [row for row in labelled if row["audit_label"] == "natural_recovery_control"]
    positives = [row for row in labelled if row["audit_label"] == "selector_failure"]
    excluded = [
        row
        for row in labelled
        if row["audit_label"] == "confirmed_stall_not_selector_failure"
    ]
    if not controls or not positives:
        raise ValueError("stall trigger audit lacks controls or selector failures")
    trial_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for threshold in thresholds:
        for row in controls + positives:
            trigger_index = same_neighborhood_trigger_index(
                row.get("history_attempts") or [], threshold
            )
            trial_rows.append(
                {
                    "task_id": row.get("task_id"),
                    "map_id": row.get("map_id"),
                    "layout_mode": row.get("layout_mode"),
                    "agent_count": row.get("agent_count"),
                    "before_repair_fingerprint": row.get(
                        "before_repair_fingerprint"
                    ),
                    "audit_label": row["audit_label"],
                    "threshold": threshold,
                    "triggered": trigger_index is not None,
                    "trigger_attempt_index": trigger_index,
                    "run_length": row.get("run_length"),
                    "distinct_neighborhood_count": row.get(
                        "distinct_neighborhood_count"
                    ),
                }
            )
        natural_trigger_count = sum(
            same_neighborhood_trigger_index(
                row.get("history_attempts") or [], threshold
            )
            is not None
            for row in controls
        )
        selector_trigger_count = sum(
            same_neighborhood_trigger_index(
                row.get("history_attempts") or [], threshold
            )
            is not None
            for row in positives
        )
        upper = wilson_upper(natural_trigger_count, len(controls))
        summaries.append(
            {
                "threshold": threshold,
                "natural_control_count": len(controls),
                "false_trigger_count": natural_trigger_count,
                "false_trigger_rate": natural_trigger_count / len(controls),
                "false_trigger_wilson_95_upper": upper,
                "selector_failure_count": len(positives),
                "selector_trigger_count": selector_trigger_count,
                "selector_recall": selector_trigger_count / len(positives),
                "point_false_rate_gate_passed": (
                    natural_trigger_count / len(controls)
                    <= maximum_false_trigger_rate
                ),
                "wilson_false_rate_gate_passed": (
                    upper is not None and upper <= maximum_false_trigger_rate
                ),
            }
        )
    zero_observed_false = [row for row in summaries if row["false_trigger_count"] == 0]
    recommended = (
        max(
            zero_observed_false,
            key=lambda row: (float(row["selector_recall"]), -int(row["threshold"])),
        )
        if zero_observed_false
        else None
    )
    passing = [row for row in summaries if row["wilson_false_rate_gate_passed"]]
    required_controls = _zero_failure_sample_requirement(maximum_false_trigger_rate)
    report = {
        "schema": STALL_TRIGGER_POLICY_AUDIT_SCHEMA,
        "complete": True,
        "evidence_level": "action-preserving policy_train shadow audit",
        "natural_recovery_control_count": len(controls),
        "selector_failure_count": len(positives),
        "excluded_non_selector_stall_count": len(excluded),
        "thresholds": summaries,
        "recommended_shadow_threshold": (
            recommended["threshold"] if recommended else None
        ),
        "recommended_shadow_selector_recall": (
            recommended["selector_recall"] if recommended else None
        ),
        "minimum_zero_failure_controls_for_one_percent_wilson_gate": (
            required_controls
        ),
        "additional_zero_failure_controls_needed": max(
            0, required_controls - len(controls)
        ),
        "promotion_eligible": bool(passing),
        "decision": (
            "threshold_ready_for_runtime_confirmation"
            if passing
            else "shadow_only_collect_more_natural_recovery_controls"
        ),
        "controller_actions_changed": False,
        "training_started": False,
    }
    atomic_write_csv(output_root / "stall_trigger_trials.csv", trial_rows)
    atomic_write_csv(output_root / "stall_trigger_summary.csv", summaries)
    _write_json(output_root / "stall_trigger_policy_audit_report.json", report)
    return report


__all__ = [
    "DEFAULT_SAME_NEIGHBORHOOD_THRESHOLDS",
    "audit_stall_trigger_policies",
    "same_neighborhood_trigger_index",
    "validate_source_threshold_coverage",
    "wilson_upper",
]
