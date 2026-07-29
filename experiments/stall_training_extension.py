from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.lns2_bottleneck import validate_manifest_trace
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.run_output_guard import prepare_run_output
from experiments.stall_escape_continuation import compare_continuation_prefix
from experiments.stall_escape_qualification import (
    classify_stall_sequences,
    normalize_episode_decisions,
)
from experiments.trace_replay import decision_rows


STALL_TRAINING_EXTENSION_SCHEMA = "lns2.stall_training_extension.v1"


def unfinished_training_manifests(
    manifests: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    keys: set[tuple[str, int]] = set()
    for raw in manifests:
        row = dict(raw)
        summary = row.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("training extension manifest summary is missing")
        if str(row.get("split")) != "policy_train":
            raise ValueError("training extension accepts policy_train only")
        if str(row.get("policy")) != "realized_dynamic":
            raise ValueError("training extension requires realized_dynamic traces")
        if str(row.get("status")) not in {"ok", "resumed"}:
            raise ValueError("training extension source manifest is incomplete")
        if bool(summary.get("success")):
            continue
        if str(summary.get("stop_reason")) != "repair_limit":
            raise ValueError("unfinished training episode did not stop at repair limit")
        key = (str(row.get("task_id") or ""), int(row.get("solver_seed", -1)))
        if not key[0] or key[1] < 0 or key in keys:
            raise ValueError("training extension source keys are invalid or duplicated")
        keys.add(key)
        selected.append(row)
    return sorted(selected, key=lambda row: (str(row["task_id"]), int(row["solver_seed"])))


def _source_protocol_path(
    source_root: Path,
    explicit_config: str | Path | None = None,
) -> Path:
    if explicit_config is not None:
        explicit = Path(explicit_config).resolve()
        if not explicit.is_file():
            raise ValueError("training extension explicit source config is missing")
        return explicit
    candidates = (
        source_root.parent / "protocol" / "policy_train_extension.json",
        source_root.parent.parent / "protocol" / "source_policy_train.json",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError("training extension source protocol is missing or ambiguous")
    return existing[0]


def _validate_source_protocol(
    source: dict[str, Any],
    source_configuration: dict[str, Any],
) -> None:
    mismatches = [
        key
        for key, value in source.items()
        if key not in source_configuration or source_configuration[key] != value
    ]
    if mismatches:
        raise ValueError(
            "training extension source config does not match the collection: "
            + ", ".join(sorted(mismatches))
        )


def _derived_config(
    source_path: Path,
    output_root: Path,
    target_decisions: int,
) -> Path:
    source = _read_json(source_path)
    original = int(source["max_decisions"])
    if target_decisions <= original:
        raise ValueError("training extension target must exceed the source budget")
    budget_scale = target_decisions / original
    source["max_decisions"] = target_decisions
    source["metric_iteration_budget"] = target_decisions
    for field in (
        "wall_time_budget_seconds",
        "episode_process_timeout_seconds",
    ):
        if field in source:
            source[field] = float(source[field]) * budget_scale
    environment = dict(source["environment"])
    environment["max_repair_iterations"] = target_decisions
    if "time_limit" in environment:
        environment["time_limit"] = float(environment["time_limit"]) * budget_scale
    source["environment"] = environment
    destination = output_root / "protocol" / "policy_train_extension.json"
    _write_json(destination, source)
    return destination


def _sequence_row(
    sequence: dict[str, Any],
    manifest: dict[str, Any],
    *,
    source_transition_count: int,
    source_sequence_classes: dict[str, str],
    continued_root: Path,
) -> dict[str, Any]:
    trigger = dict(sequence["trigger"])
    history = [dict(row) for row in sequence["history"]]
    maximum_history_index = max(int(row["decision_index"]) for row in history)
    repair_fingerprint = str(trigger["before_repair_fingerprint"])
    source_class = source_sequence_classes.get(repair_fingerprint)
    existed_in_source = source_class is not None
    qualification_class = str(sequence["qualification_class"])
    return {
        "schema": STALL_TRAINING_EXTENSION_SCHEMA,
        "qualification_class": qualification_class,
        "resolution": str(sequence["resolution"]),
        "split": "policy_train",
        "map_id": str(manifest["map_id"]),
        "layout_mode": str(manifest.get("layout_mode", "unknown")),
        "agent_count": int(manifest.get("agent_count", 0)),
        "task_id": str(manifest["task_id"]),
        "solver_seed": int(manifest["solver_seed"]),
        "episode_id": str(manifest["episode_id"]),
        "source_collection": str(continued_root),
        "anchor_decision_index": int(trigger["decision_index"]),
        "before_fingerprint": str(trigger["before_fingerprint"]),
        "before_repair_fingerprint": repair_fingerprint,
        "selected_candidate_id": str(trigger["selected_candidate_id"]),
        "candidate_count": len(trigger["candidate_pool"]),
        "threshold": int(sequence["threshold"]),
        "run_length": int(sequence["run_length"]),
        "observed_after_trigger": int(sequence["observed_after_trigger"]),
        "recovery_delay_decisions": sequence["recovery_delay_decisions"],
        "distinct_attempt_count": int(sequence["distinct_attempt_count"]),
        "distinct_neighborhood_count": int(sequence["distinct_neighborhood_count"]),
        "source_transition_count": source_transition_count,
        "maximum_history_decision_index": maximum_history_index,
        "new_evidence_after_source": maximum_history_index >= source_transition_count,
        "existing_sequence_in_source": existed_in_source,
        "new_independent_state": not existed_in_source,
        "source_qualification_class": source_class,
        "classification_changed_from_source": bool(
            source_class is not None and source_class != qualification_class
        ),
        "prefix_actions": list(trigger["prefix_actions"]),
        "candidate_pool": list(trigger["candidate_pool"]),
        "history_attempts": [
            {
                "decision_index": int(row["decision_index"]),
                "attempt_key": str(row["attempt_key"]),
                "neighborhood_key": str(row["actual_neighborhood_key"]),
                "selected_candidate_id": str(row["selected_candidate_id"]),
                "repair_outcome": str(row["repair_outcome"]),
            }
            for row in history
        ],
    }


def run_stall_training_extension(
    source: str | Path,
    output: str | Path,
    *,
    source_config: str | Path | None = None,
    target_decisions: int = 60,
    thresholds: tuple[int, ...] = (3, 4, 6),
    future_observation_decisions: int = 3,
    minimum_distinct_attempts: int = 2,
    workers: int = 4,
    task_ids: Iterable[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    if workers <= 0:
        raise ValueError("training extension workers must be positive")
    source_run_path = source_root / "run_config.json"
    source_manifest_path = source_root / "realized_dynamic_manifest.jsonl"
    source_run = _read_json(source_run_path)
    configuration = dict(source_run["configuration"])
    source_protocol_path = _source_protocol_path(source_root, source_config)
    source_protocol = _read_json(source_protocol_path)
    _validate_source_protocol(source_protocol, configuration)
    if (
        str(source_run.get("controller")) != "v2-full"
        or str(configuration.get("split")) != "policy_train"
        or bool(configuration.get("formal"))
    ):
        raise ValueError("training extension source must be non-formal policy_train v2")
    source_manifests = unfinished_training_manifests(
        _read_jsonl(source_manifest_path)
    )
    requested_tasks = set(map(str, task_ids or ()))
    if requested_tasks:
        available_tasks = {str(row["task_id"]) for row in source_manifests}
        missing_tasks = requested_tasks - available_tasks
        if missing_tasks:
            raise ValueError(
                "training extension requested unavailable tasks: "
                + ", ".join(sorted(missing_tasks))
            )
        source_manifests = [
            row for row in source_manifests if str(row["task_id"]) in requested_tasks
        ]
    if not source_manifests:
        raise ValueError("training extension source has no unfinished episodes")
    job_keys = {
        (str(row["task_id"]), int(row["solver_seed"])) for row in source_manifests
    }
    identity = {
        "schema": STALL_TRAINING_EXTENSION_SCHEMA,
        "source_run_config_sha256": sha256_file(source_run_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_protocol_sha256": sha256_file(source_protocol_path),
        "source_run_fingerprint": str(source_run["run_fingerprint"]),
        "target_decisions": int(target_decisions),
        "thresholds": list(thresholds),
        "future_observation_decisions": int(future_observation_decisions),
        "minimum_distinct_attempts": int(minimum_distinct_attempts),
        "requested_task_ids": sorted(requested_tasks),
        "job_keys": [list(key) for key in sorted(job_keys)],
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    runner = prepare_run_output(output_root, resume=resume, identity=identity)
    config_path = _derived_config(
        source_protocol_path,
        output_root,
        target_decisions,
    )
    collection_root = output_root / "collection"
    common = {
        "workers": workers,
        "task_ids": sorted(key[0] for key in job_keys),
        "controller": "v2-full",
        "feature_backend": str(configuration["feature_backend"]),
        "controller_bundle": str(configuration["controller_bundle"]),
        "controller_runtime": str(configuration["controller_runtime"]),
        "verification_profile": str(configuration["verification_profile"]),
        "job_keys": job_keys,
        "cohort_job_keys": job_keys,
        "stopping_rule": "historical",
    }
    run_closed_loop_collection(
        str(source_run["dataset"]),
        config_path,
        collection_root,
        phase="qualify",
        resume=resume or (collection_root / "run_config.json").is_file(),
        **common,
    )
    run_closed_loop_collection(
        str(source_run["dataset"]),
        config_path,
        collection_root,
        phase="realized_dynamic",
        resume=True,
        **common,
    )
    continued_run = _read_json(collection_root / "run_config.json")
    continued_manifests = {
        str(row["episode_id"]): dict(row)
        for row in _read_jsonl(collection_root / "realized_dynamic_manifest.jsonl")
    }
    source_by_episode = {str(row["episode_id"]): row for row in source_manifests}
    if set(source_by_episode) != set(continued_manifests):
        raise ValueError("training extension episode coverage is incomplete")

    prefix_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    for episode_id, source_manifest in sorted(source_by_episode.items()):
        continued_manifest = continued_manifests[episode_id]
        _path, source_events, _blob = validate_manifest_trace(
            source_root,
            source_manifest,
            run_fingerprint=str(source_run["run_fingerprint"]),
            expected_policy="realized_dynamic",
        )
        _path, continued_events, _blob = validate_manifest_trace(
            collection_root,
            continued_manifest,
            run_fingerprint=str(continued_run["run_fingerprint"]),
            expected_policy="realized_dynamic",
        )
        source_decisions, _events = decision_rows(source_root, source_manifest)
        continued_decisions, _events = decision_rows(
            collection_root, continued_manifest
        )
        prefix = compare_continuation_prefix(
            source_decisions,
            source_events,
            continued_decisions,
            continued_events,
        )
        prefix_rows.append(
            {
                "episode_id": episode_id,
                "task_id": source_manifest["task_id"],
                "solver_seed": source_manifest["solver_seed"],
                **prefix,
            }
        )
        if not bool(prefix["passed"]):
            continue
        normalized = normalize_episode_decisions(
            continued_decisions, continued_events
        )
        source_normalized = normalize_episode_decisions(
            source_decisions, source_events
        )
        source_sequences, _source_ordinary = classify_stall_sequences(
            source_normalized,
            thresholds=thresholds,
            future_observation_decisions=future_observation_decisions,
            minimum_distinct_attempts=minimum_distinct_attempts,
        )
        source_sequence_classes = {
            str(sequence["state_fingerprint"]): str(
                sequence["qualification_class"]
            )
            for sequence in source_sequences
        }
        sequences, _ordinary = classify_stall_sequences(
            normalized,
            thresholds=thresholds,
            future_observation_decisions=future_observation_decisions,
            minimum_distinct_attempts=minimum_distinct_attempts,
        )
        sequence_rows.extend(
            _sequence_row(
                sequence,
                continued_manifest,
                source_transition_count=len(source_decisions),
                source_sequence_classes=source_sequence_classes,
                continued_root=collection_root,
            )
            for sequence in sequences
        )
    semantic_mismatches = sum(
        int(row["semantic_mismatch_count"]) for row in prefix_rows
    )
    new_rows = [row for row in sequence_rows if bool(row["new_independent_state"])]
    extended_existing_rows = [
        row
        for row in sequence_rows
        if bool(row["existing_sequence_in_source"])
        and (
            bool(row["new_evidence_after_source"])
            or bool(row["classification_changed_from_source"])
        )
    ]
    counts = collections.Counter(
        str(row["qualification_class"]) for row in new_rows
    )
    confirmed = [
        row
        for row in new_rows
        if str(row["qualification_class"]) == "confirmed_long_stall"
    ]
    oracle_plan = [
        {
            "source": str(collection_root),
            "task_id": row["task_id"],
            "solver_seed": row["solver_seed"],
            "decision_index": row["anchor_decision_index"],
            "before_repair_fingerprint": row["before_repair_fingerprint"],
            "all_candidates": True,
            "trials_per_branch": 4,
        }
        for row in confirmed
    ]
    report = {
        "schema": STALL_TRAINING_EXTENSION_SCHEMA,
        "complete": semantic_mismatches == 0,
        "run_fingerprint": str(runner["identity_fingerprint"]),
        "source_episode_count": len(source_manifests),
        "target_decisions": target_decisions,
        "semantic_mismatch_count": semantic_mismatches,
        "budget_boundary_exclusion_count": sum(
            int(row["budget_boundary_exclusion_count"]) for row in prefix_rows
        ),
        "sequence_count": len(sequence_rows),
        "new_sequence_count": len(new_rows),
        "extended_existing_sequence_count": len(extended_existing_rows),
        "new_state_count_by_class": dict(sorted(counts.items())),
        "confirmed_training_stall_count": len(confirmed),
        "oracle_plan_count": len(oracle_plan),
        "controller_actions_changed": False,
        "training_started": False,
        "validation_or_movingai_labels_seen": False,
        "decision": (
            "prefix_mismatch_stop"
            if semantic_mismatches
            else "collect_exact_policy_train_stall_oracles"
            if confirmed
            else "no_confirmed_train_stalls_extend_or_redesign_sources"
        ),
        "collection": str(collection_root),
    }
    atomic_write_csv(output_root / "prefix_equivalence.csv", prefix_rows)
    atomic_write_csv(
        output_root / "stall_sequences.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"candidate_pool", "prefix_actions", "history_attempts"}
            }
            for row in sequence_rows
        ],
    )
    _write_jsonl(output_root / "stall_sequences.jsonl", sequence_rows)
    _write_json(output_root / "oracle_plan.json", oracle_plan)
    _write_json(output_root / "training_extension_report.json", report)
    lines = [
        "# Policy-train stall extension",
        "",
        f"- Episodes: `{report['source_episode_count']}`; target repairs: `{target_decisions}`.",
        f"- Prefix semantic mismatches: `{semantic_mismatches}`.",
        f"- New classified sequences: `{len(new_rows)}`; by class: `{dict(sorted(counts.items()))}`.",
        f"- Confirmed train stalls: `{len(confirmed)}`.",
        f"- Decision: `{report['decision']}`.",
        "- Training started: `false`; v2 actions changed: `false`.",
        "",
    ]
    (output_root / "training_extension_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = [
    "STALL_TRAINING_EXTENSION_SCHEMA",
    "run_stall_training_extension",
    "unfinished_training_manifests",
]
