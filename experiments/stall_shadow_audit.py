from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import (
    atomic_write_csv,
    sha256_file,
    strict_bool as _strict_bool,
    strict_int as _strict_int,
)
from experiments.lns2_bottleneck import _configured_cohort, validate_manifest_trace
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.repair_aware import REPAIR_OUTCOMES
from experiments.stall_shadow import (
    STALL_SHADOW_SUMMARY_SCHEMA,
    STALL_SHADOW_TRANSITION_SCHEMA,
    load_stall_shadow_config,
    neighborhood_key,
    pp_attempt_key,
)


STALL_SHADOW_AUDIT_SCHEMA = "lns2.stall_shadow_audit.v3"
STALL_SHADOW_AUDIT_VERSION = 3


def _optional_native_seed(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer or null")
    return value if value >= 0 else None


def _int_list(value: Any, *, field: str, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise ValueError(f"{field} must be a non-negative integer list")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    return list(value)


def _expected_cohort(root: Path, run: dict[str, Any]) -> tuple[set[tuple[str, int]], str]:
    configured = _configured_cohort(run)
    if bool(configured.get("available")):
        if int(configured.get("invalid_entry_count", 0)) or int(
            configured.get("duplicate_key_count", 0)
        ):
            raise ValueError("stall shadow run config contains an invalid cohort")
        keys = {
            (str(value[0]), int(value[1]))
            for value in configured.get("keys", [])
            if isinstance(value, list) and len(value) == 2
        }
        if not keys:
            raise ValueError("stall shadow run config contains an empty cohort")
        return keys, str(configured.get("source"))

    schedule_paths = [
        path
        for path in (
            root.parent / "execution_schedule.json",
            root.parent.parent / "execution_schedule.json",
        )
        if path.is_file()
    ]
    if len(schedule_paths) != 1:
        raise ValueError("stall shadow audit cannot identify one registered cohort")
    schedule = _read_json(schedule_paths[0])
    if str(schedule.get("schema")) != "lns2.controller_execution_schedule.v1" or not isinstance(
        schedule.get("entries"), list
    ):
        raise ValueError("stall shadow execution schedule is invalid")
    keys: list[tuple[str, int]] = []
    for entry in schedule["entries"]:
        if not isinstance(entry, dict) or not str(entry.get("task_id") or ""):
            raise ValueError("stall shadow execution schedule entry is invalid")
        seed = entry.get("solver_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("stall shadow execution schedule seed is invalid")
        keys.append((str(entry["task_id"]), seed))
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("stall shadow execution schedule is empty or duplicated")
    return set(keys), "execution_schedule"


def _wilson_upper(false_count: int, resolved_count: int) -> float | None:
    if resolved_count <= 0:
        return None
    z = 1.959963984540054
    proportion = false_count / resolved_count
    denominator = 1.0 + z * z / resolved_count
    center = (proportion + z * z / (2.0 * resolved_count)) / denominator
    margin = (
        z
        * math.sqrt(
            (
                proportion * (1.0 - proportion)
                + z * z / (4.0 * resolved_count)
            )
            / resolved_count
        )
        / denominator
    )
    return min(1.0, center + margin)


def _validate_episode_shadow(
    events: list[dict[str, Any]], *, config: Any
) -> tuple[dict[str, Any], dict[int, Counter[str]]]:
    if not events or str(events[-1].get("event")) != "finish":
        raise ValueError("stall shadow trace has no finish event")
    raw_summary = dict(events[-1].get("summary") or {}).get("stall_shadow")
    if not isinstance(raw_summary, dict):
        raise ValueError("trace is missing the registered stall shadow summary")
    summary = dict(raw_summary)
    if str(summary.get("schema")) != STALL_SHADOW_SUMMARY_SCHEMA:
        raise ValueError("trace has an unsupported stall shadow summary")
    if summary.get("config") != config.payload() or str(
        summary.get("config_fingerprint")
    ) != config.fingerprint:
        raise ValueError("stall shadow summary config identity mismatch")
    if (
        str(summary.get("mode")) != "shadow"
        or summary.get("deployment_enabled") is not False
        or summary.get("promotion_eligible") is not False
        or summary.get("most_conservative_passing_threshold") is not None
    ):
        raise ValueError("stall shadow summary claims an unsupported deployment mode")

    transitions = [event for event in events if str(event.get("event")) == "transition"]
    outcome_counts: Counter[str] = Counter()
    threshold_counts = {
        threshold: Counter(
            {"triggers": 0, "premature": 0, "confirmed": 0, "unresolved": 0}
        )
        for threshold in config.unchanged_attempt_thresholds
    }
    open_triggers: set[tuple[int, int]] = set()
    current_attempts: list[str] = []
    longest_unchanged_streak = 0
    reset_count = 0
    for event in transitions:
        controller = event.get("controller")
        if not isinstance(controller, dict) or not isinstance(
            controller.get("stall_shadow"), dict
        ):
            raise ValueError("transition is missing its stall shadow evidence")
        shadow = dict(controller["stall_shadow"])
        if (
            str(shadow.get("schema")) != STALL_SHADOW_TRANSITION_SCHEMA
            or str(shadow.get("config_fingerprint")) != config.fingerprint
            or str(shadow.get("mode")) != "shadow"
            or shadow.get("deployment_enabled") is not False
            or str(shadow.get("route")) != "model"
        ):
            raise ValueError("transition stall shadow identity is invalid")
        for field in ("action_preserved", "base_selection_preserved"):
            if not _strict_bool(shadow.get(field), field=field):
                raise ValueError("stall shadow modified the frozen v2 action")
        base_id = str(shadow.get("base_selected_candidate_id") or "")
        effective_id = str(shadow.get("effective_selected_candidate_id") or "")
        if not base_id or base_id != effective_id:
            raise ValueError("stall shadow selected candidate identity mismatch")

        decision_index = _strict_int(
            event.get("decision_index"), field="decision_index"
        )
        before_count = _strict_int(
            shadow.get("unchanged_attempt_count_before"),
            field="unchanged_attempt_count_before",
        )
        distinct_before = _strict_int(
            shadow.get("distinct_pp_attempt_count_before"),
            field="distinct_pp_attempt_count_before",
        )
        if before_count != len(current_attempts) or distinct_before != len(
            set(current_attempts)
        ):
            raise ValueError("stall shadow pre-repair attempt counts are inconsistent")
        _strict_int(
            shadow.get("cooldown_remaining_before"),
            field="cooldown_remaining_before",
        )

        triggered = _int_list(
            shadow.get("triggered_thresholds"), field="triggered_thresholds"
        )
        if any(threshold not in threshold_counts for threshold in triggered):
            raise ValueError("stall shadow triggered an unconfigured threshold")
        for threshold in triggered:
            trigger = (threshold, decision_index)
            if trigger in open_triggers:
                raise ValueError("stall shadow repeats one trigger identity")
            open_triggers.add(trigger)
            threshold_counts[threshold]["triggers"] += 1

        outcome = str(shadow.get("repair_outcome") or "")
        if outcome not in REPAIR_OUTCOMES:
            raise ValueError("stall shadow transition outcome is invalid")
        no_progress = _strict_bool(shadow.get("no_progress"), field="no_progress")
        state_unchanged = _strict_bool(
            shadow.get("state_unchanged"), field="state_unchanged"
        )
        expected_no_progress = outcome in {"hard_failure", "accepted_noop"}
        if no_progress != expected_no_progress or state_unchanged != expected_no_progress:
            raise ValueError("stall shadow outcome flags are inconsistent")
        outcome_counts[outcome] += 1

        attempt = shadow.get("attempt")
        if no_progress:
            if not isinstance(attempt, dict):
                raise ValueError("stall shadow no-progress transition lacks an attempt")
            agents = _int_list(
                attempt.get("agents"), field="attempt.agents", allow_empty=False
            )
            if (
                str(attempt.get("candidate_id") or "") != base_id
                or str(attempt.get("neighborhood_key") or "")
                != neighborhood_key(agents)
                or str(attempt.get("outcome") or "") != outcome
            ):
                raise ValueError("stall shadow attempt metadata is inconsistent")
            action = event.get("action")
            metrics = event.get("metrics")
            if not isinstance(action, dict) or not isinstance(metrics, dict):
                raise ValueError("stall shadow transition lacks action/metric evidence")
            step_seed = _strict_int(
                attempt.get("step_random_seed"), field="attempt.step_random_seed"
            )
            if _strict_int(action.get("random_seed"), field="action.random_seed") != step_seed:
                raise ValueError("stall shadow attempt uses a different action seed")
            requested_seed = _optional_native_seed(
                attempt.get("requested_pp_seed"), field="attempt.requested_pp_seed"
            )
            applied_seed = _optional_native_seed(
                attempt.get("applied_pp_seed"), field="attempt.applied_pp_seed"
            )
            if requested_seed != _optional_native_seed(
                metrics.get("requested_pp_random_seed"),
                field="metrics.requested_pp_random_seed",
            ) or applied_seed != _optional_native_seed(
                metrics.get("applied_pp_random_seed"),
                field="metrics.applied_pp_random_seed",
            ):
                raise ValueError("stall shadow attempt PP seeds differ from native metrics")
            repair_order = _int_list(
                metrics.get("repair_order", []), field="metrics.repair_order"
            )
            expected_attempt_key = pp_attempt_key(
                state_fingerprint=str(shadow.get("state_anchor_fingerprint") or ""),
                agents=agents,
                step_random_seed=step_seed,
                requested_pp_seed=requested_seed,
                applied_pp_seed=applied_seed,
                repair_order=repair_order,
            )
            if str(attempt.get("attempt_key") or "") != expected_attempt_key:
                raise ValueError("stall shadow attempt key is inconsistent")
            current_attempts.append(expected_attempt_key)
            longest_unchanged_streak = max(
                longest_unchanged_streak, len(current_attempts)
            )
        else:
            if attempt is not None:
                raise ValueError("stall shadow progress transition claims an attempt")
            reset_count += int(outcome == "state_changed_no_reduction")
            current_attempts.clear()
        after_count = _strict_int(
            shadow.get("unchanged_attempt_count_after"),
            field="unchanged_attempt_count_after",
        )
        distinct_after = _strict_int(
            shadow.get("distinct_pp_attempt_count_after"),
            field="distinct_pp_attempt_count_after",
        )
        if after_count != len(current_attempts) or distinct_after != len(
            set(current_attempts)
        ):
            raise ValueError("stall shadow post-repair attempt counts are inconsistent")

        resolved = shadow.get("resolved_triggers")
        if not isinstance(resolved, list):
            raise ValueError("stall shadow resolved trigger evidence is invalid")
        seen_resolved: set[tuple[int, int]] = set()
        for raw in resolved:
            if not isinstance(raw, dict):
                raise ValueError("stall shadow resolved trigger is not an object")
            threshold = _strict_int(raw.get("threshold"), field="resolved.threshold")
            trigger_decision = _strict_int(
                raw.get("decision_index"), field="resolved.decision_index"
            )
            observed = _strict_int(
                raw.get("observed_decisions"),
                field="resolved.observed_decisions",
                minimum=1,
            )
            resolution = str(raw.get("resolution") or "")
            trigger = (threshold, trigger_decision)
            if (
                threshold not in threshold_counts
                or trigger not in open_triggers
                or trigger in seen_resolved
                or observed > config.future_observation_decisions
                or resolution not in {"premature_trigger", "confirmed_stall"}
            ):
                raise ValueError("stall shadow resolved trigger is inconsistent")
            if resolution == "confirmed_stall" and observed != config.future_observation_decisions:
                raise ValueError("stall shadow confirmed a trigger before its horizon")
            if resolution == "premature_trigger" and no_progress:
                raise ValueError("stall shadow marked a no-progress repair premature")
            seen_resolved.add(trigger)
            open_triggers.remove(trigger)
            threshold_counts[threshold][
                "premature" if resolution == "premature_trigger" else "confirmed"
            ] += 1

    for threshold, _decision_index in open_triggers:
        threshold_counts[threshold]["unresolved"] += 1

    if _strict_int(summary.get("action_override_count"), field="action_override_count") != 0:
        raise ValueError("stall shadow summary reports an action override")
    if _strict_int(summary.get("decision_count"), field="decision_count") != len(
        transitions
    ):
        raise ValueError("stall shadow summary decision count is inconsistent")
    _strict_int(summary.get("cooldown_decision_count"), field="cooldown_decision_count")
    if _strict_int(
        summary.get("state_changed_no_reduction_reset_count"),
        field="state_changed_no_reduction_reset_count",
    ) != reset_count:
        raise ValueError("stall shadow reset count is inconsistent")
    if _strict_int(
        summary.get("longest_unchanged_streak"), field="longest_unchanged_streak"
    ) != longest_unchanged_streak:
        raise ValueError("stall shadow longest unchanged streak is inconsistent")
    raw_outcomes = summary.get("outcome_counts")
    if not isinstance(raw_outcomes, dict) or set(raw_outcomes) != set(
        REPAIR_OUTCOMES
    ):
        raise ValueError("stall shadow summary outcome coverage is invalid")
    if any(
        _strict_int(raw_outcomes[name], field=f"outcome_counts.{name}")
        != outcome_counts[name]
        for name in REPAIR_OUTCOMES
    ):
        raise ValueError("stall shadow summary outcome counts are inconsistent")
    raw_thresholds = summary.get("thresholds")
    if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != {
        str(value) for value in config.unchanged_attempt_thresholds
    }:
        raise ValueError("stall shadow summary threshold coverage is invalid")
    for threshold, counts in threshold_counts.items():
        values = raw_thresholds[str(threshold)]
        if not isinstance(values, dict):
            raise ValueError("stall shadow threshold summary is not an object")
        expected = {
            "trigger_count": counts["triggers"],
            "premature_trigger_count": counts["premature"],
            "confirmed_stall_count": counts["confirmed"],
            "unresolved_trigger_count": counts["unresolved"],
        }
        for field, count in expected.items():
            if _strict_int(values.get(field), field=f"threshold.{field}") != count:
                raise ValueError("stall shadow threshold summary is inconsistent")
        resolved_count = counts["premature"] + counts["confirmed"]
        expected_rate = (
            counts["premature"] / resolved_count if resolved_count else None
        )
        rate = values.get("false_trigger_rate")
        if expected_rate is None:
            if rate is not None:
                raise ValueError("stall shadow threshold false rate is inconsistent")
        elif isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isclose(
            float(rate), expected_rate, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("stall shadow threshold false rate is inconsistent")
        if (
            values.get("false_trigger_gate_passed") is not False
            or str(values.get("gate_status")) != "external_audit_required"
        ):
            raise ValueError("stall shadow runtime summary claims an audit gate")
    return summary, threshold_counts


def audit_stall_shadow_collection(
    source: str | Path, output: str | Path
) -> dict[str, Any]:
    root = Path(source).resolve()
    output_root = Path(output).resolve()
    run_path = root / "run_config.json"
    manifest_path = root / "realized_dynamic_manifest.jsonl"
    run = _read_json(run_path)
    if str(run.get("controller")) != "v2-stall-shadow" or not str(
        run.get("run_fingerprint") or ""
    ):
        raise ValueError("stall shadow audit requires a registered collection")
    configuration = run.get("configuration")
    if not isinstance(configuration, dict) or not isinstance(
        configuration.get("stall_shadow_config"), dict
    ):
        raise ValueError("stall shadow run config lacks its shadow configuration")
    config = load_stall_shadow_config(configuration["stall_shadow_config"])
    expected_keys, cohort_source = _expected_cohort(root, run)
    manifests = _read_jsonl(manifest_path)
    observed_keys: list[tuple[str, int]] = []
    for manifest in manifests:
        if not isinstance(manifest, dict):
            raise ValueError("stall shadow manifest contains a non-object")
        if str(manifest.get("status")) not in {"ok", "resumed"}:
            raise ValueError("stall shadow collection is incomplete or contains errors")
        if str(manifest.get("policy")) != "realized_dynamic":
            raise ValueError("stall shadow manifest policy is invalid")
        task_id = str(manifest.get("task_id") or "")
        solver_seed = manifest.get("solver_seed")
        if not task_id or isinstance(solver_seed, bool) or not isinstance(solver_seed, int):
            raise ValueError("stall shadow manifest key is invalid")
        observed_keys.append((task_id, solver_seed))
    if len(observed_keys) != len(set(observed_keys)):
        raise ValueError("stall shadow manifest repeats an episode key")
    if set(observed_keys) != expected_keys:
        raise ValueError("stall shadow manifest does not cover its registered cohort")

    thresholds = {
        threshold: Counter(
            {"triggers": 0, "premature": 0, "confirmed": 0, "unresolved": 0}
        )
        for threshold in config.unchanged_attempt_thresholds
    }
    episode_rows: list[dict[str, Any]] = []
    for manifest in manifests:
        _path, events, _blob = validate_manifest_trace(
            root,
            manifest,
            run_fingerprint=str(run["run_fingerprint"]),
            expected_policy="realized_dynamic",
        )
        summary, episode_thresholds = _validate_episode_shadow(events, config=config)
        for threshold, counts in episode_thresholds.items():
            thresholds[threshold].update(counts)
        transition_count = sum(
            str(event.get("event")) == "transition" for event in events
        )
        episode_rows.append(
            {
                "task_id": manifest["task_id"],
                "solver_seed": int(manifest["solver_seed"]),
                "agent_count": int(manifest.get("agent_count", 0)),
                "transition_count": transition_count,
                "action_override_count": 0,
                "state_changed_no_reduction_count": int(
                    dict(summary["outcome_counts"])["state_changed_no_reduction"]
                ),
                "longest_unchanged_streak": int(
                    summary["longest_unchanged_streak"]
                ),
            }
        )

    threshold_rows: list[dict[str, Any]] = []
    passing: list[int] = []
    for threshold, values in sorted(thresholds.items()):
        resolved = values["premature"] + values["confirmed"]
        false_rate = values["premature"] / resolved if resolved else None
        upper = _wilson_upper(values["premature"], resolved)
        passed = bool(
            resolved > 0
            and values["unresolved"] == 0
            and upper is not None
            and upper <= config.maximum_false_trigger_rate
        )
        if passed:
            passing.append(threshold)
        threshold_rows.append(
            {
                "threshold": threshold,
                "trigger_count": values["triggers"],
                "premature_trigger_count": values["premature"],
                "confirmed_stall_count": values["confirmed"],
                "unresolved_trigger_count": values["unresolved"],
                "resolved_trigger_count": resolved,
                "false_trigger_rate": false_rate,
                "false_trigger_wilson_95_upper": upper,
                "false_trigger_gate_passed": passed,
            }
        )
    selected = max(passing) if passing else None
    report = {
        "schema": STALL_SHADOW_AUDIT_SCHEMA,
        "schema_version": STALL_SHADOW_AUDIT_VERSION,
        "source": str(root),
        "source_run_config_sha256": sha256_file(run_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "run_fingerprint": str(run["run_fingerprint"]),
        "cohort_source": cohort_source,
        "expected_episode_count": len(expected_keys),
        "episode_count": len(manifests),
        "error_count": 0,
        "action_override_count": 0,
        "invalid_outcome_count": 0,
        "thresholds": threshold_rows,
        "most_conservative_passing_threshold": selected,
        "shadow_integrity_passed": True,
        "deployment_promoted": False,
        "decision": (
            "shadow_candidate_threshold_found"
            if selected is not None
            else "keep_shadow_collecting"
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_root / "stall_shadow_thresholds.csv", threshold_rows)
    atomic_write_csv(output_root / "stall_shadow_episodes.csv", episode_rows)
    _write_json(output_root / "stall_shadow_audit_report.json", report)
    lines = [
        "# v2 stall-shadow audit",
        "",
        f"- Episodes: `{report['episode_count']}`; registered cohort: `{report['expected_episode_count']}`.",
        "- Action overrides: `0`.",
        f"- Most conservative passing threshold: `{selected}`.",
        f"- Decision: `{report['decision']}`; deployment promoted: `false`.",
        "",
        "The external gate uses the 95% Wilson upper bound and requires complete trigger resolution. A passing threshold remains a shadow candidate and does not change the default v2 controller.",
        "",
    ]
    (output_root / "stall_shadow_audit_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = [
    "STALL_SHADOW_AUDIT_SCHEMA",
    "STALL_SHADOW_AUDIT_VERSION",
    "audit_stall_shadow_collection",
]
