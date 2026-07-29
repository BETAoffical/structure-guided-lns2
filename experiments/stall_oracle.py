from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from experiments._common import sha256_file, strict_int as _object_int
from experiments.repair_collection import _fingerprint, _read_json, _write_json
from experiments.repair_aware import REPAIR_OUTCOMES, classify_repair_outcome
from experiments.run_output_guard import (
    RUNNER_CONFIG_SCHEMA,
    runner_identity_fingerprint,
)
from experiments.stall_shadow import neighborhood_key, pp_attempt_key
from experiments.stalled_state_probe import (
    STALLED_STATE_PROBE_SCHEMA,
    STALLED_STATE_PROBE_VERSION,
    TRIAL_STATE_RESTORE,
    paired_probe_seed,
)


STALL_ORACLE_SCHEMA = "lns2.stall_oracle_audit.v2"
STALL_ORACLE_VERSION = 2
def _strict_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{field} must be boolean")


def _strict_int(
    value: Any, *, field: str, minimum: int | None = 0
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error
    if str(value).strip() not in {str(number), f"+{number}"} and not isinstance(
        value, int
    ):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return number


def _optional_int(value: Any, *, field: str) -> int | None:
    if value in {None, "", "None"}:
        return None
    return _strict_int(value, field=field)


def _finite_nonnegative(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _json_ints(value: Any, *, field: str) -> list[int]:
    if value is None or value == "":
        return []
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is not valid JSON") from error
    if not isinstance(raw, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw
    ):
        raise ValueError(f"{field} must be an integer list")
    result = list(raw)
    if any(value < 0 for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique non-negative integers")
    return result


def read_probe_trials(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).resolve().open("r", encoding="utf-8", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError("stall Oracle source has no trial rows")
    return rows


def _validated_trial(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("schema")) != STALLED_STATE_PROBE_SCHEMA or _strict_int(
        row.get("schema_version"), field="schema_version"
    ) != STALLED_STATE_PROBE_VERSION:
        raise ValueError("stall Oracle trial schema is unsupported")
    if not _strict_bool(row.get("complete"), field="complete"):
        raise ValueError("stall Oracle trial is incomplete")
    run_fingerprint = str(row.get("run_fingerprint") or "")
    before_fingerprint = str(row.get("before_fingerprint") or "")
    before_repair_fingerprint = str(row.get("before_repair_fingerprint") or "")
    after_fingerprint = str(row.get("after_fingerprint") or "")
    after_repair_fingerprint = str(row.get("after_repair_fingerprint") or "")
    if not all(
        (
            run_fingerprint,
            before_fingerprint,
            before_repair_fingerprint,
            after_fingerprint,
            after_repair_fingerprint,
        )
    ):
        raise ValueError("stall Oracle trial is missing identity fingerprints")
    if not _strict_bool(
        row.get("replay_fingerprint_match"), field="replay_fingerprint_match"
    ):
        raise ValueError("stall Oracle trial replay did not match its source")
    if (
        str(row.get("trial_state_restore")) != TRIAL_STATE_RESTORE
        or not _strict_bool(
            row.get("repair_state_fingerprint_match"),
            field="repair_state_fingerprint_match",
        )
    ):
        raise ValueError("stall Oracle trial lacks exact repair-state restoration")
    source_before_fingerprint = str(
        row.get("source_before_fingerprint") or ""
    )
    restored_before_fingerprint = str(
        row.get("restored_before_fingerprint") or ""
    )
    if (
        source_before_fingerprint != before_fingerprint
        or not restored_before_fingerprint
    ):
        raise ValueError("stall Oracle trial restore fingerprints are invalid")
    full_state_match = _strict_bool(
        row.get("full_state_fingerprint_match"),
        field="full_state_fingerprint_match",
    )
    if full_state_match != (
        restored_before_fingerprint == source_before_fingerprint
    ):
        raise ValueError("stall Oracle trial full-state evidence is inconsistent")

    branch_key = str(row.get("branch_key") or "")
    branch_mode = str(row.get("branch_mode") or "")
    if not branch_key or branch_mode not in {"explicit_neighborhood", "official"}:
        raise ValueError("stall Oracle trial has invalid branch metadata")
    trial_index = _strict_int(row.get("trial_index"), field="trial_index")
    expected_seed = paired_probe_seed(before_fingerprint, trial_index)
    step_seed = _strict_int(row.get("random_seed"), field="random_seed")
    requested_seed = _optional_int(
        row.get("requested_pp_random_seed"), field="requested_pp_random_seed"
    )
    applied_seed = _optional_int(
        row.get("applied_pp_random_seed"), field="applied_pp_random_seed"
    )
    if step_seed != expected_seed or requested_seed != expected_seed:
        raise ValueError("stall Oracle trial is not paired by state and trial index")

    agents = _json_ints(row.get("candidate_agents"), field="candidate_agents")
    if not agents or _strict_int(
        row.get("candidate_size"), field="candidate_size", minimum=1
    ) != len(agents):
        raise ValueError("stall Oracle trial has an invalid actual neighborhood")
    repair_order = _json_ints(row.get("repair_order"), field="repair_order")
    if repair_order:
        if set(repair_order) != set(agents) or applied_seed != expected_seed:
            raise ValueError("stall Oracle trial PP order or applied seed is invalid")
    elif applied_seed is not None:
        raise ValueError("stall Oracle trial applied a PP seed without a repair order")

    replan_success = _strict_bool(row.get("replan_success"), field="replan_success")
    terminated = _strict_bool(row.get("terminated"), field="terminated")
    truncated = _strict_bool(row.get("truncated"), field="truncated")
    if terminated and truncated:
        raise ValueError("stall Oracle trial cannot terminate and truncate")
    conflicts_before = _strict_int(
        row.get("conflicts_before"), field="conflicts_before"
    )
    conflicts_after = _strict_int(row.get("conflicts_after"), field="conflicts_after")
    if terminated != (conflicts_after == 0):
        raise ValueError("stall Oracle termination disagrees with the after state")
    conflict_delta = _strict_int(
        row.get("conflict_delta"), field="conflict_delta", minimum=None
    )
    if conflict_delta != conflicts_before - conflicts_after:
        raise ValueError("stall Oracle conflict delta is inconsistent")
    outcome = classify_repair_outcome(
        before_fingerprint=before_repair_fingerprint,
        after_fingerprint=after_repair_fingerprint,
        replan_success=replan_success,
        conflicts_before=conflicts_before,
        conflicts_after=conflicts_after,
        feasible=terminated,
    )
    if outcome not in REPAIR_OUTCOMES or str(row.get("repair_outcome")) != outcome:
        raise ValueError("stall Oracle repair outcome is inconsistent")
    total_seconds = _finite_nonnegative(
        row.get("total_decision_seconds"), field="total_decision_seconds"
    )
    aliases = tuple(
        sorted(alias for alias in str(row.get("branch_aliases") or "").split(",") if alias)
    )
    candidate_rank = _optional_int(row.get("candidate_rank"), field="candidate_rank")
    candidate_id = str(row.get("candidate_id") or "")
    candidate_score: float | None = None
    if branch_mode == "explicit_neighborhood":
        if not candidate_id or candidate_rank is None or candidate_rank < 1:
            raise ValueError("explicit stall Oracle branch lacks candidate identity")
        candidate_score = float(row.get("candidate_score"))
        if not math.isfinite(candidate_score):
            raise ValueError("stall Oracle candidate score is non-finite")
    elif branch_mode == "official":
        if candidate_id or candidate_rank is not None:
            raise ValueError("official stall Oracle branch claims a model candidate")

    result = dict(row)
    result.update(
        {
            "run_fingerprint": run_fingerprint,
            "before_fingerprint": before_fingerprint,
            "before_repair_fingerprint": before_repair_fingerprint,
            "after_fingerprint": after_fingerprint,
            "after_repair_fingerprint": after_repair_fingerprint,
            "branch_key": branch_key,
            "branch_alias_tuple": aliases,
            "branch_mode": branch_mode,
            "candidate_id": candidate_id or None,
            "candidate_rank": candidate_rank,
            "candidate_score": candidate_score,
            "candidate_size": len(agents),
            "candidate_agents_list": agents,
            "trial_index": trial_index,
            "random_seed": step_seed,
            "requested_pp_random_seed": requested_seed,
            "applied_pp_random_seed": applied_seed,
            "repair_order_list": repair_order,
            "replan_success": replan_success,
            "terminated": terminated,
            "truncated": truncated,
            "conflicts_before": conflicts_before,
            "conflicts_after": conflicts_after,
            "conflict_delta": conflict_delta,
            "repair_outcome": outcome,
            "total_decision_seconds": total_seconds,
            "escaped_state": outcome not in {"hard_failure", "accepted_noop"},
            "neighborhood_key": neighborhood_key(agents),
            "attempt_key": pp_attempt_key(
                state_fingerprint=before_repair_fingerprint,
                agents=agents,
                step_random_seed=step_seed,
                requested_pp_seed=requested_seed,
                applied_pp_seed=applied_seed,
                repair_order=repair_order,
            ),
        }
    )
    return result


def classify_stall_oracle_trials(
    rows: list[dict[str, Any]],
    *,
    minimum_trials: int = 4,
    stable_fraction: float = 0.75,
) -> dict[str, Any]:
    if minimum_trials < 3:
        raise ValueError("stall Oracle needs at least three PP trials")
    if not math.isfinite(stable_fraction) or not 0.5 < stable_fraction <= 1.0:
        raise ValueError("stall Oracle stable fraction must be in (0.5, 1]")
    if not rows:
        raise ValueError("stall Oracle has no trials")
    validated = [_validated_trial(dict(source)) for source in rows]
    for field in (
        "run_fingerprint",
        "before_fingerprint",
        "before_repair_fingerprint",
    ):
        if len({str(row[field]) for row in validated}) != 1:
            raise ValueError(f"stall Oracle trials do not share one {field}")

    by_branch: dict[str, list[dict[str, Any]]] = {}
    for row in validated:
        by_branch.setdefault(str(row["branch_key"]), []).append(row)
    trial_index_sets: set[frozenset[int]] = set()
    summaries: list[dict[str, Any]] = []
    for branch_key, trials in sorted(by_branch.items()):
        trial_indexes = [int(row["trial_index"]) for row in trials]
        if len(trial_indexes) != len(set(trial_indexes)):
            raise ValueError(f"stall Oracle branch repeats a trial index: {branch_key}")
        attempt_keys = [str(row["attempt_key"]) for row in trials]
        if len(attempt_keys) != len(set(attempt_keys)):
            raise ValueError(f"stall Oracle branch repeats a PP attempt: {branch_key}")
        trial_index_sets.add(frozenset(trial_indexes))
        metadata = {
            (
                row["branch_alias_tuple"],
                row["branch_mode"],
                row["candidate_id"],
                row["candidate_rank"],
                row["candidate_score"],
            )
            for row in trials
        }
        if len(metadata) != 1:
            raise ValueError(f"stall Oracle branch metadata changes: {branch_key}")
        fixed_neighborhood = str(trials[0]["branch_mode"]) == "explicit_neighborhood"
        if fixed_neighborhood:
            neighborhoods = {
                (
                    row["candidate_size"],
                    tuple(row["candidate_agents_list"]),
                    row["neighborhood_key"],
                )
                for row in trials
            }
            if len(neighborhoods) != 1:
                raise ValueError(
                    f"stall Oracle model neighborhood changes: {branch_key}"
                )
        if len(trials) < minimum_trials:
            raise ValueError(f"stall Oracle branch has too few trials: {branch_key}")
        outcomes: dict[str, int] = {}
        for row in trials:
            outcome = str(row["repair_outcome"])
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        escape_count = sum(bool(row["escaped_state"]) for row in trials)
        failure_count = len(trials) - escape_count
        escape_fraction = escape_count / len(trials)
        failure_fraction = failure_count / len(trials)
        first = trials[0]
        summaries.append(
            {
                "branch_key": branch_key,
                "aliases": list(first["branch_alias_tuple"]),
                "branch_mode": str(first["branch_mode"]),
                "candidate_id": first["candidate_id"],
                "candidate_rank": first["candidate_rank"],
                "candidate_size": (
                    int(first["candidate_size"]) if fixed_neighborhood else None
                ),
                "neighborhood_key": (
                    str(first["neighborhood_key"]) if fixed_neighborhood else None
                ),
                "unique_actual_neighborhood_count": len(
                    {str(row["neighborhood_key"]) for row in trials}
                ),
                "trial_count": len(trials),
                "distinct_pp_attempt_count": len(attempt_keys),
                "escape_count": escape_count,
                "escape_fraction": escape_fraction,
                "stable_escape": escape_fraction >= stable_fraction - 1e-12,
                "stable_failure": failure_fraction >= stable_fraction - 1e-12,
                "pp_order_sensitive": bool(escape_count and failure_count),
                "outcome_counts": outcomes,
                "mean_conflict_delta": sum(
                    int(row["conflict_delta"]) for row in trials
                )
                / len(trials),
                "mean_total_decision_seconds": sum(
                    float(row["total_decision_seconds"]) for row in trials
                )
                / len(trials),
            }
        )
    if len(trial_index_sets) != 1:
        raise ValueError("stall Oracle branches do not cover the same paired trials")

    rank1 = [row for row in summaries if "rank1" in row["aliases"]]
    if len(rank1) != 1 or rank1[0]["branch_mode"] != "explicit_neighborhood":
        raise ValueError("stall Oracle source does not identify exactly one v2 rank1")
    selected = rank1[0]
    explicit = [
        row for row in summaries if row["branch_mode"] == "explicit_neighborhood"
    ]
    alternatives = [
        row
        for row in explicit
        if row["branch_key"] != selected["branch_key"] and row["stable_escape"]
    ]
    all_explicit_stable_fail = bool(explicit) and all(
        bool(row["stable_failure"]) for row in explicit
    )
    if not bool(selected["stable_failure"]):
        classification = "no_confirmed_v2_failure"
    elif alternatives:
        classification = "selector_failure"
    elif all_explicit_stable_fail:
        classification = "candidate_pool_failure"
    else:
        classification = "inconclusive"
    first = validated[0]
    return {
        "schema": STALL_ORACLE_SCHEMA,
        "schema_version": STALL_ORACLE_VERSION,
        "run_fingerprint": str(first["run_fingerprint"]),
        "before_fingerprint": str(first["before_fingerprint"]),
        "before_repair_fingerprint": str(first["before_repair_fingerprint"]),
        "minimum_trials": minimum_trials,
        "stable_fraction": stable_fraction,
        "trial_count": len(validated),
        "paired_trial_indexes": sorted(next(iter(trial_index_sets))),
        "arm_failure_count": sum(not row["escaped_state"] for row in validated),
        "neighborhood_failure_count": sum(
            bool(row["stable_failure"]) for row in explicit
        ),
        "pp_order_sensitive_neighborhood_count": sum(
            bool(row["pp_order_sensitive"]) for row in explicit
        ),
        "classification": classification,
        "selector_failure": classification == "selector_failure",
        "candidate_pool_failure": classification == "candidate_pool_failure",
        "repairer_failure": False,
        "repairer_failure_evaluated": False,
        "repairer_failure_status": "not_evaluated_by_pp_only_probe",
        "selected_v2_branch": selected,
        "stable_alternatives": alternatives,
        "branches": summaries,
    }


def _validate_probe_binding(
    probe_report: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    if str(probe_report.get("schema")) != STALLED_STATE_PROBE_SCHEMA or _object_int(
        probe_report.get("schema_version"), field="probe schema_version"
    ) != STALLED_STATE_PROBE_VERSION:
        raise ValueError("stall Oracle requires a current v2 probe report")
    if probe_report.get("all_candidates") is not True:
        raise ValueError("stall Oracle requires a probe generated with --all-candidates")
    if str(probe_report.get("trial_state_restore")) != TRIAL_STATE_RESTORE:
        raise ValueError("stall Oracle probe uses an unsupported trial restore")
    reproduction = probe_report.get("source_reproduction")
    if not isinstance(reproduction, dict) or not reproduction or any(
        value is not True for value in reproduction.values()
    ):
        raise ValueError("stall Oracle source reproduction is incomplete")
    run_fingerprint = str(probe_report.get("run_fingerprint") or "")
    before_fingerprint = str(probe_report.get("before_fingerprint") or "")
    before_repair_fingerprint = str(
        probe_report.get("before_repair_fingerprint") or ""
    )
    if not all((run_fingerprint, before_fingerprint, before_repair_fingerprint)):
        raise ValueError("stall Oracle probe report lacks state identity")
    producer = probe_report.get("producer_identity")
    if not isinstance(producer, dict) or not producer or _fingerprint(producer) != str(
        probe_report.get("producer_identity_fingerprint") or ""
    ):
        raise ValueError("stall Oracle probe producer identity is invalid")
    if any(str(row.get("run_fingerprint")) != run_fingerprint for row in rows):
        raise ValueError("stall Oracle rows differ from the probe run fingerprint")
    if any(str(row.get("before_fingerprint")) != before_fingerprint for row in rows):
        raise ValueError("stall Oracle rows differ from the probe state fingerprint")
    if any(
        str(row.get("before_repair_fingerprint")) != before_repair_fingerprint
        for row in rows
    ):
        raise ValueError("stall Oracle rows differ from the probe repair state")

    pool = probe_report.get("candidate_pool")
    if not isinstance(pool, list) or not pool:
        raise ValueError("stall Oracle probe report lacks its candidate pool")
    if _object_int(
        probe_report.get("candidate_count"), field="probe candidate_count", minimum=1
    ) != len(pool):
        raise ValueError("stall Oracle candidate count is inconsistent")
    normalized_pool: list[dict[str, Any]] = []
    for candidate in pool:
        if not isinstance(candidate, dict):
            raise ValueError("stall Oracle candidate pool contains a non-object")
        agents = _json_ints(candidate.get("agents"), field="candidate_pool.agents")
        raw_score = candidate.get("score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError("stall Oracle candidate pool score is not numeric")
        score = float(raw_score)
        if (
            not agents
            or _object_int(
                candidate.get("actual_size"),
                field="candidate_pool.actual_size",
                minimum=1,
            )
            != len(agents)
            or not math.isfinite(score)
            or not str(candidate.get("candidate_id") or "")
        ):
            raise ValueError("stall Oracle candidate pool is invalid")
        normalized_pool.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "agents": sorted(agents),
                "actual_size": len(agents),
                "score": round(score, 12),
            }
        )
    normalized_pool.sort(key=lambda candidate: candidate["candidate_id"])
    if len({candidate["candidate_id"] for candidate in normalized_pool}) != len(
        normalized_pool
    ):
        raise ValueError("stall Oracle candidate pool repeats a candidate id")
    if _fingerprint(normalized_pool) != str(
        probe_report.get("candidate_pool_fingerprint") or ""
    ):
        raise ValueError("stall Oracle candidate pool fingerprint mismatch")

    raw_aliases = probe_report.get("branch_aliases")
    if not isinstance(raw_aliases, dict):
        raise ValueError("stall Oracle probe report lacks branch mapping")
    aliases = {str(alias): str(key) for alias, key in raw_aliases.items()}
    expected_aliases = {"rank1", "official_adaptive"} | {
        f"rank_{rank}" for rank in range(2, len(pool) + 1)
    }
    if set(aliases) != expected_aliases or any(not key for key in aliases.values()):
        raise ValueError("stall Oracle branch mapping does not cover the full pool")
    if aliases["official_adaptive"] != "official_adaptive":
        raise ValueError("stall Oracle official branch mapping is invalid")
    expected_branch_keys = set(aliases.values())
    if _object_int(
        probe_report.get("unique_branch_count"),
        field="probe unique_branch_count",
        minimum=1,
    ) != len(
        expected_branch_keys
    ):
        raise ValueError("stall Oracle unique branch count is inconsistent")
    trials_per_branch = _object_int(
        probe_report.get("trials_per_branch"),
        field="probe trials_per_branch",
        minimum=1,
    )
    if len(rows) != len(expected_branch_keys) * trials_per_branch:
        raise ValueError("stall Oracle trial row coverage is incomplete")
    actual_branch_keys = {str(row.get("branch_key") or "") for row in rows}
    if actual_branch_keys != expected_branch_keys:
        raise ValueError("stall Oracle trial branches differ from the probe report")

    ranked = sorted(
        pool,
        key=lambda candidate: (
            -round(float(candidate["score"]), 12),
            str(candidate["candidate_id"]),
        ),
    )
    for branch_key in expected_branch_keys:
        branch_rows = [row for row in rows if str(row.get("branch_key")) == branch_key]
        if {
            _strict_int(row.get("trial_index"), field="trial_index")
            for row in branch_rows
        } != set(range(trials_per_branch)):
            raise ValueError("stall Oracle branch trial coverage is incomplete")
        reported_aliases = {
            alias for alias, key in aliases.items() if key == branch_key
        }
        row_aliases = {
            alias
            for alias in str(branch_rows[0].get("branch_aliases") or "").split(",")
            if alias
        }
        if row_aliases != reported_aliases:
            raise ValueError("stall Oracle branch aliases differ across artifacts")
        if branch_key == aliases["official_adaptive"]:
            if str(branch_rows[0].get("branch_mode")) != "official":
                raise ValueError("stall Oracle official branch metadata is invalid")
            continue
        ranks = sorted(
            1 if alias == "rank1" else int(alias.removeprefix("rank_"))
            for alias in reported_aliases
        )
        expected = ranked[ranks[0] - 1]
        if (
            str(branch_rows[0].get("branch_mode")) != "explicit_neighborhood"
            or str(branch_rows[0].get("candidate_id"))
            != str(expected["candidate_id"])
            or _strict_int(
                branch_rows[0].get("candidate_rank"), field="candidate_rank", minimum=1
            )
            != ranks[0]
            or _json_ints(
                branch_rows[0].get("candidate_agents"), field="candidate_agents"
            )
            != sorted(map(int, expected["agents"]))
        ):
            raise ValueError("stall Oracle candidate branch differs from its pool")


def _validate_runner_binding(
    source_root: Path, probe_report: dict[str, Any]
) -> tuple[Path, Path]:
    runner_path = source_root / "runner_config.json"
    reproduction_path = source_root / "source_reproduction.json"
    runner = _read_json(runner_path)
    if (
        str(runner.get("schema")) != RUNNER_CONFIG_SCHEMA
        or _object_int(runner.get("schema_version"), field="runner schema_version")
        != 1
        or not isinstance(runner.get("identity"), dict)
    ):
        raise ValueError("stall Oracle probe runner identity is invalid")
    identity = dict(runner["identity"])
    fingerprint = runner_identity_fingerprint(identity)
    if (
        str(runner.get("identity_fingerprint") or "") != fingerprint
        or str(probe_report.get("run_fingerprint") or "") != fingerprint
    ):
        raise ValueError("stall Oracle probe runner fingerprint mismatch")
    expected_report_fields = {
        "schema": "schema",
        "schema_version": "schema_version",
        "candidate_pool_fingerprint": "candidate_pool_fingerprint",
        "branch_aliases": "branch_aliases",
        "task_id": "task_id",
        "solver_seed": "solver_seed",
        "decision_index": "decision_index",
        "trials": "trials_per_branch",
        "all_candidates": "all_candidates",
        "trial_state_restore": "trial_state_restore",
        "producer_identity": "producer_identity",
        "producer_identity_fingerprint": "producer_identity_fingerprint",
    }
    for identity_field, report_field in expected_report_fields.items():
        if identity.get(identity_field) != probe_report.get(report_field):
            raise ValueError(
                f"stall Oracle probe runner/report mismatch: {identity_field}"
            )
    reproduction = _read_json(reproduction_path)
    if reproduction != probe_report.get("source_reproduction"):
        raise ValueError("stall Oracle source reproduction artifacts differ")
    return runner_path, reproduction_path


def audit_stall_probe(
    source: str | Path,
    output: str | Path,
    *,
    minimum_trials: int = 4,
    stable_fraction: float = 0.75,
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    report_path = source_root / "stalled_state_probe_report.json"
    trials_path = source_root / "stalled_state_trials.csv"
    probe_report = _read_json(report_path)
    rows = read_probe_trials(trials_path)
    _validate_probe_binding(probe_report, rows)
    runner_path, reproduction_path = _validate_runner_binding(
        source_root, probe_report
    )
    report = classify_stall_oracle_trials(
        rows,
        minimum_trials=minimum_trials,
        stable_fraction=stable_fraction,
    )
    if report["run_fingerprint"] != str(probe_report["run_fingerprint"]):
        raise ValueError("stall Oracle classification lost its probe binding")
    report.update(
        {
            "source_probe": str(source_root),
            "source_probe_report_sha256": sha256_file(report_path),
            "source_probe_trials_sha256": sha256_file(trials_path),
            "source_probe_runner_sha256": sha256_file(runner_path),
            "source_reproduction_sha256": sha256_file(reproduction_path),
            "source_candidate_pool_fingerprint": str(
                probe_report["candidate_pool_fingerprint"]
            ),
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "stall_oracle_report.json", report)
    lines = [
        "# v2 stalled-state Oracle audit",
        "",
        f"- Classification: `{report['classification']}`.",
        f"- Trials: `{report['trial_count']}`; arm failures: `{report['arm_failure_count']}`.",
        f"- Stable neighborhood failures: `{report['neighborhood_failure_count']}`.",
        f"- PP-order-sensitive neighborhoods: `{report['pp_order_sensitive_neighborhood_count']}`.",
        f"- Repairer failure evaluated: `{report['repairer_failure_evaluated']}`.",
        "",
        "This is a full-pool, same-state, paired one-repair Oracle diagnostic. It is not an end-to-end solver promotion result.",
        "",
    ]
    (output_root / "stall_oracle_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = [
    "STALL_ORACLE_SCHEMA",
    "STALL_ORACLE_VERSION",
    "audit_stall_probe",
    "classify_stall_oracle_trials",
    "read_probe_trials",
]
