from __future__ import annotations

import collections
import csv
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file
from experiments.repair_aware import classify_repair_outcome
from experiments.repair_collection import (
    _fingerprint,
    _low_level_delta,
    _plain,
    _read_jsonl,
    _write_json,
    state_fingerprint,
)
from experiments.stall_guard import repair_structure_fingerprint
from experiments.trace_replay import replay_prefix
from experiments.v3_s3_collection import (
    _paired_repair_action,
    _paired_seed,
    _source_replay_job,
)


V3_VALUE_PILOT_SCHEMA = "lns2.v3_value_label_pilot.v1"
DEFAULT_OVERHEAD_GRID = (0.0, 0.05, 0.10, 0.15)
ARM_PRIORITY = (
    "v2_full",
    "model_s3",
    "oracle_s3_efficiency",
    "oracle_s3_quality_time",
    "oracle_h1_efficiency",
)


def _atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path.name}")
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


def _candidate_from_sequence(
    payload: dict[str, Any], sequence_id: str
) -> dict[str, Any]:
    matches = [
        row
        for row in payload["trials"]
        if str(row["sequence_id"]) == str(sequence_id)
    ]
    if not matches:
        raise ValueError(f"state lacks S3 sequence: {sequence_id}")
    identities = set()
    template_keys = set()
    for row in matches:
        steps = [
            dict(step)
            for step in row["steps"]
            if int(step["step"]) == 1 and bool(step.get("executed"))
        ]
        if len(steps) != 1:
            raise ValueError(f"S3 sequence has invalid first-step coverage: {sequence_id}")
        step = steps[0]
        identities.add(
            (
                str(step["candidate_id"]),
                tuple(sorted(map(int, step["agents"]))),
            )
        )
        template_keys.add(str(dict(row["templates"][0])["template_key"]))
    if len(identities) != 1 or len(template_keys) != 1:
        raise ValueError(f"S3 sequence first action is not deterministic: {sequence_id}")
    candidate_id, agents = identities.pop()
    return {
        "candidate_id": candidate_id,
        "agents": list(agents),
        "template_key": template_keys.pop(),
        "sequence_id": str(sequence_id),
    }


def _candidate_from_template(
    payload: dict[str, Any], template_key: str
) -> dict[str, Any]:
    candidates = {}
    for row in payload["trials"]:
        if str(dict(row["templates"][0])["template_key"]) != str(template_key):
            continue
        steps = [
            dict(step)
            for step in row["steps"]
            if int(step["step"]) == 1 and bool(step.get("executed"))
        ]
        if len(steps) != 1:
            continue
        step = steps[0]
        key = (
            str(step["candidate_id"]),
            tuple(sorted(map(int, step["agents"]))),
        )
        candidates[key] = {
            "candidate_id": key[0],
            "agents": list(key[1]),
            "template_key": str(template_key),
            "sequence_id": "",
        }
    if len(candidates) != 1:
        raise ValueError(
            f"state template does not map to one first action: {template_key}"
        )
    return next(iter(candidates.values()))


def _v2_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in payload["external_baselines"]
        if str(row["controller"]) == "v2-full"
    ]
    identities = set()
    for row in rows:
        steps = [dict(step) for step in row["steps"] if int(step["step"]) == 1]
        if len(steps) != 1:
            raise ValueError("v2 baseline has invalid first-step coverage")
        step = steps[0]
        action = dict(step["action"])
        identities.add(
            (
                str(step["candidate_id"]),
                tuple(sorted(map(int, action["agents"]))),
            )
        )
    if len(identities) != 1:
        raise ValueError("v2 baseline first action is not deterministic")
    candidate_id, agents = identities.pop()
    return {
        "candidate_id": candidate_id,
        "agents": list(agents),
        "template_key": "",
        "sequence_id": "",
    }


def _shared_initial_selection_seconds(payload: dict[str, Any]) -> float:
    values = []
    for row in payload["trials"]:
        for step in row["steps"]:
            if int(step["step"]) == 1 and bool(step.get("executed")):
                values.append(float(step.get("selection_seconds", 0.0)))
    if not values:
        raise ValueError("state has no measured initial candidate selection time")
    return statistics.median(values)


def build_state_arms(
    payload: dict[str, Any], oracle_row: dict[str, Any]
) -> list[dict[str, Any]]:
    requested = [
        ("v2_full", _v2_candidate(payload)),
        (
            "model_s3",
            _candidate_from_sequence(payload, str(oracle_row["model_sequence_id"])),
        ),
        (
            "oracle_s3_efficiency",
            _candidate_from_sequence(
                payload,
                str(oracle_row["oracle_s3_efficiency_sequence_id"]),
            ),
        ),
        (
            "oracle_s3_quality_time",
            _candidate_from_sequence(
                payload,
                str(oracle_row["oracle_s3_quality_time_sequence_id"]),
            ),
        ),
        (
            "oracle_h1_efficiency",
            _candidate_from_template(
                payload,
                str(oracle_row["oracle_h1_efficiency_first_template"]),
            ),
        ),
    ]
    by_agents: dict[tuple[int, ...], dict[str, Any]] = {}
    for arm_id, candidate in requested:
        agents = tuple(sorted(map(int, candidate["agents"])))
        existing = by_agents.get(agents)
        if existing is None:
            by_agents[agents] = {
                "arm_id": arm_id,
                "aliases": [arm_id],
                **candidate,
            }
        else:
            existing["aliases"].append(arm_id)
    order = {name: index for index, name in enumerate(ARM_PRIORITY)}
    result = sorted(
        by_agents.values(),
        key=lambda row: (
            min(order[name] for name in row["aliases"]),
            tuple(row["agents"]),
        ),
    )
    for row in result:
        row["aliases"] = sorted(row["aliases"], key=order.get)
        row["arm_id"] = row["aliases"][0]
    return result


def _balanced_state_sample(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if int(count) <= 0:
        raise ValueError("state_count must be positive")
    cells: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(
        list
    )
    for row in rows:
        cells[(str(row["layout_mode"]), int(row["agent_count"]))].append(row)
    for values in cells.values():
        values.sort(
            key=lambda row: (
                -int(row["initial_conflicts"]),
                _fingerprint(str(row["state_id"])),
            )
        )
    selected = []
    offset = 0
    while len(selected) < int(count):
        added = False
        for cell in sorted(cells):
            values = cells[cell]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) >= int(count):
                    break
        if not added:
            break
        offset += 1
    return selected


def build_value_pilot_plan(
    *,
    source: str | Path,
    oracle_state_comparison: str | Path,
    state_count: int,
    split: str = "policy_train",
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    oracle_path = Path(oracle_state_comparison).resolve()
    oracle_rows = {
        str(row["state_id"]): row
        for row in csv.DictReader(oracle_path.open(encoding="utf-8", newline=""))
        if str(row["split"]) == str(split)
    }
    decisions = {
        str(row["state_id"]): row
        for row in _read_jsonl(source_root / "collection" / "state_selection.jsonl")
        if str(row["split"]) == str(split)
    }
    candidates = []
    state_files = sorted(
        (source_root / "collection" / "states" / split).glob("*.json")
    )
    for path in state_files:
        payload = dict(read_json(path))
        state_id = str(payload["state_id"])
        if state_id not in oracle_rows or state_id not in decisions:
            continue
        arms = build_state_arms(payload, oracle_rows[state_id])
        if len(arms) < 2:
            continue
        first_trial = dict(payload["trials"][0])
        initial_conflicts = int(first_trial["conflict_trajectory"][0])
        if initial_conflicts <= 0:
            continue
        decision = dict(decisions[state_id])
        candidates.append(
            {
                "state_id": state_id,
                "state_file": str(path),
                "split": split,
                "map_id": str(decision["map_id"]),
                "layout_mode": str(decision["layout_mode"]),
                "agent_count": int(decision["agent_count"]),
                "source_stratum": str(decision["source_stratum"]),
                "initial_conflicts": initial_conflicts,
                "before_fingerprint": str(decision["before_fingerprint"]),
                "before_repair_fingerprint": str(
                    decision["before_repair_fingerprint"]
                ),
                "shared_initial_selection_seconds": (
                    _shared_initial_selection_seconds(payload)
                ),
                "decision": decision,
                "arms": arms,
            }
        )
    selected = _balanced_state_sample(candidates, int(state_count))
    if len(selected) != int(state_count):
        raise ValueError(
            f"value pilot could select only {len(selected)}/{int(state_count)} states"
        )
    return {
        "schema": V3_VALUE_PILOT_SCHEMA,
        "source": str(source_root),
        "source_sequence_trials_sha256": sha256_file(
            source_root / "collection" / "sequence_trials.jsonl"
        ),
        "oracle_state_comparison": str(oracle_path),
        "oracle_state_comparison_sha256": sha256_file(oracle_path),
        "split": split,
        "requested_state_count": int(state_count),
        "selected_state_count": len(selected),
        "states": selected,
    }


def _rollout_file_name(state_id: str, arm_id: str, trial_index: int) -> str:
    digest = _fingerprint(
        {
            "state_id": state_id,
            "arm_id": arm_id,
            "trial_index": int(trial_index),
        }
    )[:20]
    return f"{digest}.json"


def _extend_replay_repair_budget(
    replay: dict[str, Any], *, prefix_length: int, max_repairs: int
) -> dict[str, Any]:
    prepared = dict(replay)
    environment = dict(prepared["environment"])
    environment["max_repair_iterations"] = max(
        int(environment.get("max_repair_iterations", 0)),
        int(prefix_length) + int(max_repairs),
    )
    prepared["environment"] = environment
    return prepared


def run_value_rollout(job: dict[str, Any]) -> dict[str, Any]:
    state_row = dict(job["state"])
    arm = dict(job["arm"])
    trial_index = int(job["trial_index"])
    decision = dict(state_row["decision"])
    replay, _configuration = _source_replay_job(decision)
    replay = _extend_replay_repair_budget(
        replay,
        prefix_length=len(decision["prefix_actions"]),
        max_repairs=int(job["max_repairs"]),
    )
    environment, state = replay_prefix(replay, decision["prefix_actions"])
    if state_fingerprint(state) != str(state_row["before_fingerprint"]):
        raise RuntimeError("value pilot replay fingerprint mismatch")
    if repair_structure_fingerprint(state) != str(
        state_row["before_repair_fingerprint"]
    ):
        raise RuntimeError("value pilot repair fingerprint mismatch")

    initial_repair = repair_structure_fingerprint(state)
    initial_conflicts = int(state["num_of_colliding_pairs"])
    trajectory = [initial_conflicts]
    steps = []
    rollout_started = time.perf_counter()
    total_pp_seconds = 0.0
    total_low_level = collections.Counter()
    stop_reason = "repair_limit"
    for offset in range(int(job["max_repairs"])):
        before = state
        before_repair = repair_structure_fingerprint(before)
        conflicts_before = int(before["num_of_colliding_pairs"])
        seed = _paired_seed(initial_repair, trial_index, offset + 1)
        if offset == 0:
            action = _paired_repair_action(
                "explicit_neighborhood",
                agents=arm["agents"],
                random_seed=seed,
            )
            route = "explicit_first_action"
        else:
            action = _paired_repair_action("official", random_seed=seed)
            route = "official_adaptive_continuation"
        started = time.perf_counter()
        transition = _plain(environment.step(action))
        repair_seconds = time.perf_counter() - started
        state = dict(transition["observation"])
        metrics = dict(transition["metrics"])
        conflicts_after = int(state["num_of_colliding_pairs"])
        after_repair = repair_structure_fingerprint(state)
        outcome = classify_repair_outcome(
            before_fingerprint=before_repair,
            after_fingerprint=after_repair,
            replan_success=bool(metrics.get("replan_success")),
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=bool(state.get("feasible")),
        )
        low_level = _low_level_delta(before, state)
        for name in ("generated", "expanded", "reopened", "runs"):
            total_low_level[name] += int(low_level.get(name, 0))
        pp_seconds = max(
            0.0, float(metrics.get("pp_replan_seconds", repair_seconds))
        )
        total_pp_seconds += pp_seconds
        trajectory.append(conflicts_after)
        steps.append(
            {
                "step": offset + 1,
                "route": route,
                "action": action,
                "repair_outcome": outcome,
                "conflicts_before": conflicts_before,
                "conflicts_after": conflicts_after,
                "conflict_reduction": conflicts_before - conflicts_after,
                "repair_seconds": repair_seconds,
                "pp_replan_seconds": pp_seconds,
                "before_fingerprint": state_fingerprint(before),
                "after_fingerprint": state_fingerprint(state),
                "before_repair_fingerprint": before_repair,
                "after_repair_fingerprint": after_repair,
            }
        )
        if bool(state.get("feasible")):
            stop_reason = "feasible"
            break
        if bool(state.get("done")):
            stop_reason = "environment_terminal"
            break
        if time.perf_counter() - rollout_started >= float(
            job["wall_clock_seconds"]
        ):
            stop_reason = "wall_clock_limit"
            break
    rollout_wall = time.perf_counter() - rollout_started
    conflict_auc = math.fsum(
        float(step["conflicts_before"]) * float(step["repair_seconds"])
        for step in steps
    )
    shared_selection = float(state_row["shared_initial_selection_seconds"])
    return {
        "schema": V3_VALUE_PILOT_SCHEMA,
        "state_id": str(state_row["state_id"]),
        "split": str(state_row["split"]),
        "map_id": str(state_row["map_id"]),
        "layout_mode": str(state_row["layout_mode"]),
        "agent_count": int(state_row["agent_count"]),
        "source_stratum": str(state_row["source_stratum"]),
        "arm_id": str(arm["arm_id"]),
        "arm_aliases": list(arm["aliases"]),
        "candidate_id": str(arm["candidate_id"]),
        "agents": list(map(int, arm["agents"])),
        "actual_size": len(arm["agents"]),
        "template_key": str(arm.get("template_key", "")),
        "trial_index": trial_index,
        "initial_fingerprint": str(state_row["before_fingerprint"]),
        "initial_repair_fingerprint": initial_repair,
        "final_fingerprint": state_fingerprint(state),
        "final_repair_fingerprint": repair_structure_fingerprint(state),
        "initial_conflicts": initial_conflicts,
        "final_conflicts": int(state["num_of_colliding_pairs"]),
        "conflict_trajectory": trajectory,
        "conflict_reduction": initial_conflicts
        - int(state["num_of_colliding_pairs"]),
        "steps": steps,
        "repair_iterations": len(steps),
        "continuation_iterations": max(0, len(steps) - 1),
        "feasible": bool(state.get("feasible")),
        "censored": not bool(state.get("feasible")),
        "stop_reason": stop_reason,
        "shared_initial_selection_seconds": shared_selection,
        "rollout_wall_seconds": rollout_wall,
        "observed_total_seconds": shared_selection + rollout_wall,
        "pp_replan_seconds": total_pp_seconds,
        "conflict_auc_seconds": conflict_auc,
        "normalized_conflict_auc_seconds": conflict_auc
        / max(1, initial_conflicts),
        "low_level": {
            name: int(total_low_level[name])
            for name in ("generated", "expanded", "reopened", "runs")
        },
        "complete": True,
    }


def _rollout_flat(row: dict[str, Any]) -> dict[str, Any]:
    low_level = dict(row["low_level"])
    return {
        name: value
        for name, value in row.items()
        if name not in {"steps", "low_level", "arm_aliases", "agents", "conflict_trajectory"}
    } | {
        "arm_aliases": ",".join(map(str, row["arm_aliases"])),
        "agents": ",".join(map(str, row["agents"])),
        "conflict_trajectory": ",".join(map(str, row["conflict_trajectory"])),
        **{f"low_level_{name}": value for name, value in low_level.items()},
    }


def _winner_key(row: dict[str, Any], overhead: float) -> tuple[Any, ...]:
    adjusted = float(row["observed_total_seconds"]) + float(overhead) * int(
        row["continuation_iterations"]
    )
    return (
        not bool(row["feasible"]),
        (
            adjusted
            if bool(row["feasible"])
            else float(row["final_conflicts"])
            / max(1.0, float(row["initial_conflicts"]))
        ),
        float(row["normalized_conflict_auc_seconds"]),
        adjusted,
        str(row["arm_id"]),
    )


def analyze_value_rollouts(
    rows: list[dict[str, Any]],
    *,
    expected_jobs: int,
    smoke_only: bool,
    overhead_grid: Iterable[float] = DEFAULT_OVERHEAD_GRID,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) != int(expected_jobs):
        raise ValueError(
            f"value pilot rollout coverage mismatch: {len(rows)}/{expected_jobs}"
        )
    keys = [
        (str(row["state_id"]), str(row["arm_id"]), int(row["trial_index"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("value pilot contains duplicate rollout keys")
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)

    sensitivity_rows = []
    agreement_by_overhead = {}
    for overhead in map(float, overhead_grid):
        winners: dict[tuple[str, int], str] = {}
        for state_id, state_rows in sorted(grouped.items()):
            trials = sorted({int(row["trial_index"]) for row in state_rows})
            for trial in trials:
                subset = [
                    row
                    for row in state_rows
                    if int(row["trial_index"]) == trial
                ]
                winner = min(subset, key=lambda row: _winner_key(row, overhead))
                winners[(state_id, trial)] = str(winner["arm_id"])
                sensitivity_rows.append(
                    {
                        "state_id": state_id,
                        "trial_index": trial,
                        "assumed_continuation_selection_seconds": overhead,
                        "winner_arm_id": str(winner["arm_id"]),
                        "winner_feasible": bool(winner["feasible"]),
                        "winner_final_conflicts": int(winner["final_conflicts"]),
                        "winner_adjusted_seconds": float(
                            winner["observed_total_seconds"]
                        )
                        + overhead * int(winner["continuation_iterations"]),
                    }
                )
        state_agreement = []
        for state_id in sorted(grouped):
            values = [
                arm
                for (candidate_state, _trial), arm in winners.items()
                if candidate_state == state_id
            ]
            state_agreement.append(len(set(values)) == 1)
        agreement_by_overhead[str(overhead)] = statistics.fmean(
            map(float, state_agreement)
        )

    state_diagnostics = []
    for state_id, state_rows in sorted(grouped.items()):
        arm_ids = sorted({str(row["arm_id"]) for row in state_rows})
        feasible_values = {bool(row["feasible"]) for row in state_rows}
        final_ratios = [
            float(row["final_conflicts"])
            / max(1.0, float(row["initial_conflicts"]))
            for row in state_rows
        ]
        auc_values = [
            float(row["normalized_conflict_auc_seconds"]) for row in state_rows
        ]
        feasible_times = [
            float(row["observed_total_seconds"])
            for row in state_rows
            if bool(row["feasible"])
        ]
        time_spread = (
            (max(feasible_times) - min(feasible_times))
            / max(1e-12, min(feasible_times))
            if len(feasible_times) >= 2
            else 0.0
        )
        auc_spread = (
            (max(auc_values) - min(auc_values))
            / max(1e-12, min(auc_values))
            if len(auc_values) >= 2
            else 0.0
        )
        action_sensitive = (
            len(feasible_values) > 1
            or max(final_ratios) - min(final_ratios) >= 0.05
            or time_spread >= 0.10
            or auc_spread >= 0.10
        )
        state_diagnostics.append(
            {
                "state_id": state_id,
                "map_id": str(state_rows[0]["map_id"]),
                "layout_mode": str(state_rows[0]["layout_mode"]),
                "agent_count": int(state_rows[0]["agent_count"]),
                "arm_count": len(arm_ids),
                "trial_count": len(
                    {int(row["trial_index"]) for row in state_rows}
                ),
                "action_sensitive": action_sensitive,
                "feasible_outcome_differs": len(feasible_values) > 1,
                "final_conflict_ratio_spread": max(final_ratios)
                - min(final_ratios),
                "feasible_time_spread_fraction": time_spread,
                "auc_spread_fraction": auc_spread,
            }
        )
    action_sensitive_fraction = statistics.fmean(
        float(row["action_sensitive"]) for row in state_diagnostics
    )
    uncensored_fraction = statistics.fmean(
        float(not bool(row["censored"])) for row in rows
    )
    minimum_agreement = min(agreement_by_overhead.values())
    checks = {
        "coverage_complete": len(rows) == int(expected_jobs),
        "at_least_two_actions_per_state": all(
            int(row["arm_count"]) >= 2 for row in state_diagnostics
        ),
        "action_sensitive_state_fraction_at_least_50pct": (
            action_sensitive_fraction >= 0.50
        ),
        "uncensored_branch_fraction_at_least_50pct": (
            uncensored_fraction >= 0.50
        ),
        "winner_seed_agreement_at_least_50pct": minimum_agreement >= 0.50,
    }
    decision = (
        "smoke_completed_not_scientific"
        if smoke_only
        else (
            "cost_to_go_labels_promising"
            if all(checks.values())
            else "cost_to_go_labels_insufficient"
        )
    )
    report = {
        "schema": V3_VALUE_PILOT_SCHEMA,
        "decision": decision,
        "smoke_only": bool(smoke_only),
        "rollout_count": len(rows),
        "state_count": len(grouped),
        "action_sensitive_state_fraction": action_sensitive_fraction,
        "uncensored_branch_fraction": uncensored_fraction,
        "winner_seed_agreement_by_overhead": agreement_by_overhead,
        "checks": checks,
        "limitations": [
            "The continuation teacher is official Adaptive, so labels estimate Q under that teacher rather than an optimal or on-policy v3 continuation.",
            "Candidate arms were selected from existing v2, model-S3, and retrospective Oracle diagnostics; this pilot tests label signal and is not promotion evidence.",
            "Replay and prefix reconstruction time is excluded from cost-to-go labels.",
            "Censored branches require survival-aware handling before model training.",
        ],
    }
    return report, state_diagnostics, sensitivity_rows


def _report_markdown(report: dict[str, Any]) -> str:
    checks = dict(report["checks"])
    return "\n".join(
        [
            "# Variable-horizon cost-to-go label pilot",
            "",
            f"Decision: `{report['decision']}`",
            "",
            (
                f"- States: {int(report['state_count'])}; rollouts: "
                f"{int(report['rollout_count'])}."
            ),
            (
                "- Action-sensitive state fraction: "
                f"{float(report['action_sensitive_state_fraction']):.3%}."
            ),
            (
                "- Uncensored branch fraction: "
                f"{float(report['uncensored_branch_fraction']):.3%}."
            ),
            "",
            "## Checks",
            "",
            *[
                f"- {name}: `{str(bool(value)).lower()}`"
                for name, value in sorted(checks.items())
            ],
            "",
            "## Boundary",
            "",
            *[f"- {value}" for value in report["limitations"]],
            "",
        ]
    )


def run_value_label_pilot(
    *,
    source: str | Path,
    oracle_state_comparison: str | Path,
    output: str | Path,
    state_count: int,
    trials: int = 2,
    max_repairs: int = 30,
    wall_clock_seconds: float = 60.0,
    split: str = "policy_train",
    smoke_only: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if int(trials) <= 0 or int(max_repairs) <= 0:
        raise ValueError("trials and max_repairs must be positive")
    if float(wall_clock_seconds) <= 0.0:
        raise ValueError("wall_clock_seconds must be positive")
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not bool(resume):
        raise FileExistsError("value pilot output is non-empty; pass resume")
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "plan.json"
    requested_plan = build_value_pilot_plan(
        source=source,
        oracle_state_comparison=oracle_state_comparison,
        state_count=int(state_count),
        split=split,
    )
    config = {
        "schema": V3_VALUE_PILOT_SCHEMA,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "state_count": int(state_count),
        "trials": int(trials),
        "max_repairs": int(max_repairs),
        "wall_clock_seconds": float(wall_clock_seconds),
        "split": str(split),
        "smoke_only": bool(smoke_only),
        "plan_fingerprint": _fingerprint(requested_plan),
    }
    if plan_path.is_file():
        existing_plan = dict(read_json(plan_path))
        existing_config = dict(read_json(output_root / "run_config.json"))
        if _fingerprint(existing_plan) != _fingerprint(requested_plan):
            raise ValueError("value pilot resume plan fingerprint mismatch")
        if existing_config != config:
            raise ValueError("value pilot resume configuration mismatch")
    else:
        _write_json(plan_path, requested_plan)
        _write_json(output_root / "run_config.json", config)

    rollout_root = output_root / "rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for state in requested_plan["states"]:
        for arm in state["arms"]:
            for trial_index in range(int(trials)):
                jobs.append(
                    {
                        "state": state,
                        "arm": arm,
                        "trial_index": trial_index,
                        "max_repairs": int(max_repairs),
                        "wall_clock_seconds": float(wall_clock_seconds),
                    }
                )
    completed = []
    errors = []
    for index, job in enumerate(jobs):
        state_id = str(job["state"]["state_id"])
        arm_id = str(job["arm"]["arm_id"])
        trial_index = int(job["trial_index"])
        path = rollout_root / _rollout_file_name(
            state_id, arm_id, trial_index
        )
        try:
            if bool(resume) and path.is_file():
                row = dict(read_json(path))
                if (
                    str(row.get("state_id")) == state_id
                    and str(row.get("arm_id")) == arm_id
                    and int(row.get("trial_index", -1)) == trial_index
                    and bool(row.get("complete"))
                ):
                    completed.append(row)
                    continue
            row = run_value_rollout(job)
            partial = path.with_name(path.name + ".partial")
            _write_json(partial, row)
            os.replace(partial, path)
            completed.append(row)
        except Exception as error:
            errors.append(
                {
                    "state_id": state_id,
                    "arm_id": arm_id,
                    "trial_index": trial_index,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            break
        finally:
            _write_json(
                output_root / "status.json",
                {
                    "schema": V3_VALUE_PILOT_SCHEMA,
                    "status": "error" if errors else "running",
                    "completed_rollout_count": len(completed),
                    "total_rollout_count": len(jobs),
                    "error_count": len(errors),
                    "last_job_index": index,
                },
            )
    if errors:
        _write_json(output_root / "errors.json", {"errors": errors})
        raise RuntimeError(errors[0]["error"])
    (output_root / "errors.json").unlink(missing_ok=True)
    report, state_diagnostics, sensitivity = analyze_value_rollouts(
        completed,
        expected_jobs=len(jobs),
        smoke_only=bool(smoke_only),
    )
    report["run_config"] = config
    report["plan_coverage"] = {
        "selected_by_layout": dict(
            collections.Counter(
                str(row["layout_mode"]) for row in requested_plan["states"]
            )
        ),
        "selected_by_agent_count": dict(
            collections.Counter(
                int(row["agent_count"]) for row in requested_plan["states"]
            )
        ),
        "unique_map_count": len(
            {str(row["map_id"]) for row in requested_plan["states"]}
        ),
    }
    _atomic_write_csv(
        output_root / "value_rollouts.csv",
        [_rollout_flat(row) for row in completed],
    )
    _atomic_write_csv(
        output_root / "state_label_diagnostics.csv", state_diagnostics
    )
    _atomic_write_csv(
        output_root / "selection_overhead_sensitivity.csv", sensitivity
    )
    _write_json(output_root / "value_label_pilot_report.json", report)
    (output_root / "value_label_pilot_report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    _write_json(
        output_root / "status.json",
        {
            "schema": V3_VALUE_PILOT_SCHEMA,
            "status": "complete",
            "completed_rollout_count": len(completed),
            "total_rollout_count": len(jobs),
            "error_count": 0,
            "decision": str(report["decision"]),
        },
    )
    return report
