from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


SCHEMA = "lns2.stride.platform_action_robustness.v1"
EXPECTED_CASES = 45
EXPECTED_ACTIONS = 325
EXPECTED_EPISODES = 2600
EXPECTED_TRIALS = tuple(range(8))
DETERMINISTIC_ROLES = {
    "deterministic_compact_augment",
    "deterministic_same_size_exchange",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def _action_summary(
    rows: list[dict[str, Any]], reference_by_trial: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["trial_index"]))
    by_trial = {int(row["trial_index"]): row for row in ordered}
    roles = sorted(
        {
            str(role)
            for row in ordered
            for role in (row.get("logical_roles") or ())
        }
    )
    platform_count = sum(bool(row["entered_platform"]) for row in ordered)
    success_count = sum(bool(row["success"]) for row in ordered)
    half_deltas: list[float] = []
    for trials in (range(4), range(4, 8)):
        half_deltas.append(
            mean(
                float(bool(by_trial[trial]["entered_platform"]))
                - float(bool(reference_by_trial[trial]["entered_platform"]))
                for trial in trials
            )
        )
    reference_rows = [reference_by_trial[trial] for trial in EXPECTED_TRIALS]
    platform_delta = mean(
        float(bool(by_trial[trial]["entered_platform"]))
        - float(bool(reference_by_trial[trial]["entered_platform"]))
        for trial in EXPECTED_TRIALS
    )
    success_delta = mean(
        float(bool(by_trial[trial]["success"]))
        - float(bool(reference_by_trial[trial]["success"]))
        for trial in EXPECTED_TRIALS
    )
    auc_delta = _mean(ordered, "normalized_fixed_auc") - _mean(
        reference_rows, "normalized_fixed_auc"
    )
    restricted_decisions = mean(
        min(64, int(row["repair_iterations"])) for row in ordered
    )
    reference_restricted_decisions = mean(
        min(64, int(row["repair_iterations"])) for row in reference_rows
    )
    restricted_delta = restricted_decisions - reference_restricted_decisions
    stable = all(delta < 0.0 for delta in half_deltas)
    quality = (
        success_delta >= 0.0 and auc_delta <= 0.0 and restricted_delta <= 0.0
    )
    return {
        "candidate_id": str(ordered[0]["candidate_id"]),
        "candidate_size": int(ordered[0]["candidate_size"]),
        "logical_roles": roles,
        "frontier_variant": ordered[0].get("frontier_variant"),
        "trial_indices": [int(row["trial_index"]) for row in ordered],
        "platform_count": platform_count,
        "platform_rate": platform_count / len(ordered),
        "success_count": success_count,
        "success_rate": success_count / len(ordered),
        "mean_normalized_fixed_auc": _mean(ordered, "normalized_fixed_auc"),
        "mean_restricted_repair_decisions": restricted_decisions,
        "paired_platform_delta": platform_delta,
        "paired_platform_half_deltas": half_deltas,
        "paired_success_delta": success_delta,
        "paired_normalized_fixed_auc_delta": auc_delta,
        "paired_restricted_repair_decisions_delta": restricted_delta,
        "seed_half_stable_platform_improvement": stable,
        "quality_preserving": quality,
        "quality_preserving_stable_improvement": stable and quality,
        "mixed_platform_outcomes": 0 < platform_count < len(ordered),
    }


def analyze_rows(
    episodes: list[dict[str, Any]], causal_by_case: dict[str, str]
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in episodes:
        grouped[(str(row["case_id"]), str(row["candidate_id"]))].append(row)
    case_ids = sorted({case_id for case_id, _candidate_id in grouped})
    cases: list[dict[str, Any]] = []
    action_count = 0
    for case_id in case_ids:
        case_groups = {
            candidate_id: rows
            for (group_case_id, candidate_id), rows in grouped.items()
            if group_case_id == case_id
        }
        historical_ids = [
            candidate_id
            for candidate_id, rows in case_groups.items()
            if any(
                "historical_actual" in (row.get("logical_roles") or ())
                for row in rows
            )
        ]
        if len(historical_ids) != 1:
            raise ValueError(
                f"Case {case_id} has {len(historical_ids)} historical concrete actions"
            )
        historical_id = historical_ids[0]
        reference_by_trial = {
            int(row["trial_index"]): row for row in case_groups[historical_id]
        }
        action_summaries: list[dict[str, Any]] = []
        for candidate_id in sorted(case_groups):
            rows = case_groups[candidate_id]
            trials = sorted(int(row["trial_index"]) for row in rows)
            if trials != list(EXPECTED_TRIALS):
                raise ValueError(
                    f"Case {case_id} action {candidate_id} has trials {trials}"
                )
            action_summaries.append(_action_summary(rows, reference_by_trial))
        action_count += len(action_summaries)
        reference = next(
            row for row in action_summaries if row["candidate_id"] == historical_id
        )
        alternatives = [
            row for row in action_summaries if row["candidate_id"] != historical_id
        ]
        stable = [
            row
            for row in alternatives
            if row["seed_half_stable_platform_improvement"]
        ]
        qualified = [
            row for row in alternatives if row["quality_preserving_stable_improvement"]
        ]
        deterministic_qualified = [
            row
            for row in qualified
            if DETERMINISTIC_ROLES.intersection(row["logical_roles"])
        ]
        if int(reference["platform_count"]) == 0:
            classification = "reference_no_platform"
        elif deterministic_qualified:
            classification = "deterministic_rule_success"
        elif qualified:
            classification = "selection_headroom"
        elif stable:
            classification = "platform_only_headroom"
        elif any(
            int(row["platform_count"]) < int(reference["platform_count"])
            or bool(row["mixed_platform_outcomes"])
            for row in alternatives
        ):
            classification = "seed_unstable_headroom"
        else:
            classification = "pool_or_repairer_gap"
        first = next(iter(case_groups.values()))[0]
        cases.append(
            {
                "case_id": case_id,
                "map_id": str(first["map_id"]),
                "causal_classification": causal_by_case.get(case_id, "missing"),
                "classification": classification,
                "historical_candidate_id": historical_id,
                "historical_platform_count": int(reference["platform_count"]),
                "concrete_action_count": len(action_summaries),
                "stable_platform_alternative_count": len(stable),
                "quality_preserving_stable_alternative_count": len(qualified),
                "deterministic_quality_preserving_stable_count": len(
                    deterministic_qualified
                ),
                "stable_candidate_ids": [row["candidate_id"] for row in stable],
                "quality_preserving_stable_candidate_ids": [
                    row["candidate_id"] for row in qualified
                ],
                "actions": action_summaries,
            }
        )

    classification_counts = collections.Counter(
        str(row["classification"]) for row in cases
    )
    causal_counts = collections.Counter(str(row["causal_classification"]) for row in cases)
    cross_tab: dict[str, dict[str, int]] = {}
    for causal in sorted(causal_counts):
        cross_tab[causal] = dict(
            sorted(
                collections.Counter(
                    str(row["classification"])
                    for row in cases
                    if str(row["causal_classification"]) == causal
                ).items()
            )
        )
    map_tab: dict[str, dict[str, int]] = {}
    for map_id in sorted({str(row["map_id"]) for row in cases}):
        map_tab[map_id] = dict(
            sorted(
                collections.Counter(
                    str(row["classification"])
                    for row in cases
                    if str(row["map_id"]) == map_id
                ).items()
            )
        )
    integrity = {
        "case_count": len(cases) == EXPECTED_CASES,
        "unique_action_count": action_count == EXPECTED_ACTIONS,
        "episode_count": len(episodes) == EXPECTED_EPISODES,
        "all_trials_0_to_7": all(
            row["trial_indices"] == list(EXPECTED_TRIALS)
            for case in cases
            for row in case["actions"]
        ),
        "causal_annotation_complete": all(
            row["causal_classification"] != "missing" for row in cases
        ),
    }
    return {
        "schema": SCHEMA,
        "scientific_status": "completed_zero_solver_action_robustness_diagnostic",
        "claim_boundary": (
            "existing paired episode diagnosis only; no selector, runtime, or TTF claim"
        ),
        "case_count": len(cases),
        "unique_action_count": action_count,
        "episode_count": len(episodes),
        "classification_counts": dict(sorted(classification_counts.items())),
        "causal_classification_counts": dict(sorted(causal_counts.items())),
        "classification_by_causal_mechanism": cross_tab,
        "classification_by_map": map_tab,
        "cases_with_any_stable_platform_alternative": sum(
            int(row["stable_platform_alternative_count"] > 0) for row in cases
        ),
        "cases_with_quality_preserving_stable_alternative": sum(
            int(row["quality_preserving_stable_alternative_count"] > 0)
            for row in cases
        ),
        "cases_where_deterministic_rule_is_stable": sum(
            int(row["deterministic_quality_preserving_stable_count"] > 0)
            for row in cases
        ),
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "cases": cases,
    }


def analyze_action_robustness(
    frontier_report_path: str | Path,
    witness_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    frontier_report_path = Path(frontier_report_path).resolve()
    witness_path = Path(witness_path).resolve()
    output_path = Path(output_path).resolve()
    frontier = json.loads(frontier_report_path.read_text(encoding="utf-8"))
    if not frontier.get("integrity_passed"):
        raise ValueError("Frontier report integrity did not pass")
    witnesses = [
        json.loads(line)
        for line in witness_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    causal_by_case = {
        str(row["case_id"]): str(row["causal_intervention_classification"])
        for row in witnesses
    }
    report = analyze_rows(list(frontier.get("episodes") or ()), causal_by_case)
    report["artifact_sha256"] = {
        "frontier_report": _sha256(frontier_report_path),
        "platform_entry_witnesses": _sha256(witness_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["analyze_action_robustness", "analyze_rows"]
