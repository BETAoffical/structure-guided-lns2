from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .candidate_retrieval import actual_utility


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return sum(rows) / len(rows)


def _manifest_lookup(
    dataset: str | Path | None,
    split: str,
) -> dict[str, dict[str, Any]]:
    if dataset is None:
        return {}
    path = Path(dataset).resolve() / split / "manifest.jsonl"
    return {
        str(row["task_id"]): row
        for row in _read_jsonl(path)
    }


def _group_stats(
    cases: list[dict[str, Any]],
    key_name: str,
) -> list[dict[str, Any]]:
    grouped: collections.defaultdict[str, list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for case in cases:
        grouped[str(case.get(key_name, ""))].append(case)
    rows = []
    for key, values in sorted(grouped.items()):
        rows.append(
            {
                key_name: key,
                "case_count": len(values),
                "valid_probability": _mean(
                    float(
                        row["outcome"].get(
                            "valid_probability",
                            row["outcome"]["candidate_valid"],
                        )
                    )
                    for row in values
                ),
                "mean_utility": _mean(
                    actual_utility(row["outcome"]) for row in values
                ),
                "mean_conflict_reduction": _mean(
                    float(row["outcome"].get("conflict_reduction") or 0.0)
                    for row in values
                ),
                "mean_cost_improvement": _mean(
                    float(row["outcome"].get("cost_improvement") or 0.0)
                    for row in values
                ),
                "mean_runtime_ms": _mean(
                    float(row["outcome"]["total_runtime_ms"])
                    for row in values
                ),
            }
        )
    return rows


def _oracle_stats(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_state: collections.defaultdict[str, list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for case in cases:
        by_state[str(case["state_id"])].append(case)
    generator_wins: collections.Counter[str] = collections.Counter()
    candidate_wins: collections.Counter[int] = collections.Counter()
    better_than_baseline = 0
    baseline_is_oracle = 0
    for state_cases in by_state.values():
        by_index = {
            int(case["candidate_index"]): case for case in state_cases
        }
        baseline = by_index[0]
        baseline_utility = actual_utility(baseline["outcome"])
        oracle = max(
            state_cases,
            key=lambda case: (
                actual_utility(case["outcome"]),
                -int(case["candidate_index"]),
            ),
        )
        oracle_index = int(oracle["candidate_index"])
        generator_wins[str(oracle["generator"])] += 1
        candidate_wins[oracle_index] += 1
        if oracle_index == 0:
            baseline_is_oracle += 1
        if any(
            actual_utility(case["outcome"]) > baseline_utility
            for case in state_cases
            if int(case["candidate_index"]) != 0
        ):
            better_than_baseline += 1
    state_count = len(by_state)
    return {
        "state_count": state_count,
        "baseline_is_oracle_count": baseline_is_oracle,
        "baseline_is_oracle_ratio": baseline_is_oracle
        / max(1, state_count),
        "states_with_better_alternative": better_than_baseline,
        "states_with_better_alternative_ratio": better_than_baseline
        / max(1, state_count),
        "oracle_wins_by_generator": dict(sorted(generator_wins.items())),
        "oracle_wins_by_candidate_index": {
            str(key): value for key, value in sorted(candidate_wins.items())
        },
    }


def _order_noise(order_cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not order_cases:
        return {
            "candidate_with_multiple_orders": 0,
            "mean_conflict_reduction_range": 0.0,
            "mean_utility_range": 0.0,
            "by_order_seed": [],
        }
    by_candidate: collections.defaultdict[str, list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in order_cases:
        base_id = str(row["case_id"]).rsplit("__order_", 1)[0]
        by_candidate[base_id].append(row)
    conflict_ranges = []
    utility_ranges = []
    for rows in by_candidate.values():
        if len(rows) < 2:
            continue
        reductions = [
            float(row["outcome"].get("conflict_reduction") or 0.0)
            for row in rows
        ]
        utilities = [actual_utility(row["outcome"]) for row in rows]
        conflict_ranges.append(max(reductions) - min(reductions))
        utility_ranges.append(max(utilities) - min(utilities))
    return {
        "candidate_with_multiple_orders": sum(
            len(rows) > 1 for rows in by_candidate.values()
        ),
        "mean_conflict_reduction_range": _mean(conflict_ranges),
        "mean_utility_range": _mean(utility_ranges),
        "by_order_seed": _group_stats(order_cases, "order_seed"),
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Candidate Diagnostics",
        "",
        f"Split: `{summary['split']}`",
        f"Candidate cases: `{summary['candidate_case_count']}`",
        f"States: `{summary['oracle']['state_count']}`",
        "",
        "## Oracle",
        "",
        (
            "States with a better alternative than candidate0: "
            f"`{summary['oracle']['states_with_better_alternative']}` "
            f"({summary['oracle']['states_with_better_alternative_ratio']:.1%})"
        ),
        (
            "Baseline is oracle: "
            f"`{summary['oracle']['baseline_is_oracle_count']}` "
            f"({summary['oracle']['baseline_is_oracle_ratio']:.1%})"
        ),
        "",
        "## Generators",
        "",
        "| Generator | Cases | Valid prob. | Utility | Conflict red. | Cost imp. | Runtime ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["by_generator"]:
        lines.append(
            f"| `{row['generator']}` | {row['case_count']} | "
            f"{row['valid_probability']:.3f} | "
            f"{row['mean_utility']:.3f} | "
            f"{row['mean_conflict_reduction']:.3f} | "
            f"{row['mean_cost_improvement']:.3f} | "
            f"{row['mean_runtime_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Replan Order Noise",
            "",
            (
                "Candidates with multiple orders: "
                f"`{summary['order_noise']['candidate_with_multiple_orders']}`"
            ),
            (
                "Mean conflict-reduction range: "
                f"`{summary['order_noise']['mean_conflict_reduction_range']:.3f}`"
            ),
            (
                "Mean utility range: "
                f"`{summary['order_noise']['mean_utility_range']:.3f}`"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_candidate_diagnostics(
    memory: str | Path,
    output: str | Path,
    dataset: str | Path | None = None,
) -> dict[str, Any]:
    memory_root = Path(memory).resolve()
    output_root = Path(output).resolve()
    memory_summary = _read_json(memory_root / "candidate_summary.json")
    cases = _read_jsonl(memory_root / "candidate_cases.jsonl")
    order_cases = _read_jsonl(memory_root / "candidate_order_cases.jsonl")
    split = str(memory_summary["split"])
    manifest = _manifest_lookup(dataset, split)
    for case in cases:
        row = manifest.get(str(case["task_id"]), {})
        case["layout_mode"] = row.get("layout_mode", case["map_id"])
        case["task_variant"] = row.get(
            "task_variant", row.get("scenario_type", case["task_id"])
        )
    summary = {
        "schema_version": 1,
        "split": split,
        "usage": memory_summary.get("usage"),
        "candidate_case_count": len(cases),
        "candidate_order_case_count": len(order_cases),
        "oracle": _oracle_stats(cases),
        "by_generator": _group_stats(cases, "generator"),
        "by_candidate_index": _group_stats(cases, "candidate_index"),
        "by_layout": _group_stats(cases, "layout_mode"),
        "by_task_variant": _group_stats(cases, "task_variant"),
        "order_noise": _order_noise(order_cases),
    }
    _write_json(output_root / "candidate_diagnostics.json", summary)
    _write_text(output_root / "candidate_diagnostics.md", _report(summary))
    return summary
