from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, sha256_file
from experiments.repair_collection import _write_json


V3_WALL_CLOCK_HISTORY_AUDIT_SCHEMA = "lns2.v3_wall_clock_history_audit.v1"
PAIRWISE_RELATIVE_TOLERANCE = 1e-12


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_pairwise_run(
    *, status_path: Path, episode_path: Path
) -> dict[str, Any]:
    status = _read_json(status_path)
    if str(status.get("status")) != "complete" or str(
        status.get("phase")
    ) != "complete":
        raise ValueError(f"historical wall-clock run is not complete: {status_path}")
    if int(status.get("bottleneck_validation", {}).get("error_episode_count", -1)) != 0:
        raise ValueError("historical wall-clock run contains errors")
    if not bool(status.get("bottleneck_validation", {}).get("coverage_passed")):
        raise ValueError("historical wall-clock run lacks paired coverage")
    rows = _read_csv(episode_path)
    expected = int(status["bottleneck_validation"]["expected_task_seed_count"])
    pairs = {str(row["pair"]) for row in rows}
    if len(rows) != expected * len(pairs):
        raise ValueError("historical pairwise CSV row count is inconsistent")
    if any(str(row["initial_fingerprint_match"]).lower() != "true" for row in rows):
        raise ValueError("historical pairwise CSV has fingerprint mismatch")
    keys = {
        (str(row["pair"]), str(row["task_id"]), int(row["solver_seed"]))
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("historical pairwise CSV has duplicate keys")
    return {
        "row_count": len(rows),
        "pair_count": len(pairs),
        "task_seed_count": expected,
        "status_sha256": sha256_file(status_path),
        "episode_sha256": sha256_file(episode_path),
    }


def _boolean(row: dict[str, Any], name: str) -> bool:
    return str(row[name]).lower() == "true"


def summarize_pairwise_rows(
    rows: Iterable[dict[str, Any]], *, pair: str, group: str
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("wall-clock summary group is empty")
    candidate_times = [
        float(row["candidate_restricted_time_to_feasible"])
        for row in materialized
    ]
    reference_times = [
        float(row["reference_restricted_time_to_feasible"])
        for row in materialized
    ]
    common = [row for row in materialized if _boolean(row, "common_success")]
    candidate_common = [
        float(row["candidate_restricted_time_to_feasible"]) for row in common
    ]
    reference_common = [
        float(row["reference_restricted_time_to_feasible"]) for row in common
    ]
    candidate_mean = statistics.fmean(candidate_times)
    reference_mean = statistics.fmean(reference_times)
    candidate_common_mean = (
        statistics.fmean(candidate_common) if candidate_common else None
    )
    reference_common_mean = (
        statistics.fmean(reference_common) if reference_common else None
    )
    deltas = [
        candidate - reference
        for candidate, reference in zip(candidate_times, reference_times)
    ]
    return {
        "pair": pair,
        "group": group,
        "episode_count": len(materialized),
        "candidate_success_count": sum(
            int(_boolean(row, "candidate_success")) for row in materialized
        ),
        "reference_success_count": sum(
            int(_boolean(row, "reference_success")) for row in materialized
        ),
        "candidate_capped_ttf_mean": candidate_mean,
        "reference_capped_ttf_mean": reference_mean,
        "capped_ttf_ratio": candidate_mean / max(1e-12, reference_mean),
        "capped_ttf_relative_change": candidate_mean
        / max(1e-12, reference_mean)
        - 1.0,
        "common_success_count": len(common),
        "candidate_common_success_ttf_mean": candidate_common_mean,
        "reference_common_success_ttf_mean": reference_common_mean,
        "common_success_ttf_ratio": (
            candidate_common_mean / max(1e-12, reference_common_mean)
            if candidate_common_mean is not None
            and reference_common_mean is not None
            else None
        ),
        "candidate_faster_count": sum(
            int(delta < -PAIRWISE_RELATIVE_TOLERANCE) for delta in deltas
        ),
        "reference_faster_count": sum(
            int(delta > PAIRWISE_RELATIVE_TOLERANCE) for delta in deltas
        ),
        "tie_count": sum(
            int(abs(delta) <= PAIRWISE_RELATIVE_TOLERANCE) for delta in deltas
        ),
    }


def pairwise_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for pair in sorted({str(row["pair"]) for row in rows}):
        pair_rows = [row for row in rows if str(row["pair"]) == pair]
        groups: list[tuple[str, list[dict[str, Any]]]] = [
            ("all", pair_rows),
            (
                "agents_lt_600",
                [row for row in pair_rows if int(row["agent_count"]) < 600],
            ),
            (
                "agents_600",
                [row for row in pair_rows if int(row["agent_count"]) == 600],
            ),
        ]
        groups.extend(
            (
                f"agents_{agent_count}",
                [
                    row
                    for row in pair_rows
                    if int(row["agent_count"]) == agent_count
                ],
            )
            for agent_count in sorted(
                {int(row["agent_count"]) for row in pair_rows}
            )
        )
        for group, group_rows in groups:
            if group_rows:
                result.append(
                    summarize_pairwise_rows(group_rows, pair=pair, group=group)
                )
    return result


def validate_s3_runtime(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        if str(row.get("status")) != "ok":
            raise ValueError("v3-S3 runtime contains a non-ok episode")
        grouped.setdefault(
            (str(row["task_id"]), int(row["solver_seed"])), []
        ).append(row)
    for key, episodes in grouped.items():
        controllers = {str(row["controller"]) for row in episodes}
        if len(episodes) != 2 or controllers != {"v2-full", "v3-s3"}:
            raise ValueError(f"v3-S3 runtime pair is incomplete: {key}")
    return {
        "episode_count": len(rows),
        "pair_count": len(grouped),
        "sha256": sha256_file(path),
    }


def summarize_s3_runtime(
    rows: list[dict[str, Any]], *, run: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["task_id"]), int(row["solver_seed"])), []
        ).append(row)
    pair_rows = []
    for (task_id, seed), episodes in sorted(grouped.items()):
        indexed = {str(row["controller"]): row for row in episodes}
        v2 = indexed["v2-full"]
        v3 = indexed["v3-s3"]
        v2_time = float(v2["capped_wall_time_to_feasible"])
        v3_time = float(v3["capped_wall_time_to_feasible"])
        pair_rows.append(
            {
                "run": run,
                "task_id": task_id,
                "solver_seed": seed,
                "agent_count": int(v2["agent_count"]),
                "v2_success": _boolean(v2, "success"),
                "v3_s3_success": _boolean(v3, "success"),
                "v2_capped_ttf": v2_time,
                "v3_s3_capped_ttf": v3_time,
                "v3_s3_to_v2_ratio": v3_time / max(1e-12, v2_time),
                "v2_repair_iterations": int(v2["repair_iterations"]),
                "v3_s3_repair_iterations": int(v3["repair_iterations"]),
            }
        )
    v2_times = [float(row["v2_capped_ttf"]) for row in pair_rows]
    v3_times = [float(row["v3_s3_capped_ttf"]) for row in pair_rows]
    v2_mean = statistics.fmean(v2_times)
    v3_mean = statistics.fmean(v3_times)
    return {
        "run": run,
        "pair_count": len(pair_rows),
        "v2_success_count": sum(int(row["v2_success"]) for row in pair_rows),
        "v3_s3_success_count": sum(
            int(row["v3_s3_success"]) for row in pair_rows
        ),
        "v2_capped_ttf_mean": v2_mean,
        "v3_s3_capped_ttf_mean": v3_mean,
        "v3_s3_to_v2_ratio": v3_mean / max(1e-12, v2_mean),
        "v3_s3_faster_count": sum(
            int(v3 < v2) for v2, v3 in zip(v2_times, v3_times)
        ),
    }, pair_rows


def _render_report(report: dict[str, Any]) -> str:
    summaries = {
        (row["pair"], row["group"]): row
        for row in report["v3_full_pairwise_summaries"]
    }
    versus_v2 = summaries[("v3-full_vs_v2-full", "all")]
    sub600_v2 = summaries[("v3-full_vs_v2-full", "agents_lt_600")]
    sub600_lns2 = summaries[
        ("v3-full_vs_official_adaptive", "agents_lt_600")
    ]
    agents_200_v2 = summaries[("v3-full_vs_v2-full", "agents_200")]
    lines = [
        "# Historical V3 real wall-clock audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## One-step v3-full, three solver seeds",
        "",
        f"- All 24 pairs vs v2: capped TTF ratio {versus_v2['capped_ttf_ratio']:.4f}.",
        f"- Under 600 agents vs v2: ratio {sub600_v2['capped_ttf_ratio']:.4f}.",
        f"- Under 600 agents vs original LNS2: ratio {sub600_lns2['capped_ttf_ratio']:.4f}.",
        (
            "- The only pooled agent-count subgroup faster than v2 was 200 agents: "
            f"ratio {agents_200_v2['capped_ttf_ratio']:.4f} "
            f"({agents_200_v2['candidate_faster_count']}/"
            f"{agents_200_v2['episode_count']} paired episodes faster)."
        ),
        "",
        "## Evidence boundary",
        "",
        "- H3, Value, and Receding-Q have no complete-episode wall-clock result.",
        "- v3-S3 has two four-task diagnostics and no promotion-scale quick.",
        "- Offline efficiency and Oracle results are excluded from the speed decision.",
        "",
    ]
    return "\n".join(lines)


def audit_v3_wall_clock_history(
    *, project_root: Path, output: Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("V3 wall-clock history output is non-empty")
    pairwise_runs = [
        (
            "seed1_feature124_fixed",
            root / "build/initlns-v3-wall-clock-diagnostic-seed1-v2-fixed",
        ),
        (
            "seeds23_feature94",
            root / "build/initlns-v3-wall-clock-diagnostic-seeds23-feature94-v1",
        ),
    ]
    pairwise_rows = []
    pairwise_sources = []
    for name, directory in pairwise_runs:
        status_path = directory / "status.json"
        episode_path = directory / "report/controller_pairwise_episodes.csv"
        verification = validate_pairwise_run(
            status_path=status_path, episode_path=episode_path
        )
        rows = _read_csv(episode_path)
        for row in rows:
            row["historical_run"] = name
        pairwise_rows.extend(rows)
        pairwise_sources.append(
            {"run": name, "path": str(directory), **verification}
        )
    keys = {
        (str(row["pair"]), str(row["task_id"]), int(row["solver_seed"]))
        for row in pairwise_rows
    }
    if len(keys) != len(pairwise_rows):
        raise ValueError("V3 historical wall-clock runs overlap")
    pairwise_summaries = pairwise_summary_rows(pairwise_rows)

    s3_runs = [
        (
            "deterministic_run_1",
            root / "build/initlns-v3-s3-stable-runtime-v1/episode_runtime.csv",
        ),
        (
            "deterministic_repeat",
            root
            / "build/initlns-v3-s3-stable-runtime-v2-repeat/episode_runtime.csv",
        ),
    ]
    s3_summaries = []
    s3_pairs = []
    s3_sources = []
    for name, path in s3_runs:
        verification = validate_s3_runtime(path)
        summary, rows = summarize_s3_runtime(_read_csv(path), run=name)
        s3_summaries.append(summary)
        s3_pairs.extend(rows)
        s3_sources.append({"run": name, "path": str(path), **verification})

    summary_index = {
        (row["pair"], row["group"]): row for row in pairwise_summaries
    }
    v3_v2_all = summary_index[("v3-full_vs_v2-full", "all")]
    v3_v2_sub600 = summary_index[
        ("v3-full_vs_v2-full", "agents_lt_600")
    ]
    v3_lns2_sub600 = summary_index[
        ("v3-full_vs_official_adaptive", "agents_lt_600")
    ]
    v3_v2_agents_200 = summary_index[("v3-full_vs_v2-full", "agents_200")]
    no_s3_speedup = all(
        float(row["v3_s3_to_v2_ratio"]) >= 1.0 for row in s3_summaries
    )
    verified_faster_than_v2 = bool(
        float(v3_v2_all["capped_ttf_ratio"]) < 1.0 or not no_s3_speedup
    )
    decision = (
        "historical_v3_verified_faster_than_v2_overall"
        if verified_faster_than_v2
        else "no_v3_variant_verified_faster_than_v2_overall"
    )
    report = {
        "schema": V3_WALL_CLOCK_HISTORY_AUDIT_SCHEMA,
        "decision": decision,
        "historical_complete_episode_only": True,
        "v3_full_pairwise_summaries": pairwise_summaries,
        "v3_s3_summaries": s3_summaries,
        "key_findings": {
            "v3_full_all_to_v2_capped_ratio": v3_v2_all[
                "capped_ttf_ratio"
            ],
            "v3_full_sub600_to_v2_capped_ratio": v3_v2_sub600[
                "capped_ttf_ratio"
            ],
            "v3_full_sub600_to_lns2_capped_ratio": v3_lns2_sub600[
                "capped_ttf_ratio"
            ],
            "v3_full_agents200_to_v2_capped_ratio": v3_v2_agents_200[
                "capped_ttf_ratio"
            ],
            "v3_full_agents200_faster_count": v3_v2_agents_200[
                "candidate_faster_count"
            ],
            "v3_full_agents200_episode_count": v3_v2_agents_200[
                "episode_count"
            ],
            "sub600_memory_resolution": (
                "v3-full was faster than original LNS2, not v2-full"
            ),
        },
        "excluded_as_non_end_to_end": [
            {
                "variant": "v3-H3",
                "reason": "offline horizon labels and diagnostics only",
            },
            {
                "variant": "v3-S3 training pilots",
                "reason": "offline sequence metrics; runtime evidence reported separately",
            },
            {
                "variant": "v3-Value",
                "reason": "cost-to-go label pilot only",
            },
            {
                "variant": "Receding-Q",
                "reason": "fresh label and stability pilots only",
            },
            {
                "variant": "v2 Top-3 cost tiebreak",
                "reason": "offline one-repair audit; no runtime controller",
            },
        ],
        "invalid_or_superseded": [
            {
                "path": str(
                    root / "build/initlns-v3-wall-clock-diagnostic-seed1-v1"
                ),
                "reason": "status=error with seven v3-full episode errors",
            }
        ],
        "sources": {
            "v3_full": pairwise_sources,
            "v3_s3": s3_sources,
        },
        "new_solver_execution_started": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "v3_full_pairwise_episodes.csv", pairwise_rows)
    atomic_write_csv(
        output / "v3_full_wall_clock_summary.csv", pairwise_summaries
    )
    atomic_write_csv(output / "v3_s3_pairwise_episodes.csv", s3_pairs)
    atomic_write_csv(output / "v3_s3_wall_clock_summary.csv", s3_summaries)
    _write_json(output / "v3_wall_clock_history_report.json", report)
    (output / "v3_wall_clock_history_report.md").write_text(
        _render_report(report), encoding="utf-8", newline="\n"
    )
    return report
