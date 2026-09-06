"""Path-derived reports; absent or failed episodes never become zero-time rows."""
from __future__ import annotations

import collections
import math
import random
import statistics
from pathlib import Path

from experiments._common import atomic_write_csv, atomic_write_text, quantile, read_json, sha256_file, write_json
from experiments.closed_loop_confirmation_analysis import map_paired_bootstrap
from experiments.repair_collection import state_fingerprint
from lns2_selector.evaluation.anytime_handoff import modeled_completion
from lns2_selector.evaluation.path_quality_execution import PathJournal, read_artifact
from lns2_selector.evaluation.path_quality_preflight import audit_paths, contained, scenario_prefix

COMPARISONS = (("official_adaptive", "v2-full"), ("official_adaptive", "dual16"), ("v2-full", "dual16"))
METRICS = ("ttf_seconds", "dispatch_seconds", "soc", "makespan", "waits", "completion_0.5s", "completion_1.0s", "completion_2.0s")


def absolute_map_bootstrap(pairs, metric, samples):
    groups = collections.defaultdict(list)
    for left, right in pairs:
        groups[left["map_id"]].append(right[metric] - left[metric])
    maps, rng = sorted(groups), random.Random(20260907)
    values = [statistics.fmean(v for _ in maps for v in groups[rng.choice(maps)]) for _ in range(samples)]
    return [quantile(values, 0.025), quantile(values, 0.975)]


def finite_duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid result duration")
    return float(value)


def inspect_episode(root: Path, case: dict, item: dict, folder: Path, expected_binding: str, anchor: dict) -> dict:
    row = {**item, "status": "pending", "success": False, "flow_completed": False,
           "found_within_budget": False, "delivered_within_budget": False,
           **{metric: None for metric in METRICS}}
    if not (folder / "binding.json").exists():
        return row
    identity = read_json(folder / "binding.json")
    if identity["binding"] != expected_binding or identity["item"] != item:
        raise ValueError("episode binding differs from schedule")
    journal = PathJournal(folder, expected_binding, contained(root, case["files"]["map_file"]),
                          contained(root, case["files"]["scenario_file"]))
    evidence = {}
    for name in ("initial", "terminal", "first_feasible", "first_phase_result", "final_paths", "result", "supervisor"):
        file = folder / f"{name}.json"
        if file.exists():
            evidence[name] = read_artifact(file, expected_binding)
    for name in ("initial", "terminal", "first_feasible"):
        if name in evidence:
            payload = evidence[name]
            if journal.quality(payload["observation"]) != payload["quality"]:
                raise ValueError("saved path quality mismatch")
            if state_fingerprint(payload["observation"]) != payload["state_fingerprint"]:
                raise ValueError("saved state fingerprint mismatch")
    if "initial" in evidence:
        initial = evidence["initial"]
        if initial["state_fingerprint"] != anchor["state_fingerprint"]:
            raise ValueError("initial state differs from reset admission")
        row["initial_fingerprint"] = initial["state_fingerprint"]
        row["initial_conflicts"] = initial["quality"]["colliding_pairs"]
    if "first_feasible" in evidence:
        first = evidence["first_feasible"]
        if first["quality"]["feasible"] is not True:
            raise ValueError("first feasible checkpoint is not feasible")
        found_at = finite_duration(first["available_elapsed_seconds"])
        row["found_within_budget"] = found_at <= item["budget_seconds"]
        row["observed_first_feasible_seconds"] = found_at
        row.update({f"first_{key}": first["quality"][key] for key in ("soc_steps", "makespan_steps", "wait_steps")})
    if "supervisor" not in evidence:
        row["status"] = "interrupted"
        return row
    supervisor = evidence["supervisor"]
    row["status"] = supervisor["status"]
    if bool(supervisor["first_feasible_within_budget"]) != row["found_within_budget"]:
        raise ValueError("supervisor feasible status mismatch")
    if row["status"] != "completed":
        if supervisor["success_by_deadline"]:
            raise ValueError("non-completed episode claims normal success")
        return row
    if not {"initial", "terminal", "first_feasible", "first_phase_result", "final_paths", "result"}.issubset(evidence):
        raise ValueError("completed episode is missing artifacts")
    final, result, first = evidence["final_paths"], evidence["result"], evidence["first_feasible"]
    phase = evidence["first_phase_result"]
    if phase["status"] != "ok" or result["status"] != "completed":
        raise ValueError("completed status disagrees with first phase/result")
    grid = journal.grid
    endpoints = scenario_prefix(journal.map_path, journal.scenario, grid, case["static_audit"]["agent_count"])
    quality = audit_paths(grid, [(int(r[5]), int(r[4])) for r in endpoints],
                          [(int(r[7]), int(r[6])) for r in endpoints], final["paths"])
    if quality != final["final_quality"] or not quality["feasible"]:
        raise ValueError("invalid final paths/quality")
    first_paths = [a["path"] for a in sorted(first["observation"]["agents"], key=lambda a: a["id"])]
    if item["protocol"] == "first_feasible":
        if final["paths"] != first_paths:
            raise ValueError("first-feasible protocol changed its solution")
    elif (final["initial_paths"] != first_paths or final["stage2_seed"] != item["stage2_seed"]
          or quality["soc_steps"] > first["quality"]["soc_steps"]):
        raise ValueError("invalid official second-stage handoff")
    ready = finite_duration(result["paths_saved_elapsed_seconds"])
    dispatch = max(item["budget_seconds"], ready) if item["protocol"] == "fixed_budget" else ready
    if result["dispatch_wall_seconds"] != dispatch or result["first_feasible_elapsed_seconds"] != first["available_elapsed_seconds"]:
        raise ValueError("result clock disagreement")
    success = row["found_within_budget"]
    if any(bool(x["success_by_deadline"]) != success for x in (result, supervisor)):
        raise ValueError("result success disagreement")
    if result["delivered_within_budget"] != (ready <= item["budget_seconds"]):
        raise ValueError("delivery deadline disagreement")
    row.update(flow_completed=True, success=success, delivered_within_budget=result["delivered_within_budget"],
               actual_dispatch_seconds=dispatch, cold_start_to_paths_seconds=finite_duration(result["cold_start_to_paths_seconds"]),
               startup_before_reset_seconds=finite_duration(result["startup_before_reset_seconds"]),
               environment_construct_seconds=finite_duration(result["environment_construct_seconds"]),
               repair_iterations=phase["summary"]["repair_iterations"],
               final_soc=quality["soc_steps"], final_makespan=quality["makespan_steps"], final_waits=quality["wait_steps"],
               soc_change=quality["soc_steps"] - first["quality"]["soc_steps"],
               makespan_change=quality["makespan_steps"] - first["quality"]["makespan_steps"],
               budget_overshoot_seconds=max(0.0, ready - item["budget_seconds"]))
    if success:
        row.update(ttf_seconds=first["available_elapsed_seconds"], dispatch_seconds=dispatch,
                   soc=quality["soc_steps"], makespan=quality["makespan_steps"], waits=quality["wait_steps"])
        for step in (0.5, 1.0, 2.0):
            row[f"completion_{step}s"] = modeled_completion(dispatch, quality["makespan_steps"], step)
    row["artifact_sha256"] = {name: sha256_file(folder / f"{name}.json") for name in evidence}
    return row


def summarize(rows: list[dict], *, samples=5000) -> dict:
    result = {"by_protocol": {}, "comparisons": []}
    for protocol in sorted({(r["protocol"], r["budget_seconds"]) for r in rows}):
        selected = [r for r in rows if (r["protocol"], r["budget_seconds"]) == protocol]
        label = f"{protocol[0]}_{protocol[1]:g}s"
        strata = {"all": selected}
        for kind in ("family", "map_id"):
            strata.update({f"{kind}:{value}": [r for r in selected if r[kind] == value] for value in sorted({r[kind] for r in selected})})
        for stratum, subset in strata.items():
            key = f"{label}/{stratum}"
            stats = {}
            for controller in ("official_adaptive", "v2-full", "dual16"):
                group = [r for r in subset if r["controller"] == controller]
                good = [r for r in group if r["success"]]
                stats[controller] = {"scheduled": len(group), "successes": len(good),
                    "status_counts": dict(collections.Counter(r["status"] for r in group)),
                    "found_within_budget": sum(r["found_within_budget"] for r in group),
                    "delivered_within_budget": sum(r["delivered_within_budget"] for r in group),
                    "metrics_on_own_successes": {m: {"mean": statistics.fmean(r[m] for r in good),
                        "median": statistics.median(r[m] for r in good)} if good else None for m in METRICS}}
            result["by_protocol"][key] = stats
            for baseline, challenger in COMPARISONS:
                left = {(r["task_id"], r["solver_seed"]): r for r in subset if r["controller"] == baseline}
                right = {(r["task_id"], r["solver_seed"]): r for r in subset if r["controller"] == challenger}
                if left.keys() != right.keys():
                    raise ValueError("paired schedule incomplete")
                pairs = [(left[k], right[k]) for k in sorted(left) if left[k]["success"] and right[k]["success"]]
                comparison = {"group": key, "baseline": baseline, "challenger": challenger,
                    "scheduled_pairs": len(left), "common_success_pairs": len(pairs),
                    "non_common_success_keys": [list(k) for k in sorted(left) if not (left[k]["success"] and right[k]["success"])], "metrics": {}}
                for metric in METRICS:
                    if not pairs:
                        comparison["metrics"][metric] = None
                        continue
                    a, b = [x[metric] for x, _ in pairs], [y[metric] for _, y in pairs]
                    boot = (map_paired_bootstrap([(dict(map_id=x["map_id"], summary={metric: x[metric]}),
                         dict(map_id=y["map_id"], summary={metric: y[metric]})) for x, y in pairs], metric, samples, seed=20260907)
                            if all(x > 0 for x in a) else {"map_count": len({x['map_id'] for x,_ in pairs}),
                                "improvement_95_ci": None, "reason": "zero baseline; use absolute paired delta interval"})
                    boot["paired_delta_95_ci"] = absolute_map_bootstrap(pairs, metric, samples)
                    comparison["metrics"][metric] = {"baseline_mean": statistics.fmean(a), "challenger_mean": statistics.fmean(b),
                        "baseline_median": statistics.median(a), "challenger_median": statistics.median(b),
                        "mean_paired_delta": statistics.fmean(y-x for x, y in zip(a,b)),
                        "improvement_fraction": 1-statistics.fmean(b)/statistics.fmean(a) if statistics.fmean(a) else None,
                        "wins": sum(y < x for x,y in zip(a,b)), "ties": sum(y == x for x,y in zip(a,b)),
                        "losses": sum(y > x for x,y in zip(a,b)), "map_bootstrap": boot}
                result["comparisons"].append(comparison)
    return result


def publish_analysis(output: Path, rows: list[dict], report: dict) -> None:
    write_json(output / "report.json", report)
    keys = sorted({k for r in rows for k,v in r.items() if not isinstance(v, (dict,list))})
    atomic_write_csv(output / "episodes.csv", [{k: r.get(k) for k in keys} for r in rows])
    lines = ["# 路径质量与任务完成时间", "", "这是复用案例的开发性诊断，不是独立泛化确认或真实机器人测试。", "",
             f"排程 {len(rows)} 项；正常成功 {sum(r['success'] for r in rows)} 项。", "",
             "主时间从 reset 前起算；冷启动单列。失败不填零，配对比较只使用共同成功任务。", "",
             "|协议/分层|对比|共同成功|交付时间改善|1秒步长完成时间改善|", "|---|---|---:|---:|---:|"]
    for c in report["statistics"]["comparisons"]:
        def fmt(metric):
            value = c["metrics"][metric]
            return f"{100*value['improvement_fraction']:.3f}%" if value and value["improvement_fraction"] is not None else "NA"
        lines.append(f"|{c['group']}|{c['challenger']} vs {c['baseline']}|{c['common_success_pairs']}|{fmt('dispatch_seconds')}|{fmt('completion_1.0s')}|")
    lines += ["", "正值代表挑战方法更低；均值来自共同成功任务，必须结合全组成功数解释。",
              "SOC 降低不保证 makespan 降低；Maze 只有一张有效地图，不外推整个地图族。"]
    atomic_write_text(output / "report_zh.md", "\n".join(lines)+"\n")
    from html import escape
    charts = [c for c in report["statistics"]["comparisons"] if c["group"].endswith("/all")]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{max(160, 60+60*len(charts))}" viewBox="0 0 1000 {max(160, 60+60*len(charts))}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="20" y="24" font-family="sans-serif" font-size="16">Paired modeled completion: baseline / challenger (seconds; step=1s)</text>']
    maximum = max([v for c in charts if c["metrics"]["completion_1.0s"] for v in (c["metrics"]["completion_1.0s"]["baseline_mean"], c["metrics"]["completion_1.0s"]["challenger_mean"])] or [1]) or 1
    for index, c in enumerate(charts):
        y = 50 + 60*index
        svg.append(f'<text x="10" y="{y+12}" font-family="sans-serif" font-size="11">{escape(c["group"]+" "+c["challenger"]+" vs "+c["baseline"])}</text>')
        metric = c["metrics"]["completion_1.0s"]
        if metric:
            for j, name in enumerate(("baseline_mean", "challenger_mean")):
                value = metric[name]
                width = 480*value/maximum
                svg.append(f'<rect x="410" y="{y+18*j}" width="{width:.3f}" height="14" fill="{("#676767", "#00866b")[j]}"/>')
                svg.append(f'<text x="{420+width:.3f}" y="{y+12+18*j}" font-family="sans-serif" font-size="11">{value:.3f}</text>')
    svg.append('</svg>')
    atomic_write_text(output / "completion.svg", "\n".join(svg))
