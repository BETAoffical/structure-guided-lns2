"""Read-only verification and analysis of the registered diagnostic results."""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file, write_json
from experiments.feedback_exploration_diagnostic import FeedbackSelection, paired_summary, classify_rounds
from scripts import run_feedback_exploration_diagnostics as run


def verify_rollout(result, job):
    run.checked(result["conflicts"][0] == job["case"]["state"]["num_of_colliding_pairs"], "initial conflict mismatch")
    selector = FeedbackSelection(job["arm"])
    counters = Counter()
    seen = set()
    for d, event in enumerate(result["transitions"]):
        pool, metrics = event["candidate_pool"], event["metrics"]
        ids = [c["candidate_id"] for c in pool]
        run.checked(len(set(ids)) == len(ids) and len(ids) == len(event["keys"]), "invalid pool")
        base = ids.index(event["base_id"])
        run.checked(pool[base]["score"] == max(c["score"] for c in pool), "base is not model winner")
        chosen, diagnostic = selector.select(event["keys"], ids, base,
            run.seed(job["phase"], job["case"]["case_id"], job["trial"], d, "selection"))
        run.checked(diagnostic == event["feedback"] and ids[chosen] == event["candidate_id"], "feedback cannot be reproduced")
        key = event["keys"][chosen]
        run.checked(key == event["selected_key"], "selected key mismatch")
        pp_seed = run.seed(job["phase"], job["case"]["case_id"], job["trial"], d, "pp")
        run.checked(event["action"] == {"mode": "explicit_neighborhood", "agents": pool[chosen]["agents"],
                                      "random_seed": pp_seed}, "unregistered action")
        run.checked(metrics["requested_random_seed"] == pp_seed and
                    metrics["requested_pp_time_limit_seconds"] == run.PP_SECONDS and
                    not metrics["requested_repair_order"] and metrics["requested_pp_random_seed"] == -1,
                    "repair protocol changed")
        before, after = result["conflicts"][d:d+2]
        run.checked((metrics["conflicts_before"], metrics["conflicts_after"]) == (before, after), "conflict summary mismatch")
        run.checked(metrics["action_valid"] and metrics["step_applied"] and
                    sorted(metrics["neighborhood"]) == sorted(pool[chosen]["agents"]), "action integrity")
        censor = metrics["pp_failure_reason"] not in ("none", "conflict_bound_exceeded") or metrics["native_replan_seconds"] >= run.PP_SECONDS
        selector.observe(key, before, after, censor)
        counters["steps"] += 1
        counters["strict_drops"] += after < before
        counters["rollbacks"] += bool(metrics["pp_rolled_back"])
        counters["repeated_selected_keys"] += key in seen
        counters["active_steps"] += diagnostic["active"]
        counters["departures_from_base"] += chosen != base
        counters["equal_conflict_accepts"] += after == before and not metrics["pp_rolled_back"]
        if diagnostic["active"]:
            counters["active_pool_slots"] += len(pool)
            counters["active_unseen_slots"] += sum(sum(p) == 0 for p in diagnostic["prior"])
            counters["active_success_evidence_slots"] += sum(p[0] > 0 for p in diagnostic["prior"])
            counters["active_selected_unseen"] += sum(diagnostic["prior"][chosen]) == 0
            counters["active_selected_previously_nondrop"] += diagnostic["prior"][chosen][1] > 0
        counters["selected_size_" + str(len(pool[chosen]["agents"]))] += 1
        seen.add(key)
    run.validate_final(result["final_state"])
    return dict(counters)


def summarize_cases(rows):
    cases = []
    for case in sorted({r["case_id"] for r in rows}):
        selected = [r for r in rows if r["case_id"] == case]
        summary = paired_summary(selected)
        cases.append({"case_id": case, "map_id": selected[0]["map_id"],
                      "initial_conflicts": selected[0]["conflicts"][0], "arms": summary,
                      "trajectories": {arm: [r["conflicts"] for r in sorted(selected, key=lambda r: r["trial"])
                                             if r["arm"] == arm] for arm in FeedbackSelection.ARMS}})
    return cases


def analyze():
    plan = run.verify()
    report = read_json(run.OUT / "report.json")
    run.checked(report["plan_sha256"] == sha256_file(run.OUT / "plan.json"), "report plan binding mismatch")
    stages, diagnostic_counts, case_rows = {}, {}, {}
    for stage in ("round1", "round2", "round3"):
        jobs = read_json(run.OUT / stage / "schedule.json")
        if stage != "round2":
            cases = plan["discovery" if stage == "round1" else "validation"]
            run.checked(jobs == run.stage_jobs(plan, cases, stage), "schedule differs from registration")
        expected = {j["job_id"] + ".json" for j in jobs}
        actual = {p.name for p in (run.OUT / stage / "outcomes").glob("*.json")}
        run.checked(actual == expected, "missing or extra outputs")
        rows, counts = [], {a: Counter() for a in FeedbackSelection.ARMS}
        for job in jobs:
            p = run.OUT / stage / "outcomes" / (job["job_id"] + ".json")
            run.checked(sha256_file(p) == report["outcome_sha256"][p.relative_to(run.OUT).as_posix()], "outcome SHA mismatch")
            r = run.load_result(p, job)
            if "arm" in job:
                counts[job["arm"]].update(verify_rollout(r, job))
            else:
                run.checked(job["repair_key"] == run.repair_keys(job["case"]["state"], [job["candidate"]],
                                                                 job["case"]["instance"])[0], "probe condition mismatch")
                run.checked((r["before_conflicts"], r["after_conflicts"]) ==
                            (r["metrics"]["conflicts_before"], r["metrics"]["conflicts_after"]), "probe metrics mismatch")
            rows.append(r)
        stages[stage] = rows
        diagnostic_counts[stage] = {a: dict(c) for a, c in counts.items()}
        if stage != "round2":
            run.checked(paired_summary(rows) == report[stage], "aggregate mismatch")
            case_rows[stage] = summarize_cases(rows)
    decision = classify_rounds(report["round1"], report["round3"])
    run.checked(decision == report["decision"], "decision mismatch")
    opportunities = []
    for case, candidate in sorted({(r["case_id"], r["candidate_id"]) for r in stages["round2"]}):
        rows = [r for r in stages["round2"] if (r["case_id"], r["candidate_id"]) == (case, candidate)]
        run.checked({r["trial"] for r in rows} == set(range(run.TRIALS)) and len(rows) == run.TRIALS, "incomplete probe trials")
        opportunities.append({"case_id": case, "candidate_id": candidate, "size": len(rows[0]["agents"]),
                              "before_conflicts": rows[0]["before_conflicts"],
                              "after_conflicts": [r["after_conflicts"] for r in rows],
                              "feasible": sum(r["after_conflicts"] == 0 for r in rows),
                              "censored": sum(r["censored"] for r in rows)})
    result = {"schema": "lns2.feedback_exploration_analysis.v1", "plan_sha256": report["plan_sha256"],
              "report_sha256": sha256_file(run.OUT / "report.json"), "decision": decision,
              "verified_episodes": len(stages["round1"]) + len(stages["round3"]),
              "verified_probe_jobs": len(stages["round2"]), "cases": case_rows,
              "diagnostics": diagnostic_counts, "successor_opportunities": opportunities,
              "new_repairs": sum(len(r["transitions"]) for phase in ("round1", "round3") for r in stages[phase]) + len(stages["round2"]),
              "source_prefix_repairs": sum(len(j["case"]["prefix"]) for phase in ("round1", "round2", "round3")
                                           for j in read_json(run.OUT / phase / "schedule.json")),
              "no_controller_promotion": True}
    write_json(run.OUT / "analysis.json", result)
    lines = ["# 三轮反馈探索诊断结果", "", "分类：`" + decision + "`。仅为历史状态开发证据，不是TTF或总体成功率确认。", "",
             "| 轮次 | 案例 | 初始冲突 | Frozen清零 | Uniform清零 | Feedback清零 | Frozen/Uniform/Feedback AUC总和 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for phase, rows in case_rows.items():
        for c in rows:
            arms = c["arms"]
            lines.append(f"| {phase} | {c['case_id']} | {c['initial_conflicts']} | " +
                         " | ".join(str(arms[a]["feasible"]) + "/4" for a in FeedbackSelection.ARMS) + " | " +
                         "/".join(str(arms[a]["auc_sum"]) for a in FeedbackSelection.ARMS) + " |")
    lines += ["", f"核验 {result['verified_episodes']} 条轨迹、{result['verified_probe_jobs']} 个单步分支，"
              f"新增修复 {result['new_repairs']} 次、来源prefix重放 {result['source_prefix_repairs']} 次。",
              "", "逐动作反馈、随机seed、候选成员、分数首选、请求预算、冲突序列、终态路径及冲突图均已重新核验。",
              "", "第二轮完整候选结果见 analysis.json 的 successor_opportunities。未清零表示固定32步内未完成，不是无解证明。"]
    (run.OUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("decision", "verified_episodes", "verified_probe_jobs", "new_repairs", "source_prefix_repairs")}, indent=2))


if __name__ == "__main__":
    analyze()
