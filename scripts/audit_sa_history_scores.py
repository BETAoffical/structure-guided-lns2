"""Post-hoc score diagnosis from frozen labels/OOF predictions; no solver or fit."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, write_json, sha256_file
from experiments.sa_history_selector import select_prediction, pareto_indices

HEADS = ("conflicts", "persistence", "feasible")
SOURCE = ROOT / "build/sa-history-selector-sampling-v2/training"
EXPECTED = {
    "index.jsonl": "3406ea64d63e2c5cd504826224ed836e914c76d3bc91b67e3e2c143202f987cb",
    "oof_predictions.jsonl": "08f66467fd0ec64103c90a0082550effc67a986908acb1a3d39bccab37cc9ef6",
    "report.json": "2fe4cd315ce5938bd232e7b8cf026ede0435ccc3a9225d7cfaa53804b6ff5ab5",
}


from lns2_selector.runtime.contracts import require


def substitutions(predictions, labels, ids, c0):
    """Mask bits replace heads with finite-sample labels, not new predictions."""
    return [select_prediction([
        {head: (label if mask & (1 << j) else pred)[head] for j, head in enumerate(HEADS)}
        for pred, label in zip(predictions, labels)], ids, c0)
        for mask in range(8)]


def shapley_replacement(losses):
    require(len(losses) == 8, "eight substitutions required")
    result = {}
    for j, head in enumerate(HEADS):
        gain = 0.0
        for mask in range(8):
            if mask & (1 << j):
                continue
            size = bin(mask).count("1")
            weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
            gain += weight * (losses[mask] - losses[mask | (1 << j)])
        result[head] = gain
    require(math.isclose(sum(result.values()), losses[0] - losses[7], abs_tol=1e-10), "decomposition")
    return result


def analyze_state(rows, predicted):
    rows = sorted(rows, key=lambda r: r["candidate_id"])
    ids = [r["candidate_id"] for r in rows]
    require(len(set(ids)) == len(ids), "duplicate candidate")
    require(len({r["initial_conflicts"] for r in rows}) == 1, "inconsistent initial conflicts")
    c0 = rows[0]["initial_conflicts"]
    require(c0 > 0, "nonrepair state")
    labels = [r["labels"] for r in rows]
    for r in rows:
        require(len(r["trial_labels"]) == 4, "four trials required")
        for head in HEADS:
            require(math.isclose(r["labels"][head], sum(t[head] for t in r["trial_labels"]) / 4, abs_tol=1e-12), "trial aggregation drift")
    empirical_risk = select_prediction(labels, ids, c0)
    empirical_conflict = select_prediction(labels, ids, c0, risk=False)
    minimum = min(l["conflicts"] for l in labels)
    pareto = pareto_indices(labels)
    choices = {}

    def record(name, i):
        choices[name] = dict(candidate_id=ids[i], **labels[i],
                             regret=labels[i]["conflicts"] - minimum, pareto_hit=float(i in pareto))

    record("empirical_risk", empirical_risk)
    record("empirical_conflict", empirical_conflict)
    record("frozen", next(i for i, r in enumerate(rows) if r["old_selected"]))
    profiles = {}
    for profile in ("dynamic", "history"):
        preds = [predicted[profile, cid] for cid in ids]
        picked = substitutions(preds, labels, ids, c0)
        conflict_only = select_prediction(preds, ids, c0, risk=False)
        record(profile, picked[0])
        record(profile + "_conflict_only", conflict_only)
        losses = [labels[i]["conflicts"] - minimum for i in picked]
        cost = labels[empirical_risk]["conflicts"] - labels[empirical_conflict]["conflicts"]
        residual = labels[picked[0]]["conflicts"] - labels[empirical_risk]["conflicts"]
        total = labels[picked[0]]["conflicts"] - labels[empirical_conflict]["conflicts"]
        require(math.isclose(total, cost + residual, abs_tol=1e-12), "score decomposition")
        excess = c0 * (labels[picked[0]]["conflicts"] - labels[conflict_only]["conflicts"])
        profiles[profile] = dict(
            masks=[dict(mask=m, replaced=[h for j, h in enumerate(HEADS) if m & (1 << j)],
                        candidate_id=ids[i], regret=losses[m]) for m, i in enumerate(picked)],
            score_cost=cost, prediction_residual=residual, total_gap=total,
            shapley_gains=shapley_replacement(losses),
            actual_excess_pairs_vs_predicted_conflict=excess,
            actual_excess_over_one_pair=excess > 1 + 1e-10,
            risk_changes_choice=picked[0] != conflict_only)
    # Disjoint halves reduce same-trial oracle optimism, but remain only two trials.
    split = []
    for select_trials, eval_trials in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
        halves = [[{h: sum(r["trial_labels"][i][h] for i in part) / 2 for h in HEADS}
                   for r in rows] for part in (select_trials, eval_trials)]
        a = select_prediction(halves[0], ids, c0)
        b = select_prediction(halves[0], ids, c0, risk=False)
        split.append(dict(select_trials=list(select_trials), eval_trials=list(eval_trials),
                          risk_id=ids[a], conflict_id=ids[b],
                          delta={h: halves[1][a][h] - halves[1][b][h] for h in HEADS}))
    return dict(state_id=rows[0]["state_id"], map_id=rows[0]["map_id"], stratum=rows[0]["stratum"],
                initial_conflicts=c0, choices=choices, profiles=profiles, split_halves=split,
                split_delta={h: sum(s["delta"][h] for s in split) / 2 for h in HEADS})


def bootstrap(states, values, draws=5000):
    import numpy as np
    maps = sorted({s["map_id"] for s in states})
    require(len(values) == len(states) and len(states) > 0, "bootstrap alignment")
    totals = np.array([sum(v for s, v in zip(states, values) if s["map_id"] == m) for m in maps])
    counts = np.array([sum(s["map_id"] == m for s in states) for m in maps])
    picks = np.random.default_rng(20260916).integers(0, len(maps), size=(draws, len(maps)))
    means = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return dict(mean=sum(values) / len(values), ci95=np.quantile(means, [.025, .975]).tolist(),
                map_means={m: float(t / n) for m, t, n in zip(maps, totals, counts)})


def summarize(states):
    out = dict(states=len(states), maps=len({s["map_id"] for s in states}))
    out["choices"] = {name: {metric: sum(s["choices"][name][metric] for s in states) / len(states)
        for metric in (*HEADS, "regret", "pareto_hit")} for name in states[0]["choices"]}
    out["decomposition"] = {}
    for profile in ("dynamic", "history"):
        parts = {key: bootstrap(states, [s["profiles"][profile][key] for s in states])
                 for key in ("score_cost", "prediction_residual", "total_gap")}
        parts["shapley_gains"] = {h: bootstrap(states, [s["profiles"][profile]["shapley_gains"][h] for s in states]) for h in HEADS}
        parts["risk_changes_choice"] = sum(s["profiles"][profile]["risk_changes_choice"] for s in states)
        parts["actual_excess_over_one_pair"] = sum(s["profiles"][profile]["actual_excess_over_one_pair"] for s in states)
        out["decomposition"][profile] = parts
    out["split_half_delta"] = {h: bootstrap(states, [s["split_delta"][h] for s in states]) for h in HEADS}
    out["empirical_risk_nonpareto_states"] = sum(not s["choices"]["empirical_risk"]["pareto_hit"] for s in states)
    return out


def load_verified():
    from scripts.train_sa_history_selector import verify, index_rows
    plan = verify(ROOT / "configs/sa_history_selector_corrected.json")
    for name, digest in EXPECTED.items():
        require(sha256_file(SOURCE / name) == digest, "frozen input changed: " + name)
    report = read_json(SOURCE / "report.json")
    status = read_json(SOURCE / "status.json")
    require(status["status"] == "complete" and status["report_sha256"] == EXPECTED["report.json"], "training status")
    require(report["binding"] == plan["binding"], "source binding")
    for bundle in report["bundles"].values():
        require(sha256_file(ROOT / bundle["path"]) == bundle["sha256"], "model changed")
    rows = [json.loads(line) for line in (SOURCE / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    rebuilt, excluded = index_rows(plan)
    require(not excluded and rows == rebuilt, "receipt/index mismatch")
    predictions = [json.loads(line) for line in (SOURCE / "oof_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    lookup = {(p["state_id"], p["profile"], p["candidate_id"]): p["prediction"] for p in predictions}
    expected_keys = {(r["state_id"], profile, r["candidate_id"]) for r in rows for profile in ("dynamic", "history")}
    require(len(lookup) == len(predictions) and set(lookup) == expected_keys, "prediction coverage")
    groups = defaultdict(list)
    for row in rows:
        groups[row["state_id"]].append(row)
    states = [analyze_state(groups[sid], {(profile, cid): p for (s, profile, cid), p in lookup.items() if s == sid}) for sid in sorted(groups)]
    original = {s["state_id"]: s for s in report["state_predictions"]}
    require(len(states) == 29 and len(rows) == 174, "source size drift")
    for s in states:
        for name in ("frozen", "dynamic", "history", "history_conflict_only"):
            require(s["choices"][name]["candidate_id"] == original[s["state_id"]]["choices"][name]["candidate_id"], "original choice drift")
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/sa-history-score-diagnosis-v1")
    args = parser.parse_args()
    out = args.output.resolve()
    require((ROOT / "build").resolve() in out.parents and not out.exists(), "new output inside build required")
    states = load_verified()
    result = dict(schema="lns2.sa_history_score_diagnosis.v1", post_hoc=True, runtime_allowed=False,
                  new_training_runs=0, new_solver_calls=0, input_sha256=EXPECTED,
                  implementation_sha256=sha256_file(Path(__file__)),
                  protocol_sha256=sha256_file(ROOT / "docs/SA_HISTORY_SCORE_DIAGNOSIS_ZH.md"),
                  summary=summarize(states),
                  by_stratum={s: summarize([r for r in states if r["stratum"] == s]) for s in sorted({r["stratum"] for r in states})},
                  states=states, decision="diagnostic_only_no_policy_promotion",
                  limitation="empirical_four_trial_oracle_not_expected_value_or_long_term_value")
    out.mkdir()
    write_json(out / "report.json", result)
    write_json(out / "status.json", dict(status="complete", report_sha256=sha256_file(out / "report.json")))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
