"""Post-hoc action-support inventory; no new labels, fits, or policy changes."""
import json
from itertools import combinations
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file
from scripts.audit_sa_history_information import atomic, require
from scripts.collect_sa_history_candidate_bridge import verify

TARGETS = ("sustained_progress", "completion")


def support(rows, size, conflicts, excluded_map):
    train = [r for r in rows if r["map_id"] != excluded_map]
    same_size = [r for r in train if r["base"]["proposal.actual_size"] == size]
    # Descriptive overlap only, not an admission threshold or a policy rule.
    comparable = [r for r in same_size if conflicts / 2 <= r["current_conflicts"] <= conflicts * 2]
    return dict(train_rows=len(train), same_size_rows=len(same_size),
                same_size_episodes=len({r["episode"] for r in same_size}),
                same_size_maps=len({r["map_id"] for r in same_size}),
                same_size_comparable_conflict_rows=len(comparable))


def directions(predictions, labels, target):
    rows = []
    for a, b in combinations(sorted(labels), 2):
        require(len(labels[a]) == len(labels[b]) == 8, "trial coverage")
        half = [sum(labels[a][i][target] - labels[b][i][target]
                    for i in range(start, start + 4)) / 4 for start in (0, 4)]
        difference = sum(half) / 2
        predicted = predictions[a] - predictions[b]
        rows.append(dict(pair=[a, b], observed_difference=difference,
                         predicted_difference=predicted, half_differences=half,
                         stable_strict=half[0] * half[1] > 0,
                         prediction_strict=predicted != 0,
                         agrees=predicted * difference > 0))
    return rows


def build(train, preflight, report):
    rows = [r for r in train if r["labels"] is not None]
    distribution = []
    for size in sorted({r["base"]["proposal.actual_size"] for r in rows}):
        selected = [r for r in rows if r["base"]["proposal.actual_size"] == size]
        conflicts = [r["current_conflicts"] for r in selected]
        distribution.append(dict(size=int(size), rows=len(selected),
            episodes=len({r["episode"] for r in selected}), maps=len({r["map_id"] for r in selected}),
            conflicts_min=min(conflicts), conflicts_median=median(conflicts), conflicts_max=max(conflicts),
            positive={t:sum(r["labels"][t] for r in selected) for t in TARGETS}))
    states = []
    for entry, result in zip(preflight["roots"], report["states"], strict=True):
        require(entry["target"]["id"] == result["id"], "root order mismatch")
        require(not result["censored"], "do not invent missing candidate labels")
        candidates = {r["candidate_id"]:r for r in entry["rows"]}
        ids = list(candidates)
        predictions = {name:dict(zip(ids, ps, strict=True)) for name, ps in entry["predictions"].items()}
        conflicts = entry["rows"][0]["base"]["state.colliding_pairs"]
        selected = dict(frozen=entry["old_selected_id"], **entry["choices"])
        selection = {}
        for name, cid in selected.items():
            actual = result["labels"].get(cid)
            selection[name] = dict(candidate_id=cid, size=len(candidates[cid]["agents"]),
                support=support(rows, len(candidates[cid]["agents"]), conflicts, result["map_id"]),
                collected=actual is not None,
                prediction=predictions[name][cid] if name in predictions else None,
                observed_rates={t:sum(v[t] for v in actual) / 8 for t in TARGETS} if actual else None)
        direction = {name:directions(ps, result["labels"],
                     "completion" if name in ("dynamic_completion", "temporal_bag") else "sustained_progress")
                     for name, ps in predictions.items()}
        states.append(dict(id=result["id"], map_id=result["map_id"], stratum=result["stratum"],
            current_conflicts=conflicts, previous_best=entry["previous_best"], selected=selection,
            pair_directions=direction,
            constant_targets={t:len({v[t] for vs in result["labels"].values() for v in vs}) == 1 for t in TARGETS}))
    summary = {}
    for name in preflight["roots"][0]["predictions"]:
        pairs = [p for s in states for p in s["pair_directions"][name]]
        strict = [p for p in pairs if p["stable_strict"]]
        summary[name] = dict(stable_strict_pairs=len(strict),
            correct=sum(p["agrees"] for p in strict), predicted_ties=sum(not p["prediction_strict"] for p in strict),
            note="selected_three_candidate_pool_only; pairs_not_independent_samples")
    return dict(schema="sa-history-bridge-support-v1", post_hoc=True, training_rows=len(rows),
        training_size_distribution=distribution, states=states, direction_summary=summary,
        constant_target_states={t:sum(s["constant_targets"][t] for s in states) for t in TARGETS},
        no_training=True, no_solver_calls=True, no_policy_change=True,
        boundary="descriptive_support_and_within_state_directions_not_a_causal_proof_or_new_gate")


def main():
    plan, out = verify()
    paths = [ROOT / "build/sa-history-information-audit-v1/index.json",
             out / "preflight.json", out / "report.json", Path(__file__),
             ROOT / "tests/evaluation/test_sa_history_bridge_support.py"]
    inputs = {p.relative_to(ROOT).as_posix():sha256_file(p) for p in paths}
    result = build(read_json(paths[0])["result"]["rows"], read_json(paths[1]), read_json(paths[2]))
    result.update(binding=plan["binding"], inputs=inputs)
    verify()
    require(all(sha256_file(ROOT / p) == sha for p, sha in inputs.items()), "input changed during support audit")
    target = out / "support_report.json"
    if target.exists():
        require(read_json(target) == result, "existing support report differs")
    else:
        atomic(target, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("states", "inputs")}, indent=2))


if __name__ == "__main__":
    main()
