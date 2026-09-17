"""Post-hoc description only; never selects models or alters rollout labels."""
import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file
from experiments.sa_paired_completion import require
from experiments.sa_policy_aligned_update import aggregate
from scripts.run_sa_paired_closed_loop import check_seal, once, sealed
from scripts.run_sa_policy_aligned_update import verify, root_read, check_result


def describe_root(root, rows, label, new_choice):
    ids = [c["candidate_id"] for c in root["candidates"]]
    target = label["target"]
    require(label["complete"] and set(target) == set(ids), "incomplete labels")
    halves = []
    for trial_ids in ({0, 1, 2, 3}, {4, 5, 6, 7}):
        halves.append({cid: sum(int(r["success"]) for r in rows
                                 if r["candidate_id"] == cid and r["trial"] in trial_ids)/8
                       for cid in ids})
    best = lambda rates: min(ids, key=lambda cid: (-rates[cid], cid))
    anchor = root["anchor_id"]
    gain = sum(halves[1-i][best(halves[i])]-halves[1-i][anchor] for i in (0, 1))/2
    informative, repeated = 0, 0
    for a, b in combinations(ids, 2):
        x, y = halves[0][a]-halves[0][b], halves[1][a]-halves[1][b]
        informative += int(x != 0 or y != 0)
        repeated += int(x*y > 0)
    old_choice = root["source_event"]["ranking"]["selected"]
    return dict(root_id=root["id"], map_id=root["map_id"], decision=root["decision"],
        initial_conflicts=root["state"]["num_of_colliding_pairs"], target=target,
        rates=label["rates"], all_tied=len(set(target.values())) == 1,
        all_zero=all(v == 0 for v in target.values()),
        first_half=halves[0], second_half=halves[1],
        half_informative_pairs=informative, half_repeated_strict_pairs=repeated,
        cross_half_gain_vs_anchor=gain,
        choices=dict(anchor=anchor, old_gbdt=old_choice, updated=new_choice),
        training_completion={"anchor": target[anchor], "old_gbdt": target[old_choice],
            "updated": target[new_choice], "uniform": sum(target.values())/len(ids),
            "sample_oracle": max(target.values())},
        training_updated_best=target[new_choice] == max(target.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args()
    if args.recovery:
        from scripts.recover_sa_policy_aligned_update_evaluation import active, CONFIG
        active.CONFIG = CONFIG
    plan, out = verify()
    cfg = plan["config"]
    source = ROOT/plan["recovery"]["source"] if "recovery" in plan else out
    require(cfg["trials"] == 8 and cfg["reference_weight"] == .5, "descriptive split protocol changed")
    audit = check_seal(read_json(source/"labels.audit.json"))
    report = check_seal(read_json(out/"report.json"))
    training = check_seal(read_json(out/"model/training.json"))
    require(all(r["binding"] == plan["binding"] for r in (report, training)), "binding mismatch")
    label_plan = read_json(source/"plan.json")
    require(audit["binding"] == label_plan["binding"], "label binding mismatch")
    for name, digest in training["files"].items():
        require(sha256_file(out/"model"/name) == digest, "model artifact changed")
    fixtures = read_json(out/"model/fixtures.json")
    roots = [root_read(e, plan) for e in plan["roots"]]
    require(len(roots) == len(fixtures), "fixture coverage changed")
    results = []
    for jid, digest in audit["results"].items():
        folder = source/"labels"/jid
        require(sha256_file(folder/"result.json") == digest, "audited result changed")
        results.append(check_result(folder, label_plan))
    labels = {r["root_id"]:r for r in audit["roots"]}
    rows = []
    for root, fixture in zip(roots, fixtures, strict=True):
        require([c["candidate_id"] for c in fixture["state"]["candidates"]] ==
                [c["candidate_id"] for c in root["candidates"]], "fixture order")
        subset = [r for r in results if r["root_id"] == root["id"]]
        require(aggregate(root, subset, cfg) == labels[root["id"]], "aggregate changed")
        rows.append(describe_root(root, subset, labels[root["id"]], fixture["ranking"]["selected"]))
    metrics = list(rows[0]["training_completion"])
    summary = dict(roots=len(rows), maps=len({r["map_id"] for r in rows}),
        branches=len(results), repairs=sum(r["decisions"] for r in results),
        prefix_repairs=plan["replay_repairs"],
        all_tied=sum(r["all_tied"] for r in rows), all_zero=sum(r["all_zero"] for r in rows),
        roots_with_contrast=sum(not r["all_tied"] for r in rows),
        training_updated_best=sum(r["training_updated_best"] for r in rows),
        training_updated_best_nonflat=sum(r["training_updated_best"] and not r["all_tied"] for r in rows),
        training_completion={k:sum(r["training_completion"][k] for r in rows)/len(rows) for k in metrics},
        cross_half_gain_vs_anchor=sum(r["cross_half_gain_vs_anchor"] for r in rows)/len(rows),
        half_informative_pairs=sum(r["half_informative_pairs"] for r in rows),
        half_repeated_strict_pairs=sum(r["half_repeated_strict_pairs"] for r in rows),
        continuation_success={a:sum(r["success"] for r in results if r["arm"] == a) for a in cfg["continuations"]})
    evidence_paths = [source/"labels.audit.json"]+[out/name for name in (
        "plan.json", "report.json", "model/training.json", "model/fixtures.json")]
    evidence = {p.relative_to(ROOT).as_posix():sha256_file(p) for p in evidence_paths}
    payload = dict(binding=plan["binding"], evidence=evidence, summary=summary, roots=rows,
        evaluation=report["summary"], contrasts=report["contrasts"],
        diagnostics_source_sha256=sha256_file(Path(__file__)),
        post_hoc=True, independent_confirmation=False, model_selection=False, no_ttf=True)
    once(out/"diagnostics.json", sealed(payload))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
