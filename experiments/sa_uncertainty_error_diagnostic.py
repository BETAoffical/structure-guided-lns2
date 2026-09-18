"""Post-hoc error ranking at fixed original actions, not policy calibration."""
from collections import Counter
import json
from pathlib import Path
import statistics

from experiments._common import read_json, sha256_file
from experiments.sa_paired_completion import require
from experiments.sa_uncertainty_audit import rates_for


def weighted_auc(rows, score, gain, multiplicity=None):
    counts = Counter(r["map_id"] for r in rows)
    multiplicity = multiplicity if multiplicity is not None else {m: 1 for m in counts}
    positives = [r for r in rows if r[gain] < 0]
    negatives = [r for r in rows if r[gain] > 0]
    numerator = denominator = 0.
    for a in positives:
        for b in negatives:
            weight = multiplicity.get(a["map_id"], 0) * multiplicity.get(b["map_id"], 0) / (
                counts[a["map_id"]] * counts[b["map_id"]])
            denominator += weight
            numerator += weight * (int(a[score] > b[score]) + .5*int(a[score] == b[score]))
    return numerator/denominator if denominator else None


def describe(rows, seed, bootstrap):
    import numpy as np
    names = ("disagreement_sd", "small_margin", "negative_votes")
    maps = sorted({r["map_id"] for r in rows})
    metrics = {name: {g: weighted_auc(rows, name, g) for g in ("gain", "first_half", "second_half")} for name in names}
    draws = np.random.default_rng(seed).integers(0, len(maps), (bootstrap, len(maps)))
    intervals, differences = {name: [] for name in names}, []
    for draw in draws:
        multiplicity = Counter(maps[i] for i in draw)
        aucs = {name: weighted_auc(rows, name, "gain", multiplicity) for name in names}
        if aucs["disagreement_sd"] is not None:
            for name, value in aucs.items():
                intervals[name].append(value)
            differences.append(aucs["disagreement_sd"]-aucs["small_margin"])
    return dict(changed=len(rows), maps=len(maps), harmful=sum(r["gain"] < 0 for r in rows),
        helpful=sum(r["gain"] > 0 for r in rows), tied=sum(r["gain"] == 0 for r in rows), auc=metrics,
        map_bootstrap_ci95={n: np.quantile(v, [.025, .975]).tolist() if v else None for n, v in intervals.items()},
        paired_auc_difference_ci95=np.quantile(differences, [.025, .975]).tolist() if differences else None,
        undefined_bootstrap=bootstrap-len(differences), total_bootstrap=bootstrap,
        posthoc=True, independent_confirmation=False, calibrated_confidence=False,
        threshold_selected=False, new_fits=0, new_solver_calls=0,
        warning="Empirical H32 label sign; ties excluded from ROC, no continuous deployment evidence.", rows=rows)


def run():
    from scripts.run_sa_uncertainty_audit import verify, check_fit, result_path
    from scripts.run_sa_paired_closed_loop import once, sealed, check_seal
    plan, out = verify()
    data = read_json(Path(__file__).resolve().parents[1] / plan["config"]["dataset"])
    report = check_seal(read_json(out / "report.json"))
    require(report["binding"] == plan["binding"], "source report binding")
    rows, inputs = [], {}
    for held in plan["maps"]:
        members = {}
        for member in range(-1, plan["config"]["members"]):
            path = result_path(out, held, member)
            digest = sha256_file(path)
            require(report["files"][path.relative_to(out).as_posix()] == digest, "source fit changed")
            inputs[path.relative_to(out).as_posix()] = digest
            fit = check_fit(plan, data, held, member, read_json(path))
            members[member] = {r["state_id"]: r for r in fit["predictions"]}
        for s in sorted(data["states"], key=lambda s: s["state_id"]):
            if s["map_id"] != held:
                continue
            point = members[-1][s["state_id"]]
            chosen, anchor = point["selected"], s["anchor_id"]
            if chosen == anchor:
                continue
            differences = [members[i][s["state_id"]]["scores"][chosen] - members[i][s["state_id"]]["scores"][anchor]
                           for i in range(plan["config"]["members"])]
            values = [rates_for(s, ids) for ids in (range(8), range(4), range(4, 8))]
            rows.append(dict(state_id=s["state_id"], map_id=held, selected=chosen, anchor=anchor,
                disagreement_sd=statistics.pstdev(differences),
                small_margin=point["scores"][anchor]-point["scores"][chosen],
                negative_votes=sum(d < 0 for d in differences)/len(differences),
                **{key: rates[chosen]-rates[anchor] for key, rates in zip(("gain", "first_half", "second_half"), values)}))
    result = describe(rows, plan["config"]["seed"], plan["config"]["bootstrap"])
    result.update(binding=plan["binding"], source_report_sha256=sha256_file(out / "report.json"),
                  implementation_sha256=sha256_file(Path(__file__)), fits=inputs)
    once(out / "fixed-action-error-diagnostic.json", sealed(result))
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "fits"}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
