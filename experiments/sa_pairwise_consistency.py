"""Read-only diagnostics for an unchanged paired-completion model."""
from itertools import combinations
import math

from experiments.sa_paired_completion import PairedCompletionModel, _candidates, require


def select(scores, anchor):
    return min(scores, key=lambda c: (-scores[c], c != anchor, c))


def matrix_diagnostics(ids, margins, anchor):
    ids = sorted(ids)
    require(ids and len(ids) == len(set(ids)) and anchor in ids, "candidate identity")
    require(set(margins) == set(combinations(ids, 2)), "incomplete pair matrix")
    require(all(math.isfinite(v) and -1 <= v <= 1 for v in margins.values()), "invalid margin")
    n = len(ids)
    scores = dict.fromkeys(ids, 0.)
    for (a, b), value in margins.items():
        scores[a] += value / (n-1)
        scores[b] -= value / (n-1)
    def margin(a, b):
        return 0. if a == b else margins[(a, b)] if a < b else -margins[(b, a)]
    winner = select(scores, anchor)
    direct = {c: margin(c, anchor) for c in ids}
    cycles = []
    triangle_sums = []
    for a, b, c in combinations(ids, 3):
        edges = (margin(a,b), margin(b,c), margin(c,a))
        triangle_sums.append(sum(edges))
        if all(v > 0 for v in edges) or all(v < 0 for v in edges):
            cycles.append([a,b,c])
    # Complete equal-weight pair graph: least-squares utility projection.
    # It is a positive scaling of the existing Borda scores, not a new ranker.
    utilities = {c: scores[c]*(n-1)/n for c in ids}
    residuals = [v-(utilities[a]-utilities[b]) for (a,b),v in margins.items()]
    deletions = []
    for removed in ids:
        if removed in {winner, anchor} or n <= 2:
            continue
        retained = [c for c in ids if c != removed]
        sub_scores = dict.fromkeys(retained, 0.)
        for a,b in combinations(retained, 2):
            value = margin(a,b)/(len(retained)-1)
            sub_scores[a] += value
            sub_scores[b] -= value
        deletions.append(dict(removed=removed, selected=select(sub_scores,anchor)))
    selected_margin = direct[winner]
    return dict(selected=winner, scores=scores, selected_anchor_margin=selected_margin,
        changed=winner != anchor, selected_loses_anchor=winner != anchor and selected_margin < 0,
        selected_ties_anchor=winner != anchor and selected_margin == 0,
        direct_anchor_selected=select(direct,anchor),
        veto_selected=winner if winner == anchor or selected_margin > 0 else anchor,
        direct_anchor_margins=direct, cycles=cycles,
        maximum_triangle_residual=max(map(abs,triangle_sums), default=0.),
        projection_rmse=math.sqrt(sum(v*v for v in residuals)/len(residuals)) if residuals else 0.,
        utilities=utilities, deletions=deletions,
        deletion_changes=sum(d["selected"] != winner for d in deletions),
        pairs=[dict(left=a,right=b,margin=v) for (a,b),v in margins.items()])


def inspect(model, state, expected=None):
    if len(state["candidates"]) == 1:
        c = state["candidates"][0]
        require(c["candidate_id"] == state["anchor_id"] and c["agents"] and
                len(c["agents"]) == len(set(c["agents"])) and set(c["agents"]) <= set(state["agent_ids"]),
                "invalid singleton")
        require(set(c["features"]) == set(model.feature_names) and
                all(math.isfinite(v) for v in c["features"].values()), "singleton feature schema")
        result = matrix_diagnostics([c["candidate_id"]], {}, state["anchor_id"])
        if expected is not None:
            require(expected == dict(selected=c["candidate_id"], scores=result["scores"],
                                     calibrated_confidence=False), "singleton ranking changed")
        return result
    calls = []
    class Capture:
        def predict(self, vectors):
            values = model.estimator.predict(vectors)
            calls.append([float(v) for v in values])
            return values
    candidates = _candidates(state, model.feature_names)
    ranked = PairedCompletionModel(model.feature_names,Capture()).rank(state)
    if expected is not None:
        require(ranked == expected, "frozen inference/selection changed")
    ids = [c["candidate_id"] for c in candidates]
    pairs = list(combinations(ids,2))
    require(len(calls) == 2 and all(len(v) == len(pairs) for v in calls), "prediction calls")
    margins = {pair:max(-1.,min(1.,f/2-r/2)) for pair,f,r in zip(pairs,*calls,strict=True)}
    result = matrix_diagnostics(ids,margins,state["anchor_id"])
    require(result["scores"] == ranked["scores"] and result["selected"] == ranked["selected"],
            "diagnostic did not reproduce production ranking")
    return result


def evaluate(row, values):
    require(set(values) == set(row["scores"]), "label candidate mismatch")
    lengths = {len(v) for v in values.values()}
    require(len(lengths) == 1 and next(iter(lengths)) > 0 and next(iter(lengths)) % 2 == 0,
            "incomplete paired trials")
    require(all(type(v) in (int,bool) and v in (0,1) for vals in values.values() for v in vals),
            "censored/invalid outcome must not be treated as failure")
    rates = {c:sum(v)/len(v) for c,v in values.items()}
    choices = dict(anchor=row["anchor_id"], model=row["selected"],
        direct=row["direct_anchor_selected"], veto=row["veto_selected"])
    selected, anchor = choices["model"], choices["anchor"]
    half = next(iter(lengths))//2
    differences = [sum(values[selected][i:i+half])/half-sum(values[anchor][i:i+half])/half
                   for i in (0,half)]
    truth_pairs = [dict(left=p["left"],right=p["right"],prediction=p["margin"],
                        observed=rates[p["left"]]-rates[p["right"]]) for p in row["pairs"]]
    return dict(rates=rates, choices=choices, choice_rates={k:rates[c] for k,c in choices.items()},
        uniform=sum(rates.values())/len(rates), empirical_oracle=max(rates.values()),
        selected_minus_anchor=rates[selected]-rates[anchor], half_differences=differences,
        loss_in_both_halves=all(v < 0 for v in differences),
        gain_in_both_halves=all(v > 0 for v in differences), pairs=truth_pairs)


def counts(rows):
    return dict(decisions=len(rows), changed=sum(r["changed"] for r in rows),
        selected_loses_anchor=sum(r["selected_loses_anchor"] for r in rows),
        selected_ties_anchor=sum(r["selected_ties_anchor"] for r in rows),
        cyclic=sum(bool(r["cycles"]) for r in rows),
        deletion_sensitive=sum(r["deletion_changes"] > 0 for r in rows),
        direct_differs=sum(r["direct_anchor_selected"] != r["selected"] for r in rows),
        veto_differs=sum(r["veto_selected"] != r["selected"] for r in rows),
        projection_rmse_mean=sum(r["projection_rmse"] for r in rows)/len(rows) if rows else None)


def label_summary(rows, arm, seed, bootstrap):
    import numpy as np
    maps = sorted({r["map_id"] for r in rows})
    require(rows, "no labeled states")
    per_map = []
    for m in maps:
        group = [r["labels"][arm] for r in rows if r["map_id"] == m]
        rates = {k:sum(r["choice_rates"][k] for r in group)/len(group)
                 for k in ("anchor","model","direct","veto")}
        per_map.append(dict(map_id=m,states=len(group),rates=rates))
    draws = np.random.default_rng(seed).integers(0,len(maps),(bootstrap,len(maps)))
    contrasts = {}
    for challenger,baseline in (("model","anchor"),("direct","model"),("veto","model")):
        values = np.array([r["rates"][challenger]-r["rates"][baseline] for r in per_map])
        contrasts[challenger+"_minus_"+baseline] = dict(delta=float(values.mean()),
            ci95=np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist(),
            wins=int(sum(values>0)), losses=int(sum(values<0)), ties=int(sum(values==0)))
    labels = [r["labels"][arm] for r in rows]
    return dict(states=len(rows),maps=len(maps),per_map=per_map,contrasts=contrasts,
        state_mean_rates={k:sum(r["choice_rates"][k] for r in labels)/len(labels)
                          for k in ("anchor","model","direct","veto")},
        selected_loss_states=sum(r["selected_minus_anchor"]<0 for r in labels),
        selected_gain_states=sum(r["selected_minus_anchor"]>0 for r in labels),
        loss_in_both_halves=sum(r["loss_in_both_halves"] for r in labels),
        gain_in_both_halves=sum(r["gain_in_both_halves"] for r in labels))
