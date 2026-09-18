"""Pure map-bootstrap models and selection; no solver or experiment runner."""
from collections import Counter
import math

from experiments._common import json_fingerprint
from experiments.sa_linear_closed_loop import training_arrays
from experiments.sa_paired_closed_loop import choose as paired_choose
from experiments.sa_paired_completion import MODEL_PARAMS, PairedCompletionModel, require


ARMS = ("frozen", "gbdt", "posterior", "uniform")
MEMBER_COUNT = 20
MAP_DRAWS = 8
SCHEMA = "lns2.sa.bootstrap_closed_loop.v1"
CENSORED = {"wall_safety", "incomplete_pp", "external_timeout"}
CONTRASTS = (("posterior", "frozen"), ("posterior", "gbdt"), ("posterior", "uniform"),
             ("gbdt", "frozen"), ("gbdt", "uniform"), ("uniform", "frozen"))


def fit_full_members(data, cfg):
    """Return {models: 20 PairedCompletionModel objects, metadata: JSON data}.

    Metadata draws/train_ids/weights/state_weights are indexed by member;
    weights are the actual oriented-pair sample weights passed to fit.
    Zero-frequency maps are excluded, not passed to sklearn with zero weight.
    Only bootstrap draws use cfg['model_seed']; estimator seeds stay frozen.
    """
    import numpy as np
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor

    seed = cfg["model_seed"]
    require(type(seed) is int and seed >= 0, "invalid model_seed")
    require(sklearn.__version__ == "1.5.0", "registered sklearn required")
    arrays = training_arrays(data)  # Includes the existing dataset validation.
    states = sorted(data["states"], key=lambda s: s["state_id"])
    maps = arrays["train_maps"]
    require(len(states) == 47 and len(maps) == MAP_DRAWS, "expected 47 states on 8 maps")
    require(arrays["train_ids"] == [s["state_id"] for s in states], "training state order")

    # training_arrays emits both orientations contiguously for each state.
    blocks = []
    offset = 0
    for state in states:
        count = len(state["candidates"])
        end = offset + count * (count - 1)
        blocks.append((state, offset, end))
        offset = end
    require(offset == len(arrays["pairwise"]) == len(arrays["y"]) == len(arrays["weights"]),
            "training pair block mismatch")

    rng = np.random.default_rng(seed)
    draws = [[maps[int(i)] for i in row]
             for row in rng.integers(0, len(maps), size=(MEMBER_COUNT, MAP_DRAWS))]
    models, train_ids, weights, state_weights = [], [], [], []
    for draw in draws:
        frequencies = Counter(draw)
        x, y, w, ids, sw = [], [], [], [], {}
        for state, start, end in blocks:
            frequency = frequencies[state["map_id"]]
            if not frequency:
                continue
            sid = state["state_id"]
            ids.append(sid)
            sw[sid] = arrays["state_weights"][sid] * frequency
            x.extend(arrays["pairwise"][start:end])
            y.extend(arrays["y"][start:end])
            w.extend(value * frequency for value in arrays["weights"][start:end])
        # A tied bootstrap sample remains valid; never redraw based on labels.
        estimator = HistGradientBoostingRegressor(loss="squared_error", **MODEL_PARAMS)
        estimator.fit(x, y, sample_weight=w)
        models.append(PairedCompletionModel(list(data["feature_names"]), estimator))
        train_ids.append(ids)
        weights.append(w)
        state_weights.append(sw)
    metadata = dict(schema=SCHEMA, dataset_schema=data["schema"],
                    feature_names=list(data["feature_names"]),
                    continuation_binding=data["continuation_binding"],
                    model_seed=seed, model_params=dict(MODEL_PARAMS),
                    member_count=MEMBER_COUNT, map_draws=MAP_DRAWS, train_maps=maps,
                    draws=draws, train_ids=train_ids, weights=weights,
                    state_weights=state_weights, calibrated_confidence=False)
    return dict(models=models, metadata=metadata)


def _selection_index(arm, pair_id, decision, seed, count):
    """Separate hash stream with rejection sampling to avoid modulo bias."""
    limit = (1 << 256) - (1 << 256) % count
    attempt = 0
    while True:
        digest = json_fingerprint([SCHEMA, "selection", arm, pair_id, decision, seed, attempt])
        value = int(digest, 16)
        if value < limit:
            return value % count, digest
        attempt += 1


def _validate_model(model, names=None):
    require(isinstance(model, PairedCompletionModel), "invalid paired member/model")
    require(model.feature_names and model.feature_names == sorted(set(model.feature_names))
            and all(isinstance(n, str) and n for n in model.feature_names), "invalid model schema")
    require(callable(getattr(model.estimator, "predict", None)), "invalid member estimator")
    if names is not None:
        require(model.feature_names == names, "member schema mismatch")


def choose(arm, candidates, anchor, gbdt, members, features, known_ids,
           pair_id, decision, seed):
    """Select without consulting outcomes or consuming PP/SA random streams.

    members is the list in fit_full_members()['models']. selection_draw is the
    accepted SHA-256 hex digest; selection_index indexes members (posterior)
    or sorted candidate IDs (uniform). Deterministic arms/singletons use None.
    Uniform scores are zero placeholders, not model values or probabilities.
    The hash includes arm: posterior and uniform have algorithm-specific
    selection streams. Neither changes or consumes the paired PP/SA streams.
    """
    require(arm in ARMS, "unknown arm")
    require(isinstance(pair_id, str) and pair_id, "invalid pair_id")
    require(type(decision) is int and decision >= 0, "invalid decision")
    require(type(seed) is int and seed >= 0, "invalid selection seed")
    ids = [c["candidate_id"] for c in candidates]
    require(ids and all(isinstance(i, str) and i for i in ids)
            and len(ids) == len(set(ids)) and anchor in ids, "candidate identity")
    require(known_ids and all(type(i) is int and i >= 0 for i in known_ids)
            and len(known_ids) == len(set(known_ids)), "invalid known agent IDs")
    require(len(features) == len(candidates), "candidate feature count")
    memberships = set()
    for candidate, row in zip(candidates, features, strict=True):
        agents = candidate["agents"]
        require(agents and all(type(i) is int and i >= 0 for i in agents)
                and len(agents) == len(set(agents)) and set(agents) <= set(known_ids),
                "invalid candidate agents")
        membership = tuple(sorted(agents))
        require(membership not in memberships, "duplicate physical candidate")
        memberships.add(membership)
        require(row and all(isinstance(n, str) and n for n in row)
                and all(type(v) in (int, float) and math.isfinite(v) for v in row.values()),
                "invalid candidate features")

    model = None
    if arm in ("frozen", "gbdt"):
        _validate_model(gbdt)
        model = gbdt
    elif arm == "posterior":
        require(isinstance(members, (list, tuple)) and len(members) == MEMBER_COUNT,
                "expected 20 bootstrap members")
        for member in members:
            _validate_model(member)
        for member in members[1:]:
            require(member.feature_names == members[0].feature_names, "member schema mismatch")
        model = members[0]
    names = model.feature_names if model is not None else sorted(features[0])
    require(all(set(row) == set(names) for row in features), "candidate feature schema")

    member_index = selection_index = selection_draw = None
    if len(ids) == 1:
        ranking = dict(selected=anchor, scores={anchor: 0.}, calibrated_confidence=False)
    elif arm == "uniform":
        selection_index, selection_draw = _selection_index(arm, pair_id, decision, seed, len(ids))
        ranking = dict(selected=sorted(ids)[selection_index],
                       scores={i: 0. for i in sorted(ids)}, calibrated_confidence=False)
    else:
        if arm == "posterior":
            member_index, selection_draw = _selection_index(arm, pair_id, decision, seed, MEMBER_COUNT)
            selection_index = member_index
            model = members[member_index]
        ranking = dict(paired_choose("frozen" if arm == "frozen" else "paired",
                                    candidates, anchor, model, features, known_ids, pair_id, seed))
    require(ranking["selected"] in ids and set(ranking["scores"]) == set(ids),
            "ranking action membership")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in ranking["scores"].values()),
            "nonfinite ranking scores")
    ranking.update(member_index=member_index, selection_index=selection_index,
                   selection_draw=selection_draw, calibrated_confidence=False)
    return ranking


def _count(value):
    return type(value) is int and value >= 0


def _nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _cohort(episodes, cfg):
    require(episodes, "empty cohort")
    grouped = {}
    optional_counts = ("physical_revisits", "changed_from_anchor", "consecutive_repeats", "rollbacks")
    for row in episodes:
        require(all(isinstance(row[k], str) and row[k]
                    for k in ("pair_id", "map_id", "initial_fingerprint")), "missing episode identity")
        require(row["arm"] in ARMS, "unknown episode arm")
        require(type(row["success"]) is bool and row["stop"] in CENSORED | {
            "feasible", "node_budget", "decision_budget"}, "invalid termination")
        require(row["success"] == (row["stop"] == "feasible"), "termination mismatch")
        require(all(_count(row[k]) for k in ("decisions", "generated")), "invalid work counts")
        for name in optional_counts:
            if row.get(name) is not None:
                require(_count(row[name]) and row[name] <= row["decisions"], "invalid " + name)
        for name in ("final_soc", "final_makespan"):
            if row.get(name) is not None:
                require(_nonnegative(row[name]), "invalid " + name)
        if row.get("mean_feature_outside") is not None:
            require(_nonnegative(row["mean_feature_outside"]) and row["mean_feature_outside"] <= 1,
                    "invalid range diagnostic")
        arms = grouped.setdefault(row["pair_id"], {})
        require(row["arm"] not in arms, "duplicate pair arm")
        arms[row["arm"]] = row

    pairs = []
    for key, arms in sorted(grouped.items()):
        require(set(arms) == set(ARMS), "incomplete four-arm pair")
        for field in ("map_id", "initial_fingerprint", "solver_seed", "task_id"):
            if any(field in r for r in arms.values()):
                require(all(field in r for r in arms.values())
                        and len({r[field] for r in arms.values()}) == 1, "paired " + field + " mismatch")
        pairs.append(dict(pair_id=key, map_id=arms["frozen"]["map_id"],
                          arms={arm: arms[arm] for arm in ARMS}))
    maps = sorted({p["map_id"] for p in pairs})
    for name, actual in (("maps", len(maps)), ("paired_tasks", len(pairs)),
                         ("pairs", len(pairs)), ("episodes", len(episodes))):
        if name in cfg:
            require(_count(cfg[name]) and cfg[name] > 0 and cfg[name] == actual,
                    "configured " + name + " count mismatch")
    if "arms" in cfg:
        require(tuple(cfg["arms"]) == ARMS, "configured arms mismatch")
    tasks_per_map = cfg.get("tasks_per_map")
    if "densities" in cfg:
        require(cfg["densities"], "empty configured densities")
        if tasks_per_map is None:
            tasks_per_map = len(cfg["densities"])
        else:
            require(tasks_per_map == len(cfg["densities"]), "task/density count mismatch")
    if tasks_per_map is not None:
        require(_count(tasks_per_map) and tasks_per_map > 0, "invalid tasks_per_map")
    if "solver_seeds" in cfg:
        seeds = cfg["solver_seeds"]
        require(seeds and all(_count(s) for s in seeds) and len(seeds) == len(set(seeds)),
                "invalid solver_seeds")
        if tasks_per_map is None and "maps" in cfg and ("paired_tasks" in cfg or "pairs" in cfg):
            cells = len(maps) * len(seeds)
            require(len(pairs) % cells == 0, "configured map/seed grid is incomplete")
            tasks_per_map = len(pairs) // cells
        for p in pairs:
            require(p["arms"]["frozen"].get("solver_seed") in seeds, "unregistered solver seed")
        if tasks_per_map is not None:
            for m in maps:
                actual = Counter(p["arms"]["frozen"]["solver_seed"] for p in pairs if p["map_id"] == m)
                require(actual == Counter({s: tasks_per_map for s in seeds}), "map/seed task count mismatch")
    return pairs, maps


def _diagnostics(rows):
    result = {}
    for name in ("changed_from_anchor", "physical_revisits", "consecutive_repeats", "rollbacks"):
        reported = [r for r in rows if r.get(name) is not None]
        total = sum(r[name] for r in reported)
        decisions = sum(r["decisions"] for r in reported)
        result[name] = dict(reported_episodes=len(reported), missing_episodes=len(rows) - len(reported),
                            observed_total=total, total=total if len(reported) == len(rows) else None,
                            observed_decisions=decisions, fraction=total / decisions if decisions else None)
    reported = [r for r in rows if r.get("mean_feature_outside") is not None]
    decisions = sum(r["decisions"] for r in reported)
    result["mean_feature_outside"] = dict(
        reported_episodes=len(reported), missing_episodes=len(rows) - len(reported),
        episode_mean=sum(r["mean_feature_outside"] for r in reported) / len(reported) if reported else None,
        decision_weighted_mean=sum(r["mean_feature_outside"] * r["decisions"] for r in reported) / decisions
        if decisions else None)
    return result


def summarize(episodes, cfg):
    """Four-arm non-TTF report, compatible with audited linear-runner records.

    Optional cfg counts: maps, paired_tasks (or pairs), episodes. When provided,
    tasks_per_map/densities plus solver_seeds also validate each map/seed cell;
    maps and paired_tasks/pairs with solver_seeds imply a balanced cell count.
    Censor bounds and observed bootstrap deltas are map-equal; episode-equal
    bounds are separately named. Bootstrap CIs are NOT censor-adjusted.
    Optional consecutive_repeats means adjacent same-neighborhood selections;
    it is never inferred from physical_revisits. Missing diagnostics stay null.
    No solver, fitting, file writes, threshold search, or promotion is performed.
    """
    import numpy as np

    pairs, maps = _cohort(episodes, cfg)
    bootstrap, seed = cfg.get("bootstrap", 5000), cfg.get("master_seed", 0)
    require(_count(bootstrap) and bootstrap > 0 and _count(seed), "invalid summary bootstrap")
    draws = np.random.default_rng(seed).integers(0, len(maps), (bootstrap, len(maps)))
    summaries = {}
    for arm in ARMS:
        rows = [p["arms"][arm] for p in pairs]
        success = sum(r["success"] for r in rows)
        censored = sum(r["stop"] in CENSORED for r in rows)
        diagnostics = _diagnostics(rows)
        completed = [r for r in rows if r["success"]]
        summaries[arm] = dict(
            episodes=len(rows), success=success, censored=censored,
            known_noncompletion=len(rows) - success - censored,
            observed_success_rate=success / len(rows),
            completion_bounds=[success / len(rows), (success + censored) / len(rows)],
            stops=dict(sorted(Counter(r["stop"] for r in rows).items())),
            decisions=sum(r["decisions"] for r in rows), generated=sum(r["generated"] for r in rows),
            revisits=diagnostics["physical_revisits"]["total"],
            changed_from_anchor=diagnostics["changed_from_anchor"]["total"],
            repeats=diagnostics["consecutive_repeats"]["total"], diagnostics=diagnostics,
            completion_curve={str(k): sum(r["success"] and r["decisions"] <= k for r in rows)
                              for k in (32, 64, 128, 256, 512)},
            successful_decision_median=float(np.median([r["decisions"] for r in completed])) if completed else None)

    contrasts = {}
    for challenger, baseline in CONTRASTS:
        per_map, map_bounds, episode_bounds, common = {}, {}, [], []
        wins = losses = known_pairs = 0
        for m in maps:
            values, bounds = [], []
            for p in (p for p in pairs if p["map_id"] == m):
                a, b = p["arms"][challenger], p["arms"][baseline]
                unknown_a, unknown_b = a["stop"] in CENSORED, b["stop"] in CENSORED
                values.append(int(a["success"]) - int(b["success"]))
                bounds.append(((0 if unknown_a else int(a["success"])) - (1 if unknown_b else int(b["success"])),
                               (1 if unknown_a else int(a["success"])) - (0 if unknown_b else int(b["success"]))))
                if not unknown_a and not unknown_b:
                    known_pairs += 1
                    wins += int(a["success"] and not b["success"])
                    losses += int(b["success"] and not a["success"])
                if a["success"] and b["success"]:
                    common.append((p["pair_id"], a, b))
            per_map[m] = sum(values) / len(values)
            map_bounds[m] = [sum(v[i] for v in bounds) / len(bounds) for i in (0, 1)]
            episode_bounds.extend(bounds)
        values = np.asarray(list(per_map.values()))
        metrics = {}
        for name, field in (("generated", "generated"), ("decisions", "decisions"),
                            ("soc", "final_soc"), ("makespan", "final_makespan")):
            available = [(a, b) for _, a, b in common if a.get(field) is not None and b.get(field) is not None]
            complete = bool(common) and len(available) == len(common)
            numerator = sum(a[field] for a, _ in available) if complete else None
            denominator = sum(b[field] for _, b in available) if complete else None
            metrics[name] = dict(reported_pairs=len(available), missing_pairs=len(common) - len(available),
                                 challenger_sum=numerator, baseline_sum=denominator,
                                 ratio=numerator / denominator if denominator else None)
        contrasts[challenger + "_vs_" + baseline] = dict(
            challenger=challenger, baseline=baseline, observed_map_delta=float(values.mean()),
            observed_ci95=np.quantile(values[draws].mean(1), [.025, .975]).tolist(),
            ci_target="observed_completion_not_censor_adjusted", per_map=per_map,
            map_wins=int(sum(values > 0)), map_losses=int(sum(values < 0)), map_ties=int(sum(values == 0)),
            known_pairs=known_pairs, known_pair_wins=wins, known_pair_losses=losses,
            censor_delta_bounds=[sum(v[i] for v in map_bounds.values()) / len(maps) for i in (0, 1)],
            episode_censor_delta_bounds=[sum(v[i] for v in episode_bounds) / len(pairs) for i in (0, 1)],
            per_map_censor_delta_bounds=map_bounds, common_success=len(common),
            common_success_pair_ids=[key for key, _, _ in common], common_success_metrics=metrics,
            **{"common_" + name + "_ratio": record["ratio"] for name, record in metrics.items()})

    positive = [arm for arm in ARMS if arm != "frozen"
                and contrasts[arm + "_vs_frozen"]["observed_map_delta"] > 0]
    work = [arm for arm in ARMS if arm != "frozen"
            and contrasts[arm + "_vs_frozen"]["observed_map_delta"] == 0
            and contrasts[arm + "_vs_frozen"]["common_generated_ratio"] is not None
            and contrasts[arm + "_vs_frozen"]["common_generated_ratio"] < 1]
    decision = ("resource_censored_inconclusive" if any(s["censored"] for s in summaries.values())
                else "bounded_positive_signal" if "posterior" in positive
                else "bounded_work_signal" if "posterior" in work
                else "no_positive_completion_or_work_signal")
    return dict(schema=SCHEMA, summary=summaries, contrasts=contrasts,
                primary_contrasts=["posterior_vs_" + arm for arm in ("frozen", "gbdt", "uniform")],
                maps=len(maps), paired_tasks=len(pairs), episodes=len(episodes),
                positive_observed_completion=positive, positive_observed_work=work,
                decision=decision, decision_is_descriptive=True, no_ttf=True,
                production_changed=False, automatic_promotion=False, paired_cases=pairs)
