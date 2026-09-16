"""Offline paired-completion prototype; not imported by a live controller."""

from dataclasses import dataclass
from itertools import combinations
import math


SCHEMA = "lns2.sa.paired_completion.v1"
MODEL_PARAMS = dict(learning_rate=0.05, max_iter=100, max_leaf_nodes=7,
                    min_samples_leaf=10, l2_regularization=0.1,
                    early_stopping=False, random_state=20260916)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _candidates(state, names):
    candidates = state["candidates"]
    ids = [c["candidate_id"] for c in candidates]
    require(len(ids) >= 2 and len(ids) == len(set(ids)), "duplicate or insufficient candidates")
    require(all(isinstance(c, str) and c for c in ids), "invalid candidate id")
    require(state["anchor_id"] in ids, "missing frozen anchor")
    known = state["agent_ids"]
    require(known and all(_integer(i) for i in known) and len(known) == len(set(known)), "invalid agent ids")
    memberships = set()
    for c in candidates:
        agents = c["agents"]
        require(agents and all(_integer(i) for i in agents) and len(agents) == len(set(agents)), "invalid neighborhood")
        require(set(agents) <= set(known), "unknown agent")
        membership = tuple(sorted(agents))
        require(membership not in memberships, "duplicate physical neighborhood")
        memberships.add(membership)
        require(set(c["features"]) == set(names), "feature schema mismatch")
        require(all(type(v) in (float, int) and math.isfinite(v) for v in c["features"].values()),
                "nonfinite or nonnumeric feature")
    return sorted(candidates, key=lambda c: c["candidate_id"])


def validate_dataset(data):
    require(data["schema"] == SCHEMA, "dataset schema mismatch")
    names = data["feature_names"]
    require(names and names == sorted(set(names)) and all(isinstance(n, str) and n for n in names),
            "feature names must be sorted and unique")
    require(_digest(data["continuation_binding"]), "missing continuation identity")
    horizon, trials = data["horizon"], data["trial_count"]
    require(_integer(horizon, 1) and _integer(trials, 2), "invalid horizon or trials")
    require(data["states"], "empty state dataset")
    ids, occurrences, episodes = set(), set(), {}
    for state in data["states"]:
        require(isinstance(state["state_id"], str) and state["state_id"], "missing state id")
        require(state["state_id"] not in ids, "duplicate state id")
        ids.add(state["state_id"])
        require(all(isinstance(state[k], str) and state[k] for k in ("episode", "map_id")), "missing map/episode")
        require(_integer(state["decision"]), "invalid occurrence decision")
        occurrence = (state["episode"], state["decision"])
        require(occurrence not in occurrences, "duplicate state occurrence")
        occurrences.add(occurrence)
        require(episodes.setdefault(state["episode"], state["map_id"]) == state["map_id"], "episode crosses maps")
        require(_digest(state["state_fingerprint"]) and _digest(state["history_fingerprint"]), "missing root identity")
        paired_keys = {}
        for c in _candidates(state, names):
            observations = c["trials"]
            require(len(observations) == trials and {r["trial"] for r in observations} == set(range(trials)),
                    "incomplete or duplicate trial grid")
            for row in observations:
                require(_integer(row["trial"]), "invalid trial id")
                require(_digest(row["randomization_key"]), "missing paired randomization identity")
                require(paired_keys.setdefault(row["trial"], row["randomization_key"]) == row["randomization_key"],
                        "unpaired trial randomization")
                require(_integer(row["steps"]) and row["steps"] <= horizon, "invalid observed step count")
                require(_integer(row["final_conflicts"]), "invalid final conflicts")
                if row["stop"] == "feasible":
                    require(row["completed"] is True and row["final_conflicts"] == 0 and row["steps"] > 0,
                            "inconsistent completion event")
                elif row["stop"] == "horizon":
                    require(row["completed"] is False and row["steps"] == horizon and row["final_conflicts"] > 0,
                            "incomplete horizon is censoring, not failure")
                else:
                    require(row["stop"] in ("node_budget", "wall_safety", "incomplete_pp", "external_timeout"),
                            "unexplained branch error")
                    require(row["completed"] is None, "censored outcome cannot become a failure label")
    return data


def paired_labels(state):
    """Keep all discordant and tied trials; no empirical winner filtering."""
    result = []
    candidates = sorted(state["candidates"], key=lambda c: c["candidate_id"])
    for a, b in combinations(candidates, 2):
        left = {r["trial"]: r for r in a["trials"]}
        right = {r["trial"]: r for r in b["trials"]}
        require(left.keys() == right.keys(), "unpaired trial ids")
        delta, censored = [], []
        for trial in sorted(left):
            x, y = left[trial], right[trial]
            require(x["randomization_key"] == y["randomization_key"], "unpaired trial randomization")
            if x["completed"] is None or y["completed"] is None:
                censored.append(trial)
            else:
                delta.append(int(x["completed"]) - int(y["completed"]))
        result.append(dict(left=a["candidate_id"], right=b["candidate_id"],
                           wins=delta.count(1), losses=delta.count(-1), ties=delta.count(0),
                           censored_trials=censored,
                           mean_difference=None if censored else math.fsum(delta) / len(delta)))
    return result


def inspect_dataset(data):
    validate_dataset(data)
    states = [dict(state_id=s["state_id"], map_id=s["map_id"], pairs=paired_labels(s)) for s in data["states"]]
    reasons = []
    if data["role"] != "prospective_development":
        reasons.append("diagnostic_source_not_training_data")
    if data["sampling"] != "outcome_blind_same_state":
        reasons.append("model_selected_candidate_pool")
    if data["source_kind"] != "registered_prospective_collection":
        reasons.append("no_prospective_collection_registration")
    if any(p["censored_trials"] for s in states for p in s["pairs"]):
        reasons.append("censoring_policy_not_registered")
    if not any(p["mean_difference"] not in (None, 0) for s in states for p in s["pairs"]):
        reasons.append("no_observed_completion_contrast")
    return dict(states=states, state_count=len(states), map_count=len({s["map_id"] for s in states}),
                candidate_count=sum(len(s["candidates"]) for s in data["states"]),
                trial_count=sum(len(c["trials"]) for s in data["states"] for c in s["candidates"]),
                pair_count=sum(len(s["pairs"]) for s in states),
                training_data_contract_passed=not reasons, rejection_reasons=reasons,
                research_admission="not_established_by_data_contract",
                runtime_integration_allowed=False)


def pair_vector(a, b, names):
    left, right = [a["features"][n] for n in names], [b["features"][n] for n in names]
    # Shared levels retain temperature and absolute candidate scale even when
    # both candidates have identical values; differences alone would erase them.
    vector = [x-y for x, y in zip(left, right)] + [x/2+y/2 for x, y in zip(left, right)]
    require(all(math.isfinite(v) for v in vector), "pair feature overflow")
    return vector


def training_matrix(data, held_out_maps):
    validate_dataset(data)
    known_maps = {s["map_id"] for s in data["states"]}
    held = set(held_out_maps)
    require(held and held < known_maps, "nonempty held-out and train maps required")
    train = sorted((s for s in data["states"] if s["map_id"] not in held), key=lambda s: s["state_id"])
    report = inspect_dataset(dict(data, states=train))
    require(report["training_data_contract_passed"], "training refused: " + ", ".join(report["rejection_reasons"]))
    x, y, weights, state_ids = [], [], [], []
    for state in train:
        candidates = {c["candidate_id"]: c for c in state["candidates"]}
        pairs = paired_labels(state)
        for p in pairs:
            a, b = candidates[p["left"]], candidates[p["right"]]
            for left, right, target in ((a, b, p["mean_difference"]), (b, a, -p["mean_difference"])):
                x.append(pair_vector(left, right, data["feature_names"]))
                y.append(target)
                weights.append(1 / (2 * len(pairs)))
                state_ids.append(state["state_id"])
    require(any(v != 0 for v in y), "train fold has no observed completion contrast")
    return dict(x=x, y=y, weights=weights, state_ids=state_ids,
                train_maps=sorted(known_maps-held), held_out_maps=sorted(held),
                state_count=len(train), feature_names=data["feature_names"])


@dataclass
class PairedCompletionModel:
    feature_names: list
    estimator: object

    @classmethod
    def fit(cls, data, held_out_maps):
        matrix = training_matrix(data, held_out_maps)
        import sklearn
        from sklearn.ensemble import HistGradientBoostingRegressor
        require(sklearn.__version__ == "1.5.0", "registered sklearn 1.5.0 required")
        model = HistGradientBoostingRegressor(loss="squared_error", **MODEL_PARAMS)
        model.fit(matrix["x"], matrix["y"], sample_weight=matrix["weights"])
        return cls(list(data["feature_names"]), model)

    def rank(self, state):
        candidates = _candidates(state, self.feature_names)
        pairs = list(combinations(candidates, 2))
        forward = self.estimator.predict([pair_vector(a, b, self.feature_names) for a, b in pairs])
        reverse = self.estimator.predict([pair_vector(b, a, self.feature_names) for a, b in pairs])
        require(len(forward) == len(reverse) == len(pairs), "prediction shape mismatch")
        scores = {c["candidate_id"]: 0.0 for c in candidates}
        for (a, b), f, r in zip(pairs, forward, reverse):
            require(math.isfinite(f) and math.isfinite(r), "nonfinite prediction")
            margin = max(-1.0, min(1.0, float(f)/2-float(r)/2)) / (len(candidates)-1)
            scores[a["candidate_id"]] += margin
            scores[b["candidate_id"]] -= margin
        # A score tie is not evidence that deviating from the frozen action helps.
        selected = min(scores, key=lambda c: (-scores[c], c != state["anchor_id"], c))
        return dict(selected=selected, scores=scores, calibrated_confidence=False)
