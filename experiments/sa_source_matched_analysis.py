"""Source-matched H32 label analysis, with no fitting, I/O or admission gates.

Trial positions are paired across candidates. The caller must verify root,
generation-family, actual-size and randomization identities before supplying
records. Candidate IDs are sorted for a deterministic signed A-minus-B contrast.
"""

from itertools import combinations
import math
import random


TRIAL_COUNT = 8
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 2026091809


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def partitions():
    """Return all 35 unordered 4/4 partitions, putting trial 0 on the left."""
    universe = set(range(TRIAL_COUNT))
    return tuple(
        (left, tuple(sorted(universe - set(left))))
        for rest in combinations(range(1, TRIAL_COUNT), 3)
        for left in [(0,) + rest]
    )


def _validate(record):
    _require(isinstance(record, dict), "record must be a dict")
    for name in ("state_id", "map_id", "anchor_id"):
        _require(isinstance(record.get(name), str) and bool(record[name]), "invalid " + name)
    pair = record.get("pair_ids")
    _require(isinstance(pair, (list, tuple)) and len(pair) == 2,
             "pair_ids must contain two candidates")
    _require(all(isinstance(cid, str) and cid for cid in pair) and pair[0] != pair[1],
             "pair candidates must have distinct nonempty IDs")
    values = record.get("values")
    _require(isinstance(values, dict), "values must be a dict")
    _require(set(pair) | {record["anchor_id"]} <= set(values), "missing pair or anchor candidate")
    for cid, trials in values.items():
        _require(isinstance(cid, str) and cid, "invalid candidate ID")
        _require(isinstance(trials, (list, tuple)) and len(trials) == TRIAL_COUNT,
                 "each candidate must have exactly eight trial positions")
        _require(all(type(v) is bool or v is None for v in trials),
                 "trial outcomes must be bool or None, without coercion")
    return tuple(sorted(pair)), values


def _candidate_stats(trials):
    successes = sum(v is True for v in trials)
    unknown = sum(v is None for v in trials)
    return dict(successes=successes, failures=TRIAL_COUNT-successes-unknown,
                censored=unknown, complete=unknown == 0,
                completion_rate=successes / TRIAL_COUNT if unknown == 0 else None,
                completion_bounds=[successes / TRIAL_COUNT, (successes+unknown) / TRIAL_COUNT])


def _rate(values, cid, indices):
    return sum(values[cid][i] for i in indices) / len(indices)


def _direction(value):
    return (value > 0) - (value < 0)


def _crossfit_direction(values, pair, selection, evaluation):
    selection_rates = {cid: _rate(values, cid, selection) for cid in pair}
    chosen = min(pair, key=lambda cid: (-selection_rates[cid], cid))
    evaluated = _rate(values, chosen, evaluation)
    uniform = math.fsum(_rate(values, cid, evaluation) for cid in pair) / 2
    return dict(selected=chosen, selection_rates=selection_rates,
                selection_tied=selection_rates[pair[0]] == selection_rates[pair[1]],
                evaluation_rate=evaluated, uniform_rate=uniform, gain=evaluated-uniform)


def analyze_record(record):
    """Analyze one root. Censored pairs retain counts but have no partitions.

    Crossfit averages both selection/evaluation directions within each split,
    then all 35 splits into ONE state value. Anchor never enters selection.
    All-trial differences and half directions use sorted pair IDs, not winners.
    Paired counts classify each trial once; either missing outcome is unknown.
    Fixed half directions are -1/0/+1, or None if that half is incomplete.
    """
    pair, values = _validate(record)
    stats = {cid: _candidate_stats(trials) for cid, trials in sorted(values.items())}
    a, b = pair
    complete = stats[a]["complete"] and stats[b]["complete"]
    delta = stats[a]["completion_rate"] - stats[b]["completion_rate"] if complete else None
    uniform = (stats[a]["completion_rate"] + stats[b]["completion_rate"]) / 2 if complete else None
    paired_counts = dict(a_wins=0, b_wins=0, both_success=0, both_failure=0, unknown=0)
    for outcome_a, outcome_b in zip(values[a], values[b]):
        category = ("unknown" if outcome_a is None or outcome_b is None
                    else "both_success" if outcome_a and outcome_b
                    else "both_failure" if not outcome_a and not outcome_b
                    else "a_wins" if outcome_a else "b_wins")
        paired_counts[category] += 1
    fixed_differences = []
    for half in (range(4), range(4, 8)):
        known = all(values[cid][i] is not None for cid in pair for i in half)
        fixed_differences.append(_rate(values, a, half)-_rate(values, b, half) if known else None)
    fixed_halves = dict(left_trials=list(range(4)), right_trials=list(range(4, 8)),
                        half_differences=fixed_differences,
                        half_directions=[_direction(value) if value is not None else None
                                         for value in fixed_differences])
    split_rows = []
    counts = dict(strict_same_direction=0, strict_opposite_direction=0, both_tied=0, one_tied=0)
    if complete:
        for left, right in partitions():
            differences = [_rate(values, a, half)-_rate(values, b, half) for half in (left, right)]
            signs = [_direction(value) for value in differences]
            category = ("both_tied" if signs == [0, 0] else "one_tied" if 0 in signs
                        else "strict_same_direction" if signs[0] == signs[1]
                        else "strict_opposite_direction")
            counts[category] += 1
            forward = _crossfit_direction(values, pair, left, right)
            reverse = _crossfit_direction(values, pair, right, left)
            split_rows.append(dict(left_trials=list(left), right_trials=list(right),
                                   half_differences=differences, half_directions=signs,
                                   direction_category=category, left_to_right=forward,
                                   right_to_left=reverse, gain=(forward["gain"]+reverse["gain"])/2))
    gain = math.fsum(row["gain"] for row in split_rows) / len(split_rows) if complete else None
    anchor_rate = stats[record["anchor_id"]]["completion_rate"]
    return dict(state_id=record["state_id"], map_id=record["map_id"], pair_ids=list(pair),
                anchor_id=record["anchor_id"], candidates=stats, pair_complete=complete,
                paired_counts=paired_counts, fixed_halves=fixed_halves,
                all_trial_difference=delta,
                all_trial_difference_bounds=[stats[a]["completion_bounds"][0]-stats[b]["completion_bounds"][1],
                                             stats[a]["completion_bounds"][1]-stats[b]["completion_bounds"][0]],
                pair_uniform_rate=uniform, absolute_all_trial_difference=abs(delta) if complete else None,
                partitions=split_rows, partition_count=len(split_rows),
                half_direction_counts=counts if complete else None, crossfit_gain=gain,
                anchor_completion_rate=anchor_rate,
                anchor_minus_pair_uniform=anchor_rate-uniform
                if anchor_rate is not None and uniform is not None else None,
                anchor_descriptive_only=True)


def _quantile(sorted_values, probability):
    position = (len(sorted_values)-1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    return sorted_values[lo] + (sorted_values[hi]-sorted_values[lo]) * (position-lo)


def _aggregate(rows, field, maps, draws):
    per_map = {}
    for map_id in maps:
        values = [row[field] for row in rows if row["map_id"] == map_id]
        known = [value for value in values if value is not None]
        per_map[map_id] = dict(states=len(values), observed_states=len(known),
                              mean=math.fsum(known)/len(values) if len(known) == len(values) else None)
    means = [per_map[map_id]["mean"] for map_id in maps]
    complete = all(value is not None for value in means)
    interval = None
    if complete:
        samples = sorted(math.fsum(means[i] for i in draw)/len(maps) for draw in draws)
        interval = [_quantile(samples, .025), _quantile(samples, .975)]
    return dict(mean=math.fsum(means)/len(maps) if complete else None, ci95=interval,
                complete=complete, states=len(rows), observed_states=sum(row[field] is not None for row in rows),
                maps=len(maps), complete_maps=sum(value is not None for value in means), per_map=per_map)


def summarize(records, *, bootstrap_samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """Return state details and equal-map/equal-state descriptive aggregates.

    Bootstrap resamples maps only, sharing draws across metrics. If a metric
    lacks any state, its full-cohort mean/CI is None; no complete-case filtering
    or imputation is performed. Anchor censoring does not censor a full pair.
    Signed differences are alphabetical A-minus-B, not policy improvement.
    """
    _require(type(bootstrap_samples) is int and bootstrap_samples > 0, "invalid bootstrap_samples")
    _require(type(seed) is int and seed >= 0, "invalid bootstrap seed")
    rows = [analyze_record(record) for record in records]
    _require(rows, "empty cohort")
    _require(len({row["state_id"] for row in rows}) == len(rows), "duplicate state_id")
    rows.sort(key=lambda row: (row["map_id"], row["state_id"]))
    maps = sorted({row["map_id"] for row in rows})
    rng = random.Random(seed)
    draws = [tuple(rng.randrange(len(maps)) for _ in maps) for _ in range(bootstrap_samples)]
    fields = ("all_trial_difference", "absolute_all_trial_difference", "pair_uniform_rate",
              "crossfit_gain", "anchor_completion_rate", "anchor_minus_pair_uniform")
    metrics = {field: _aggregate(rows, field, maps, draws) for field in fields}
    return dict(schema="lns2.sa.source_matched_analysis.v1", states=len(rows), maps=len(maps),
                complete_pair_states=sum(row["pair_complete"] for row in rows),
                censored_pair_states=sum(not row["pair_complete"] for row in rows),
                metrics=metrics, rows=rows,
                bootstrap=dict(samples=bootstrap_samples, seed=seed, unit="map",
                               interval="descriptive_percentile_95", quantile="linear_interpolation"),
                weighting="equal_maps_then_equal_states", missing_policy="no_imputation_no_state_filtering",
                partitions_per_complete_state=35, directions_per_partition=2,
                partitions_are_independent_samples=False, anchor_descriptive_only=True,
                signed_difference="sorted_pair_ids[0]_minus_sorted_pair_ids[1]",
                interpretation="development_frozen_continuation_not_closed_loop_or_ttf",
                thresholds_applied=False, automatic_promotion=False)
