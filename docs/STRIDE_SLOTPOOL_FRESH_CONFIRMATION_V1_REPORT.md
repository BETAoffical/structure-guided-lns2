# STRIDE SlotPool v1 fresh-map confirmation report

## Decision

The frozen `stride-slotpool-v1` pairwise model passed all preregistered
candidate-quality gates on the six-map confirmation cohort. Runtime design,
GuardPool, and the known Maze regression are now permitted. This result does
not itself establish a TTF improvement or promote a solver controller.

The model was not retrained or recalibrated. The confirmation maps (`arena`,
`den020d`, `den404d`, `hrt002d`, `lak109d`, and `lak515d`) do not overlap the
16 development maps. All 12 fixed tasks and solver seeds 1 and 2 completed,
giving 24 states, 416 exact-deduplicated four-size candidates, and 6,656
current-step repair trials.

## Integrity

All 13 explicit collection checks passed:

- exactly four states per map and 24 registered states;
- one unique candidate identity per state and a complete candidate/trial
  product;
- trial indices 0--15 for every candidate;
- one strictly paired PP seed and repair fingerprint per state/trial index;
- recomputed reductions, outcomes, fixed-half means, and aggregate metrics;
- exact native 124-feature schema and finite values;
- exact 170-dimensional SlotPool candidate feature schema;
- legal, unique agent IDs within each native action;
- state-artifact and combined-JSONL identity;
- no TTF, runtime, future-trajectory, Cost-to-Go, or Receding-Q field;
- registered artifact hashes.

The collection contains zero execution errors. Its immutable hashes are:

- candidate aggregates: `a0a390efc59fa408af81c48abf1e2adfaf8b6d3965956c74e90d2877c03da8f5`;
- repair trials: `98819856396b58cacc0c51c4655f05a5defb20dd97303b3f41a3ccc98e2e2ed6`;
- state-artifact tree: `a35402e3b687cbd6d906bf5f1ef84afc5d7a5c8333f6f4285029b761d585a094`;
- collection report: `6fa3866e3843c8aca98d2308ecfb644f63cb3f993aba685e50b21fb6a71b4f90`.

## Frozen-model metrics

| Metric | Result | Gate |
|---|---:|---:|
| Global best retained in six slots | 1.0000 | >= 0.90 |
| Mean normalized regret | 0.0000 | <= 0.02 |
| Maximum map mean regret | 0.0000 | <= 0.05 |
| Maximum topology-group mean regret | 0.0000 | <= 0.05 |
| First 8-seed-half best retained | 1.0000 | >= 0.85 |
| Second 8-seed-half best retained | 0.9583 | >= 0.85 |
| Stable-pair accuracy | 0.9242 | >= 0.60 |
| Candidate count | 416 -> 144 | at most six/state |

All six maps and all three topology groups have zero mean normalized regret.
The frozen model retained a globally best candidate on every state. There are
3,142 stable candidate pairs, and all 24 states contribute stable pairs.

The state-evaluation SHA-256 is
`dcdd6961aa9e937d40b1ad04845be3bf5b6c3ecc4f5cdbd6dee5c0c524509d4b`.
The integrity-bearing confirmation report SHA-256 is
`8b648f38765cd0310a95a216e9ad475df97eb0c5954c8eb1d37c1652bcf880c0`.

## Interpretation and next boundary

The development result was not a map-specific artifact: a fixed model kept
the best observed one-step candidate under a six-candidate budget on this
map-disjoint cohort. This validates the SlotPool candidate reducer, not the
long-term repair trajectory.

The next controller must therefore keep `v2-full` as the anchor, use SlotPool
only to construct up to six structural challengers, and include a frozen
no-progress GuardPool fallback before the known Maze long-tail state is
rerun. Only paired run-to-completion raw-TTF experiments can establish a
solver speed improvement.
