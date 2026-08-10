# STRIDE TailSwitch v1 formal report

## Decision

TailSwitch completed all 66 frozen first-divergence states and all four
interventions per state. Integrity passed, but none of the preregistered causal
mechanism gates passed. The formal conclusion is therefore `inconclusive`:
there is no supported map-independent claim that either the first structural
action or structural continuation is the universal cause of long tails.

This does not show that topology is irrelevant. The paired directions separate
the two factors: the frozen first structural action is favorable on average,
while continued structural exposure is adverse on average and concentrated on
one large Maze map.

## Integrity

- states: 66/66;
- policies per state: 4/4;
- total episodes: 264/264;
- execution errors and process timeouts: 0;
- invalid actions and fingerprint mismatches: 0;
- exact frozen first action and paired PP seed: passed;
- first action forced exactly once: passed;
- no result-based state exclusion: passed;
- report SHA-256:
  `107f11faaa2e18f92af8e73e4a69e1ba7f8948f77b8f0e174981cc4c3c2b6754`.

`repair_limit` and `wall_timeout` remain preregistered right censoring, not
execution errors.

## Four-policy outcomes

| Policy | Success | Repair limit | Wall timeout | Mean rounds | Mean normalized fixed-200 AUC | Mean final conflicts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2-then-v2` | 58/66 | 4 | 4 | 45.30 | 0.037008 | 1.424 |
| `struct-then-v2` | 59/66 | 3 | 4 | 46.47 | 0.028957 | 0.561 |
| `v2-then-struct` | 58/66 | 2 | 6 | 33.65 | 0.046569 | 6.091 |
| `struct-then-struct` | 58/66 | 3 | 5 | 37.36 | 0.042629 | 5.424 |

Rounds alone are misleading because right-censored episodes stop at a fuse.
The fixed-horizon AUC and final conflicts are the registered causal outcomes.

## Registered contrasts

Positive deltas are adverse; lower AUC and fewer final conflicts are better.

| Contrast | Mean AUC delta | Mean final-conflict delta | Adverse / beneficial / neutral | Gate |
| --- | ---: | ---: | ---: | --- |
| First Struct action, then V2 | -0.008051 | -0.864 | 2 / 8 / 56 | fail |
| First Struct action, then Struct | -0.003940 | -0.667 | 4 / 8 / 54 | fail |
| Struct continuation after V2 | +0.009561 | +4.667 | 9 / 6 / 51 | fail |
| Struct continuation after Struct | +0.013672 | +4.864 | 8 / 5 / 53 | fail |
| Full structural adverse against both switches | -0.012110 | -0.697 | 4 / 0 / 62 | fail |

The mean factorial interaction in normalized AUC is only `+0.004111`. The
frozen first structural action is not supported as the dominant irreversible
cause: with V2 continuation it has fewer adverse states, more beneficial
states, lower mean AUC and fewer final conflicts than the first V2 action.

Structural continuation has the adverse average direction, but only 8--9 of
66 comparisons cross the registered effect threshold. This is far below the
two-thirds prevalence required for a universal mechanism.

## Localized map effect

The continuation penalty is heterogeneous:

- `maze-128-128-2`: Struct continuation after V2 has AUC `+0.04270` and final
  conflicts `+14.56`; after a Struct first action it has AUC `+0.05463` and
  final conflicts `+15.28`. The two contrasts contain 6/18 and 5/18 adverse
  cases and no beneficial cases.
- `maze-128-128-1`: the continuation penalty is smaller: AUC `+0.00644` or
  `+0.01209` and final conflicts `+1.92` or `+2.00`.
- `maze-32-32-4`: Struct continuation is favorable on average: AUC `-0.01218`
  or `-0.01547`, with essentially unchanged final conflicts.

This cross-map reversal explains why the universal gate fails. A single global
fallback rule would discard useful behavior on the small Maze while still not
fully protecting the difficult large Maze.

## Interpretation and next safe step

The evidence supports a narrower hypothesis: repeated structural selection can
create or preserve residual conflict on particular large-map state sequences,
but the first structural neighborhood is often useful. The next study should
use the already collected trajectories to compare adverse and beneficial
continuation sequences using pre-action sequence features such as repeated
agent/neighborhood overlap, unresolved conflict-pair recurrence, boundary
coverage, component persistence and family run length.

That study must remain diagnostic and map-grouped. It must not train on only the
adverse states, may not claim TTF improvement, and may not promote a new default
controller until a separately preregistered fresh-map paired evaluation passes.

## Claim boundary

TailSwitch is a causal policy-switch diagnostic. It does not authorize model
training, a TTF improvement claim, a generalization claim or replacement of
`v2-full`.
