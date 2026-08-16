# STRIDE StructShell Rollback-Aware v1 diagnostic result

## Status

The registered two-case known-platform replay completed four of four serial
episodes with zero execution errors, process timeouts, invalid actions or
fingerprint mismatches.  All integrity gates passed.  The registered mechanism
decision nevertheless failed because `no_post_escape_platform_transfer` was
false.  This is a diagnostic result, not promotion evidence; V2 remains the
runtime default.

- Implementation commit: `78128b6`.
- Run fingerprint: `18ff68a7ebd3a8e6f4c82abccd2ded132f33c033036b2b76e57f9ed37e8df868`.
- Report SHA-256: `89c86b2a3476aa29896f5dfe58285d3c5ab41fe79e0796a68531761f64868477`.
- Timed protocol: current producer, decision-zero replay, native PP order,
  `episode_stream` seed policy, one PP call per decision, strict serial arm
  rotation, and 180/240/300-second wall/process/outer bounds.

## Paired outcomes

| Known Random case | Arm | Success | Restricted TTF (s) | Repair decisions | Exact rollbacks | Final conflicts | First-platform escape latency |
|---|---|---:|---:|---:|---:|---:|---:|
| `random_01`, seed 14 | routed StructShell v1 | no | 180.000 | 1,188 | 1,142 | 17 | 108 |
| `random_01`, seed 14 | rollback-aware v2 | yes | 48.628 | 370 | 305 | 0 | 4 |
| `random_02`, seed 14 | routed StructShell v1 | no | 180.000 | 966 | 940 | 40 | 16 |
| `random_02`, seed 14 | rollback-aware v2 | yes | 33.608 | 241 | 158 | 0 | 4 |

Across the two registered failures, the challenger changed success from 0/2
to 2/2, reduced mean restricted TTF from 180.000 to 41.118 seconds (77.16%),
reduced mean repair decisions from 1,077.0 to 305.5 (71.63%), reduced exact
rollbacks from 2,082 to 463 (77.76%), and reduced normalized wall conflict AUC
from 0.10196 to 0.03922 (61.54%).  These are development-case mechanism
effects, not estimates of general runtime performance.

The reuse intervention also removed most repeated online construction work on
these paths.  Mean candidate-generation time fell from 19.780 to 1.373 seconds
(93.06%), total controller-before-repair time from 42.425 to 5.696 seconds
(86.57%), and native PP replan time from 123.696 to 30.009 seconds (75.74%).
Current 124-dimensional features, the V2 anchor and Copeland scores were still
recomputed on every cache hit; only the repair-state candidate structure was
reused.

## Why the formal gate failed

The guard did exactly bound a repeated pure StructShell candidate: its longest
exact rollback streak was three in both cases, versus 594 and 864 under the
baseline.  It also left the first registered repair platform after four
decisions in both cases.  However, success was reached by moving through
multiple later repair fingerprints rather than by eliminating all later
three-rollback stalls:

| Case | Later three-or-more rollback platforms | Longest later platform | Final later platform | Guard bans | Guard overrides |
|---|---:|---:|---:|---:|---:|
| `random_01` | 25 | 33 | 4 | 87 | 229 |
| `random_02` | 21 | 16 | 4 | 27 | 61 |

The preregistered gate required *zero* later repair fingerprints to form a
three-rollback platform.  It therefore fails even though both episodes left
their original attractor and became feasible.  This is not a trace-label
artifact: the analyzer groups by repair fingerprint, ignores candidate/source
label rotation, and rejects a transfer into any later three-rollback segment.

The evidence supports a narrower statement: the guard converted two terminal,
single-candidate long tails into a sequence of bounded stalls that eventually
succeeded.  It does not show that the controller prevents platforms, nor that
the same effect generalizes beyond the two known Random failures.

## Frozen decision

Per the preregistration, `mechanism_replay_passed=false`; the rollback-aware
branch is not promoted and no result-blind confirmation is launched
automatically.  The implementation and traces remain frozen for audit.

A scientifically defensible continuation would require a new, explicit
preregistration on unseen keys.  It must keep this implementation unchanged
and treat end-to-end success, restricted TTF, conflict AUC and terminal or
long-duration platform occupancy as primary outcomes.  The current evidence
must not be used to tune the rollback limit, select cases, or choose a
post-hoc maximum platform-duration threshold.  Until that separate decision
is authorized and passes, V2 remains the default runtime.
