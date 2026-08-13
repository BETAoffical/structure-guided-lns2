# STRIDE CycleTransition Pilot v1 Result

## Decision

The bounded collection is complete and its integrity checks pass, but the
registered label-readiness decision fails. Selector training, runtime
integration, TTF experiments, and long-tail-avoidance claims stop here.

This failure does **not** show that neighborhood choice is irrelevant. It
shows that the preregistered one-action `soft_policy_cycle` is too sparse to
serve as the next pre-action learning target on this cohort.

## Complete execution identity

- 18 result-blind first-structural states, six per registered Maze map;
- 679 exact-deduplicated actions: 319 complete V2 base actions and 360 raw
  equal-size structural actions;
- 5,432 state-candidate-seed jobs, covering trial indices 0-7;
- 16 workers and a 300-second hard bound per job;
- 5,432 valid atomic trial files, zero execution errors, and zero timeouts;
- native actions, paired PP seeds, state identity, candidate identity, repair
  fingerprints, and the frozen 124-dimensional pre-action feature rows pass;
- no runtime, TTF, or future repair trajectory was stored as a label.

The canonical report is
`build/stride-cycletransition-pilot-v1/cycletransition_report.json`, SHA-256
`dc6bbbf6f0cb62506cf5bcfdd19e8fd3f4499d103c68b27a30254398966849f8`.
Its recorded collection/analysis producer is the frozen preregistration source
at commit `38c5fecf1b140ead9d2c2307e3ef97e50b297e4c`; the later interface-only
`load_cycletransition_registration` rename was made solely to satisfy the
repository duplicate-function hygiene rule and was not used to regenerate the
report.

## What the transition audit found

The native repair problem remains visible:

- 1,016/5,432 actions returned the exact repair state unchanged;
- 1,149/5,432 did not strictly reduce conflicts;
- 1,129/5,432 both failed to reduce conflicts and retained at least 80% of the
  original conflict edges.

The joint policy-cycle label almost vanished, however:

- only 18/5,432 actions met the registered soft-cycle definition;
- only 12/5,432 met the exact-cycle definition;
- the limiting term was next-action neighborhood overlap: only 18/5,432
  shadow next selections had Jaccard at least 0.8 with the forced action.

Thus PP no-op and persistent-conflict evidence is common enough, but a single
shadow re-selection by the full-grid V2 ranker is not the mechanism that
reproduces most historical long tails. The audit must not reinterpret every
PP no-op as an impending tail, nor train on the sparse joint label.

## Registered gates

Passed:

- complete integrity, zero errors, and zero timeouts;
- global soft-cycle half-split class agreement: `1.0000 >= 0.80`;
- global soft-cycle half-split rank correlation: `0.6667 >= 0.60`;
- each-map soft-cycle half-split class agreement.

Failed:

- each-map rank correlation: `maze-32-32-4 = 0.3333 < 0.50`;
- both soft-cycle classes appeared on only 1/3 maps;
- a quality-admissible structural escape existed on only 1/18 states
  (`5.56% < 60%`), and only on `maze-32-32-4`;
- the sole escape was one nominal-size-24 action, so the registered
  non-collapse gate also failed.

The only registered escape action was candidate
`neighborhood-c68785df0eac24c1` on state
`5766ee8b8959ad1255d22ac76e8fad5063f754f886c9f55caf2805cc4329583`.
One isolated action cannot justify a model or a size rule.

## Additional diagnostic boundary

When the frozen V2 ranker scored the complete equal-size mixed pool, its
initial full-pool choice was a nominal-size-32 structural action in all 18
states. It matched the historical source-selected action in 11/18 states.
This is evidence that the old ranker is not a neutral evaluator of the
expanded grid and has a strong largest-shell preference in this cohort. It is
not evidence that size 32 is optimal, and it cannot be used to introduce a new
preferred-size rule.

## Consequence

Do not tune the `0.8` Jaccard or `0.25` escape thresholds post hoc, select the
single successful state, or replace the target with exact no-op alone. The
valid conclusion is narrower:

1. candidate repairability and PP failure remain real;
2. immediate one-step quality remains insufficient for tail avoidance;
3. one forced action plus one memoryless shadow selection does not reconstruct
   the historical repeated-decision mechanism;
4. the next study must return to the complete recorded pre-tail decision
   sequence and identify the earliest **observed** transition into a recurrent
   conflict/neighborhood basin, while keeping candidate generation, ranking,
   native repair, and TTF evidence separate.

Any such successor requires a new result-blind preregistration. This failed
pilot authorizes no automatic continuation.

## Validation

- WSL Python, 16-process xdist: 857 passed, 35 skipped;
- Linux native CTest, `-j16`: 11/11 passed;
- Windows native `lns2_tests.exe`: passed;
- CycleTransition artifact integrity: 18 states, 679 candidates, 5,432 paired
  trials, zero errors, and zero timeouts;
- retention manifest and repository hygiene: passed.
