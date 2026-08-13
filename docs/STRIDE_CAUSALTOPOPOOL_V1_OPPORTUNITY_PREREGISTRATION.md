# STRIDE CausalTopoPool v1 opportunity preregistration

## Purpose

Compactness established only that the hybrid pool is generated correctly.  It
did not establish that the 366 natural-topology additions are repairable.  This
stage replays every new topology action directly, without a ranker, and compares
it with the complete frozen V2 base pool under strictly paired PP seeds.

All 930 already measured CausalClosure actions remain part of the logical
hybrid pool.  Their frozen report is joined only at the state level; they are
not rerun and their outcomes do not select which topology actions are tested.

## Frozen execution

- 78 states and all 366 new topology candidates;
- trial indices 0--15 for every candidate;
- the same state/repair fingerprint and trial index produce the same PP seed;
- 5,856 atomic state-candidate-trial jobs;
- 16 workers and a 300-second hard limit per job;
- first error or timeout stops the collection; and
- no runtime, TTF or future-trajectory field enters the action label.

The registered preflight uses the first two frozen states and trials 0--1.  It
tests execution only and cannot change the cohort, thresholds or candidate
generator.

## Stable opportunity definition

A topology action robustly dominates the best V2 action only when its 16-seed
mean normalized conflict reduction is at least 0.02 higher, its no-progress
rate is no worse, and both fixed eight-seed halves are strictly better.  The
hybrid state opportunity is the logical union of this result and the frozen
CausalClosure state opportunity.

Because the earlier 10% gate was too weak to establish that lost StructPool
coverage had been restored, progression now requires all of:

- topology opportunity in at least 30% of states;
- hybrid Causal-plus-topology opportunity in at least 40% of states;
- a topology stable-frontier addition in at least 30% of states;
- hybrid opportunity in at least 25% of every map group; and
- complete integrity with zero errors or timeouts.

Passing permits only an independently registered, result-blind bounded
forced-continuation confirmation.  It does not permit ranker training, runtime
integration, a TTF claim or a long-tail avoidance claim.
