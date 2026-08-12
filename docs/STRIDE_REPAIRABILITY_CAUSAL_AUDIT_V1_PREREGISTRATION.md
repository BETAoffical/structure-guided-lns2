# STRIDE Repairability Causal Audit v1 Preregistration

## Question

The frozen long-tail evidence shows a deterministic loop: the same structural
neighborhood is ranked first, PP returns a path-level no-op, and the unchanged
controller selects that neighborhood again. Existing candidate replay found
many better immediate actions, but first-divergence replay also showed that an
immediately good action can still start an adverse continuation. Static
closure, a single residual-risk target, novelty rules, and multi-horizon value
labels were not stable across maps or seeds.

This audit therefore asks a narrower causal question before any new pool or
ranker is designed:

> Does PP fail because the selected agent set omits the agents that block the
> repair, because the repair order is wrong, or because PP remains unstable
> even after set and order are corrected?

The audit instruments PP internals and jointly intervenes on agent set and
repair order. It is not a controller, a training set, a TTF experiment, or a
candidate-pool promotion test.

## Frozen cohort

- All 45 `first_repeat_stall` logical checkpoints from the preregistered
  MarginalPool root diagnostic are retained.
- They are 45 distinct state fingerprints on three Maze maps.
- Classification, prior one-step outcome, and later continuation outcome may
  describe strata but cannot remove a checkpoint or change its trials.
- The selected neighborhood is the exact candidate selected at that frozen
  checkpoint. Candidate generation and ranking are not rerun.
- The state blob, source run configuration, logical manifest, prior analysis,
  and every input SHA-256 must match before collection.

## Diagnostic-only native instrumentation

Normal solver actions keep `collect_pp_diagnostics=false`. The audit sets it
to true and records, without changing PP acceptance or rollback:

- PP failure reason: `none`, `conflict_bound_exceeded`, or `time_limit`;
- attempted and inserted agent counts;
- failed agent and repair-order index;
- for each attempted agent: old/new path cost, whether the path changed,
  low-level collision count, cumulative conflict-pair count, and whether its
  path entered the path table;
- newly introduced conflict pairs;
- blocker agents inside and outside the selected neighborhood;
- whether the whole PP attempt rolled back.

Runtime fields may be used only for worker preflight. They are forbidden from
the causal trial rows and all scientific labels.

## Paired interventions

For every state and trial index 0 through 15, all arms use the same frozen PP
seed. The base arm first obtains the native randomized applied order. The
following interventions are then restored independently from the identical
pre-state:

1. `selected_native_order`: selected set with the native seeded order.
2. `selected_reverse_order`: identical set and the exact reversed base order.
3. `selected_conflict_priority`: identical set ordered by descending pre-state
   conflict degree, then ascending agent id.
4. `blocker_augmented_tail`: base set plus up to eight external blockers first
   observed in the base PP attempt, preserving base order and appending
   blockers by first failed-order position then id.
5. `blocker_augmented_head`: identical augmented set, but blockers precede the
   preserved base order.

The blocker-derived arms are explicitly outcome-enriched mechanism probes.
They cannot be used as online features, learned labels, or a deployable pool
rule. If the base attempt exposes no external blocker, both augmented arms are
recorded as not applicable rather than duplicated.

## Causal contrasts

- Order effect: arms 2 or 3 versus arm 1, with the agent set fixed.
- Set effect: arm 4 versus arm 1, preserving the original-agent order prefix.
- Joint order effect after closure: arm 5 versus arm 4, with the augmented set
  fixed.
- Residual PP instability: neither set nor order interventions produce a
  seed-stable improvement.

An intervention is seed-stable only when both fixed eight-seed halves have the
same strict direction and the full 16-seed mean normalized conflict reduction
improves by at least 0.02 without increasing no-progress rate. Replan success,
failure reason, failed position, new-conflict count, and external-blocker count
are reported separately; they are not compressed into a single target.

## Worker preflight and bounded execution

Before full collection, run the identical first 16 lexical states, base arm,
trial 0 with worker counts 12, 14, and 16. Select the highest-throughput count
that has:

- zero execution or fingerprint errors;
- peak measured memory below 75% of available memory;
- no lower completed-job count;
- positive throughput relative to the next lower safe count.

If 16 workers are safe but not faster, use the fastest measured safe count.
The preflight is execution tuning only and cannot alter cohort, arms, seeds, or
scientific thresholds. Full collection uses atomic per-state checkpoints, a
1,800-second state-attempt limit, at most four attempts, and stops for diagnosis
after two consecutive attempts without new completed trials.

## Decision rules

- Predominant stable order effect: repair-order selection is the first defect;
  design an outcome-blind order policy before changing the pool.
- Predominant stable set effect: the pool misses repair dependencies; derive
  an outcome-blind blocker/closure predictor from pre-state geometry and
  conflict timing, then validate it independently.
- Predominant joint effect: candidate membership and repair order must be
  generated and ranked together; separate pool and ranker fixes are invalid.
- Predominant residual instability: the current PP operator/seed uncertainty
  is limiting; changing only candidate ranking is insufficient.

No result permits model training, direct online blocker lookup, controller
promotion, TTF claims, or deletion of failed/timeout states. Any deployable
candidate rule must be preregistered later and use only pre-action state data.

