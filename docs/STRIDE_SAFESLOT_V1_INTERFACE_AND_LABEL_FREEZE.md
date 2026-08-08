# STRIDE SafeSlot v1 Interface and Label Freeze

## Unified scope

`stride-safeslot-v1` replaces the artificial fixed-size StructPool rule without
treating SlotPool as an unrelated competing controller. Its fixed pipeline is:

1. generate the full structural family-by-size search grid;
2. select the exact V2 anchor using only the original candidate pool;
3. reduce structural candidates under the learned slot budget;
4. compare each retained challenger directly with the fixed V2 anchor;
5. execute at most one challenger, otherwise abstain to V2;
6. retain an independent checkpoint/rollback safety path.

The V2 pairwise ranker must never be rerun over the mixed original-plus-
structural pool. Adding candidates changes Borda scores and can change the base
ordering, which destroys the meaning of an exact V2 fallback.

## Frozen selector interface

For one state, the selector receives:

- the original V2 candidate pool and their 124-dimensional features;
- the full structural family-by-size drafts and their 124-dimensional features;
- candidate support, size, event/pair/component coverage, family provenance,
  and overlap with the V2 anchor;
- current no-progress streak and overlap with recently failed neighborhoods.

It returns:

- `v2_anchor_candidate_id`;
- `challenger_candidate_id` or null;
- calibrated safe-replacement probability;
- `selected_candidate_id`, equal to the anchor when abstaining;
- a stable abstention reason and artifact identity.

Inference is pointwise and anchor-relative, O(M) in the retained structural
candidate count. It does not perform all-pairs Borda ranking over a mixed pool.

## Frozen current-step label boundary

For challenger `c` and V2 anchor `a`, all trial indices use strictly paired PP
seeds. A challenger can be a positive safe-replacement example only when:

```text
mean_reduction(c) - mean_reduction(a) >= 0.02
no_progress_rate(c) <= no_progress_rate(a)
both fixed seed halves support the replacement
post_repair_residual_risk(c) <= post_repair_residual_risk(a)
```

`post_repair_residual_risk` is a deterministic teacher field derived from the
state immediately after that one PP repair. It summarizes residual conflict
concentration, component/bottleneck concentration, and unresolved structural
support. It is not an inference feature unless a separate pre-action predictor
is later registered and validated. Ambiguous or half-inconsistent cases are
negative/abstain examples, not weak positives.

The following are forbidden as primary labels:

- TTF or runtime;
- terminal remaining repair rounds;
- Cost-to-Go or Receding-Q;
- future controller actions;
- the retrospective eight-repair trajectory.

## Training and validation boundary

- Whole maps, states, and task seeds must not cross training/validation folds.
- State weighting is uniform; candidates from a large state do not multiply
  that state's influence.
- Threshold calibration minimizes unsafe replacements subject to replacement
  precision and success constraints; failure to find a valid threshold means
  full abstention to V2.
- Candidate-generation quality, gate quality, runtime overhead, repair rounds,
  and raw TTF are reported separately.

## Runtime safety boundary

The gate is preventive; rollback is a second line of defense. Before executing
a structural challenger, the runtime stores a restorable solver checkpoint and
the conflict signature. If the frozen hazard rule later fires, it restores that
checkpoint, marks the challenger tabu for that signature, and resumes exact
V2. Hazard parameters must be frozen on development trajectories that exclude
the known Maze regression.

Rollback cannot make an unsafe training label safe, and immediate progress may
not release the checkpoint by itself. The exact checkpoint release rule remains
a separate preregistered implementation milestone.
