# STRIDE Maze Tail Full-Episode v1 Protocol

## Purpose

This block measures whether harmful StructPool or SlotPool departures from the
frozen V2 controller recur across the independent Maze cohort selected by the
reset-only preflight.  It is an incidence and mechanism-evidence block, not a
model-training or speed-promotion experiment.

## Frozen cohort and execution

The 11 selected tasks and three solver seeds form 33 paired keys.  Every key is
run with `v2-full`, `v2-plus-structpool`, and `v2-plus-slotpool`, giving 99
complete episodes.  Controller order rotates by sorted key.  Runs use one
worker, deterministic PP replay, paired reset state fingerprints, and the raw
reset-inclusive run-to-completion TTF clock.  Scientific, environment, and
episode-process time caps are disabled.  No task may be removed after any
controller outcome is observed.

## Tail labels

For each challenger and paired V2 episode:

- beneficial or tied: challenger repair iterations are no greater than V2;
- adverse: challenger uses at least one more repair iteration;
- severe: challenger uses at least 10 more repair iterations and at least
  twice the V2 repair iterations.

The full set of 66 challenger comparisons is retained.  These episode labels
are used only to decide whether there is enough independent tail coverage to
justify a later first-divergence action replay.  They are not training labels.

## Evidence gate

The 16-seed action replay is allowed only if the complete block has at least 12
adverse comparisons, including at least 4 severe tails, covering at least 2
maps, 6 tasks, and 2 solver seeds, with at least 3 adverse comparisons for each
challenger.  Failure causes a hard stop before replay or training; the full
result is retained and a second independent task-seed block must be registered.

## Claim boundary

Raw TTF, repair rounds, AUC, PP time, candidate-generation time, and selection
overhead are reported descriptively.  This block cannot promote StructPool or
SlotPool, establish fresh-map generalization, tune a gate, or train a tail
predictor.  Predictor design remains forbidden until the subsequent 16-seed
paired action replay demonstrates stable current-step labels.
