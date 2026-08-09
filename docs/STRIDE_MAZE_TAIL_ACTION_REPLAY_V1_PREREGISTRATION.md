# STRIDE Maze Tail Action Replay v1 Preregistration

## Purpose

The fused Maze state collection found enough independent harmful-tail coverage
to justify an exact action-level diagnostic.  This block asks whether the first
structural departure from V2 has a stable current-step effect across PP seeds.
It does not train a model or make a TTF claim.

## Frozen selection

All 66 challenger-versus-V2 comparisons are inspected.  For each comparison,
the first decision whose realized agent set differs is retained only when:

- every preceding state, action, PP seed, and post-state fingerprint is equal;
- the challenger action is structural;
- the exact V2 action remains in the challenger's base candidate pool; and
- the pre-action fingerprints and conflict counts match.

No comparison is selected or removed using the new repair outcomes.  A
comparison with no action divergence remains in the preflight report.  Before
any repair trial, the full selection and its SHA-256 are frozen in a second
configuration.

Preflight requires at least 24 divergences.  At least 12 must belong to the
already frozen adverse/severe episode set, including four severe cases, two
maps, six tasks, two solver seeds, and three cases per challenger.

## Paired action replay

At every frozen divergence state, the exact V2 action and structural challenger
action each receive trial indices 0--15.  Both actions use the same PP seed at a
given state and index.  The primary outcome is one-step conflict reduction
normalized by the common before-conflict count.  Runtime is descriptive only;
TTF, future trajectory, remaining rounds, and Cost-to-Go are not labels.

An action pair is robustly ordered only when the winner:

- wins at least 12/16 paired seeds;
- has an absolute mean advantage of at least 0.02;
- has the same advantage direction in both fixed eight-seed halves; and
- has a no-progress rate no worse than the loser.

## Decision gate

Predictor design, but not training, is allowed only when at least 70% of frozen
action pairs are robustly ordered and at least eight independently adverse or
severe comparisons have a robust order.  Those tail comparisons must cover two
maps, four tasks, two solver seeds, and two examples per challenger.

Failure retains every artifact and stops predictor design.  The next step is a
second independently registered task-seed block, not post-hoc case filtering or
threshold adjustment.  Passing authorizes only an outcome-blind residual-hazard
predictor design; training, runtime integration, default promotion, and TTF
claims remain forbidden.

## Frozen preflight result

The extraction preflight passed before any new repair trial was executed.  All
66 comparisons had a genuine first realized-neighborhood divergence: 63 at
decision zero, one at decision one, and two at decision two.  The frozen set
contains all 24 adverse/severe comparisons and all 10 severe comparisons, with
12 tail cases per challenger across three maps, eight tasks, and three solver
seeds.

The immutable preflight report SHA-256 is
`74dad85c81fadec8c9a44932bf3c0cb7ae942a078058776964bc6f00f1ac0330`;
the 66-row state/action selection SHA-256 is
`453fd4702e104b74ed032c62aae152017afb32401dc2191e9f90701872ef1377`.
The subsequent replay config pins both artifacts and requires exactly 2,112
single-step repairs.
