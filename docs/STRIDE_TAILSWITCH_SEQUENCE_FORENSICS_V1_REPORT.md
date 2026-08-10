# STRIDE TailSwitch sequence-forensics report

## Decision

The new evidence points to a neighborhood-selection control problem, but not to
a general failure of structural candidate generation. Structural neighborhoods
usually target the current conflict graph more aggressively than base
candidates. The adverse behavior is associated with repeatedly selecting the
same or nearly identical structural neighborhoods without a temporal diversity
penalty or an enabled exit rule.

The registered interpretation is
`selection_memory_and_missing_exit_association`. This is an existing-trajectory
diagnostic, not a per-action causal proof or a TTF improvement claim.

## Integrity and coverage

- frozen states: 66/66;
- TailSwitch policies: 4/4 per state;
- episodes: 264/264;
- reconstructed transitions: 10,744;
- continuation transitions: 10,480;
- augmented-policy continuation transitions: 4,555;
- actual structural continuation selections: 1,757 (38.57%);
- every trace SHA-256, state fingerprint and conflict-edge count: verified;
- report SHA-256:
  `5461e88a9348a38b25b20305ae77f1a09f1bdba736d282cf27bf9d60ac2b9fc4`.

No new solver, PP or candidate-repair trial was run.

## What the structural neighborhoods select

Across the augmented-policy continuation transitions:

| Metric | Base selected action | Structural selected action |
| --- | ---: | ---: |
| Current conflict edges touched | 0.8682 | 0.9765 |
| Current edges with both endpoints selected | 0.3737 | 0.6152 |
| Selected agents currently in conflict | 0.3514 | 0.8850 |
| Immediate normalized conflict progress | 0.1086 | 0.0718 |
| Unresolved-edge fraction | 0.8105 | 0.8067 |
| New-edge fraction after repair | 0.0992 | 0.1687 |
| Recent-three neighborhood reuse | 0.5848 | 0.8881 |
| Exact candidate repeated within three steps | 0.2059 | 0.7234 |

The structural actions do not miss the active conflict graph. They cover more
of it, but produce less immediate conflict reduction on average, create a
higher fraction of new conflict edges, and reuse recent agents far more often.
Therefore the supported defect is not simply "the neighborhood contains the
wrong conflict agents". It is closer to over-concentrated and repeatedly reused
repair sets with insufficient state-sequence feedback.

## Original-pool V2 anchor comparison

For each continuation state, the original-pool V2 anchor was reconstructed by
removing all `structpool-*` candidates and applying the same recorded V2 score
and deterministic tie rule.

- all 1,757 actual structural selections differ from the original-pool anchor;
- mean selected-versus-anchor agent-set Jaccard is only 0.2909;
- the structural candidate has a mean recorded-score advantage of 3.3227;
- it touches 0.3641 more of the current conflict edges than the anchor.

Thus the augmented pool is materially changing the action, not adding a nearly
duplicate candidate. The V2 ranker strongly favors broad structural conflict
coverage when those candidates are present. This comparison does not repair
the unselected anchor on the same continuation state, so it is selection and
coverage evidence rather than a per-step causal quality label.

## StructPool versus SlotPool

The candidate-targeting gate does not pass globally because the two challenger
families behave differently:

- SlotPool structural actions touch 0.1045 more conflict edges than base
  actions, have normalized progress `+0.0063`, and unresolved-edge fraction
  `-0.1126` relative to base actions.
- Full StructPool structural actions touch 0.1287 more conflict edges, but have
  normalized progress `-0.0811` and unresolved-edge fraction `+0.0978`.

SlotPool filtering improves the one-step direction, while Full StructPool is
more prone to broad but inefficient repairs. Neither result is a same-state
repair counterfactual, so it cannot by itself promote SlotPool.

## Sequence lock-in evidence

The two registered continuation contrasts contain 17 adverse, 11 beneficial
and 104 neutral pairs.

| Pair class | Mean structural selections | Mean maximum structural-run delta | Recent-three reuse delta | Exact-repeat delta |
| --- | ---: | ---: | ---: | ---: |
| Adverse | 62.94 | 62.76 | +0.4681 | +0.4898 |
| Beneficial | 23.09 | 20.55 | +0.0173 | +0.0466 |
| Neutral | 4.16 | 4.03 | +0.0374 | +0.0013 |

All 17 adverse pairs contain at least two post-first structural selections.
Some beneficial sequences also use structural candidates repeatedly, so the
number of structural actions alone is not a sufficient hazard signal. The
discriminating signal is repeated reuse of the same or highly overlapping
agent set.

The strongest adverse runs are localized but not limited to one challenger:

- `maze-128-128-1` Full StructPool: three adverse pairs average 199 consecutive
  structural selections;
- `maze-128-128-1` SlotPool: two adverse pairs average 111.5;
- `maze-128-128-2` Full StructPool: six adverse pairs average 14.0 maximum run;
- `maze-128-128-2` SlotPool: five adverse pairs average 9.8;
- `maze-32-32-4` has one adverse Full StructPool pair with a run of 114.

## Code-level cause

The current high-stress gate decides whether structural candidates enter the
pool. Once admitted, the frozen V2 ranker scores the augmented pool one state
at a time. It has no recent-neighborhood history, exact-repeat penalty, agent
reuse penalty or per-conflict-signature tabu state.

The runtime contains optional `stall_guard` support, but the frozen TailSwitch
parent configuration has no `stall_guard` block. Its StructPool gate only uses
current stress thresholds: at least 16 conflict pairs plus an agent, active
conflict-agent or component-size condition. A persistently difficult state can
therefore keep the gate open while a similar structural candidate remains the
highest-scored action every round.

This creates the observed feedback loop:

1. a difficult state opens the structural gate;
2. V2 gives a high score to a broad structural candidate outside its original
   candidate distribution;
3. PP changes paths but leaves a similar high-stress conflict structure;
4. deterministic structural generation proposes the same or overlapping agent
   set;
5. the memoryless ranker selects it again, with no mechanism to force a diverse
   repair or return temporarily to the original-pool anchor.

## Recommended next diagnostic

Do not retrain the full ranker yet. First audit an outcome-blind diversity guard
on the existing transitions:

- trigger inputs: exact candidate repetition, recent-three agent reuse and
  conflict-signature persistence;
- action: abstain from structural candidates for one decision and use the
  reconstructed original-pool V2 anchor;
- release: allow structural candidates again after a material conflict-graph
  change;
- report trigger coverage separately for adverse, beneficial and neutral
  pairs, by map and challenger;
- freeze thresholds before any new solver run.

Only if that trigger separates adverse lock-in without broadly suppressing the
beneficial structural sequences should it be implemented and tested with
paired run-to-completion raw TTF.

## Claim boundary

This result identifies a sequence-level association and a policy-level missing
exit mechanism. It does not prove that every repeated structural action is
wrong, does not provide the counterfactual repair result for the unselected
V2 anchor at every state, and does not authorize training, generalization or
default-controller claims.
