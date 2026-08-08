# STRIDE SafeSlot First-Divergence Diagnostic v1

## Purpose

This retrospective diagnostic is the first implementation step of the unified
`stride-safeslot-v1` line. It does not improve StructPool and SlotPool as two
independent methods. Instead, it asks where either treatment first replaces
the exact V2 action, what happens immediately after that replacement, and
whether immediate conflict reduction agrees with the later repair trajectory.

The result will be used only to freeze the SafeSlot label and runtime interface.
It cannot train a model, tune a threshold, promote a controller, or make a TTF
claim.

## Fixed evidence

The cohort is the 29-key SlotPool-versus-StructPool diagnostic cohort. It keeps
the single previously registered Maze exclusion and permits no additional
result-based deletion. For every task and solver seed, the analysis combines:

1. a newly completed V2 trace using the same current runtime and registered
   qualification state;
2. the full StructPool trace from the strictly alternating diagnostic;
3. the frozen SlotPool trace from the same strictly alternating diagnostic.

The V2 baseline was collected separately from the earlier two-controller timing
diagnostic. Wall-clock values are therefore provenance only and are never
compared as performance evidence. Conflict states, candidate IDs, paired PP
seeds, fingerprints, and repair counts are valid for the divergence audit.

## First-divergence contract

Before the first different action, V2 and the challenger must have identical:

- state fingerprints and conflict counts;
- selected candidate IDs;
- candidate-bound PP seeds;
- post-step fingerprints and conflict counts.

At the first different action:

- the challenger action must be a StructPool family candidate;
- the V2 action must exist unchanged in the challenger's base-candidate pool;
- SlotPool's separately logged base-only V2 anchor must equal the candidate
  actually selected by the current-runtime V2 trace;
- full StructPool has no independent anchor, so the matched V2 trace is the
  authoritative base-only anchor and the audit separately reports whether
  mixed-pool Borda scoring changed the best base candidate;
- candidate size, family, V2 score, anchor overlap, one-step conflict change,
  the following eight-repair trajectory, terminal repair count, and trace hash
  are recorded.

These checks isolate the earliest causal policy decision. Later states are
descriptive trajectories, not matched counterfactual states.

## Interpretation boundary

The key failure signature is not simply “the structural action reduced fewer
conflicts.” A structural action may win the immediate conflict count and still
leave a harder residual state with more repair rounds. The diagnostic therefore
separates:

- immediate one-step advantage;
- eight-repair progress and no-progress streak;
- terminal repair-round advantage;
- cases where one-step improvement and terminal behavior disagree.

No Cost-to-Go or remaining-round target is introduced. The output only decides
which conservative, anchor-relative current-step and short-hazard fields must
be present in the later SafeSlot training schema.
