# STRIDE-LNS quality V2 protocol

`stride-quality-v1` passed the registered pairwise-consistency gate but failed
the Top-3 stability gate. It remains untrained and unpromoted. Quality V2 is a
new label contract rather than a retrospective change to V1.

## Design boundary

- Controller ID: `stride-quality-v2`.
- Label schema: `lns2.stride.quality_label.v2`.
- The 48 states used by the V1 stability analysis are consumed as the V2 design
  cohort. They cannot provide confirmation evidence.
- Runtime and PP duration do not enter the quality label.
- Candidate generation, the 124 frozen realized features, and native PP repair
  remain unchanged.
- Cost-to-Go, future repair rounds, and downstream rollout outcomes remain
  diagnostics rather than training labels.

## Label

Each candidate is repaired with eight deterministic paired PP seeds. Let
`r_bar` be mean conflict reduction divided by pre-repair conflicts. Let `s` be
the mean within-state percentile of four lower-is-better post-repair metrics:
largest conflict component, conflict-edge density, conflict-event density, and
degree concentration. The registered score is

```text
quality_score = r_bar - 0.02 * s
```

The bounded structure term can reverse a conflict-reduction difference only
when the normalized difference is below two percentage points. Pairwise labels
follow the score direction; reverse rows are emitted and every state has total
sample weight one. The design-cohort four-versus-four diagnostic was 0.8017
pairwise consistency and 0.7708 mean Top-3 overlap. These are design evidence,
not confirmation evidence.

## Eight-seed design check

Trial indices 8 through 15 are added to the consumed design cohort. The first
half is indices 0 through 7 and the second is 8 through 15. This check asks
whether increasing from four to eight seeds is plausible before spending the
fresh confirmation cohort. It cannot promote the label.

## Fresh confirmation

The confirmation cohort is selected without repair outcomes from the original
240-state Pilot after excluding every design state and every design episode.
It contains 48 states, 24 per source policy and at most one per episode. The
selection passed all disjointness and balance gates. Its selection SHA-256 is
`6f6c1a0b2f1288875b76d5e3d81f6bdb23acfa2d52bdfca196b6348c7e5093e3`.

For confirmation, indices 4 through 15 are collected and combined with the
existing indices 0 through 3. Independent halves 0--7 and 8--15 must satisfy:

- aggregate pairwise direction consistency at least 70 percent;
- mean Top-3 intersection divided by three at least 80 percent;
- all 48 states complete with zero collection errors.

The metric, thresholds, cohort, and hash are frozen before confirmation repair
outcomes are read. Failure keeps Stage 3 blocked. Passing permits a separate
Stage 3 expansion using eight PP seeds per candidate; it does not by itself
show end-to-end solver improvement or promote a runtime controller.
