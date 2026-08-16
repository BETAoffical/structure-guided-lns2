# Warehouse Fixed16 Development V2 Result

## Decision

No fixed16 challenger passed the preregistered development gate. The selected
controller is therefore `null`, V2 remains the default, and neither the
8/16/24/32 size study nor the final Official/V2/challenger confirmation is
authorized.

This is a qualification-conditioned development ablation on one narrow-aisle,
high-density Warehouse layout. It is not a four-map result, a Warehouse-wide
speed claim, or a runtime promotion result.

## Registered identity and integrity

- Collection source commit: `a6bb528`.
- Output: `build/wh-f16-v2-r2`.
- Run fingerprint: `390f550db3c8a9d89c555e1b35600c80070f22a86ef4830218ece4573fcdc6c1`.
- Map/task slice: `warehouse-10-20-10-2-1`, 600 agents, OE and uniform-random
  task seeds 233/277, solver seeds 19--22.
- Q0 regenerated and hash-checked all four registered tasks.
- Q1 completed 16/16 fresh resets with no errors. Minimum initial conflicts
  were 23 for opposite-exchange and 12 for uniform-random.
- Formal collection completed 48/48 episodes, eight per controller, with one
  timed worker and strict six-arm rotation.
- There were zero execution errors, process timeouts, invalid actions,
  fingerprint mismatches, and semantic mismatches. One Boundary16 episode
  reached the registered 180-second wall limit and was retained as valid
  right-censoring.
- The failed four-layout `v1-r2` qualification was used only as registered design
  provenance (`passed=false`, formal episodes `0`); no `v1-r2` reset, state blob,
  manifest, or episode was imported.

Primary report SHA-256:
`b487a3d7d4f1843b3ec4c6f181301f6788079c094318104d9608dc5412a87697`.

## Primary TTF results

All TTF values below are reset-inclusive mean 180-second restricted TTF.
Half deltas are challenger minus V2, so a negative value is faster.

| Controller | Success | TTF (s) | Change vs V2 | Paired faster | Half A delta (s) | Half B delta (s) | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| V2 | 8/8 | 11.781 | -- | -- | -- | -- | baseline |
| Boundary16 | 7/8 | 34.008 | +188.66% slower | 3/8 | -1.448 | +45.900 | fail |
| Bottleneck16 | 8/8 | 10.955 | 7.01% faster | 5/8 | -1.789 | +0.137 | fail |
| Component16 | 8/8 | 10.431 | 11.46% faster | 7/8 | -3.175 | +0.474 | fail |
| Hotspot16 | 8/8 | 9.995 | 15.16% faster | 6/8 | -3.591 | +0.018 | fail |
| Path-overlap16 | 8/8 | 12.828 | 8.88% slower | 0/8 | +1.452 | +0.641 | fail |

The three best point estimates all failed only because their Half B mean was
not strictly below V2. Hotspot16 was closest: Half B was 0.0177 seconds slower.
The registered rule is strict, so this near-tie cannot be rounded into a pass.

## Flow diagnostics

| Controller | OE TTF (s) | Uniform-random TTF (s) |
|---|---:|---:|
| V2 | 17.696 | 5.867 |
| Boundary16 | 62.668 (3/4 success) | 5.348 |
| Bottleneck16 | 16.508 | 5.402 |
| Component16 | 16.523 | 4.339 |
| Hotspot16 | 15.807 | 4.183 |
| Path-overlap16 | 18.899 | 6.757 |

Hotspot16 and Component16 improved the aggregate point estimate in both flow
types, but their gains were concentrated in Half A and did not reproduce as a
strict mean improvement in Half B. This is useful mechanism evidence, not an
eligible winner.

## Why the controllers were faster or slower

The report-only timing diagnostics show that the favorable fixed16 families
reduced native repair work enough to offset candidate handling:

| Controller | Mean repairs | PP wall (s) | Selection wall (s) | Diagnostic AUC |
|---|---:|---:|---:|---:|
| V2 | 25.000 | 4.680 | 0.668 | 0.04405 |
| Boundary16 | 32.125 | 25.603 | 1.729 | 0.11062 |
| Bottleneck16 | 15.125 | 4.038 | 0.782 | 0.04368 |
| Component16 | 13.250 | 3.740 | 0.659 | 0.03991 |
| Hotspot16 | 12.125 | 3.232 | 0.624 | 0.04158 |
| Path-overlap16 | 26.750 | 4.757 | 1.599 | 0.04499 |

Hotspot16 is not faster because its Python selector is free; it is faster in
the aggregate because it approximately halves repair decisions and lowers PP
wall time. Boundary16 exhibits the opposite failure mode: a selected trajectory
entered a 180-second tail, and both PP and selection time rose sharply.

AUC, repair count, PP time, selection time, and paired-faster fraction are
diagnostics only. None was used to accept or reject a challenger.

## Next step

Stop this branch as registered. Do not choose Hotspot16 post hoc, do not run a
size study, and do not consume the reserved final Warehouse namespace. V2
remains the project default. A future attempt would require a separately
preregistered question and fresh development evidence; it cannot reinterpret
this screen as a pass.
