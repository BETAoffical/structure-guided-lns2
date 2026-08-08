# STRIDE StructPool Size Ablation v1

## Scope and claim boundary

This experiment evaluates only the current-step repair quality of StructPool
family-by-size candidates.  It excludes the known Maze long-tail state, future
repair trajectories, remaining repair rounds, Cost-to-Go, runtime, and TTF.  It
therefore cannot establish an end-to-end solver speedup.

## Cohort and integrity

- 98 topology-active, non-Maze states from 16 DAO/Game maps.
- Four nominal sizes per available family variant: 8, 16, 24, and 32.
- 1,680 raw drafts, 1,368 unique candidate agent sets, and 312 exact duplicates.
- 16 paired PP trials per unique candidate: 21,888 trials total.
- 9,424 exact trials reused and 12,464 newly executed trials.
- 0 collection errors and 0 timeouts.
- The current-code immutable audit passed: exact trial indices 0-15, one paired
  PP seed per state/trial index, 124 features, legal native agent sets, matching
  state and repair fingerprints, no forbidden fields, and matching SHA-256 for
  every retained source artifact.

Important source hashes:

- Grid manifest: `de4248402ed009e458fe282a25b22ce3c39edea46650757bfad3490a32d20a27`
- Grid state tree: `3a2e67a8f851891b12ee1fc8230f19352875f7e961f92d93991bca17bd3e0bb2`
- Repair trials: `e80c45317510eb91d3e69fd7c6f56cdef8445a8203f8737d9bd9231956b09537`
- Candidate aggregates: `e04f885f95972c95dd0f7922efa223cb2bb25d23c8d93ae3ec29ef55b4a58ed4`
- Label state tree: `89c74fde89807d8741ff260746ac79016f882cebd6f633f603e724917354a82b`

## Results

There is no uniform best size.  Size 32 has the highest aggregate seed mean for
most families, but the winning size remains state-dependent, especially for
path overlap and low-degree boundary candidates.

| Family | Best aggregate size | Mean seed score | Mean no-progress rate |
|---|---:|---:|---:|
| Bottleneck crossing | 32 | 0.4348 | 0.1275 |
| Conflict component | 32 | 0.5133 | 0.0886 |
| Spatiotemporal hotspot | 32 | 0.5139 | 0.0867 |
| Boundary articulation | 32 | 0.5976 | 0.0582 |
| Boundary low degree | 32 | 0.4858 | 0.1008 |
| Path overlap | 32 | 0.0459 | 0.6129 |

The frozen StructPool fixed-size policy has mean normalized regret `0.07464`,
maximum regret `0.35897`, and positive regret on `66.33%` of states relative to
the complete four-size grid.  The two fixed eight-seed halves agree on the exact
global winner in `64.29%` of states and have mean Top-3 overlap `69.80%`.

Mixed-provenance candidate rows have higher current-step quality than pure-family
rows (`0.4700` versus `0.2740` mean seed score), but this is descriptive evidence,
not a causal or TTF result.

## Interpretation

The experiment rejects the assumption that the existing hard-coded preferred
sizes are generally optimal.  It also shows that candidate size cannot be chosen
reliably from family identity alone.  The result justified evaluating the
preregistered support-nearest ScalePool rule, but did not authorize runtime
integration.

