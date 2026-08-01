# STRIDE-LNS quality V2 independent confirmation

`stride-quality-v2` passed the frozen aggregate stability gates on a fresh
48-state cohort. The cohort was selected without repair outcomes and is
disjoint from the design cohort by both state and source episode.

## Registered cohort and collection

- 48 states: 24 official-Adaptive source states and 24 frozen-V2 source states;
- at most one state per episode, with zero design-state or design-episode
  overlap;
- all 14 development maps that produced conflicting states represented;
- selection SHA-256:
  `6f6c1a0b2f1288875b76d5e3d81f6bdb23acfa2d52bdfca196b6348c7e5093e3`;
- 48/48 states completed with zero collection errors;
- 855 candidates, 10,260 new outcomes at trial indices 4 through 15;
- confirmation extension SHA-256:
  `64a4a73baa33ebcde266d76082c8afab3b1bbe257ffbae58747df94f6f634af3`.

## Frozen eight-versus-eight result

- 13,680 outcomes across trial indices 0 through 15;
- 6,770 decisive-pair union, with 5,902 directions agreeing;
- aggregate pairwise consistency: `0.8717872969` -- pass against `0.70`;
- mean Top-3 intersection divided by three: `0.8055555556` -- pass against
  `0.80`;
- official-Adaptive source states: pairwise `0.8743`, Top-3 `0.8194`;
- frozen-V2 source states: pairwise `0.8639`, Top-3 `0.7917`;
- runtime did not enter the label.

The deterministic analysis reproduced byte-for-byte. Report SHA-256:
`b60bf7d6588f1140bc59c778f9085ff15971006aeff397117c7e029f7f5ee486`.

## Decision

The preregistered aggregate gates passed, so Stage 3 data expansion is allowed
for `stride-quality-v2`. This is label-stability evidence only. It does not
train or promote a model and does not show fewer repair rounds, lower TTF, or
lower end-to-end wall time.

The Top-3 result cleared its threshold by only `0.00556`, and the frozen-V2
source subgroup remained below 0.80. Stage 3 must therefore retain source-policy
strata, report subgroup metrics, use map-grouped splits, and must not weaken the
later Shadow, Quick, or formal OOD promotion gates.

Quality V1 remains a recorded failed label. Stage 3 uses eight paired PP seeds
per candidate and the distinct `lns2.stride.quality_label.v2` schema.

## Reproduction

```bash
PYTHONPATH=build/linux/project /usr/bin/python3 scripts/run_stride_pipeline.py \
  collect-quality-v2-confirmation \
  --selection build/stride-quality-v2-confirmation-selection-v1/confirmation_selection.jsonl \
  --collection build/stride-stage2-pilot-v2 \
  --output build/stride-quality-v2-confirmation-v1 --workers 4

python scripts/run_stride_pipeline.py analyze-quality-v2 \
  --trials build/stride-stage2-pilot-v2/repair_trials.jsonl \
  --trials build/stride-quality-v2-confirmation-v1/extra_trials.jsonl \
  --output build/stride-quality-v2-confirmation-analysis-v1 \
  --expected-state-count 48
```
