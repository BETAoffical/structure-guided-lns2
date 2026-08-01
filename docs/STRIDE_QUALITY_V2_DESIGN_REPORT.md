# STRIDE-LNS quality V2 design check

The consumed 48-state design cohort completed an independent eight-seed versus
eight-seed check. This result permits the already-frozen fresh confirmation
experiment; it does not permit Stage 3 expansion or model training.

## Collection

- 48/48 states completed with zero errors.
- 845 candidates and 6,760 new outcomes at trial indices 8 through 15.
- Combined analysis coverage: 13,520 outcomes at indices 0 through 15.
- Design extension SHA-256:
  `a0a7e5dfda852eba699e80307ff05d624784e780af05ec5e4c5a8133678e5ea5`.

## Result

- aggregate pairwise direction consistency: `0.8570783133` -- pass;
- mean Top-3 intersection divided by three: `0.8472222222` -- pass;
- official-Adaptive source states: pairwise `0.8786`, Top-3 `0.9028`;
- frozen V2 source states: pairwise `0.8300`, Top-3 `0.7917`;
- 48 states and all 14 conflict-producing development maps were represented;
- runtime was not used by the label.

The deterministic analysis report reproduced byte-for-byte. Report SHA-256:
`c5cb3b472495be98d3235d3545b24e606c29fb6ee643749e34b373a436432368`.

## Decision boundary

The aggregate design gates passed, making an eight-seed label plausible. The
V2-source Top-3 subgroup remains below 0.80 and is explicitly retained as a
risk signal. Because the label weight was designed using this cohort, none of
these outcomes can serve as confirmation evidence.

The next allowed action is the frozen 48-state, state- and episode-disjoint
confirmation. Stage 3 remains blocked until that cohort independently reaches
pairwise consistency at least 0.70 and mean Top-3 overlap at least 0.80 with
zero collection errors.

## Reproduction

```bash
PYTHONPATH=build/linux/project /usr/bin/python3 scripts/run_stride_pipeline.py \
  collect-quality-v2-design \
  --selection build/stride-stage2-stability-selection-v1/stability_selection.jsonl \
  --collection build/stride-stage2-pilot-v2 \
  --output build/stride-quality-v2-design-extension-v1 --workers 4

python scripts/run_stride_pipeline.py analyze-quality-v2 \
  --trials build/stride-stage2-pilot-v2/repair_trials.jsonl \
  --trials build/stride-stage2-stability-v1/extra_trials.jsonl \
  --trials build/stride-quality-v2-design-extension-v1/extra_trials.jsonl \
  --output build/stride-quality-v2-design-analysis-v1 \
  --expected-state-count 48
```
