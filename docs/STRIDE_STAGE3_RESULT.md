# STRIDE-LNS Stage 3 result

## Decision

Stage 3 passed its registered data and isolation gates. The final
`stride-quality-v2` cohort contains exactly 600 states, 300 from official
Adaptive source episodes and 300 from frozen `v2-full` source episodes. This
authorizes the controlled Stage 4 training and feature-ablation protocol. It is
label-data evidence, not evidence that STRIDE improves end-to-end LNS2 runtime.

## Completed repair trials

The retained Stage 2 cohort contains 240 states. Ninety-six already had trial
indices 0 through 7; the remaining 144 received exactly the missing indices 4
through 7. That completion produced 10,284 rows over 2,571 state-candidate
pairs with zero collection errors. Candidate sets match the base collection,
paired seeds are constant across candidates at each state-index slot, and no
completion seed overlaps indices 0 through 3.

The retained completion report SHA-256 is
`7b9f7cb466d6a459bb43fbbb2d4c41f8f2d4353dedb4267be5693fb2270a8e60`;
the completion-trial JSONL SHA-256 is
`fc116ddead77fffee852c2c8bbc11d1504c52cb2f4f893e720c9e3437f179eb3`.
The mutually exclusive outcomes are 1,486 feasible repairs, 5,140
conflict-reduced repairs, 2,525 state changes without conflict reduction, one
accepted no-op, and 1,132 hard failures. Hard failures are valid label outcomes
rather than collection errors.

The 360-state extension separately completed indices 0 through 7 as recorded
in `docs/STRIDE_STAGE3_YIELD_RECOVERY.md`. Across the final cohort there are
10,713 candidates and exactly 85,704 consumed repair trials, eight per
candidate. Stored indices 8 through 15 from the earlier label-design
experiment remain diagnostic-only and are excluded from the Stage 3 label.

## Final quality labels

The label build produced:

- 600 states with at least one dominance pair;
- 10,713 candidate aggregates;
- 83,201 unordered dominance pairs;
- 166,402 oriented training rows;
- total oriented sample weight one per state, within floating-point tolerance;
- frozen 124-dimensional features and `runtime_used_in_label=false`.

Artifact SHA-256 values are:

- candidate aggregates:
  `e4976b2da43e617452bde0fa753ac01d8626d263f40495ea54773ae5847d994c`;
- dominance pairs:
  `c2b028285c446555ec75c99a79452a62d31eebad331fa1edf01460a8193b62b7`;
- label-build summary:
  `2d334e6d51a05760e24389ad8193272ae8129c70180a97c9105d9be23ed4b4c4`.

## Cohort audit

The repeatable audit entry point is:

```powershell
python scripts/run_stride_pipeline.py audit-stage3-labels `
  --config configs/stride_stage3_label_audit.json `
  --output build/stride-stage3-label-audit-v1
```

The executed audit passed every gate:

- 300 official-Adaptive and 300 frozen-V2 states;
- 36 labelled maps, exceeding the registered minimum of 24;
- 14 Stage 2 maps, all ten first-extension maps, and all 12 recovery maps;
- agent counts: 26 at 100, 221 at 200, 129 at 300, 76 at 400, and
  148 at 600 agents;
- 400 early, 100 middle, and 100 late decision states;
- at most one selected state per extension source-policy episode;
- zero exact or conservative name overlap with the 12 formal MovingAI/OOD
  maps;
- no duplicate candidates, invalid pair orientations, bad references, feature
  dimension errors, metadata mismatches, or trial-source hash mismatches.

The audit configuration SHA-256 is
`8a1f3ff942ee9fbb437268f85bf0e48136d6e7414ec1af49da710df161750f24`;
the audit report SHA-256 is
`1fc63b3b85a3616da97a7b675a4c7294ea148ceca5010f7a1f1ab7ea1072a9db`.

## Next boundary

Stage 4 must use the same 600 states and fixed train/validation grouping to
compare the frozen quality anchor, an old-label retraining control, and the new
quality-V2 model. Feature-family ablations must be trained and evaluated under
the same splits and seeds. No Stage 4 model may replace `v2-full` before the
registered offline, Shadow, and Quick gates pass.
