# SlotPool versus Full StructPool Raw-TTF Diagnostic v1 Result

## Decision

Full StructPool is faster in the preregistered known-tail-excluded diagnostic.
SlotPool solved every episode but increased mean reset-inclusive raw wall TTF by
`4.5860%` (`6.715697` versus `6.421219` seconds), increased median TTF and was
faster on only `10/29` paired keys. SlotPool therefore fails the diagnostic
gates and is not promoted over full StructPool.

## Scope and integrity

The fixed cohort contains five maps, ten tasks and solver seeds 1--3. Exactly
one key was excluded before timing, as requested: Maze task seed 233 with solver
seed 3, the previously known 61/301-repair regression. The remaining 29 paired
keys produced 58 strictly alternating, single-worker, run-to-completion
episodes. Three reset states had zero initial conflicts and 26 required repair.

Both controllers succeeded on all 29 keys. Initial fingerprints and conflict
counts matched pairwise; there were zero execution errors, invalid actions,
semantic mismatches, capped TTF values or clock-schema violations. Both
StructPool treatments activated and selected structural candidates, and the
frozen SlotPool reduction was exercised. Reanalysis reproduced the report
deterministically.

## Aggregate result

| Metric | Full StructPool | SlotPool |
| --- | ---: | ---: |
| Mean raw TTF (s) | 6.421219 | 6.715697 |
| Median raw TTF (s) | 2.921570 | 3.642969 |
| Mean repair iterations | 9.103448 | 9.206897 |
| Mean PP time (s) | 2.474097 | 2.709703 |
| Mean neighborhood selection (s) | 0.326120 | 0.424105 |
| Mean normalized wall AUC | 0.716304 | 0.712207 |
| Successes | 29/29 | 29/29 |

SlotPool adds `0.294478` seconds per episode on average. Its neighborhood
selection time is about `30.05%` higher and PP time about `9.52%` higher. The
mean repair-count increase is small (`+0.103448`), but the aggregate hides
opposite seed-level trajectories.

## Map-level result

| Group | SlotPool TTF improvement vs full | Paired wins | Mean repair delta |
| --- | ---: | ---: | ---: |
| Den300 | -4.5542% | 1/6 | 0.000 |
| Maze100 | -25.4994% | 3/5 | +1.600 |
| Random500 | +5.2986% | 2/6 | 0.000 |
| Room400 | -14.1748% | 1/6 | +0.667 |
| Warehouse600 | +4.2292% | 3/6 | -1.500 |

The excluded catastrophic key did not re-enter the result. Nevertheless, the
remaining Maze keys still expose trajectory risk. Maze task seed 277 with
solver seed 3 takes `18.858097` seconds and 46 repairs under SlotPool versus
`5.322878` seconds and 13 repairs under full StructPool. Maze task seed 233 with
solver seed 1 takes `16.867582` seconds and 40 repairs versus `7.827975` seconds
and 19 repairs. These observations were not deleted after timing.

SlotPool also has real wins: on Maze task seed 233 with solver seed 2 it reduces
TTF from `13.400436` to `6.102532` seconds and repairs from 42 to 16. The same
candidate reduction can therefore help or hurt depending on the realized state
and PP trajectory; a higher-quality current-step candidate budget is not a
stable end-to-end speed advantage.

## Interpretation and boundary

SlotPool materializes the full four-size structural grid and runs an additional
pairwise model before the unchanged V2 final ranker. That extra work is not
offset by a reliable reduction in repair rounds in this cohort. Full StructPool
uses the cheaper fixed six-candidate reduction and wins the primary raw-TTF
comparison.

Because one key was excluded using an already-known outcome, this remains a
focused diagnostic rather than formal speed or generalization evidence. No
additional result-based exclusions are permitted. The result supports retaining
full StructPool as the faster of these two research treatments on this scope;
`v2-full` remains the project default because it was not re-evaluated here and
earlier evidence did not promote either topology treatment.

## Artifacts

- report: `build/stride-slotpool-structpool-ttf-diagnostic-v1/slotpool_structpool_ttf_report.json`
- report SHA-256: `4c82e4283ec3626976f9944ec26f55757973d063ab1ff861dc2fa7c8a16f3c3d`
- schedule SHA-256: `37346a3453f5ed387b36862f31330b13a97f610652fadd04a17f37ea033808d4`
