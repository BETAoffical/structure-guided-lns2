# STRIDE Compact-Blocker Escape Durability v1 Result

## Outcome

The read-only 16-process audit completed over all 280 registered arm traces.
All identity and integrity checks passed. Among the 81 paired rescue-trigger
events, 41 had a trial-local compact plan and 40 were paired compact-ineligible
fallback observations. No solver call or new PP collection was performed.

The result is `transient_escape_signal`, not a durable recovery result.
Compact plus blocker has a stable immediate effect and the effect remains
detectable through three repair decisions, but horizon-8 durability and edge
clearance do not pass the frozen gates.

## Paired results

| Horizon | Complete pairs | Sustained escape, full blocker | Sustained escape, compact + blocker | Paired difference (95% state-cluster bootstrap) | Improved / worse / same |
| --- | ---: | ---: | ---: | ---: | ---: |
| H=1 | 81 | 24.69% | 38.27% | +13.58 pp `[+2.82, +25.35]` | 13 / 2 / 66 |
| H=3 | 76 | 25.97% | 40.79% | +14.47 pp `[+2.94, +26.92]` | 13 / 2 / 61 |
| H=8 | 63 | 12.70% | 25.40% | +12.70 pp `[0.00, +25.81]` | 10 / 2 / 51 |

The H=8 lower confidence limit is exactly zero, so it fails the preregistered
strictly-positive requirement. The original-platform re-entry difference at
H=8 is favorable and stable at -17.46 pp `[-32.76, -4.29]`, but this alone is
not enough: original conflict-edge retention AUC changes by -0.0532
`[-0.1277, +0.0183]`, and normalized conflict AUC changes by -0.0489
`[-0.1294, +0.0135]`. Both long-horizon intervals still cross zero.

At H=8, the new-platform formation difference is 0.00 pp
`[-10.29, +9.23]`, while feasibility changes by +1.59 pp
`[-3.33, +7.27]`; neither supplies stable additional evidence.

## Map boundary and censoring

- `maze-128-128-1`: H=8 sustained-escape point difference +8.11 pp.
- `maze-32-32-4`: H=8 sustained-escape point difference +19.23 pp.
- `maze-128-128-2`: all 18 eligible events in each arm are right-censored
  before H=8, leaving no complete H=8 pair.

The third map is reported as unavailable and fails the per-map safety gate; it
is not discarded. Overall H=8 uses 63 complete pairs from 24 state clusters,
whereas H=1 uses all 81 pairs from 31 state clusters.

## Interpretation

Semantic compaction is not merely creating a one-decision fingerprint change:
the sustained-escape contrast remains statistically positive through H=3.
However, current data do not establish that this advantage persists to H=8 or
clears the original conflict structure. The mechanism therefore remains a
useful diagnostic component, not a deployable rescue policy.

This audit does not reopen the previous rescue-wall failure: compact + blocker
still did not reduce rescue PP wall time, so no trial-4--7 extension, runtime
integration, training or TTF test is authorized. The compact-blocker branch
stops here rather than collecting more seeds or tuning on the same cohort.

## Integrity and operational notes

- Source run fingerprint:
  `3d1275576be4ca73d79861f936a0b23306da20b3f0fe384654ad0b2dd1bbc0c5`.
- Report SHA-256:
  `73dc1b3a8d362d00129663db58d59df7eb2438c69728592eef2b98ff0072603a`.
- Audit rows SHA-256:
  `0ac67ea48f2b5984fba1acbc46d14051203968795b3c182c8081dff398e3cadb`.
- Audit run config SHA-256:
  `0d9c3b9560d47a9c0197b3ee27097a00900a7cf8115d06a9947b9c5071893d61`.
- Windows Python reported a false missing-file error on a deep trace path that
  exists; the registered WSL workflow read all traces successfully. This was a
  Windows path-length limitation, not source corruption.

## Validation

- WSL Python suite with 16 workers: 961 passed, 35 skipped.
- Linux CTest with 16-way scheduling: 13/13 passed.
- Windows native `build/windows/Release/lns2_tests.exe`: passed.
- Repository hygiene: 24/24 evidence entries verified, zero errors.
