# STRIDE Platform-Entry Order V1 Result

## Outcome

The corrected `r4` initial phase completed all 288 registered episodes with
zero execution errors and zero process timeouts.  Collection integrity passed,
but the order mechanism did not: a deterministic conflict-priority repair order
increased rather than decreased the observed platform-entry rate.  The uniform
trial 4--7 extension is therefore forbidden, and this branch stops without
training, runtime integration, or TTF testing.

## Execution identity

- Source HEAD: `90349e6` (`step_with_time_limit` outer-budget propagation), on
  top of the low-level deadline and canonical-module corrections.
- Output: `build/stride-platformentry-order-v1-r4`.
- Native producer:
  `build/linux/project/lns2_env.cpython-310-x86_64-linux-gnu.so`.
- Native SHA-256:
  `d0ad5b99ee3573b2cfdfb00c7fed16d4191f968ce82597b97add50fd69f97b3e`.
- Qualification: 17/17, with zero errors and zero process timeouts.
- Initial collection: 288/288 episodes, 36 cases, two arms and trials 0--3,
  with zero errors and zero process timeouts.
- Collection throughput: approximately 10.23 scheduled episodes/minute with
  16 workers.

The run imported no result from the superseded `v1`, `r2`, or `r3` outputs.
`wall_timeout` and `repair_limit` remain valid right-censoring outcomes rather
than execution failures.

## Integrity

Every frozen integrity condition passed: complete paired arms, paired
first-action PP seeds, the same forced agent set and initial repair fingerprint,
the forced action exactly once, the requested conflict-priority order applied
exactly, legal actions, matching fingerprints, valid stop reasons, and complete
schedule and case counts.

## Initial results

| Metric | Native order | Conflict-priority order | Order minus native |
|---|---:|---:|---:|
| Platform-entry rate | 56.25% | 59.03% | +2.78 pp |
| Success rate | 41.67% | 43.06% | +1.39 pp |
| Mean normalized fixed AUC | 0.28514 | 0.27379 | -0.01135 |
| Restricted mean repair decisions | 40.1447 | 38.9240 | -1.2207 |
| Right-censored episodes | 84 | 82 | -2 |

The case-cluster paired bootstrap interval for the platform-risk difference
(`order - native`) was `[-4.17 pp, +10.42 pp]`.  Its upper endpoint is not below
zero, and the point estimate is in the wrong direction.

Per-map platform-risk differences were:

- `maze-128-128-1`: 0.00 pp (95.00% versus 95.00%);
- `maze-128-128-2`: 0.00 pp (0.00% versus 0.00%);
- `maze-32-32-4`: +6.25 pp (73.44% versus 67.19%).

The last result also violates the preregistered rule forbidding a map-level
worsening greater than five percentage points.

## Gate decision and interpretation

The success, AUC, and restricted-mean conditions passed, but both primary
platform conditions failed:

1. overall platform entry did not decrease; and
2. one map worsened by more than five percentage points.

Thus `initial_extension_gate=false`, `mechanism_passed=false`, and
`extension_allowed=false`.  Conflict-priority ordering can sometimes shorten
successful continuations, but it is not a reliable pre-entry self-loop
prevention mechanism.  This result concerns only the order branch on the
outcome-enriched same-agent-set cohort; it neither validates nor invalidates a
separately designed compact dependency-aware candidate pool.

## Frozen evidence hashes

- `initial_report.json`:
  `c8b902ed9da1f9d7423c92d2db8187d792b58ffac8d077ffb2b52521c91628ef`
- `initial_run_config.json`:
  `35d7eb59e22aae3d185e8a8eae2ba72655a6c445a5ba3a9909cfc3541e457fb4`
- `initial_schedule.jsonl`:
  `edabec01b30863f46153b9079adac6961a0b5c0723bfef8b3d5956daf68de5e8`
- `collection_status.json`:
  `1d7306e17e6d5a1e072ee90616b2ce7ae9ef498a3d9e666f126932d319cd3ef6`

## Verification

- WSL Python: `892 passed, 35 skipped` with 16 pytest workers.
- Linux native: CTest `11/11` passed with `-j16`.
- Windows native: `build/windows/Release/lns2_tests.exe` passed.
- Repository hygiene: passed with 1,188 tracked files, one expected untracked
  result report, and all 24 retained-evidence entries verified.
