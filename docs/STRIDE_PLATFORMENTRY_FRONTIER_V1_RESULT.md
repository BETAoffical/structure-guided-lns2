# STRIDE Platform-Entry Frontier V1 Result

## Outcome

The corrected `r2` run completed all 2,600 registered episodes across the
initial and uniform extension phases with zero execution errors and zero
process timeouts. All integrity conditions passed. Neither deterministic
FrontierDependency intervention passed the final platform-entry gate, so this
branch stops without training, runtime integration, independent confirmation,
or TTF testing.

The result rejects two simple explanations. Blindly adding supported frontier
agents is not a reliable platform-prevention rule, and preserving the original
neighborhood size while exchanging members is not sufficient either. Candidate
membership still affects continuation quality, but the observed platform-risk
changes were one episode out of 360 and were not statistically supported.

## Execution identity

- Source milestones: `06cd249` (preregistration), `da8a8b5` (complete
  qualification cohort), and `72c5693` (hash-domain analysis correction).
- Output: `build/stride-platformentry-frontier-v1-r2`.
- Qualification: 19/19 keys, zero errors and zero process timeouts.
- Initial phase: 1,300/1,300 episodes, trial indices 0--3.
- Uniform extension: 1,300/1,300 episodes, trial indices 4--7.
- Combined analysis: 45 states, 325 unique actions and 2,600 episodes.
- Execution used 16 workers. Every case/trial used a strictly paired first PP
  seed across all actions and native PP order only.
- Every forced action was applied exactly once before returning to the frozen
  continuation policy.

`wall_timeout` and `repair_limit` remain valid right-censoring outcomes rather
than execution errors. The extended analysis used 16 independent parser
processes and completed in approximately four minutes.

## Integrity

All 15 frozen checks passed: schedule, case, action and episode counts;
complete candidate sets; legal and exactly-once forced actions; scheduled
neighborhood equality; paired PP seeds; common runtime initial identity within
each case/trial; registered repair-state identity; native PP order; valid stop
reasons; and zero action or fingerprint errors.

The earlier initial-report failure was an analysis defect, not corrupt data. It
compared a context-free runtime state hash with a context-rich registered
checkpoint hash. The corrected analysis compares runtime hashes only within
their own domain and separately verifies the registered repair fingerprint.

## Extended results

| Metric | Historical action | V2 anchor | Compact augment | Same-size exchange |
|---|---:|---:|---:|---:|
| Episodes | 360 | 360 | 360 | 360 |
| Mean neighborhood size | 30.40 | 16.00 | 36.27 | 30.40 |
| Platform-entry rate | 61.67% | 62.78% | 61.39% | 61.39% |
| Success rate | 43.89% | 45.56% | 45.83% | 43.06% |
| Mean normalized fixed AUC | 0.27803 | 0.25133 | 0.24514 | 0.25113 |
| Restricted mean repair decisions | 42.4906 | 41.1254 | 41.8695 | 42.4650 |
| Right-censored episodes | 202 | 196 | 195 | 205 |

Both deterministic variants changed the platform rate by only `-0.28` points
relative to the historical action. The case-cluster paired 95% bootstrap
intervals were:

- compact augment: `[-2.78 pp, +1.94 pp]`;
- same-size exchange: `[-4.17 pp, +3.33 pp]`.

Both upper bounds are above zero, so the primary requirement of supported
platform-risk reduction failed. Same-size exchange also failed the success
non-inferiority condition (`43.06%` versus `43.89%`). Compact augment passed the
secondary success, AUC and restricted-mean conditions, but those improvements
cannot substitute for the failed platform endpoint.

## Map heterogeneity

Platform-risk differences relative to the historical action were:

| Map | Compact augment | Same-size exchange |
|---|---:|---:|
| `maze-128-128-1` | +0.78 pp | +1.56 pp |
| `maze-128-128-2` | 0.00 pp | +1.04 pp |
| `maze-32-32-4` | -1.47 pp | -2.94 pp |

The variants helped the small Maze cohort but did not reproduce that direction
on either 128x128 cohort. All map changes stayed within the preregistered
five-point safety bound, but the cross-map inconsistency prevents a mechanism
claim.

## Size and pool diagnostics

Size-band results are post-hoc and confounded by state and candidate family.
They do not establish a monotone size rule. The `<=24` band had a lower platform
rate (`36.57%`) but also the lowest success rate (`30.09%`) and the worst mean
AUC (`0.37203`). The `33--40` band had a `64.82%` platform rate but a higher
`41.88%` success rate. The small V2 anchor likewise had a worse platform rate
than the historical action despite better AUC and success.

Across 336 case/trial groups with frontier candidates, a post-hoc pool oracle
found at least one candidate avoiding the platform in `60.12%` and at least one
successful candidate in `66.67%`. This is only a pool upper bound. It is not an
online selector and cannot be used as a runtime or solver-improvement claim.

## Gate decision and interpretation

`final_gate=false`, `mechanism_passed=false`, and there are no passing roles.
The experiment therefore establishes:

1. larger supported neighborhoods are not automatically safer;
2. same-size membership exchange is not a stable platform-prevention rule;
3. one-step continuation quality can improve without a supported reduction in
   repeated exact rollback; and
4. candidate availability is not the same as a deterministic pre-entry policy.

No training, runtime integration, result-blind confirmation, or TTF test is
authorized. Any next branch must explain the cross-seed and cross-map platform
transition itself rather than rescore these actions by one-step quality or add
more agents indiscriminately.

## Frozen evidence hashes

- `platformentry_frontier_report.json`:
  `eedb680ad3a2b6052692883ea0c1255b3679806e00004ca2a8d8375423e010f7`
- `initial_report.json`:
  `20be16faa84bf17eefc32229a6928de688edf3aa2c7af84bdaa36641031d4e95`
- `initial_schedule.jsonl`:
  `72412c1874af883d31e3d27df840463b6ed212c5eb89175e3fced68159b2e47c`
- `extension_schedule.jsonl`:
  `ab5305295f26d401fc903d19ae1f6ca768644e7b6ee7c05091a584ad4d8df0fe`
- `initial_run_config.json`:
  `6ccfc13e436f45d55dca3e29190c35ca4268e8858251b78d027a39f42c60e0cf`
- `extension_run_config.json`:
  `0371564147bbb032fc12bb94eedfd530d0f9b7e0d4ddbf03fcb9a51c420b791e`
- `collection_status.json`:
  `48291e3486b1abe61c99571ce3e9759524b8d5cf4cd3b99c17c0d6f94340a0a5`

## Verification

- WSL Python: `899 passed, 35 skipped` with 16 pytest workers.
- Linux native: CTest `11/11` passed with `-j16`.
- Windows native: `build/windows/Release/lns2_tests.exe` passed.
- Repository hygiene: passed with 1,199 tracked files, one expected untracked
  result report, and all 24 retained-evidence entries verified.
