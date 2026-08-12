# STRIDE TransactionalRepair v1 Report

## Decision

`stride-transactionalrepair-v1-r2` completed with full integrity and passed every frozen mechanism-readiness gate. The primary `transactional_set_order_retry` policy is therefore eligible only for an independently preregistered, result-blind confirmation cohort.

This result does **not** authorize model training, runtime integration, a default-controller change, or a TTF experiment. It is a bounded causal mechanism result on the frozen 45-state cohort.

## Experiment identity

- Registration: `configs/stride_transactionalrepair_v1_registration.json`
- Output: `build/stride-transactionalrepair-v1-r2`
- Execution unit: `state x trial_index x policy`
- Coverage: 45 states x 16 paired trial indices x 3 policies = 2,160 policy jobs
- Parallelism: 16 workers; parallel execution was retained through the final 32 completions
- Per-policy-job hard timeout: 300 seconds
- Run fingerprint: `af718dae...`
- Registration SHA-256: `ec23a0b5448ba5bb839a8392b21f45af8153b36dfa152d4dc8c0ea811730bc15`
- Native module SHA-256: `7f937f5c4647f621db7e855f7d6dd6a807c5d5a127c074106cdd56494aa1bcfd`

The old aggregate output under `build/stride-transactionalrepair-v1` was not imported. The r2 run used fresh atomic policy artifacts and preserved strict first-attempt parity across all three policies.

## Integrity

All integrity checks passed:

- 45 states, 720 complete state-trials, 2,160 policy rows, and 2,160 atomic policy files;
- zero execution errors and zero timeouts;
- exact initial-attempt parity across policies;
- exact native rollback before every retry;
- registered attempt caps and complete policy coverage;
- no commit with worse conflict count;
- no runtime, TTF, or future-trajectory fields in the scientific records.

## Main result

| Policy | Success | Returned unchanged | Strict conflict reduction | Mean normalized conflict reduction | Mean attempts | Mean added-agent ratio | Max neighborhood fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single attempt | 35.83% | 64.17% | 32.64% | 0.0676 | 1.000 | 0.000 | 0.400 |
| Set retry | 65.97% | 34.03% | 60.69% | 0.1349 | 1.639 | 0.188 | 0.500 |
| Set + order retry | **80.69%** | **19.31%** | **74.58%** | **0.1674** | 1.981 | 0.189 | 0.500 |

Relative to the registered single attempt, the primary policy:

- raised native replan success by 44.86 percentage points;
- reduced returned-unchanged outcomes by 44.86 percentage points;
- raised strict conflict reduction by 41.94 percentage points;
- increased mean normalized conflict reduction from 0.0676 to 0.1674.

Adding the registered order retry on top of set expansion also improved every principal mechanism metric: success rose by 14.72 points, returned-unchanged fell by 14.72 points, strict conflict reduction rose by 13.89 points, and normalized conflict reduction rose by 0.0324.

## Robustness breakdown

The returned-unchanged rate improved on every map:

| Map | Single attempt | Set + order retry | Absolute reduction |
|---|---:|---:|---:|
| `maze-128-128-1` | 67.97% | 21.09% | 46.88 points |
| `maze-128-128-2` | 57.81% | 6.77% | 51.04 points |
| `maze-32-32-4` | 65.07% | 26.47% | 38.60 points |

The two frozen seed halves agreed: returned-unchanged fell by 45.28 points in trials 0-7 and 44.44 points in trials 8-15. The mechanism also improved all four discovery root-cause groups. The residual-PP-instability group remained the hardest, but success still rose from 26.39% to 59.03% and returned-unchanged fell from 73.61% to 40.97%.

## Frozen gates

All gates passed:

- overall returned-unchanged reduction >= 0.15;
- overall success improvement >= 0.15;
- each seed-half returned-unchanged reduction >= 0.10;
- strict returned-unchanged improvement on every map;
- no decrease in strict conflict-reduction rate;
- joint set-and-order policy not worse than set-only;
- mean attempts <= 2.25;
- mean added-agent ratio <= 0.50;
- maximum total neighborhood fraction < 0.80;
- zero errors/timeouts, 16-worker execution, and tail parallelism.

## Interpretation

The experiment rejects the simplest explanation that these failures are merely unavoidable PP randomness after an otherwise complete neighborhood choice. In a large fraction of paired trials, the same first attempt failed, exact rollback restored the pre-state, and a bounded retry succeeded after adding observed external blockers and, when necessary, changing repair order.

The evidence therefore supports a joint defect in the selected agent set and repair transaction/order. A nominally strong neighborhood can still omit agents that block the attempted paths, or present a repair order that makes a feasible joint change difficult for sequential PP. This provides a concrete mechanism by which repeated selection can return unchanged and contribute to solver stagnation.

It does not yet prove that the policy prevents long tails end to end. The cohort is outcome-enriched and was used for mechanism discovery; it is not an independent generalization cohort. The next valid step is a fresh, result-blind preregistration that freezes the same policies, blocker cap, order rule, retry limits, and gates. Only confirmation can justify considering a separately preregistered runtime/TTF evaluation.

## Artifact hashes

- `collection_status.json`: `a110cf3761e3e82dce357d40eeb98392f351f98f3dffb1546f34d72ff984e2`
- `transactionalrepair_report.json`: `65bfc9a2f696ffe01a308b1898f37c32e93241ad4de3820dd3c16eb48134eeff`
- causal registration input: `61d63da5ee33cb531149afe6adec31ea13c3883de1185b996bb419e2cf4e79a8`
- causal report input: `f2a5b0f5a99ea58382539681179b948783e1ab7c18a9bc2fc3abe775402e2ee7`
- causal state effects input: `507746850954dd893dce939441da38cc6f41ffe6a2399c109871a38010f61c78`
- causal state manifest input: `45a2d484ad85797172bdae82f73e978b454c95451bb1eb7a04db88c94c32c125`
- predictability report input: `2396310b3ac75ab519aa4015442ba44880c1e159162f50c63a81fd70ea9297e8`

## Validation

- WSL Python: `812 passed, 35 skipped` with 16 workers.
- Linux native CTest: `11/11` passed with `-j16`.
- Windows native `lns2_tests.exe`: passed.
- Repository hygiene and retention/hash audit: passed.
