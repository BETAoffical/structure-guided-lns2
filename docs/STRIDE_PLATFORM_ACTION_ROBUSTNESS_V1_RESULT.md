# STRIDE Platform Action Robustness v1 result

## Outcome

The zero-solver cross-seed diagnostic completed over the full frozen
Platform-entry Frontier collection.  Input integrity passed for all 45 cases,
325 concrete actions, 2,600 episodes and trial indices 0--7.  No PP,
continuation, training, runtime integration, or TTF run was performed.

The principal result is that the current bottleneck is native-realization
instability, not a large population of stable actions missed by the frozen
deterministic selectors.

| Case classification | Cases | Fraction of all 45 | Fraction of 33 applicable |
|---|---:|---:|---:|
| Historical action never entered a platform in these eight trials | 12 | 26.67% | -- |
| Frozen deterministic rule selected a stable, quality-preserving action | 3 | 6.67% | 9.09% |
| Stable, quality-preserving action existed but deterministic rules missed it | 4 | 8.89% | 12.12% |
| Stable platform action existed only by sacrificing quality | 0 | 0.00% | 0.00% |
| Only seed/order-sensitive headroom was observed | 25 | 55.56% | 75.76% |
| No tested action supplied even unstable headroom | 1 | 2.22% | 3.03% |

Only 7/33 applicable cases therefore contained any concrete action that
reduced platform entry in both fixed four-seed halves while preserving
success, normalized fixed AUC and restricted repair decisions.  Three of those
seven were already selected by a frozen deterministic rule.  The remaining
deployable selection headroom is only 4/33 applicable cases.

## Causal-mechanism cross-tabulation

The causal annotation comes from the later registered platform checkpoint and
is used only as outcome-enriched mechanism evidence.

| Causal class | No reference platform | Deterministic success | Selection headroom | Seed-unstable | Pool/repairer gap |
|---|---:|---:|---:|---:|---:|
| Set | 10 | 0 | 1 | 9 | 1 |
| Order | 1 | 1 | 0 | 11 | 0 |
| Joint | 0 | 0 | 0 | 2 | 0 |
| Residual PP | 1 | 2 | 3 | 3 | 0 |

This resolves the apparent conflict with the earlier 23/45 set-related causal
result.  That result describes interventions at an already formed platform.
At the earlier first-structural-action checkpoint, 10/21 set-classified cases
did not reproduce a platform under any of these eight historical-action
trials, nine exposed only unstable seed-dependent alternatives, one had stable
selection headroom and one had a pool/repairer gap.  A later set defect is not
equivalent to a reliably preventable first-action membership defect.

## Map heterogeneity

- `maze-128-128-2`: all 12 cases were `reference_no_platform`; this map supplied
  no first-action prevention contrast under the registered eight seeds.
- `maze-128-128-1`: 12 seed-unstable cases, two deterministic successes, one
  selection-headroom case and the single pool/repairer gap.
- `maze-32-32-4`: 13 seed-unstable cases, one deterministic success and three
  selection-headroom cases.

The seven stable actions had sizes 16, 18, 24, 32, 37, 39 and 40 (one case had
two stable actions).  They do not support a monotone small- or large-neighborhood
rule.

## Interpretation

The earlier per-case/trial post-hoc oracle reported an avoiding candidate in
60.12% of groups.  This analysis shows why that number is not deployable:
most wins change with the native PP seed/order half.  Once the same concrete
action must improve in both halves and retain downstream quality, robust pool
opportunity falls to 7/33 applicable cases.

The evidence therefore separates the current causes as follows:

1. **Selector deficiency is real but narrow at entry.**  Four applicable cases
   contain a robust quality-preserving action missed by both deterministic
   rules.
2. **Candidate composition is not the dominant first-entry limitation in this
   tested pool.**  Only one applicable case has no observed action headroom.
   This does not prove pool completeness outside the frozen 45 cases.
3. **Native PP realization is the dominant limitation.**  In 25/33 applicable
   cases, apparent headroom is not stable across the two seed/order halves.
   A set-only deterministic selector cannot turn those outcomes into a reliable
   prevention guarantee.
4. **The later causal set diagnosis remains valid.**  It explains an already
   formed platform, but cannot be projected backward into a universal
   first-action pool rule.

## Decision

Do not train a new pre-entry selector, enlarge the candidate pool, or run more
first-action continuation from this result.  The maximum current deployable
selection headroom is 4/33 applicable cases and is concentrated in residual-PP
and one set-classified case; it cannot justify another broad model or pool
cycle.

Strict zero-loop prevention by selecting only an agent set is rejected for the
current action interface.  Any further branch must change the controllable
repair mechanism itself or establish an outcome-blind, cross-map predictor of
the *distribution* over native PP realizations while retaining hard success,
AUC and repair-decision non-inferiority.  Such a branch requires separate
preregistration; it must not reuse the per-seed oracle as an online label.

## Artifacts

- frontier report SHA-256:
  `eedb680ad3a2b6052692883ea0c1255b3679806e00004ca2a8d8375423e010f7`
- platform witness SHA-256:
  `202f88bc43c72f7d2c6fa48bde32dd20069906bc769e5b57d9ad238eb12cec53`
- machine-readable result:
  `build/stride-platformentry-frontier-v1-r2/platform_action_robustness_report.json`
- machine-readable result SHA-256:
  `70985c47061a3c93dce148a64185d9d931a9c4c082fba6a6b50b03222deae765`

## Verification

- WSL Python: `902 passed, 35 skipped` with 16 pytest workers.
- Linux native: CTest `11/11` passed with `-j16`.
- Windows native: `build/windows/Release/lns2_tests.exe` passed.
- Repository hygiene: passed with 1,200 tracked files, five expected new
  analysis files, and all 24 retained-evidence entries verified.
