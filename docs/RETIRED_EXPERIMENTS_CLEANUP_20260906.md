# Retired experiment cleanup (2026-09-06)

## Scope and recovery

This cleanup removes execution chains that have a recorded stop decision and
will not be rerun locally. Frozen reports, registrations, hashes, model bundles,
maps and raw result directories are retained. No native solver, PP, low-level
search, RNG, or promoted controller algorithm is changed. Component16 remains
available for development confirmation; it is not promoted by this cleanup.

Before any source edit, commit
`6e2f605fb989422bc3f15de93180cb26ef5ef836` was pushed to both:

- branch `codex/component16-threeway-ttf-v1`;
- tag `backup/retired-experiments-cleanup-00-original-20260906`.

Both remote refs were verified against the full commit SHA. Work continues on
`codex/retired-experiments-cleanup`. To recover a historical file without
reverting unrelated work:

```bash
git restore --source backup/retired-experiments-cleanup-00-original-20260906 -- path/to/file
```

Git source recovery does not cover ignored `build/` data. No build results or
external archives are removed. In particular, retain the H1 evidence ZIP
`../backups/dual16-hierarchical-admission-h1-v1-20260901.zip` with SHA256
`02b6cb36b11229193eb3bbb687e9e74aa34163b514239324ff2cfeb27ae68a5a`.
Component16 still reads the frozen H1-cohort checkpoint manifest to establish
map/task/checkpoint disjointness, not the retired H1 collector implementation.

## Stage 1: independent stopped chains

- Remove the H1 collector, label evaluator, CLI and two dedicated test files.
  The result remains `NO_GO_LABEL_SUPPORT`; see
  [H1 result](STRIDE_DUAL16_HIERARCHICAL_ADMISSION_H1_V1_RESULT.md).
- Remove the aborted TransactionalRepair confirmation runner, CLI and dedicated
  test. The original form is not to be restarted; see
  [confirmation stop](STRIDE_TRANSACTIONALREPAIR_CONFIRMATION_V1_ABORT.md).
- Remove the ScalePool-only adaptive-size generator, result class and two tests.
  Retain shared topology candidate construction and any helpers with retained
  consumers; see [ScalePool result](STRIDE_SCALEPOOL_V1_OFFLINE_REPORT.md).
- Replace duplicate repository-root functions in retained warehouse runners
  with module constants. Register the five retained Component16 modules/tests
  and remove stale Dual16 plateau-fallback descriptions from the manifest.

Historical source identities remain in their result artifacts. Cleanup changes
source hashes, so old completed runs are evidence, not new-code resume inputs.
No formal experiment is rerun as part of this work.

Stage 1 validation: full WSL Python suite `977 passed, 37 skipped`; native build
passed and CTest `13/13`, including official parity checks. An independent
focused check of Component16 evidence inputs, topology and load-extension
tests also passed (`38 passed`). The source diff contains no changes under
`src/`, `include/`, `third_party/`, or `CMakeLists.txt`.

The next phase starts only after this stage is committed, pushed and protected
by `backup/retired-experiments-cleanup-01-before-shared-pruning-20260906`.

## Stage 2: remove historical hosts after preserving their live dependencies

The phase-1 recovery tag and remote cleanup branch were both verified at
`0de922ee7157a011eb9cba8241997243c63753b5` before this phase began.

- Remove MultiValue collection/labels and their dedicated CLI/tests after the
  [label stability stop](STRIDE_MULTIVALUE_PILOT_V1_REPORT.md). Keep strict-prefix
  history reconstruction in `runtime/temporal_state.py`; keep source-trace
  lookup in its remaining FrontierDependency consumer.
- Remove ParetoPool and its gap-audit execution chain after its last retained
  production consumer (MultiValue) is removed. This also removes the test-class
  import that caused topology tests to be collected twice, and the now-unused
  ScalePool family-name/order helpers.
- Remove RepairClosurePool execution and dedicated tests after its
  [stop decision](STRIDE_REPAIRCLOSUREPOOL_V1_RESULT.md). Preserve the small
  opportunity-summary function and its behavior test in the CausalClosure
  opportunity consumer.
- Remove RepairDependency predictability execution after its
  [hard stop](STRIDE_REPAIRDEPENDENCY_PREDICTABILITY_V1_REPORT.md). The retained
  TransactionalRepair and NativeOrder consumers use the existing common
  contained-path resolver instead of importing that historical experiment.
- Remove the stopped base Hybrid/SourceRoute online execution branches and
  their private branch tests. A small `runtime/fixed_structshell.py` registry
  accepts only retained Dual16/Component16 payloads. Keep the candidate-result
  type, exact-set merge and offline budget reducer used by retained callers.

Required history/summary regression tests move with the preserved logic, rather
than disappearing with the historical test files. Producer dependency lists
and the retention manifest are updated for the new ownership. Existing
Hybrid membership/budget evidence tools remain; they do not restore the
deleted online execution routes or authorize a compressed/runtime replacement.

## Evidence and non-goals

All registered configurations, result reports, model bundles, maps and raw
experiment data remain in place. The saved H1, MultiValue and other negative
conclusions are not reinterpreted. No report metrics are recalculated or
overwritten by this cleanup. The two report-input validation weaknesses found
in the preceding audit are a separate retained-path repair task, not silently
claimed as fixed by deleting unrelated experiments.

## Size comparison

Counts use the original Git tree `6e2f605` and the final cleanup tree; Python
physical lines include comments and blank lines. Test modules are files named
`test_*.py` under `tests/`, not collected test cases. No hard reduction target
was used.

| Measure | Original | Cleaned |
| --- | ---: | ---: |
| Tracked files | 1,295 | 1,271 |
| Python files | 490 | 465 |
| Python physical lines | 171,103 | 159,877 |
| Test modules | 141 | 132 |
| Python scripts | 142 | 136 |
| Registered configuration files | 337 | 337 |
| `closed_loop_confirmation.py` physical lines | 4,704 | 4,531 |

Twenty-six old Python files are removed and one fixed-pool registry is added.
Net Python reduction is 11,226 lines. This is source cleanup, not evidence that
the current solver is faster; no new performance claim is made.

## Final verification

- Full WSL Python suite: `925 passed, 37 skipped` (235.13 seconds).
- Native build and CTest: `13/13`, including official parity hash tests.
- Repository hygiene: zero errors; all 24 registered evidence hashes verified.
- Retention manifest: 454 entries, no duplicate paths or missing files.
- No retained Python imports or producer file lists reference removed modules.
- Necessary migrated helpers match the pre-stage-2 implementation: 1,000
  synthetic trace cases, 33 opportunity summaries and seven path-containment
  cases; independent review additionally checked 1,857 prefix/window cases,
  including the resulting history hashes.
- The complete diff from `6e2f605` leaves native source, headers, third-party
  source, CMake, model artifacts and registered experiment configurations
  unchanged (only the retention manifest is updated under `configs/`).

The final cleanup is protected by
`backup/retired-experiments-cleanup-02-final-20260906` on the remote, alongside
the original and pre-shared-pruning recovery tags. The cleanup branch is
`codex/retired-experiments-cleanup`.
