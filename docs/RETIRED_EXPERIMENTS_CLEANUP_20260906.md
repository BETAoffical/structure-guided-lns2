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
