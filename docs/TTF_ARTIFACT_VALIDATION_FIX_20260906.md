# TTF report artifact validation fix (2026-09-06)

## Backup and scope

Before editing, the clean source commit
`f7c1c179c197fc9a425b578da5fd3107298a53ee` was pushed and verified remotely at
`backup/ttf-artifact-validation-00-original-20260906`. Work is on
`codex/ttf-artifact-validation`.

```bash
git restore --source backup/ttf-artifact-validation-00-original-20260906 -- path/to/file
```

This fixes the two retained report-reader defects identified before cleanup.
No retired experiment chain is restored. Native, PP, low-level search,
neighborhood generation, controller inference and frozen artifacts are not
changed. No formal experiment is rerun or saved result overwritten during
verification.

## Fixes

`evaluation/ttf_artifacts.py` provides one shared reader for the main warehouse
TTF and Component16 independent TTF analyzers. Missing or failed work remains
incomplete. Completed but malformed or inconsistent work raises before report
writing rather than contributing to statistics.

The reader checks:

- configuration, producer/native and run fingerprints against their saved
  contents; paired lanes must share producer/native and dataset identity;
- frozen model and controller manifests against the registered inputs, not
  merely self-consistent hashes inside the lane;
- base proposal and PP/SIPP environment configuration, fixed-pool augmentation,
  controller, task/seed cohort, deadlines and single-worker lane settings;
- checkpoint override identity, manifest task/map/seed/policy/episode, trace
  SHA/bytes/format/event count, initial-state reference and restore provenance;
- the existing complete trace validator and agreement between manifest and
  validated trace summaries; report input hashes are recorded in new reports.

The analyzers also reject a changed execution schedule, or completed lane data
with a missing schedule. These checks validate the saved serial-execution
contract; they are not an independent hardware concurrency measurement.

`evaluation/ttf_metrics.py` validates shared warehouse summary inputs before
aggregation. It requires real booleans and integer counts, finite nonnegative
times, the registered clock schema/deadline, and consistent stop/success/censor
semantics. Successful TTF equals the final transition timestamp (the initial
state timestamp for a zero-step episode); failed capped TTF equals the deadline.
Thus changing both summary copies and recomputing file hashes cannot lower TTF
without contradicting the validated trajectory clock. Late feasibility is
retained as failure with zero final conflicts only after the deadline.

The common summary check also protects retained heldout, boundary and
load-extension aggregation. It does not claim those historical analyzers have
all been migrated to the new full artifact reader.

## Historical evidence boundary

Read-only historical analysis checks stored producer identities internally;
it does not require old source hashes to equal the current source tree. This
is distinct from resume eligibility.

The strengthened reader was applied read-only to all 48 saved Component16,
Dual16 and Official episodes. All passed, and mean capped TTF remained:

| Controller | Mean seconds |
| --- | ---: |
| Official Adaptive | 0.8972446746 |
| Component16 | 0.4701885837 |
| Dual16 | 0.4965888286 |

No original trace, manifest, report, configuration or model was changed by that
verification. These bugs were input-validation weaknesses; the inspected
48-episode evidence did not trigger them and its scientific status is unchanged.

## Regression coverage

Shared parameterized tests cover malformed JSON, missing/duplicate rows,
non-finite/negative times, string booleans, wrong integer types, inconsistent
clocks, rehashed TTF changes, producer/run/trace/provenance mismatches, and
internally consistent but unregistered models/proposals/environment settings.
High-level report-gate tests use shared synthetic validated summaries, with
separate tests proving both analyzers actually propagate validation failures.
All synthetic trace tests run without invoking the native solver.

## Verification

- Existing Ubuntu-22.04 WSL native build: passed.
- CTest: 13/13 passed, including the official behavior boundary tests.
- Full Python suite: 1192 passed, 37 skipped in 265.53 seconds.
- Repository hygiene: zero errors; all 24 frozen evidence hashes verified.
- Native source, third-party source and frozen artifact diff against the
  pre-fix commit: empty.
- The 48 saved episodes above passed read-only artifact validation; no formal
  experiment or model training was rerun.

The tested revision is retained on `codex/ttf-artifact-validation` and at
`backup/ttf-artifact-validation-01-fixed-20260906`. The pre-fix recovery tag above
remains unchanged.
