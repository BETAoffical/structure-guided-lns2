# SA/raw maintenance and integrity repair (2026-09-26)

## Recovery and scope

Before editing, the clean commit `f1ad54cd1709d63b5fbff215cb2f1afb1b26de11`
was pushed and verified at `backup/sa-maintenance-00-original-20260926`.
The maintenance branch is `codex/sa-maintenance-hardening`.

```bash
git restore --source backup/sa-maintenance-00-original-20260926 -- path/to/file
```

This phase fixes the four issue classes from the preceding review and removes
confirmed duplicate implementations. It does not change native sources, PP,
SA acceptance, candidate generation, model weights or controller inference.
No formal experiments are rerun. Ignored experiment data and archives are not
deleted or covered by the source-code backup tag.

## Integrity changes

- Raw TTF readers bind task, solver seed, paired random-stream identity and
  all previously checked job/model identifiers. The artifact inventory must
  contain exactly the initial state, final state and compressed trace, each
  with a SHA256. Non-object JSON and non-boolean flags are rejected.
- The full trajectory audit now also checks initial/final conflict counts
  against the states and native PP time against the per-transition sum.
  Counts have strict integer types; timing fields must be finite/nonnegative.
- Both entry points use `evaluation/raw_ttf_artifacts.py`. The raw registration
  source list includes the new validator and its shared contracts dependency.
- Parameterized regression fixtures reject resealed corrupt identities,
  missing file inventories, changed conflict counts and fake timing totals.
  Native micro tests exercise the integrated audit, not just the helpers.

These are stricter readers for the existing result schema, not new solver or
metric definitions. Successful TTF and success-rate calculations are unchanged.

## Test environment isolation

The six SA native micro tests execute in independent Python processes with
the existing `build/linux/sa-wall-clock-v1` binary loaded before test imports.
An earlier import of the ordinary native can no longer select the wrong
binary or make SA tests order-dependent. Existing native identity checks are
not removed. No subprocess changes the parent process's loaded module.

Two sklearn-only tests explicitly skip when sklearn is unavailable, matching
the existing optional Torch test policy. They are also run in the existing
Windows sklearn 1.5.0 environment; no packages are installed.

## Deduplication and retention

- Bounded and uncapped credit reuse one gradient coefficient implementation,
  with distinct terminal validators. Semantic equivalence and differing cap
  behavior replace the test that required two copied function bodies.
- Repeated `require` definitions now import the shared contract helper.
- The two terminal probes share CLI dispatch while retaining their own phase
  callbacks; two files-based plan readers share the registered-input check.
- The reconstruction reader reuses the existing seal helper. Unused imports
  identified in the review, and the import made redundant by CLI extraction,
  are removed. Existing documented frozen-file hygiene exceptions are not
  expanded to conceal new duplicates.
- The retention manifest gains the 82 missing production/test entries plus
  the new validator and its regression test, each with a stated purpose.

This is not a blanket deletion of past research. Current raw code still
imports feature, actor, contract and replay helpers from historical modules.
Removing entire research chains requires extracting those live dependencies
and freezing their evidence under another remotely verified checkpoint.

## Historical evidence and resume boundary

All 108 episodes from the three-arm fast-runtime batch and the remaining
two-arm batch pass the strengthened identity/outcome checks. Their 16,479
trace steps were read and PP time sums recomputed without solver calls.
The official report hashes remain:

- Fast three-arm report: `5ff48bc57d20d9fbaf36f4b68779331a4e03d6c12f614102ab4deaa8af3bae06`
- Remaining/combined report: `0381f94751d9777bb690d10590eeb348dec13150b4ca8af72e5d3c9223eb75c9`

No registered hashes in old outputs are rewritten. Source-bound execution
commands must reject old registrations after this maintenance changes source
bytes; do not bypass those checks to resume. Historical results remain readable
with the read-only validators, and exact old execution code is recoverable from
the pre-fix Git tag. A new experiment requires a new output and registration.

## Verification

- Full WSL Python suite: **2,722 passed, 142 skipped**, 382.23 seconds. Skipped
  tests are not counted as passing; the two sklearn-only tests were additionally
  executed in the existing Windows environment and both passed.
- CTest in `build/linux/sa-wall-clock-v1`: **14/14 passed**, 15.72 seconds.
  The older ordinary build lacks the newer deadline API, while the SA build
  initially lacked its GPBS executable. Only the independent `gpbs_official`
  target was built to complete the latter directory's test prerequisites;
  neither experiment native module was rebuilt or replaced.
- The frozen SA native SHA remains
  `5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d`.
- The isolated raw native test also passes when launched directly as a Python
  script, checking that the subprocess test identity works outside pytest.
- Repository hygiene passes with zero reported errors and **24/24** frozen
  evidence hashes verified. Retention coverage and `git diff --check` pass.
- No tracked changes under `src`, `include`, `third_party`, `CMakeLists.txt`
  or `artifacts` relative to the original backup.

Production Python modules increase from 581 to 582 because artifact validation
now has a shared module. After combining duplicates and adding validation, the
production code has a net reduction of three lines. Regression tests, retention
entries and this maintenance record add lines overall; this phase does not
claim a large reduction in repository size.
