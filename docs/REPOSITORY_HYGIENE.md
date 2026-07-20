# Repository hygiene

This maintenance pass freezes the research conclusions and changes no solver,
feature, label, threshold, or model behavior. Its purpose is to keep the full
formal evidence chain while removing only demonstrably reproducible clutter.

## Ownership boundary

- `experiments/`, `configs/`, and `scripts/` contain the active controller,
  data pipeline, and supported evaluation entry points.
- `research/` retains the implementations, configurations, commands, and
  documentation for all non-current studies recorded in the evidence ledger.
- `artifacts/` contains the frozen portable policy and compact evidence ledger.
- `third_party/` contains pinned, licensed MAPF-LNS2 and GPBS sources and is not
  rewritten by repository cleanup.
- The pre-official simplified solver and Stage 3-5 source history were removed
  from the working tree and remain recoverable from the
  `pre-repository-restructure-2026-07-20` Git tag.
- Formal raw collections, datasets, frozen models, `build/venv-graph`, and the
  Windows/Linux build trees remain local under ignored `build/`.

Before refactoring, the active Python tree contained no duplicate files but did
contain 27 groups of structurally identical top-level helper implementations.
Semantics-free JSON, hashing, statistics, categorical-feature, and stable-ID
helpers now live in `experiments/_common.py`. Study-specific Pareto definitions,
bootstrap methods, labels, and acceptance gates remain in their original owners.

## Read-only audit

The audit includes controlled untracked files, so new source cannot evade the
check before it is staged:

```powershell
python scripts/audit_repository_hygiene.py --check
python scripts/audit_repository_hygiene.py `
  --emit-build-plan build/repository-hygiene-20260717
```

The first command checks file ownership, duplicate content and function bodies,
unused imports, orphan experiment modules, accidental large/generated files,
machine-specific paths in versioned manifests, and every locally available
formal evidence SHA.

The second command is also read-only. It emits:

- `repository_check.json`
- `pre_cleanup_inventory.json`
- `cleanup_plan.json`

The plan recursively discovers formal build dependencies from the evidence
ledger. It separates protected roots, directly removable roots, safe nested
paths, conditional nested paths, blocked paths, and conservatively retained
roots. Conditional paths record every evidence check and are not listed as
eligible when a size, schema, count, or migration flag differs. Protected paths
always win over temporary-name rules. Unknown build directories are retained
conservatively and reported rather than guessed to be disposable. The script
deliberately has no delete option.

The current conditional candidate is only the legacy full-trace directory:

```text
build/initlns-movingai-ood-collection-v1/episodes
```

Its parent metadata, manifests, reports, compact `delta-gzip-v2` collection,
downloaded MovingAI inputs, frozen policies, and formal evidence remain
protected. A passing plan is necessary but not sufficient for removal: the
720-episode equivalence check, a fresh quick run, timeout sensitivity, exact
path containment, inactive-process check, and explicit approval are still
required.

## Safe local cleanup

The 2026-07-17 pre-cleanup inventory found about 22.5 GiB in 161 top-level build
directories before adding its own protected log directory. Forty-seven roots
were protected by formal evidence or explicit environment policy. Eighty-eight
smoke, verification, preregistration,
superseded audit, and archived Stage 4-5 roots were eligible for deletion,
totalling 405,450,255 bytes. Twenty-seven unclassified roots were retained.
Post-cleanup verification removed all 88 roots and measured 405,298,599 bytes
released after accounting for the newly written audit records.

Recursive deletion is performed separately from the audit, only after resolving
every target below the repository's `build/` directory, rejecting reparse
points, checking for active collectors, and obtaining explicit approval.
Project-level Python caches may be removed; `build/venv-graph` is left intact.
Superseded run roots are exact configuration entries rather than inferred from
similar names. For old bottleneck quick output, only the regenerable `tracks/`
subdirectory is eligible; reports and run metadata remain.

## Acceptance

A cleanup is valid only when:

- all 24 formal result SHA256 values still match;
- strict result consolidation reproduces the frozen report and decisions;
- Python tests, Linux CTest, and official parity remain unchanged;
- no protected root appears in the deletion list;
- the post-cleanup report records every removed path and final disk usage.

## Transparent build compression

Completed JSON, JSONL, CSV, Markdown, and text collections can use NTFS
transparent compression without changing their paths or logical bytes. The
manager never compresses environments, models, maps, source, or binaries, and
it does not set directory inheritance:

```powershell
python scripts/manage_build_storage.py plan
python scripts/manage_build_storage.py compress
python scripts/manage_build_storage.py verify
```

The registered plan skips compression below 500 MiB projected savings. A
successful run records every file SHA256 before and after compression, and the
verification command rejects any changed or missing file. Decompression is
available through the explicit `decompress` command.

## 2026-07-20 space cleanup

The space-first cleanup completed under
`build/repository-hygiene-space-cleanup-20260720`. Before deletion, the audit
verified 24/24 formal evidence hashes and 45/45 conditional checks. The latter
included an exact 720/720 full-v1 versus delta-gzip-v2 trace comparison, a fresh
quick evaluation, counterfactual coverage, semantic equivalence, and the
600-second timeout-sensitivity cohort.

The exact-equivalence pass took 3,134.5 seconds and found zero scientific-field
mismatches. The fresh quick and timeout run took 14,322.3 seconds, completed all
96 primary episodes and 12/12 timeout-sensitivity episodes, and reported zero
invalid actions, fingerprint mismatches, or unexplained errors. These checks
also exposed and fixed a non-monotonic diagnostic clock and inconsistent native
module discovery; neither change alters official solver behavior or the frozen
policy.

Cleanup removed 17 exact temporary or superseded roots, three exact nested
paths, nine cache directories, and the conditional legacy directory
`build/initlns-movingai-ood-collection-v1/episodes`. The first PowerShell pass
stopped safely on a long nested path; the remaining exact targets were then
removed with Windows long-path support after repeating the same containment,
reparse-point, process, protection, and evidence checks.

The post-cleanup inventory records:

- build size before: 23,328,224,454 bytes;
- build size after: 7,026,838,533 bytes;
- released space: 16,301,385,921 bytes (about 15.18 GiB);
- remaining eligible roots, nested paths, and caches: zero.

The compact MovingAI collection, its 720 episodes, frozen controllers,
downloads, reports, manifests, environments, formal raw collections, and all 24
evidence sources remain protected. Strict result consolidation still passes;
Python reports 307 passed and 17 skipped tests, Linux CTest reports 10/10, and
official parity remains
`915ee104f0168c463f05925541fef1c22ec1eb37e9bf8df7ab09807753013ecf`.
