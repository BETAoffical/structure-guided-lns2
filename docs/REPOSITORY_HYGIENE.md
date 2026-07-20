# Repository hygiene

This maintenance pass freezes the research conclusions and changes no solver,
feature, label, threshold, or model behavior. Its purpose is to keep the full
formal evidence chain while removing only demonstrably reproducible clutter.

## Ownership boundary

- `experiments/`, `configs/`, `scripts/`, `tests/`, and `docs/` retain all 24
  formal studies recorded in `configs/result_consolidation.json`.
- `artifacts/` contains the frozen portable policy and compact evidence ledger.
- `third_party/` contains pinned, licensed MAPF-LNS2 and GPBS sources and is not
  rewritten by repository cleanup.
- `archive/legacy_stage5/` retains the pre-official simplified solver and its
  negative Stage 3-5 source history. It is excluded from active builds and tests.
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
