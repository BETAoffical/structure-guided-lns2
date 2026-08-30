# Rejected-method evidence ledger

This ledger preserves the scientific status of research branches removed from
the active source tree. The complete pre-pruning workspace is recoverable from
`backup/rejected-method-pruning-00-original` at commit
`36b1e9896fbd081c1fec0e47daf1404e385d1dad`.

The removal does not change the retained Official Adaptive, `v2-full`,
`mixed-full-v2`, `v3-s3`, or targeted Dual16 implementations.

## Removed in pruning stage 1

- Hierarchical C/H and compact-flow variants: the successive readiness,
  label-support, router, consensus and ambiguity gates ended in `NO_GO`,
  `NO_GO_LABEL_SUPPORT`, or post-hoc-control rejection. No model was
  promoted.
- Fresh-matched hierarchical supply variants: stopped at state-supply or label
  readiness and never became an online controller.
- Temporal-anchor incremental readiness: the frozen report recorded
  `offline_incremental_signal_failed_hard_stop`,
  `offline_passed=false`, no model export and no runtime integration.
- V2-first single-family/consensus rescue and SafeSlot pair-motif variants:
  offline or state-supply gates failed, later stages were stopped, and no
  runtime policy was promoted.
- Exact-v7 CausalClosure/RepairDependency prototypes: isolated implementations
  with no external caller, registered result, or promotion evidence.
- Seed25 same-set PP/GCBS execution chain: the mechanism screen reached a hard
  stop and did not establish selector or end-to-end TTF improvement. Its two
  result documents remain in `docs/`.
- The timed-GCBS extension used only by that rejected screen was removed.
  Bundled upstream GCBS source remains available, but is not an active
  controller or baseline.
- The orphan high-conflict fixed-collision configuration was removed because
  it had no caller and contained a machine-specific absolute path.

The local ignored build reports were checked before removal. They are
supporting workspace evidence, not the portable recovery mechanism; the Git
backup above is authoritative.

To recover an individual deleted file without restoring the complete branch:

```bash
git restore --source 36b1e98 -- path/to/file
```

## Retained evidence

- `docs/STRIDE_SEED25_SAME_SET_PP_GCBS_SCREEN_V1_RESULT.md`
- `docs/STRIDE_SEED25_SAME_SET_PP_GCBS_MULTISEED_V1_RESULT.md`

## Removed in pruning stage 2

- Cross-map StructShell quick screen, offline profile and profile
  falsification: the registered cross-map hypothesis was falsified and the
  branch was stopped before promotion.
- Warehouse CompactCut task generation and qualification: the registered gate
  failed and the branch was stopped before controller or TTF evaluation.
- CycleTransition: the label/readiness gate failed and no model or runtime
  controller was produced.
- PlatformEntry Frontier: stopped before training, runtime integration and TTF.
- Seed25 restored-state replay diagnostic: retained only as mechanism evidence;
  its follow-up same-set PP/GCBS screen also ended in a hard stop.

Their result/protocol documents and registration configurations remain in the
repository. Before pruning, eleven canonical local JSON reports were checked
against the SHA-256 values recorded in those documents and all matched. The
CompactCut document mentions an external tar archive that was not present in
this workspace; recovery therefore relies on the Git backup and the retained
canonical report/configuration evidence, not that missing archive.
