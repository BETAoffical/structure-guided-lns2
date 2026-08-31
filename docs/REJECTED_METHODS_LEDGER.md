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

## Removed in pruning stage 3

The following diagnostic controller IDs and their executable training,
evaluation, CLI and implementation-detail test chains were removed:

- `stride-control-v1` and `stride-quality-v1`: the registered Quick and
  multi-seed evaluations were slower than frozen `v2-full`, so neither was
  promoted.
- `stride-augcontrol-v1`: its held-out offline regret was worse than V2 and it
  did not qualify for online replacement.
- `stride-guardrank-v1`: the OOF improvement missed its preregistered gate and
  the branch stopped before promotion.
- `stride-maprank-v1`: the complete method improved over the original pool,
  but its ranker was slower than the V2 ranker on the same augmented pool.
  Only the rejected ranker/override chain was removed; the independently useful
  topology and augmented-pool infrastructure remains shared code.

The action-preserving diagnostic-shadow runner was removed with these
controllers. It never changed the executed action, so removing it does not
change the behavior of `official_adaptive`, `v2-full`, `mixed-full-v2`, or
`v3-s3`. Public controller loading now accepts only those four IDs. Historical
controller diagnostics remain readable through the compatibility layer.

Seven controller-independent dataset/statistics helpers and the historical
MapBase audit schema were moved into neutral evaluation/compatibility modules
before their old host files were deleted. This preserves current StructPool,
HybridStructPool, Boundary and repairability consumers without retaining the
rejected controller implementations.

The preregistration configurations and the following conclusion documents are
retained as evidence:

- `docs/STRIDE_STAGE4_PROTOCOL.md`
- `docs/STRIDE_STAGE4_RESULT.md`
- `docs/STRIDE_STAGE4R_PROTOCOL.md`
- `docs/STRIDE_STAGE4R_SEED_DIAGNOSTIC.md`
- `docs/STRIDE_STAGE4R_PP_REPLAY.md`
- `docs/STRIDE_STAGE4R_HIGH_LOAD_PROTOCOL.md`
- `docs/STRIDE_ROBUSTSTEP_PROTOCOL.md`

The exact pre-stage-3 source tree is protected remotely at
`backup/rejected-method-pruning-02-before-diagnostic-controller-pruning`,
commit `07f18c0448cc2c13ebf6217e609b936495fd6171`. Restore a removed file with:

```bash
git restore --source 07f18c0 -- path/to/file
```
