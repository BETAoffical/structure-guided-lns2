# Selector architecture cleanup report

## Scope

The cleanup baseline is `backup/selector-cleanup-00-original` (`e75765c`).
The final tree retains four executable controller identities:

- `official_adaptive`
- `v2-full`
- `mixed-full-v2`
- `v3-s3`

Retired value/receding-Q, old V3/H3, stall-safe/shadow/oracle, rescue, and
repair-aware execution chains were removed after their evidence was frozen.
They remain recoverable from the remote cleanup checkpoints.

## Size comparison

Counts use tracked files and physical Python lines. The final column is the
cleanup report tree; generated `build/` content is excluded.

| Area | Original | Final | Change |
| --- | ---: | ---: | ---: |
| tracked files | 578 | 461 | -117 (-20.2%) |
| production Python files | 181 | 99 | -82 (-45.3%) |
| production Python lines | 87,059 | 43,236 | -43,823 (-50.3%) |
| `experiments/` Python files | 91 | 34 | -57 (-62.6%) |
| `experiments/` Python lines | 70,705 | 31,195 | -39,510 (-55.9%) |
| `scripts/` Python files | 81 | 35 | -46 (-56.8%) |
| `scripts/` Python lines | 12,649 | 7,718 | -4,931 (-39.0%) |
| `tests/` Python files | 87 | 42 | -45 (-51.7%) |
| `tests/` Python lines | 25,346 | 14,346 | -11,000 (-43.4%) |
| configuration JSON files | 67 | 52 | -15 (-22.4%) |

The new `lns2_selector/` package contains 21 focused modules (830 lines) for
the solver boundary, shared runtime contracts, four controller adapters,
training utilities, evaluation boundaries, and read-only compatibility.

## Retention and compatibility

`configs/retention_manifest.json` explicitly classifies every production
Python module and pytest file as `active`, `shared`, `reproducibility`, or
`compatibility`. A regression test rejects missing, duplicate, non-executable,
or stale entries.

Historical controller diagnostics are validated only by
`lns2_selector/compatibility/controller_diagnostics.py`; retired controllers
cannot be loaded by the active selector API. The compatibility extraction also
fixed a stale stall-shadow schema reference that could previously raise a
`NameError` while reading a historical trace. Current learned-policy traces now
always validate their model/native route totals.

The remaining configuration files are distinct versioned registrations, not
duplicate map files. Their groups and retention reasons are documented in
`configs/README.md`. Downloaded MovingAI and generated warehouse maps remain in
ignored `build/` directories.

## Behavior boundary

There are no changes relative to the cleanup baseline under `CMakeLists.txt`,
`include/`, `src/`, or `third_party/mapf_lns2/`. The cleanup therefore does not
change native PP, low-level search, RNG, or neighborhood generation. It changes
Python organization, active controller identity validation, trace integrity,
and read-only compatibility only.

## Validation

Validation was run in Ubuntu-22.04 WSL against the mounted current working
tree, not the separate modified `/home/beta/LNS2-RL` checkout:

- native configure and build: passed;
- CTest: 11/11 passed, including official parity run and hash tests;
- full pytest: 308 passed, 34 skipped;
- repository hygiene: 0 errors, 24/24 evidence entries verified;
- runtime environment profile: all required checks passed;
- retention manifest: complete, with no unclassified executable files.

No formal experiment, model training, or baseline rerun was performed because
the official native algorithm and active selector behavior were not changed.

## Recovery checkpoints

The cleanup checkpoints are:

1. `backup/selector-cleanup-00-original`
2. `backup/selector-cleanup-01-evidence-frozen`
3. `backup/selector-cleanup-02-before-structure`
4. `backup/selector-cleanup-03-before-code-pruning`
5. `backup/selector-cleanup-04-before-test-pruning`
6. `backup/selector-cleanup-05-before-cli-config`

To inspect a checkpoint without changing the active branch:

```bash
git switch --detach backup/selector-cleanup-00-original
```
