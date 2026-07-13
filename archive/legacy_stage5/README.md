# Legacy simplified-solver experiments

This directory preserves the pre-official-kernel implementation checkpointed at Git commit
`c861ca0347358f83cbe245bf9f77509cacb46ca9`.

It contains the independent simplified C++ solver and the Stage 3-5 retrieval, supervised ranking,
candidate, and rollout experiments that were built around its trace schema. These results remain useful
as negative findings and implementation history, but they are not active MAPF-LNS2 baselines and are
excluded from the root CMake build, package exports, and default tests.

The ignored `build/` directory was deliberately left untouched because it contains raw experiment
artifacts. The authoritative remote backup branch is
`codex/pre-official-lns2-backup-2026-07-13`.
