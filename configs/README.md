# Configuration index

The JSON files in this directory are versioned experiment registrations, not
duplicate map files. A newer suffix does not automatically make an older file
deletable: registered reports bind the exact configuration that produced them.

| Prefix | Role | Retention reason |
| --- | --- | --- |
| `balanced_wall_clock_*` | paired wall-clock cohorts and qualification inputs | reproduce current V2/mixed-V2 timing conclusions |
| `closed_loop_*` | frozen V2 collection and analysis inputs | rebuild or validate the promoted V2 bundle |
| `movingai_*` | pinned external benchmark members and OOD cohorts | reproduce benchmark downloads and external evaluation |
| `repair_*` | repair collection and transfer inputs | rebuild retained training data and validate resume integrity |
| `v3_s3_*` | current V3-S3 source dataset | run the retained V3-S3 pipeline |
| `stage1_*` | warehouse map and static OD example | reproduce the registered generator contract |
| `repository_hygiene.json` | repository and evidence audit policy | protect tracked evidence and reject repository debris |
| `retention_manifest.json` | per-file executable retention decisions | prevent unclassified production modules or tests |
| `stride_stage4r_pp_replay.json` | registered 16-state paired PP-seed replay | separate first-neighborhood quality from stochastic PP repair outcomes |
| `stride_stage4r_seed_diagnostic.json` | registered four-task, four-seed Stage 4R tail diagnostic | distinguish persistent selector weakness from solver-seed sensitivity without a formal speed claim |
| `result_consolidation.json` | frozen claim/source registry | verify report sources and claim boundaries |
| `build_storage_compaction.json` | local generated-output inventory policy | manage build storage without deleting evidence |
| `stride_*` | registered STRIDE-LNS stage inputs | reproduce labels, controlled training, Shadow, and TTF-first diagnostics |

MovingAI maps themselves are downloaded into ignored `build/` directories.
Generated warehouse maps also live under `build/`. The pinned upstream solver
fixtures under `third_party/mapf_lns2` remain unchanged.

New controller-facing commands accept `official_adaptive`, `v2-full`,
`mixed-full-v2`, and `v3-s3`. Historical policy labels that still appear in a
registered configuration are inputs to a retained artifact schema; they are
not additional active controller IDs.
