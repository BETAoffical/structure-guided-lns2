# Preserved result summaries

These JSON files were copied from the ignored `build/` tree before the official MAPF-LNS2 migration.
The original directories, traces, models, and generated datasets remain in place and were not deleted.

| File | Experiment |
| --- | --- |
| `stage4_evaluation_summary.json` | Offline retrieval validation |
| `stage5_v1_selected_config.json` | Validation-selected v1 guidance threshold |
| `stage5_v1_test_summary.json` | Role-template guidance paired test |
| `stage5_v2_test_summary.json` | Fixed candidate kNN test |
| `stage5_v2_dedup20_test_summary.json` | Lower-dimensional candidate features |
| `stage5_v2_2_dedup20_test_summary.json` | Multi-order candidate labels |
| `stage5_v3_test_summary.json` | Supervised candidate ranker |
| `stage5_v4_rollout_smoke_summary.json` | Closed-loop rollout-label smoke test |

The committed experiment documentation records that v1-v3 did not improve aggregate simplified-solver
performance. These files are evidence for that negative result, not results for the official MAPF-LNS2
kernel introduced after the checkpoint.
