# Structure-Guided LNS2

This repository runs the official MAPF-LNS2 feasibility solver and two frozen
high-level neighborhood controllers. The active research result is deliberately
narrow:

> A controller using the current conflict state and a realized agent neighborhood
> generalizes on unseen maps from the same synthetic layout families. On standard
> MovingAI layouts it showed a positive signal, but missed the preregistered 5%
> conflict-AUC threshold.

Static map, OD, and density context has not shown reliable incremental value. RL is
not part of the active runtime.

## Active Layout

| Path | Purpose |
| --- | --- |
| `src/`, `include/` | C++ wrapper, observer, native feature engine, and bindings. |
| `experiments/` | Active collection, inference, trace, and evaluation runtime. |
| `generators/` | Synthetic maps and static OD task generation. |
| `scripts/` | Supported command-line entry points. |
| `configs/` | Active datasets, collection protocols, and maintenance settings. |
| `docs/` | Current operation guides and frozen research conclusions. |
| `tests/` | Runtime, data, evaluation, and maintenance regression tests. |
| `artifacts/` | Frozen v1/v2 controllers and compact evidence; hidden in Explorer. |
| `third_party/` | Pinned MAPF-LNS2 and GPBS sources; hidden in Explorer. |
| `build/` | Local datasets, traces, environments, and build products; ignored by Git and hidden. |

Historical experiments were removed from the active branch. They remain available
from Git tag `pre-minimal-runtime-2026-07-20`.

## Controllers

- `v1-full`: frozen portable pairwise GBDT using `realized_dynamic` features.
- `v2-full`: exactly equivalent action selection with accelerated feature and tree inference.
- `v2-stall-safe`: v2 plus the registered stall guard.
- Official baselines: `Adaptive`, `Target`, `Collision`, and `Random`.

The removed `v2-balanced`, `v2-cascade`, and proposal-pruner variants did not earn
promotion and are not supported by active CLIs.

## Build

The normal build includes the official solver, repair wrapper, GPBS runner, Python
environment, and native feature extension:

```bash
cmake -S . -B build/linux
cmake --build build/linux -j4
ctest --test-dir build/linux --output-on-failure
```

Build only the online feature module with:

```bash
cmake -S . -B build/native-features -DLNS2_FEATURES_ONLY=ON
cmake --build build/native-features -j4
```

The WSL policy-training environment is pinned by
`requirements-policy-training-wsl.lock`. Environment inspection is read-only:

```bash
python scripts/check_environment.py --profile runtime-wsl
```

## Common Commands

Generate or inspect data:

```bash
python scripts/generate_dataset.py --help
python scripts/inspect_dataset.py --help
```

Collect or analyze closed-loop runs:

```bash
python scripts/collect_closed_loop_confirmation.py --help
python scripts/analyze_closed_loop_confirmation.py --help
python scripts/run_lns2_tradeoff_evaluation.py --help
```

Verify the frozen evidence chain:

```bash
python scripts/consolidate_research_results.py \
  --config configs/result_consolidation.json --verify-build
```

Audit repository ownership without deleting anything:

```bash
python scripts/audit_repository_hygiene.py --check
```

## Results And Boundaries

The canonical Chinese report is
[`docs/INITLNS_RESEARCH_REPORT_ZH.md`](docs/INITLNS_RESEARCH_REPORT_ZH.md). The
MovingAI protocol and current operational interface are documented in
[`docs/MOVINGAI_OOD_CLOSED_LOOP.md`](docs/MOVINGAI_OOD_CLOSED_LOOP.md) and
[`docs/TRACE_AND_POLICY_API.md`](docs/TRACE_AND_POLICY_API.md).

The 24 frozen evidence entries preserve passed, failed, exploratory, and
insufficient-evidence outcomes. Links to removed historical implementation notes
point to the safety tag rather than to files in this branch.

## Upstream And License

`third_party/mapf_lns2` is pinned to official MAPF-LNS2 commit
`1369823985a15944f9a339226d521f61605a6d17`. Upstream license and source notices are
preserved in the third-party tree. The wrapper keeps official RNG behavior when no
custom action seed is supplied; parity is guarded by the registered SHA256.
