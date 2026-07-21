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
- `v2-repair-aware`: experimental v2 rescue controller. It preserves the first
  v2 decision on every repair-relevant state, then reuses the unchanged-state
  candidate pool and consults policy-train-only repairability/cost models after
  PP makes no structural progress.
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
ctest --test-dir build/native-features --output-on-failure
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

Train and diagnose the experimental repair-aware controller without replacing
the frozen v2 ranker:

```bash
python3 scripts/run_high_load_rescue_pipeline.py \
  --mode pilot \
  --output build/initlns-high-load-rescue-pilot-v1
python scripts/run_lns2_tradeoff_evaluation.py \
  --mode quick \
  --evaluation-tracks wall-clock \
  --controllers official_adaptive,v2-full,v2-stall-safe,v2-repair-aware \
  --repair-aware-bundle build/initlns-high-load-rescue-full-v1/controller \
  --skip-wall-clock-sensitivity \
  --output build/initlns-v2-repair-aware-quick-v1
```

The completed 60-state high-load pilot kept size 12 as exploratory evidence but
did not promote it for runtime use. Its stronger OOF plus diagnostic result did
not pass, so the active next step is an offline 4/8/16 rescue-order audit rather
than the 800/200 full collection:

```bash
python scripts/audit_rescue_policies.py \
  --source build/initlns-high-load-rescue-pilot-dense-v2 \
  --output build/initlns-rescue-policy-audit-v1
```

This command reuses paired pilot outcomes and never starts the solver. Because
the v1 trial schema did not store after-state fingerprints, promotion must pass
under both documented state-change bounds. The exposed 12-state validation
split is diagnostic only and cannot be reused as a new locked validation set.

Confirm the frozen `4>8>Adaptive` rescue order on fresh synthetic maps with
exact before/after repair fingerprints:

```bash
python3 scripts/run_rescue_lite_confirmation.py \
  --output build/initlns-rescue-lite-confirmation-v1 \
  --workers 4
```

The confirmation targets 30 balanced 400/600-agent states and four paired PP
seeds. If the two pre-registered task waves cannot supply every layout/agent
cell, it reports `insufficient_confirmation_states` and stops before branch
trials. It does not register a runtime controller or start quick/formal/v3 work.

When ordinary random tasks are too easy to supply those states, qualify stress
recipes on separate maps before opening a new locked confirmation set:

```bash
python3 scripts/qualify_rescue_confirmation_data.py \
  --output build/initlns-rescue-confirmation-qualification-v2 --workers 4

python3 scripts/run_locked_rescue_confirmation.py \
  --output build/initlns-rescue-lite-locked-confirmation-v1 --workers 4
```

The first command may inspect only source no-progress yield and freezes one task
recipe per layout/agent cell. The second command pins that report by SHA256 and
uses disjoint maps. A source-coverage shortfall stops before candidate replay or
paired PP trials; quotas must not be relaxed after seeing the locked set.

If a locked set stops for a small coverage shortfall, a balanced same-state
diagnostic may be run without relaxing the locked confirmation gate:

```bash
python3 scripts/run_rescue_lite_balanced_diagnostic.py \
  --source build/initlns-rescue-lite-locked-confirmation-v1 \
  --output build/initlns-rescue-lite-balanced-diagnostic-v1 \
  --workers 4
```

This diagnostic uses four states from each layout/agent cell and four paired PP
seeds. It is explicitly not promotion eligible, does not run complete episodes,
and cannot change the default controller. Its repair-only result may decide
whether rescue research is worth continuing, but cannot substitute for a new
independent locked confirmation or quick evaluation.

The pre-registered independent v2 confirmation uses eight new maps/tasks per
cell while retaining the frozen recipes and five-state gate:

```bash
python3 scripts/run_locked_rescue_confirmation.py \
  --output build/initlns-rescue-lite-locked-confirmation-v2 \
  --dataset-config configs/rescue_lite_locked_confirmation_dataset_v2.json \
  --expected-tasks-per-cell 8 \
  --reference-dataset build/initlns-rescue-lite-locked-confirmation-v1/dataset \
  --workers 4
```

Its master seed and task capacity are committed before execution. It must stop
again if any cell cannot provide five valid states; no task may be appended
after source outcomes are observed.

The completed v2 run passed coverage and fingerprint checks but returned
`inconclusive_collect_more`. The frozen `4>8>Adaptive` order improved aggregate
escape and efficiency over Adaptive, yet only three of six cells were
efficiency-noninferior and eight alternative fixed orders dominated it on the
aggregate gate. No rescue controller was promoted and `v2-full` remains the
default.

Audit a deliberately shallow state-conditioned selector across both completed
confirmation sets from the Windows training profile (no solver execution):

```bash
python scripts/audit_state_conditioned_rescue.py \
  --output build/initlns-state-conditioned-rescue-audit-v1
```

The audit uses leave-one-confirmation-set-out transfer plus map-group OOF. It is
design-only and cannot promote a controller; WSL runtime does not need the
scikit-learn training dependency.

The high-load auxiliary trainer uses synthetic 400/600-agent `policy_train` for
fitting and four-fold map-group calibration. MovingAI OOD/formal results are
never training inputs. The frozen v2 main ranker remains unchanged. See
[`docs/V2_REPAIR_AWARE.md`](docs/V2_REPAIR_AWARE.md).

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
