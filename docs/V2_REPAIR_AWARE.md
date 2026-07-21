# v2 high-load repair-aware controller

`v2-repair-aware` is an experimental runtime layer around the frozen `v2-full`
ranker. It does not retrain the main pairwise model and does not change the
normal 4/8/16 candidate pool.

On the first visit to a repair-relevant state, the selected candidate is exactly
the v2-full winner. A repair-relevant fingerprint excludes iteration and
low-level search counters, so a PP attempt that returns the same paths,
conflict graph, and SOC is recognized as unchanged even when diagnostics have
advanced.

After the first no-progress result, the next decision reuses the candidate
pool, feature rows, and v2 scores. If locked high-load validation promotes size
12, size-12 candidates are generated only at this point and appended to the
cache. The same candidate may receive at most two paired PP attempts. The
map-group OOF-selected rescue limit is stored in the auxiliary bundle; after it
is reached, official Adaptive remains active until the repair state changes.
An unchanged Adaptive failure therefore cannot return control to the original
v2 winner.

The high-load auxiliary bundle contains four portable
HistGradientBoosting models:

- probability that a repair reduces conflicts;
- expected conflict reduction;
- expected `log1p(repair_seconds)`;
- probability of a hard PP failure.

Candidate order maximizes predicted conflict reduction per real repair second,
including progress and hard-failure risk. Adaptive is represented by a
state-only pseudo candidate and may be selected immediately when its predicted
efficiency is better.

Training uses only synthetic 400/600-agent `policy_train` maps with four
map-group OOF folds. Disjoint `policy_validation` maps are read once for locked
validation. MovingAI OOD and formal labels are never training inputs.

## High-load pilot and training

```bash
python3 scripts/run_high_load_rescue_pipeline.py \
  --mode pilot \
  --output build/initlns-high-load-rescue-pilot-v1

# Audit fixed 4/8/16 rescue orders from the completed pilot without running PP.
python scripts/audit_rescue_policies.py \
  --source build/initlns-high-load-rescue-pilot-dense-v2 \
  --output build/initlns-rescue-policy-audit-v1
```

The synthetic protocol uses 48x64 maps, 400/600 agents, disjoint map seeds and
dense random/paired-swap pressure to obtain genuinely high-load repair states
without reading MovingAI formal outcomes. The pilot requires 48 training and
12 locked-validation failure states. Size 12
passed its exploratory Pareto/winner gate, but its efficiency was about 52.6%
below the best 4/8/16 alternative per state and it failed the stronger OOF plus
validation promotion gate. It remains in the pilot evidence but is excluded
from runtime generation. The 800/200 collection is paused.

The offline audit compares all 16 fixed non-repeating permutations of 4/8/16
followed by Adaptive. It uses only the 48 training states for map-group OOF
selection. The previously exposed 12 validation states are downgraded to a
diagnostic split. The pilot schema lacks after-state fingerprints, so every
policy must pass both a `replan-success-stop` and a stricter
`conflict-reduction-stop` interpretation before it can become a
`rescue_lite_candidate`. Passing still requires a fresh independent validation
set before any runtime controller is implemented.

## Independent rescue-lite confirmation

The fixed `4>8>Adaptive` order selected by the offline audit is confirmed on a
new `policy_confirmation` dataset before any runtime integration:

```bash
python3 scripts/run_rescue_lite_confirmation.py \
  --output build/initlns-rescue-lite-confirmation-v1 \
  --workers 4
```

The command targets 30 states balanced across the three layouts and 400/600
agents, uses four paired PP seeds, records exact full and repair-structure
fingerprints, and compares the frozen rule with Adaptive, all other fixed orders,
and the existing learned rescue selector. If either pre-registered task wave
cannot supply a cell quota, it emits `insufficient_confirmation_states` and
stops before branch trials. The previously exposed pilot validation split is not
reused. Completion produces evidence only; `v2-full` remains the default.

If this ordinary-task confirmation cannot fill all six layout/agent cells, use a
separate data-qualification stage rather than repeatedly modifying a viewed
confirmation set:

```bash
python3 scripts/qualify_rescue_confirmation_data.py \
  --output build/initlns-rescue-confirmation-qualification-v2 --workers 4

python3 scripts/run_locked_rescue_confirmation.py \
  --output build/initlns-rescue-lite-locked-confirmation-v1 --workers 4
```

Qualification requires a recipe to provide at least five capped no-progress
states across three distinct tasks and maps in every layout/400-or-600-agent
cell. The resulting six recipes and report hash are frozen before the locked
maps are generated. Locked source coverage is checked before expensive candidate
replay; fewer than five states in any cell is an inconclusive safe stop, not
permission to lower the quota or add a post-hoc map. Neither command registers a
runtime controller or starts quick, formal, or v3 work.

Complete-episode evaluation remains separate:

```bash
python3 scripts/run_lns2_tradeoff_evaluation.py \
  --mode quick \
  --evaluation-tracks wall-clock \
  --controllers official_adaptive,v2-full,v2-stall-safe,v2-repair-aware \
  --repair-aware-config configs/v2_repair_aware_v1.json \
  --repair-aware-bundle build/initlns-high-load-rescue-full-v1/controller \
  --controller-runtime optimized \
  --verification-profile deployment \
  --skip-wall-clock-sensitivity \
  --output build/initlns-v2-high-load-rescue-quick-v1
```

The report directory adds `repair_aware_usage.csv`,
`repair_aware_promotion.json`, and `repair_aware_report.md`. No controller is
promoted automatically; `v2-full` remains the default.

The historical 80/100-agent auxiliary bundle can still be reproduced with
`scripts/train_repair_aware_controller.py`; it is not the deployment candidate.
Retraining the frozen main ranker would create a separate `v3` controller and is
deferred until the high-load rescue evaluation is complete.
