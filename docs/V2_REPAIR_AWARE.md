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

# Continue with size 12 only when its pilot gate passes.
python3 scripts/run_high_load_rescue_pipeline.py \
  --mode full \
  --neighborhood-sizes 4,8,12,16 \
  --output build/initlns-high-load-rescue-full-v1

# If size 12 fails, train the rescue model without that branch.
python3 scripts/run_high_load_rescue_pipeline.py \
  --mode full \
  --neighborhood-sizes 4,8,16 \
  --output build/initlns-high-load-rescue-full-no12-v1
```

The synthetic protocol uses 48x64 maps, 400/600 agents, disjoint map seeds and
dense random/paired-swap pressure to obtain genuinely high-load repair states
without reading MovingAI formal outcomes. The pilot requires 48 training and
12 locked-validation failure states. Size 12
continues to full collection only if it is Pareto-optimal in at least one state
and wins at least three states. Full collection targets 800/200 states, starts
with two paired trials per arm, and adds trials only for ambiguous top arms.
Failure of the size-12 gate removes only size 12; it does not block the 4/8/16
high-load rescue training. Full mode defaults to the larger
`configs/high_load_rescue_dataset_full.json` design (48 train maps, 16 locked
validation maps and 20 endpoint variants per map), so the 800/200 state target
is reachable; pilot mode keeps the smaller 12/6-map design.

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
