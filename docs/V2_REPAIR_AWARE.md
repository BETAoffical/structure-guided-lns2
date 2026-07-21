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

### Balanced diagnostic after a locked coverage shortfall

The locked v1 run prepared 29 of the required 30 states: five cells reached
five states and `regular_beltway/400` reached four. It therefore stopped before
branch trials, as required. To avoid discarding the usable evidence while also
preserving the locked gate, run the explicitly non-promotional diagnostic:

```bash
python3 scripts/run_rescue_lite_balanced_diagnostic.py \
  --source build/initlns-rescue-lite-locked-confirmation-v1 \
  --output build/initlns-rescue-lite-balanced-diagnostic-v1 \
  --workers 4
```

It selects exactly four states from each of the six layout/agent cells, caps
selection at two states per task, and runs four paired PP seeds. The completed
24-state diagnostic had zero replay errors. Relative to immediate Adaptive,
the frozen `4>8>Adaptive` rule increased state escape from 68.75% to 97.92%,
reduced final hard failures from 31.25% to 2.08%, and increased conflict
reduction per PP second from 53.03 to 95.65. All six cells were non-inferior in
efficiency; the worst cell ratio was 1.58.

The learned rescue reference reached 94.79% escape and 111.60 conflict
reduction per PP second overall, but it did not dominate the fixed rule and was
less stable across cells. These numbers cover only the rescue repair calls.
They exclude normal v2 candidate/feature overhead and complete-episode effects.
The emitted decision is therefore `diagnostic_supports_fixed_rescue`, with
`promotion_eligible=false`; `v2-full` remains the default controller.

The next independent attempt is pre-registered as
`configs/rescue_lite_locked_confirmation_dataset_v2.json`. It keeps the six
frozen recipes and the five-state-per-cell gate, changes only to a new master
seed, and provisions eight disjoint tasks per cell instead of four:

```bash
python3 scripts/run_locked_rescue_confirmation.py \
  --output build/initlns-rescue-lite-locked-confirmation-v2 \
  --dataset-config configs/rescue_lite_locked_confirmation_dataset_v2.json \
  --expected-tasks-per-cell 8 \
  --reference-dataset build/initlns-rescue-lite-locked-confirmation-v1/dataset \
  --workers 4
```

Map-content isolation covers all previous training, qualification and
confirmation datasets. The task count is part of the run fingerprint. This v2
run remains one-shot: a coverage failure cannot be repaired by adding maps after
results are visible.

The pre-registered v2 run completed all 48 source episodes with no error or
timeout. It found 335 no-progress states, prepared 120/120 valid replay states,
and selected five states from each cell across at least three tasks/maps per
cell. All 568 candidate branches had complete four-seed coverage and exact
before/after repair fingerprints.

Its locked decision was `inconclusive_collect_more`, not confirmation:

- Adaptive escaped 74.17% of states, hard-failed 25.83%, and reduced 14.78
  conflicts per PP second.
- Frozen `4>8>Adaptive` escaped 96.67%, hard-failed 3.33%, and reduced 17.44
  conflicts per PP second (1.18x Adaptive).
- The frozen order was efficiency-noninferior in only three of six cells; its
  worst cell ratio was 0.760, below the 0.90 gate.
- Eight alternative fixed orders dominated the frozen order on aggregate
  efficiency while matching or improving escape and hard-failure rates.
- The learned reference reached 38.77 conflicts per PP second (2.62x Adaptive)
  but escaped 93.33% and hard-failed 6.67%, so it did not dominate the safer
  frozen order.

The prior 24-state diagnostic and this independent set disagree on whether size
4 or size 8 should be attempted first. A post-outcome cross-set check also found
that `8>4>Adaptive` has a worst cell efficiency ratio of 0.817 on the old set
and 0.704 on v2. Therefore no single global fixed order is stable enough for
runtime integration. Do not run quick/formal for rescue-lite yet; the next
design step should audit a state-conditioned repair-success/cost selector across
both datasets, with any resulting policy requiring another untouched
confirmation set.

## State-conditioned rescue audit

The next design-only audit reuses the exact branch outcomes from the balanced
v1 diagnostic and independent locked v2 confirmation. It does not execute the
solver:

```bash
python scripts/audit_state_conditioned_rescue.py \
  --output build/initlns-state-conditioned-rescue-audit-v1
```

The fixed model is a depth-2 decision tree with at least four states per leaf.
Inputs are agent count, decision stage, current conflicts/SOC, cached v2 scores,
candidate overlap, failed-base size, learned recommendation size, and layout
one-hot values. Dataset and map identity are forbidden inputs. The primary test
trains on one confirmation set and evaluates the other in both directions;
four-fold map-group OOF is a secondary check.

The audit result was `state_conditioned_rescue_not_stable`. Cross-confirmation
transfer passed both aggregate safety folds and reached 1.72x Adaptive
efficiency, but only 9/12 dataset/cell units were non-inferior and the worst was
0.519x. Pooled map OOF passed 3/4 aggregate folds, reached 1.15x efficiency,
and was non-inferior in only 7/12 units. The existing learned reference reached
1.96x overall but only 8/12 non-inferior units. In contrast, the safe per-state
oracle upper bound reached 3.34x and 12/12 units, showing useful signal exists
but the current small confirmation cohort cannot learn it reliably.

No shallow selector is integrated. The evidence supports moving to a separately
versioned v3 repair-success/real-time-cost design with broader high-load
training coverage, rather than collecting another confirmation set for a fixed
4/8 order.

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
