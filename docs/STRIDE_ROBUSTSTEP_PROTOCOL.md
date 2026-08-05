# STRIDE robust-step successor protocol

## Purpose

`stride-robuststep-v1` is the successor design after the Stage 4R diagnostics.
It does not replace `v2-full` or `stride-quality-v1`. Its objective is to
select a better current-step neighborhood while retaining PP-seed uncertainty,
rather than predicting Cost-to-Go, future repair rounds, or repair runtime.

The label schema is `lns2.stride.robust_step_label.v1`. Runtime is forbidden
from the label. The frozen candidate pool, PP repair, and 124 realized features
remain unchanged during the first design comparison.

## Consumed label-design cohort

The existing Stage 3 cohort is consumed as design evidence:

- 600 states and 36 maps;
- 10,713 candidate neighborhoods;
- exactly eight paired PP outcomes per candidate;
- 85,704 repair outcomes in total;
- no formal MovingAI/OOD map overlap.

For candidate `c` and paired PP seed `k`, define the immediate seed score as

```text
q(c,k) = normalized_conflict_reduction(c,k)
         - 0.02 * within_state_seed_structure_percentile(c,k)
```

For each candidate pair, the design uses the eight paired differences
`q(left,k)-q(right,k)`. A candidate is labelled as robustly better only when
its paired win fraction reaches the registered threshold, its mean difference
has the same sign, and its no-progress probability is within the registered
tolerance. Other pairs are uncertain and do not become hard ranking labels.

Four frozen variants are compared. Trial indices 0--3 and 4--7 form
independent design halves. A variant must achieve at least 0.80 half-to-half
pairwise consistency, 0.70 mean robust-good-set Jaccard, 0.15 full pair
coverage, and 0.80 state coverage with at least five robust pairs. Among
passing variants, the one with greatest pair coverage is selected, followed by
pairwise consistency and good-set Jaccard. This design cohort cannot confirm
or promote the selected label; fresh map/state confirmation is mandatory.

## Action uncertainty at runtime

The future runtime controller will compare its selected action with the frozen
V2 action. It may override V2 only when the calibrated robust advantage exceeds
the confirmation threshold. Otherwise it abstains and executes the V2 action.
This is a proactive uncertainty gate, not a failure-triggered rescue policy.

## Outcome-blind map/load preflight

The first preflight registers MovingAI scenarios 6 and 12 over maze, room,
warehouse, street, and game maps. Map/task selection may read topology,
agent count, initial conflicts, initial PP wall time, and initial path
statistics. It may not read either controller's actions, repair outcomes, or
relative TTF.

`room-64-64-8` is already consumed development evidence. Every other map in
the formal MovingAI/OOD registry remains excluded. Agent counts are candidate
loads only: retained Pilot tasks will be chosen by conflict-load bands rather
than by agent count alone. In particular, the preflight checks 350 and 400
agents on `maze-128-128-2` because the completed 300-agent diagnostic produced
too few initial conflicts.

## Completed eight-seed design result

The registered 600-state design analysis completed with all integrity gates
passing, but no label variant passed the scientific gates. The least strict
variant, `w625-p125`, retained 0.7823 of all candidate pairs but reached only
0.7275 half-to-half pairwise consistency and 0.6397 mean robust-good-set
Jaccard, below the registered 0.80 and 0.70 thresholds. Its robust-good set was
unique in only 50.0 percent of states. Increasing the paired-win threshold did
not improve stability: it reduced pair coverage and increased action
uncertainty.

The deterministic report is
`build/stride-robuststep-design-v1/robuststep_design_report.json`, SHA-256
`02fd839c315ce2cc7f41e9d84dc965984ea773a00e4ca8b2a2bfcefcd3649668`.
Therefore no robust-step label is frozen and no training is allowed from this
result.

Before changing the pair contract, a consumed seed-depth diagnostic uses the
96 earlier V2 design and confirmation states that already have 16 paired PP
outcomes per candidate. Indices 0--7 and 8--15 form independent halves. The
same four label variants and gates are retained, and a variant must pass both
48-state cohorts as well as their aggregate. Passing would show only that
greater PP-seed depth makes the proposed contract plausible; a new result-blind
confirmation cohort would still be required.

The 96-state seed-depth diagnostic completed with 1,700 candidates and 27,200
outcomes. No variant passed. For `w625-p125`, eight-versus-eight pairwise
consistency increased to 0.7789 and good-set Jaccard to 0.7123, but consistency
remained below 0.80; the confirmation subgroup also remained below the Jaccard
gate at 0.6946. This shows that extra seeds reduce noise but do not by
themselves make the current robust-pair contract reliable. Report SHA-256:
`3b438754f65555237ea831e85f7bca1f0aea626d03c08b3b6fe2d462a40d765f`.

The map/load preflight is separately frozen before qualification. It contains
34 tasks (eight maps, two scenarios, and registered load candidates) and four
solver seeds, for 136 reset-only jobs. For each map, tasks with mean initial
conflicts at least 10 are eligible; without controller outcomes, the selector
chooses at most two distinct tasks closest to mean conflict targets 50 and 200.
If no task reaches 10, it reports the highest-conflict task as an underloaded
fallback and requires a higher-load preflight rather than silently accepting
it. Ties prefer lower agent count, lower scenario index, then task ID.

### Completed map/load preflight result

All 136 registered reset jobs completed with zero errors, timeouts, incomplete
states, or semantic inconsistencies. The analyzer read no controller action or
repair outcome, and the only formal-OOD overlap was the preregistered consumed
development exception `room-64-64-8`. The integrity report therefore passed.

The scientific load result did not permit immediate Pilot construction. Only
44/136 resets had any initial conflict, and six of eight maps were underloaded:
`Berlin_1_256`, `Boston_0_256`, `Paris_1_256`, `brc202d`, `ost003d`, and
`warehouse-20-40-10-2-1`. Most street/game tasks had exactly zero conflicts;
even the best 600-agent warehouse task averaged only 0.5.

The two useful maps reached the registered moderate/high targets as follows:

- `maze-128-128-2`, random-6: 350 agents averaged 49.75 conflicts (20--118),
  while 400 averaged 70.75 (24--182);
- `room-64-64-8`, random-6: 400 agents averaged 52.25 conflicts (36--84),
  while 500 averaged 170.0 (128--219).

Room random-12 at 600 agents averaged 515 conflicts and is a useful extreme
stress case, but it is not selected as a target-200 training task. The next
safe action is a preregistered load/scenario extension for the six underloaded
maps; they must not enter training merely to increase map count.

Artifact SHA-256 values:

- qualification manifest:
  `7b5a186462360cd4a356a4e01d091a004fbc41721edbc104a4bddf31d7ca9885`;
- qualification report:
  `16c8f15c02478a3ea5bebeebc5f262a8534e4a9e909caff0410a2cdf943c9efd`;
- outcome-blind analysis report:
  `bff0452ee87994dd8bcdcbe486ad7d3ece2775a40c0640a717d86bae20b7cb80`.

### Registered maximum-prefix load extension

The six underloaded maps receive a final random-scenario load preflight at 750
and 1,000 agents. Both random scenario files contain exactly 1,000 available
start/goal pairs, so 1,000 is the largest reproducible prefix and no larger
agent count is valid for these files. Scenarios 6 and 12 and solver seeds 1--4
remain unchanged, producing 24 tasks and 96 reset-only jobs.

The same outcome-blind conflict targets and selection rule are retained. If a
map remains below mean initial conflict 10 at the 1,000-agent prefix, further
agent-count scaling is stopped for that map. Its next candidate must instead
use a separately preregistered congestion-oriented scenario construction; it
must not be admitted to training as a zero-conflict map.

The maximum-prefix extension completed all 96 jobs with zero errors. It did
not rescue any map: the best mean initial conflicts were 0 for all three
street maps, 0.75 for `brc202d`, 1.0 for `ost003d`, and 4.75 for the warehouse.
Only 13/96 resets had any conflict. Qualification manifest, qualification
report, and analysis SHA-256 values are respectively
`07d1c1c78c4a75f10c02ff17f65472965f5d1b1215c450d240baebc14b85560d`,
`6e1380c2cfb5ec6b6c70219289f966909a1343ecdcfc3a71b00498825e091c59`,
and `ec6a2b9e4271df373ebbe99b4c36e5bbb598b71da84d441460e3feeb6435e3c0`.

### Registered congestion-oriented scenario preflight

The next preflight keeps the six official MovingAI map files but replaces the
insufficient official random scenario prefixes with deterministic,
project-derived `opposite_exchange` OD tasks. Starts are sampled without
replacement from the largest four-connected component, sorted along the map's
longer spatial axis, and paired with the reversed ordering. This creates
bidirectional long-range flow while preserving unique starts, unique goals,
reachable pairs, and exact four-neighbor distances.

This is explicitly not an official MovingAI scenario and cannot be reported as
one. Two endpoint seeds (23 and 41), three candidate loads (250, 500, and 750),
and four solver seeds produce 36 tasks and 144 reset-only jobs. Controller
actions and repair outcomes remain forbidden. The same mean-conflict targets
50 and 200 select at most two tasks per map; loads below mean 10 are rejected.

This construction also completed without errors but did not rescue the six
maps. Only 13/144 resets had conflicts; best means were 0 for Berlin, Paris,
and `brc202d`, 0.25 for Boston, 1.0 for `ost003d`, and 4.75 for the warehouse.
The manifest, qualification report, and analysis hashes are
`644d396066137868ec2199ced10dea5bf0c79c7e40e5782d0f0fc927617f2755`,
`8572a96a85a38691c75920a607b0905cc7ce6bb2bd751028afbd830449da6036`,
and `459f52fddfdcabb92fd78ac09020747ee871fc3afd03029d0988a0f4846c9659`.
These open maps are therefore excluded from repair-label collection.

Previously consumed reset-only DAO qualification pools provide the replacement
map design evidence. Across 864 valid resets with zero errors, 15 compact or
ultra-compact maps have at least one task with mean conflicts at least 10:
`den009d`, `den020d`, `den101d`, `den201d`, `den202d`, `den203d`, `den308d`,
`den998d`, `hrt002d`, `den404d`, `den408d`, `lak101d`, `lak108d`, `lak110d`,
and `ost102d`. Examples near the target bands include `den009d` at 40.67/85,
`den101d` at 34.67/165.67, `den308d` at 52/81.33, `den408d` at 85.33/210.33,
and `lak101d` at 56/148.67. This evidence is consumed map/load design only;
fresh task seeds and map-disjoint validation remain mandatory.

## Registered distributional current-step score design

Map design is now sufficient to continue label work, but the failed hard
robust-pair contract is not reused. On the consumed 96-state, 16-seed cohort,
the next design compares a plain mean immediate score with four risk-aware
aggregates: mean minus 0.25 or 0.50 standard deviations, mean minus a 0.10
no-progress penalty, and the lower quartile. Runtime, future repair rounds, and
Cost-to-Go remain excluded.

Indices 0--7 and 8--15 form independent halves. Each aggregate must reach 0.80
pairwise consistency, 0.80 Top-3 overlap, and at most 0.10 symmetric
cross-half normalized selection regret overall; both 48-state cohorts must
also reach 0.75, 0.75, and 0.15. A risk-aware aggregate must improve regret
over the plain mean by at least 0.005. Otherwise no new label is frozen and
the next controller must retain V2 actions with an uncertainty/abstention
mechanism rather than claiming a better learned ranking.

### Completed distributional score result

The 96-state design completed with 1,700 candidates, 27,200 paired outcomes,
16 PP seeds per candidate, and zero integrity errors. The plain mean itself was
stable (0.8687 pairwise consistency, 0.8264 Top-3 overlap, and 0.0487 symmetric
normalized regret), but the registered `mean-np100` risk score was the only
risk-aware variant that also improved selection regret. It achieved 0.8580
pairwise consistency, 0.8090 Top-3 overlap, and 0.0359 regret, an absolute
regret reduction of 0.0128 from the mean. Both independent 48-state cohorts
passed: their regrets were 0.0285 and 0.0433.

Penalizing standard deviation did not help: weights 0.25 and 0.50 increased
regret to 0.0635 and 0.0750. The lower-quartile score reached only 0.7743
Top-3 overlap and failed. Therefore `mean-np100` is frozen for a fresh
confirmation, but it is not yet a training label or a runtime controller. The
deterministic report SHA-256 is
`b204e19cfb93fe71a9149382e3e5764514cf83c8f5289d808ef06c0422f0d815`.

## Registered fresh score confirmation

The confirmation uses six compact DAO maps that do not overlap the 14 maps in
the consumed score-design label cohort: `den009d`, `den101d`, `den308d`,
`den408d`, `lak101d`, and `ost102d`. These maps have previously contributed
reset-only load evidence and some belong to older runtime-development
registries, so this is explicitly not a historically untouched-map or formal
OOD claim. It is a fresh-task, fresh-state confirmation of score stability.
The formal MovingAI/OOD registry remains excluded.

Each map contributes deterministic project-derived `uniform_random` and
`opposite_exchange` tasks at new task seeds 53 and 71. Loads are 200, 160, 320,
240, 160, and 120 agents in the map order above. They were selected from
consumed reset-only evidence to cover moderate/high conflicts while avoiding
the extreme thousands-of-conflicts cells. The resulting 24 tasks run under
new solver seeds 401 and 503. Official Adaptive and frozen `v2-full` provide
source trajectories, limited to decisions 0--11.

Selection is result-blind and takes 24 states per source policy, at most one
state per source episode. Every selected state must replay deterministically,
retain the same candidate pool, and supply the frozen 124-dimensional features.
Each candidate receives exactly 16 paired PP outcomes, split 8+8. Runtime,
future repair rounds, and Cost-to-Go remain forbidden.

Only the already selected score is tested:

```text
mean normalized immediate quality
- 0.10 * no-progress probability
```

No coefficient or alternative score may be selected on this cohort. Overall,
it must reach at least 0.75 pairwise consistency and 0.75 Top-3 overlap, at
most 0.15 normalized regret, and improve regret over the plain mean by at
least 0.005. Both source-policy subgroups and both DAO compactness groups must
contain at least eight states and reach 0.70, 0.70, and 0.20 respectively.
Failure keeps `v2-full` as the action source and forbids robust-step training;
passing permits only the next small balanced Pilot, not a speed claim.

Nine eligible DAO maps are reserved for later training expansion and are not
used in this confirmation: `den020d`, `den201d`, `den202d`, `den203d`,
`den998d`, `hrt002d`, `den404d`, `lak108d`, and `lak110d`.

### Completed fresh score confirmation result

The registered collection completed without execution or integrity errors.
All 48 selected states passed three replay/candidate-signature repetitions.
The final cohort contains 854 candidates and exactly 16 paired PP outcomes per
candidate: 3,416 base rows plus 10,248 extension rows, or 13,664 outcomes in
total. Every candidate has trial indices 0--15, all candidates within a
state/index share the same PP seed, and every row retains the frozen
124-dimensional `lns2.realized_features.v2` representation. The six registered
maps are complete, old label-map and formal-OOD overlaps are zero, and every
other integrity gate passed.

The frozen `mean-np100` score did not confirm. Overall pairwise consistency
was 0.8383 and normalized regret was 0.0640, both within their gates, but
Top-3 overlap was 0.7083 versus the required 0.75. More importantly, its
regret was 0.00108 worse than the plain mean instead of improving by at least
0.005. The plain mean reached 0.8442 pairwise consistency, 0.7361 Top-3
overlap, and 0.0629 regret, so it also does not independently satisfy the
registered Top-3 requirement.

The compact-map subgroup passed with 0.8166 pairwise consistency, 0.7719
Top-3 overlap, and 0.0366 regret. The ultra-compact subgroup failed only its
Top-3 gate at 0.6667 despite 0.8514 pairwise consistency and 0.0820 regret.
The Adaptive-source subgroup similarly missed Top-3 at 0.6944, while the
V2-source subgroup passed its relaxed gates at 0.7222. This pattern says the
ranking can broadly separate candidates but cannot reproduce a stable best
three on the tighter maps, and the fixed no-progress penalty does not transfer
as a selection improvement.

The preregistered decision is therefore
`retain_v2_actions_and_do_not_train_robuststep`. No robust-step model, small
Pilot, TTF comparison, or formal Stage 5/OOD run is authorized from this
score. Any follow-up inspection of conflict band, map, or penalty sensitivity
is post-hoc diagnostic evidence only and may not retune and promote a score on
this confirmation cohort.

Artifact SHA-256 values:

- base repair trials:
  `d351ea3be30cdc4e49cfdfba980d1f099ee95bf49ba17e425b26665abfafb6a6`;
- extension repair trials:
  `cc5db24a8a0dd0fb29846df34968848eec40dec6640e4630b810f9c4fc03a54a`;
- confirmation report:
  `da1d67f04aee05136729c032ed068962ecf4175a593cb0ceee5266e342507c5b`.

### Post-hoc failure diagnostic

An explicitly post-hoc inspection used the same completed outcomes only to
explain the failed gate; it cannot tune or promote a replacement score. Across
all 16 seeds, adding the 0.10 no-progress penalty changed the selected action
in only 2/48 states (4.17 percent). Across the two independent eight-seed
halves it changed 5/96 selections (5.21 percent). It changed no compact-map or
low-conflict full-sample action; both full-sample changes occurred in the
ultra-compact/high-conflict subset.

On cross-half evaluation, the penalty reduced mean immediate quality by
0.00249 while changing mean no-progress probability by exactly zero. Thus the
failure is not evidence that 0.10 was merely the wrong coefficient. The binary
no-progress term is usually identical among leading candidates and provides
too little ranking resolution; in the few states where it changes an action,
the apparent risk advantage does not transfer across PP seeds. Further tuning
of this scalar penalty on the confirmation cohort is prohibited.

The next safe investigation must leave V2 active and be diagnostic-only. It
should measure frozen-V2 action headroom against the 16-seed immediate-quality
distribution and separate representation error from irreducible PP-seed
uncertainty. A new learned successor requires a separately registered design
and new confirmation data; the reserved nine-map training expansion is not
opened by this failed result.

### Completed frozen-V2 headroom diagnostic

The post-hoc headroom audit evaluated the frozen `v2-full` action against the
plain-mean, 16-seed immediate-quality Oracle on the same 48 fresh-confirmation
states. It is diagnostic evidence only: it does not train a model, open the
reserved maps, authorize a controller replacement, or make a TTF claim. All
48 states, 854 candidates, 13,664 outcomes, frozen candidate identities, and
124-dimensional feature rows passed the registered integrity checks.

The two independent eight-seed halves selected the exact same Oracle winner
in 34/48 states (70.83 percent). Even within these 34 stable-winner states,
frozen V2 selected the Oracle winner in only 35.29 percent of states and lay
in the Oracle Top-3 in 47.06 percent. Its mean normalized regret was 0.3789,
and 22/34 states (64.71 percent) exceeded the preregistered 0.10 meaningful-
headroom threshold. Therefore the result is not explained solely by PP-seed
winner instability: a repeatable immediate-quality gap remains after unstable
states are removed.

Across all 48 states, V2 exact-best accuracy was 33.33 percent, V2-in-Oracle-
Top-3 was 50.00 percent, mean normalized regret was 0.3469, and meaningful
headroom occurred in 62.50 percent. The medium-conflict subgroup showed the
largest gap: 17.65 percent exact-best, 0.4122 regret, and 82.35 percent
meaningful headroom. High-conflict regret was 0.3970; low-conflict regret was
lower at 0.2193. Source-policy results were effectively identical, so the
diagnosis is not confined to states collected from either Adaptive or V2
trajectories.

The registered diagnosis is
`frozen_v2_has_stable_model_or_representation_headroom`. This does not yet say
whether the old V2 ranker/label is mismatched or whether the existing 124
features are insufficient. The next safe step is a grouped, out-of-sample
feature-sufficiency probe: candidate rows from the same state must never cross
train/test boundaries, map-held-out results must be reported separately, and
the frozen V2 remains the only runtime controller. The headroom report
SHA-256 is
`75b0748d68841bc508a628dfbf7124a540fd7b2d6dc5b78e9d7fc0eeaf5a3165`.

### Registered consumed feature-sufficiency probe

The next diagnostic reuses only the consumed 48-state confirmation cohort; it
does not open the nine reserved maps. Six leave-one-whole-map-out folds keep
every state and all of its candidates in one fold. Training pairs are included
only when both independent eight-seed halves give the same strict direction,
and every state receives equal total pair weight. Labels remain the 16-seed
plain mean immediate-quality score; runtime, future repair rounds, and
Cost-to-Go remain excluded.

Three fixed, separately named ephemeral probes are evaluated without tuning:

- `stride-stepdiag-v1/exact-v2-86` retrains the exact 86-input frozen-V2
  representation;
- `stride-stepdiag-v1/full-124-delta` uses all 124 registered features as
  candidate-pair differences;
- `stride-stepdiag-v1/full-124-context` adds shared `state.*` and `context.*`
  values so candidate quality may depend on the current state.

All use the already registered Stage 4 HistGradientBoosting parameters. A
probe is diagnostically sufficient only if, relative to frozen V2, it reduces
stable-state normalized regret by at least 0.05, overall regret by at least
0.03, does not reduce stable-state Top-3 hit rate, wins at least four of six
held-out maps, and degrades no map by more than 0.10 regret. Passing the
86-input probe supports an old-label/training-distribution mismatch; passing
only the 124-delta probe supports missing candidate features; passing only the
context probe supports missing state conditioning. Failure of all three means
feature sufficiency was not demonstrated on this small consumed cohort.

This probe may not export or promote a runtime model. Even a positive result
requires a separately registered fresh-data design before successor training;
`v2-full` remains the active controller throughout.

### Completed consumed feature-sufficiency probe

All integrity gates passed: 48 states, 854 candidates, 13,664 outcomes, whole-
map OOF coverage, frozen-V2 action reproduction, and complete feature rows.
The independent halves agreed on 5,852/7,176 candidate-pair directions
(81.55 percent), and every state supplied at least one stable training pair.

Frozen V2 had 0.3469 overall normalized regret and 0.3789 regret on the 34
stable-winner states. All three ephemeral probes improved the aggregate
numbers:

- the exact-V2 86-input probe reached 0.2722 overall and 0.3074 stable regret,
  with stable Top-3 hit rate increasing from 0.4706 to 0.5588;
- the 124-delta probe reached 0.2722 overall and 0.2875 stable regret, with
  stable Top-3 hit rate 0.6176;
- the 124-context probe reached 0.2653 overall and 0.2976 stable regret, with
  stable Top-3 hit rate 0.6471.

Each probe improved four of six held-out maps and passed five of the six
registered gates. All nevertheless failed the worst-map gate. The exact-86
and 124-delta probes degraded `ost102d` by 0.1161 normalized regret versus the
allowed 0.10; the context probe degraded it by 0.1189. All three also degraded
`lak101d` by 0.0637. The failure is correlated rather than variant-specific:
all three chose the same actions on `lak101d`, and the exact-86 and 124-delta
probes chose the same actions on `ost102d`. The `ost102d` OOF pairwise accuracy
was only 0.637--0.651, the lowest held-out-map range.

The registered diagnosis is therefore
`feature_sufficiency_not_demonstrated_on_consumed_cohort`. The improvement on
four maps is evidence that the new immediate-quality target contains useful
signal, but it is not evidence that an 86-, 124-, or context-input successor
will generalize safely. Extra dimensions and existing state context did not
remove the shared bad-map choices, so no runtime model is exported and
`v2-full` remains active. The next diagnostic must distinguish missing map-
distribution coverage from missing topology/interaction representation before
opening a new training expansion. The deterministic report SHA-256 is
`04b39aee19046c5a20936b3ec6b1a54a7a823ff2834ca3d82e9e1817a30b6f08`.

### Post-hoc transfer-support inspection

A descriptive min/max support check used trial-index zero only, so repeated PP
outcomes were not counted as independent feature observations. `lak101d` had
the lowest all-feature outside-training-range rate of the six maps (0.77
percent), and only 0.02 percent on the 38 features absent from the old 86-
input subset. Its two degradations were both half-winner-unstable states. This
is more consistent with PP-label uncertainty than with ordinary covariate
support failure.

`ost102d` had a 3.72 percent all-feature outside rate and a 1.88 percent rate
on the 38 added features, but the successful `den009d` fold had an even higher
4.17 percent all-feature rate. Shared state-feature outside rates were 18.36
and 14.60 percent respectively. Therefore simple numeric range shift neither
explains the sign of transfer nor justifies opening a larger same-feature
training run by itself. The bad-map action correlation and low `ost102d`
pairwise accuracy instead point to a mixture of uncertainty and missing
map-conditioned topology/interaction information.

Before changing the feature schema, the next low-risk diagnostic should test
a conservative current-step switch: keep frozen V2 unless an outer-map OOF
probe gives strong pairwise evidence that its alternative beats the V2 action.
Any confidence threshold must be chosen only inside the outer training maps
(nested map-grouped calibration), with an abstain-to-V2 option when no safe
threshold exists. This remains a pre-PP neighborhood decision; it does not
switch after a failed repair. If that gate cannot remove held-out-map
regressions, the next expansion must combine broader topology-balanced maps
with candidate-relative bottleneck/conflict-interaction features rather than
repeating static global context alone.

### Registered nested current-step confidence gate

`stride-stepgate-v1` is an ephemeral abstention diagnostic around the smallest
probe, `stride-stepdiag-v1/exact-v2-86`. It does not use the 124-feature
variants and never exports a runtime model. The outer evaluation leaves one of
the same six consumed maps wholly unseen. Within each outer training set, five
additional leave-one-map-out models calibrate a threshold from the fixed grid
0.55 through 0.90. The outer test map contributes neither repair outcomes nor
threshold selection. If no threshold passes the inner safety gates, that
outer fold uses threshold 1.01 and therefore keeps V2 for every state.

At a decision, the challenger is first selected from the candidate pool. Its
symmetrized pairwise probability of beating the frozen-V2 action is then
computed before PP runs. The action switches only when that probability meets
the calibrated threshold; otherwise it abstains to frozen V2. This is not a
failed-repair trigger and does not use repair time, future rounds, Cost-to-Go,
or any post-action field.

An inner threshold must improve both overall and stable-winner regret by at
least 0.02, keep stable Top-3 non-inferior, switch at least two states, and
degrade no inner held-out map by more than 0.03. The final nested OOF result
must improve overall regret by 0.02 and stable regret by 0.03, preserve stable
Top-3, win at least three maps, degrade no map by more than 0.03, switch at
least four states, and make at least 60 percent of switches genuinely lower-
regret than V2. Passing permits only a new fresh-map confirmation design;
failure keeps V2 and moves to candidate-relative topology/interaction feature
design. Neither outcome can promote a controller from this consumed cohort.

### Completed nested current-step confidence gate

All state, candidate, outcome, fold-partition, frozen-action reproduction, and
outer-challenger reproduction integrity gates passed. Nevertheless the nested
gate failed. For five outer maps, no threshold passed the inner safety gates,
so those folds correctly abstained to V2 everywhere. When `ost102d` was the
outer test map, thresholds 0.65--0.90 all looked safe on the other five maps;
the registered lowest-safe rule selected 0.65.

That `ost102d` fold made exactly one switch. It occurred on stable-winner state
`stride-111baa94c49bfabe8b9eaaa0` with predicted challenger-over-V2 probability
0.9222. Frozen V2 had zero normalized regret on the state, while the challenger
had 0.7047 regret. Thus even a post-hoc increase to the largest registered
0.90 threshold would not have prevented the error, and threshold retuning on
this outer test map is prohibited.

Across all 48 states, the one switch worsened overall normalized regret by
0.0147 and stable-state regret by 0.0207. Stable Top-3 decreased by 0.0294,
no map improved, `ost102d` degraded by 0.0881, and switch precision was zero.
Only the stable-state-count gate passed. The registered diagnosis is
`nested_confidence_gate_does_not_control_consumed_map_regression`.

This result rejects confidence-only abstention with the current representation:
the pairwise probability can be highly confident and wrong on an unseen map.
No step gate or challenger model is exported, and frozen V2 remains active.
The next step is candidate-relative topology/conflict-interaction feature
design, followed by a small map-grouped diagnostic before any larger label
collection. The deterministic report SHA-256 is
`225bf2af658022b978f78f1ceb765db179aa3d0ed007343f6903bc976982a8f6`.

### Registered candidate-relative topology diagnostic

`stride-topologydiag-v1` tests whether the shared transfer failures reflect
missing interactions between the proposed neighborhood and map bottlenecks.
It reuses the consumed 48-state, 854-candidate confirmation cohort and its
fixed 16-seed immediate-quality labels. Historical prefix actions may be
replayed only to reconstruct each saved decision state; no candidate is sent
through a new PP repair trial during feature extraction.

Fourteen preregistered features measure articulation-cell and degree-at-most-
two conflict prevalence, internal/incident/boundary conflict-event coverage,
affected-agent coverage, unique path-cell coverage, and visit-heat coverage.
Unlike the failed static context probe, twelve values are relative to the
candidate neighborhood; the two state prevalence values are shared context.
The diagnostic augments only the exact-V2 86-input representation, producing
a 100-input ephemeral probe with the same fixed model parameters and six
leave-one-whole-map-out folds.

Relative to the already pinned exact-86 OOF probe, the augmented probe must
improve overall normalized regret by at least 0.01 and stable-state regret by
at least 0.02, preserve stable Top-3, win at least three maps, degrade no map
by more than 0.05, and not degrade `ost102d` at all. The stable cohort must
still contain at least 24 states. These gates are diagnostic only: a pass
permits a separately registered fresh, topology-balanced confirmation; a
failure retains V2 and shifts attention to broader map coverage or candidate
generation. This consumed-data result cannot export or promote a model and
cannot support a TTF or formal OOD claim.

### Completed candidate-relative topology diagnostic

Extraction reconstructed all 48 registered states and produced one 14-value
row for each of the 854 candidates. No candidate repair trial was executed.
Every feature was finite and non-constant over the cohort. The feature JSONL
SHA-256 is
`d6ddbedf38322691c528398f7c64d2a8b24fc204ac97bf5b957f350ecd639563`.

The 100-input topology probe did not pass. Relative to the exact-86 OOF
baseline, overall normalized regret improved by only 0.00247 against the 0.01
gate, and stable-state regret improved by only 0.00349 against the 0.02 gate.
Stable Top-3 increased by 0.0294, but the probe won only two of six maps:
`den308d` improved by 0.0593 regret and `den408d` by 0.0167, while `den009d`
degraded by 0.0486. `den101d`, `lak101d`, and `ost102d` were unchanged.

Only seven of 48 selected actions changed: two on `den009d`, three on
`den308d`, and two on `den408d`. Thus the new features were neither degenerate
nor universally harmful, but they did not change either of the two previously
problematic `lak101d`/`ost102d` folds and did not provide a transferable
selection signal on this small cohort. Five of seven gates passed; the failed
gates were overall improvement, stable improvement, and at least three map
wins.

The registered diagnosis is
`topology_interaction_features_do_not_show_consumed_oof_signal`. Frozen V2
remains active, and no topology model is exported. The deterministic analysis
report SHA-256 is
`f577e59f6230c0534151e138c991798878e5eb9b7a2a7a8cc9da9d770e8c8b95`.
The next safe step is not another fit on the same 48 states. It is a fresh,
broader topology-balanced map/task preflight that first checks map structure,
load, conflict range, and candidate-pool coverage without collecting new PP
labels. Only a qualified cohort should proceed to paired multi-seed immediate-
quality collection. Candidate generation should be changed only if that
preflight demonstrates systematic failure to propose neighborhoods covering
the relevant bottlenecks.

### Registered topology-balanced map/load preflight

The next outcome-blind preflight uses six checksum-pinned, previously unused
members of the official MovingAI DAO map archive. It does not use the six
consumed confirmation maps or the formal OOD maps. The selected topology
strata are:

- ultra-bottleneck: `lak109d` and `den404d`, with 110/150 and 130/180 agents;
- articulated: `hrt002d` and `den020d`, with 260/380 and 620/930 agents;
- low-articulation controls: `arena` and `lak515d`, with 620/1000 and 550/900
  agents.

Each map receives two fresh task seeds, both uniform-random and opposite-
exchange project-derived OD semantics, and two load levels. Two solver seeds
give 96 initial-state jobs over 48 tasks. Collection stops after initial PP:
`max_decisions=0`, so neither a neighborhood action nor a candidate repair
outcome can affect map/load qualification.

A task is load-qualified at mean initial conflicts of at least 10; targets are
25 and 100 conflicts. Both ultra-bottleneck maps and both articulated maps
must qualify, giving at least four qualified maps in total. Underloaded maps
are allowed only in the registered low-articulation control stratum. A pass
does not permit label collection immediately; it permits a proposal-only
candidate coverage check on the qualified states. Only that second no-repair
gate can decide whether the existing Target/Collision/Random pool covers the
relevant bottlenecks or candidate generation must change first.

### Completed topology-balanced map/load preflight

The frozen preflight completed all 96 initialization-only jobs with zero
errors and zero timeouts. No neighborhood action was executed and no candidate
repair outcome was collected. Initial conflicting-pair counts had mean
377.823, median 202, and maximum 2557; only 2 of 96 initial states were already
feasible. All 15 registered integrity and qualification gates passed.

All six maps qualified, including both ultra-bottleneck maps, both articulated
maps, and both low-articulation controls. The registered selector retained two
tasks per map, giving 12 tasks and 24 paired solver-seed states for the next
diagnostic. This result establishes useful topology and load coverage only; it
does not establish neighborhood quality or solver speed.

The deterministic report SHA-256 is
`9673f77e85a9866eb0ec23347b0906f7c835d223766f529749cd904b0605e9e8`.
Its next decision is `register_proposal_only_candidate_coverage`: generate the
existing proposal pool twice from each retained initial state, verify exact
state preservation and proposal determinism, and measure candidate-relative
coverage of articulation and low-degree bottleneck conflict endpoints without
calling candidate PP repair. Multi-seed quality labels remain prohibited until
that coverage gate passes.

### Registered proposal-only candidate coverage diagnostic

`stride-topocoverage-v1` freezes the 12 recommended tasks and both solver seeds
from the completed topology-balanced preflight. On each of the 24 initial
states it generates the existing Target/Collision/Random proposal pool twice,
using sizes 4/8/16 and the optimized read-only proposal backend. It executes no
controller action and no candidate PP repair.

Every state must preserve its exact fingerprint, reproduce the exact candidate
IDs/agents/families, represent all three requested sizes, and retain at least
12 candidates. At least 8 states must contain articulation-cell conflict events
and at least 12 must contain low-degree-cell conflict events. Among relevant
states, the mean best incident-event coverage must reach 0.60 for articulation
and 0.75 for low degree; at least 60% and 75%, respectively, must reach 0.50.
Both non-control topology groups must contain articulation-relevant states and
reach 0.50 mean best articulation coverage. These gates were fixed before any
fresh candidates were generated.

A pass permits registration of paired multi-PP-seed immediate-quality labels;
it does not promote a model or establish a TTF improvement. A failure instead
requires revising candidate generation before paying for repair labels.

### Completed proposal-only candidate coverage result

The frozen diagnostic completed all 24 paired initial states with 431 retained
candidates. Every state fingerprint was preserved, both proposal repetitions
were identical, every state represented sizes 4/8/16, and the candidate count
was 17--18 per state. The independent repeat reproduced the candidate rows,
state rows, and full report exactly. No controller action or candidate repair
was executed.

The coverage gates did not pass. Fourteen states were relevant to each topology
definition. Best-candidate articulation incident coverage averaged 0.6158, but
only 57.14% of relevant states reached 0.50, below the registered 60% gate.
Low-degree incident coverage averaged 0.6316 and only 57.14% reached 0.50,
below the registered 0.75 and 75% gates. The ultra-bottleneck group was strong
(0.9167 articulation, 0.9643 low degree), while the articulated group reached
only 0.4944 articulation coverage and the low-articulation control group only
0.1667 low-degree coverage on its relevant states.

The deterministic report SHA-256 is
`e868539074485222f0a3ed7d6d2164e65fc7fc3f2d4512279b8cae95c4ae7c32`;
candidate and state row hashes are
`1dada28a25b2ce52c40ad34093e6fe0e8f9e4d70dd886cc87434a6a9707ee624`
and `9a5d3892d763f1b901df6aec4d78fdb2a63f92a3ea48b5cd31c0bcf9cf9a0776`.
The registered next decision is therefore
`revise_candidate_generation_before_quality_labels`. Frozen V2 remains the
active baseline and paired PP quality-label collection remains blocked.

### Registered topology-anchor candidate revision

`stride-topoanchor-v1` is an additive, non-default proposal family. For each
of articulation and low-degree relevant conflict events, it greedily selects
the conflict pair that covers the most still-uncovered events per newly added
agent, then fills to sizes 4/8/16 by conflict-graph adjacency, conflict degree,
and event weight. It keeps both endpoints when capacity permits, so it targets
repairable conflict pairs rather than merely adding a map-level feature.

The follow-up `stride-topoanchor-coverage-v1` repeats the same 24-state,
no-repair diagnostic with the six possible anchor candidates added to the
unchanged Target/Collision/Random pool. The original coverage thresholds are
unchanged. A new cap requires no more than 6 additions and 24 total candidates
per state, limiting later controller overhead. Exact replay, proposal
determinism, state preservation, size-family, overall coverage, and topology-
group gates all remain mandatory. These rules are frozen before the augmented
candidate results are observed.

Only a pass permits a separately registered paired multi-PP-seed immediate-
quality Pilot for `stride-topoanchor-v1`; it still cannot promote a model or
claim lower TTF.

### Completed topology-anchor coverage result

The augmented diagnostic passed every registered gate and reproduced exactly
on an independent run. It retained 503 candidates over 24 states: 17--24 per
state, mean 20.958, with 72 unique additions in total and no more than the
registered six additions per state. All repeated proposal signatures and state
fingerprints matched, and no candidate repair was executed.

Across the 14 relevant states, best-candidate incident coverage rose to 1.0
for both articulation and low degree. Mean best internal coverage was 0.9880
for articulation and 0.8948 for low degree. All relevant states in the ultra-
bottleneck, articulated, and control groups reached incident coverage 1.0.
This establishes that `stride-topoanchor-v1` closes the measured proposal-pool
coverage gap under the candidate cap; it does not establish repair quality.

The deterministic report SHA-256 is
`13510170e21d4b31eb44743a6e11ed07fdcbad4baef7d270a4fbe37df53c537d`;
candidate and state row hashes are
`6c0d2aba662300361e68c1bf427e50d7e384c7090e9d4e4313cc7bab303f48d0`
and `fc07313d3d3a7043cafe314ef6711d0bfe8b7042feb23d5169ee330371448fbe`.
The next permitted step is a separately frozen paired multi-PP-seed immediate-
quality Pilot. Frozen V2 remains the active controller until that Pilot and
later paired TTF gates pass.

### Registered topology-anchor immediate-quality Pilot

`stride-topoanchor-quality-pilot-v1` retains the 16 initial states that contain
at least one articulation or low-degree relevant conflict event. The cohort
contains 5 ultra-bottleneck, 8 articulated, and 3 low-articulation-control
states, with 360 total candidates: 288 frozen Target/Collision/Random candidates
and 72 unique `stride-topoanchor-v1` additions.

Every candidate receives repair trials 0--3. Within one state and trial index,
all candidates use the same deterministic PP seed. The 1440 outcomes are scored
only from the current repair: normalized conflict reduction minus 0.02 times
post-repair structural score, aggregated as the four-seed mean minus 0.10 times
the no-progress rate (`stride-topoanchor-mean-np100-v1`). Repair runtime, future
rounds, Cost-to-Go, and TTF are excluded from the label.

The augmented pool must strictly beat the frozen pool on at least 20% of states,
obtain mean normalized pool gain at least 0.01, place an anchor candidate in the
Top-3 on at least 35% of states, and keep mean anchor-best normalized regret at
most 0.15. Ultra-bottleneck and articulated groups each require at least one
strict win. Two-seed halves also report pairwise consistency, Top-3 overlap,
winner agreement, and cross-half regret, but these four-seed uncertainty values
are diagnostic only and cannot promote or train a model.

A pass permits extension to eight paired PP seeds. A failure stops this anchor
candidate line before training or TTF evaluation. These gates were frozen
before any anchor candidate repair outcome was collected.

### Completed topology-anchor immediate-quality Pilot

The Pilot completed all 16 states, 360 candidates, and 1440 paired repair
outcomes with zero collection errors. The augmented pool strictly beat the
frozen pool on 5 of 16 states (31.25%) and achieved mean normalized pool gain
0.0749. An anchor candidate entered the full four-seed Top-3 on 68.75% of
states. These three registered gates passed.

The Pilot nevertheless failed two mandatory gates. Mean anchor-best normalized
regret was 0.1842, above the 0.15 maximum. More importantly, the ultra-
bottleneck group produced zero strict wins over five states, below its required
one; articulated maps produced 3/8 wins and low-articulation controls 2/3.
Thus complete incident-event coverage did not imply good PP repair quality,
especially on ultra-bottleneck maps.

Four-seed uncertainty remained material: half-seed pairwise consistency was
0.8001, mean Top-3 overlap 0.6875, exact winner agreement 0.4375, and mean
cross-half normalized regret 0.1501. These are diagnostic and do not override
the failed quality gates. The registered decision is
`do_not_train_or_run_ttf_with_stride_topoanchor_v1`; no eight-seed extension is
permitted for this candidate generator.

The repair-trial SHA-256 is
`28e2a793eb4edca470a90879ae96ceb677caef56a47711cc3d94cb4e0da245e7`;
the deterministic analysis-report SHA-256 is
`799c21413db13f54ba8932d17d1486f205b880d0a6abd358b64a28fbfa356e67`.
Frozen V2 remains active. The next safe step is an outcome decomposition of
anchor size, topology kind, PP progress, conflict reduction, and structure
penalty, not additional training or seed collection.

### Completed topology-anchor failure decomposition

The read-only post-hoc decomposition reused the 1440 registered outcomes and
executed no new repair. The best anchor used size 16 on 14/16 states; across all
anchor candidates, mean label score rose from 0.0116 at size 4 and 0.0954 at
size 8 to 0.2299 at size 16. Removing the small sizes is therefore supported by
the existing Pilot rather than by a new outcome search.

The ultra-bottleneck failure came primarily from realized repair quality. Its
best-anchor minus best-base normalized conflict-reduction delta was -0.1138,
while the weighted structure-penalty delta was only +0.0041 and PP time differed
by only +0.0024 seconds. Articulated states had the same direction but smaller
magnitude (-0.0214 conflict reduction), while controls improved by +0.0325.
Thus complete topology-event coverage was not a sufficient repair objective;
selecting both endpoints appears to over-close mutually dependent bottleneck
conflicts. This is a mechanism hypothesis, not a causal or TTF result.

The deterministic decomposition-report SHA-256 is
`991741b2e703cf56ced4655cf91204bf7463f06290146a7f4a0fc707ae7f488e`.

### Registered topology-boundary candidate revision

`stride-topoboundary-v1` is one final proposal-only revision of this line. It
retains only size 16 and adds at most two candidates per state. For each topology
kind it first selects at most four individual incident endpoints, explicitly
penalizing conversion of already covered conflicts from boundary to internal,
then fills from the global conflict graph using the same boundary objective and
component diversity. It does not read repair outcomes, call PP, train a model,
or change the active controller.

The diagnostic reuses the 24 consumed states only to audit proposal shape. It
requires mean relevant incident coverage at least 0.70, mean relevant boundary
ratio at least 0.65 overall and on ultra-bottleneck states, no more than 0.02
mean loss in global event incidence versus the outcome-free base frontier, no
loss in global boundary ratio, and no more than 0.05 mean loss in conflict-
component reach. The total pool is capped at 20 candidates. These gates are
frozen before generating the revised candidates. A pass permits construction
of a fresh topology-balanced cohort; it does not permit repair on these reused
states, training, runtime export, or a TTF claim. A failure terminates this
topology-specific proposal line.

### Completed topology-boundary proposal audit

The proposal-only audit passed every frozen gate and reproduced byte-for-byte
on an independent run. It retained 452 candidates across 24 states (17--20 per
state), of which 21 were unique boundary additions. Eight states had no relevant
topology event and correctly received no addition. Every generated boundary
candidate had size 16; proposal repetitions, state fingerprints, task pairing,
and candidate caps all passed. No PP repair or controller action was executed.

Mean articulation incident coverage was 0.9793 with boundary ratio 0.9300;
low-degree incident coverage was 1.0 with boundary ratio 0.9612. On the failed
ultra-bottleneck group, the two boundary ratios were 0.9086 and 0.9833. Against
the outcome-free base frontier, the revised candidates increased global event
incidence by 0.4066, global boundary ratio by 0.1032, and conflict-component
reach by 0.3066 on average. Thus the revision successfully changes proposal
shape without sacrificing conflict reach, but repair quality remains unknown.

The deterministic report, state-row, and candidate-row SHA-256 values are
`dcc8320c02a7769764b683f3d4efcf2fcc861d7500185b5e1ccfd54e8e96f582`,
`4084e313e4e59cb96f1fbabd7cbbfd01a083376b970e5466508a66aee5d657f9`,
and `a482eaa598b6563267b3d475084de6b677981eec06e1d3e0b3d9fdc0480b3134`.
The next permitted step is an outcome-blind fresh-task preflight. The consumed
24 states may not be used to claim boundary repair quality.

### Registered topology-boundary fresh-task preflight

The next cohort keeps the same six DAO maps, two registered load levels per
map, and both uniform-random and opposite-exchange OD variants, but replaces
master seed `20260804` and task seeds 109/131 with master seed `20260805` and
task seeds 157/181. This creates 48 new task IDs and 96 solver-seed initial
states. Task IDs and endpoint seeds must be disjoint from the consumed cohort.

Qualification and selection remain outcome blind. Only map topology, agent
count, initial conflicts, initial PP time, and initial path statistics may be
used; candidate repairs and controller outcomes are forbidden. At most two
tasks per map are selected against the same 25/100-conflict targets. The purpose
is a controlled within-map mechanism confirmation in which map family and load
are comparable while OD tasks are fresh. It is explicitly not cross-map
generalization evidence; a successful quality Pilot must still pass later
held-out-map Stage 5/OOD evaluation.

### Completed topology-boundary fresh-task preflight

Qualification completed all 96 solver-seed resets over 48 new tasks with zero
errors and zero timeouts. The new and consumed task-ID sets have zero overlap.
The outcome-blind preflight passed every registered gate, found no forbidden
outcome field, read no controller outcome, represented all six maps and three
topology groups, and selected exactly two tasks per map (12 tasks, 24 initial
states). None of the six maps was underloaded under the registered rule.

The preflight report, qualification manifest, qualification report, and dataset
manifest SHA-256 values are
`eb8e36020c413b1c514c5a8b74bf7dfa6728259088f85cad46ac6a80bc0159e3`,
`76ea8a63a0e3ed4c8a7933603b38463ebce1b29fb46296914d23beb63a20106b`,
`8055a6a78581597736a6a707f0ace01e8db4dcd8e36858926e09b2fa2bea623e`,
and `df01c9bba9c5dd6c618140ee423fb508df879c9e71fc071f80ad7cdf320ea8ca`.

### Registered fresh-task topology-boundary proposal audit

The 12 selected tasks and both solver seeds now form a separately pinned
24-state proposal-only cohort. `stride-topoboundary-v1`, its size-16/core-4
construction, two-addition and 20-total-candidate caps, and every incident,
boundary, global-reach, group, determinism, and state-preservation gate remain
unchanged from the consumed-state audit. Task overlap is fixed at zero and no
quality outcome may be read. Only a pass permits registration of a four-seed
immediate-quality Pilot on the subset with relevant topology events.

### Completed fresh-task topology-boundary proposal audit

The fresh audit passed every frozen gate and independently reproduced all three
artifacts byte-for-byte. It covered 24 initial states and 455 candidates, with
23 unique boundary additions. Six states had no relevant topology event and
received no addition, leaving 18 eligible states for a later quality Pilot. No
candidate repair or controller action was executed.

Mean articulation and low-degree incident coverage were 0.9934 and 0.9926;
their boundary ratios were 0.9314 and 0.8301. Ultra-bottleneck boundary ratios
were 0.9744 and 0.9630. Relative to the outcome-free base frontier, global event
incidence, global boundary ratio, and conflict-component reach improved by
0.3514, 0.2539, and 0.4365. This confirms that the intended proposal shape
transfers to fresh OD tasks on the same maps; repair quality and TTF remain
unmeasured.

The report, state-row, and candidate-row SHA-256 values are
`a8b856c2be07e930866572c41f2dd4aa4f75cc2e15b18d719fe5e127f634bd18`,
`d1238c0508018905bd1fe7a57ca7813bb92490ce2b560f61f438d5b8ad3004d7`,
and `f0e5075663c43072e757a2f7a19831fa94471bb57b6fc5e5bfb19bbc34d6857a`.
The next permitted step is a separately frozen, paired four-PP-seed immediate-
quality Pilot over the 18 relevant fresh states. It still cannot train a model,
replace V2, or claim lower TTF.

### Registered fresh-state topology-boundary quality Pilot

`stride-topoboundary-quality-pilot-v1` freezes the 18 relevant fresh states:
six ultra-bottleneck, eight articulated, and four low-articulation control
states. Their 347-candidate product contains 324 unchanged Target/Collision/
Random V2 candidates and 23 unique `stride-topoboundary-v1` additions. Trial
indices 0--3 use one paired PP seed per state and index for every candidate,
giving exactly 1,388 current-step repair outcomes. State order is not used to
choose seeds, and candidates are traversed in reverse order on odd trials as an
order-sensitivity guard.

The frozen label remains `stride-topoboundary-mean-np100-v1`: for each seed it
is normalized current conflict reduction minus 0.02 times the post-repair
structural score; the four-seed aggregate subtracts 0.10 times the no-progress
rate. Runtime, future repair rounds, Cost-to-Go, controller outcomes, and prior
quality outcomes are excluded. This is the same quality and gate family used
by the failed anchor Pilot, with only the preregistered fresh cohort and
boundary candidate identity changed.

The Pilot passes only if the augmented pool strictly wins on at least 20% of
states, mean normalized pool gain is at least 0.01, a boundary-only candidate
enters the Top-3 on at least 35% of states, mean best-boundary normalized regret
is at most 0.15, and there is at least one strict win in both ultra-bottleneck
and articulated groups. Four-seed half-versus-half ranking stability remains
diagnostic only. A pass permits an eight-seed confirmation; it does not permit
training, runtime export, V2 replacement, or a TTF claim. A failure terminates
this topology-boundary revision line.

### Completed fresh-state topology-boundary quality Pilot

All 18 states and all 1,388 registered candidate/seed outcomes completed with
zero errors. The boundary additions strictly beat the frozen V2 pool on 13/18
states (72.22%), achieved mean normalized pool gain 0.2180, entered the Top-3
on 77.78% of states, and had mean best-boundary normalized regret 0.1304. Every
gate passed. Strict wins were 7/8 on articulated states, 5/6 on ultra-bottleneck
states, and 1/4 on controls. Half-versus-half pairwise consistency was 0.8146,
Top-3 overlap 0.8333, exact winner agreement 0.6111, and cross-half normalized
regret 0.0794. This is strong current-step evidence but still not a training,
runtime, or TTF result.

The collection report, repair trials, and analysis report SHA-256 values are
`2057c75a9fd72b72c5308432dcfeb972069dddace8f9f9b352240855be57f8c7`,
`bb54473d4306b3d538064d8f82b319407948a6922c6a65ab683b2ea4b4b288fe`,
and `41486ded4ce598850722c219588ce613b25b4aa356dee0088ce41cfa264c66e3`.

### Registered eight-seed topology-boundary quality confirmation

`stride-topoboundary-quality-confirmation-v1` preserves the same 18 states,
347 candidates, label, and outcome-free selection. It collects only previously
unseen trial indices 4--7 (1,388 new outcomes) and combines them with registered
indices 0--3 for an exact 2,776-outcome product. The unseen half must
independently pass every original quality gate, so the known first-half result
cannot hide failure under new PP seeds. The full eight-seed aggregate must also
pass the original gates.

The two four-seed halves additionally require pairwise consistency at least
0.75, mean Top-3 overlap at least 0.75, and mean cross-half normalized regret at
most 0.15. Exact winner agreement remains reported but is not a gate. Runtime,
future repair rounds, Cost-to-Go, training, runtime export, V2 replacement, and
TTF claims remain forbidden. Only a complete pass permits registration of the
boundary-aware training-data design. The frozen confirmation-config SHA-256 is
`185b6d83b557ba19680a7b2d6cd3d8795b02bdba217291b4bdbff0ad7efad177`.

### Completed eight-seed topology-boundary quality confirmation

The extension completed all 1,388 new outcomes with zero errors and produced
the exact 2,776-outcome eight-seed product. Both the full aggregate and the
previously unseen four-seed half passed every original immediate-quality gate.
The full aggregate had 12/18 strict augmented-pool wins, mean normalized pool
gain 0.2158, boundary Top-3 rate 0.7778, and mean best-boundary normalized
regret 0.1316. The unseen half had 11/18 strict wins, gain 0.2095, Top-3 rate
0.7222, and regret 0.1500.

The formal confirmation nevertheless failed its separately frozen action-
stability gate: half-to-half Top-3 overlap was 0.7222, below 0.75. Pairwise
consistency (0.8678) and cross-half normalized regret (0.0549) passed; exact
winner agreement was 0.6667 and remained diagnostic. This failure is retained:
the revision is not eligible for training, runtime export, V2 replacement, or
a formal TTF claim. The report and combined-trial SHA-256 values are
`ac618d8387404034d62f6319a97602ab0fe9ce2882257e9009a7925478244a4e`
and `c9e1a52d5b242871ac70e9e6f89bc8ad03d4744dbeb9ec8852ba58257cae4b01`.

### Registered non-promoting frozen-V2 boundary Shadow

At the user's request, `stride-boundary-explore-v1` is a bounded exploratory
exception after that formal failure. It does not relax or overwrite the failed
gate. For each of the same 18 states, frozen `v2-full` first scores only the
324 original candidates and then separately scores the 347-candidate pool with
23 boundary-only additions. Both routes use the same model; outcome fields are
forbidden until both action files have been written. The selected actions are
then compared on the already collected eight paired PP-seed immediate-quality
yardstick. No new repair trial, training, future-round label, runtime label, or
TTF measurement is used.

A small paired TTF Quick may be registered only if V2 actually selects a
boundary-only action at least once, improves at least one state, achieves mean
normalized selected-action gain at least 0.01, is nonnegative on the unseen
seed half, and worsens at most 35% of states. Even a pass is only a signal to
run the Quick; it is not promotion evidence. A failure stops this boundary
runtime route while preserving the formal confirmation result.

The first pre-outcome executions exposed an integrity-specification error: the
raw online feature dictionary is sparse, omits zero-valued one-hot fields, and
retains derivable aliases. The registered feature-v2 canonicalization expands
it to the fixed 124-feature schema, while the frozen compact V2 ranker consumes
its registered 86-feature subset. Both runs stopped before an outcome selection
file or outcome join was created. The corrected registration applies the
existing `canonicalize_features` contract and verifies the 124/86 dimensions
separately; no action or quality gate changed. The final corrected frozen
Shadow-config SHA-256 is
`c2a76b38fd46ffa705b7469774222f22914a3fe84c7e7ef57908c7d44cea5b9f`.

### Completed non-promoting frozen-V2 boundary Shadow

The Shadow completed all 18 states with every integrity and exploratory Quick
gate passing. Adding the 23 boundary candidates changed the frozen V2 action
on 18/18 states, and every augmented-pool action was boundary-only. Against the
same eight paired PP-seed immediate-quality yardstick, 12/18 actions improved
and 6/18 worsened. Mean normalized selected-action gain was 0.2999 and remained
0.2940 on the previously unseen seed half. Exact-best rate increased from
0.0556 to 0.6111 and quality Top-3 rate from 0.4444 to 0.7778; mean normalized
regret decreased from 0.4332 to 0.1333.

The benefit was concentrated in topology-relevant groups: articulated states
improved on 6/8 and ultra-bottleneck states on 5/6. The low-articulation
control group improved on only 1/4 and worsened on 3/4. In addition, the chosen
boundary actions had 0.2726 mean frozen-V2 feature-range violation fraction.
Thus the result shows that frozen V2 can exploit this candidate family on the
same states, but also signals out-of-distribution confidence and a necessary
control-group safeguard. It remains offline immediate-quality evidence, not a
TTF result or promotion evidence.

An independent rerun reproduced both artifacts byte-for-byte. The outcome-
blind selections and report SHA-256 values are
`ab0ee0f247ab220fdcc10dda21365d246a8a2b5a852391032552d732bad76dac`
and `9baafd6ca06ec0bad43f1961cc17f3452b5789fd3cac415f20876af1e178cced`.
The registered next step is a small paired runtime Quick that retains both
topology-relevant and low-articulation control tasks, uses capped wall TTF as
the primary diagnostic, and cannot promote or replace V2.

### Registered paired boundary-candidate TTF Quick

`stride-boundary-ttf-quick-v1` keeps the same 18 Shadow states because it is a
mechanism diagnostic, not independent confirmation. It compares frozen
`v2-full` against the same frozen ranker with `stride-topoboundary-v1` enabled
at every repair decision. The augmentation is additive, deterministic, size
16/core 4, and capped at two candidates per decision. It is disabled by
default, has a separate run fingerprint, and does not create a new production
controller identity. Both routes use deterministic same-state PP replay.

The 36 episodes use strict alternating pair order, one worker, a reset-inclusive
60-second wall TTF budget, a 60-second native limit, and a 90-second process
safety timeout. The cohort contains four low-articulation controls, eight
articulated states, and six ultra-bottleneck states. Primary evidence is mean
capped wall time to feasible under success noninferiority; the report also
records common-success TTF, repair iterations, wall AUC, PP time, controller
time, proposal/feature/inference time, boundary additions/selections, and each
topology subgroup.

Continuation requires at least 5% lower mean capped TTF, no loss of successes,
non-worse mean repair iterations, no more than 10% capped-TTF regression in
either topology-relevant group, no more than 20% regression in the control
group, and at least one executed boundary repair. A pass only permits a larger
map-disjoint development Quick. The frozen config SHA-256 is
`66b2a6aa19bcb7ea7760993c37e13f378ee5d38952952e75aa507228f7ad8a5a`.

### Completed paired boundary-candidate TTF Quick

All 36 paired episodes completed with zero execution errors, invalid actions,
semantic mismatches, initial-state mismatches, or fingerprint mismatches. The
boundary route executed 234 boundary-selected repairs and reduced mean repair
iterations from 49.72 to 35.28 (29.05%). It did not convert that action-quality
gain into the preregistered primary runtime result: successes decreased from
17/18 to 16/18, and mean capped wall TTF increased from 13.21 to 15.06 seconds
(14.01% regression). On the 15 common-success episodes, mean TTF changed from
6.43 to 6.30 seconds, which is only a 2.06% improvement and does not repair the
success loss.

The failure mechanism is visible in the timing split. Mean PP replan time was
essentially unchanged (2.62 versus 2.63 seconds), while boundary analysis added
4.22 seconds per episode and increased total controller time from 7.30 to 9.40
seconds. Ultra-bottleneck tasks improved capped TTF by 10.58%, articulated
tasks were effectively flat with a 0.41% regression, and the low-articulation
control group regressed by 23.74% while losing one success. Thus the repair-
iteration, topology-group, and activation gates passed, but the primary TTF,
success-noninferiority, and control-safeguard gates failed.

This exploratory same-cohort Quick is retained as a negative runtime result.
It does not warrant a larger development Quick, training, V2 replacement, or a
speed claim; `v2-full` remains the default and this boundary runtime route is
stopped. The controller-manifest SHA-256 values are
`801c3d42c4e8957bc6fd884f0ad7523dc8f10170ff044fb31b97bd9695e96189`
for V2 and
`7db6ee30d4cdbcb0c6029403b12c18180282e5383525d1c88b81e118c34506dc`
for the boundary route. The report SHA-256 is
`013e7f6d9911aff904db167fa7f9b32c31f57b3a090fcce93da937531e753aeb`.

### Boundary runtime profiling and action-preserving static reuse

The stopped runtime result motivated a non-promoting implementation diagnostic,
not a relaxation of its gates. Per-decision timing now separates static-grid
analysis, dynamic path/conflict reconstruction, boundary-candidate construction,
and candidate merging while preserving the original aggregate timer. On the two
`lak515d` 900-agent failures, an unchanged legacy rerun spent 21.14 and 22.09
seconds in boundary analysis. Static-grid work accounted for only 3.45 and 2.64
seconds; dynamic reconstruction accounted for 16.40 and 14.24 seconds, candidate
construction for 1.28 and 5.20 seconds, and merging for less than 0.01 seconds.

`stride-boundary-static-cache-v1` is an explicit opt-in runtime revision. It
reuses the already validated per-episode `StaticGridAnalysis`; the historical
`stride-topoboundary-v1` configuration retains its original uncached behavior.
Across the two diagnostic episodes the cache hit on every decision and reduced
per-round controller cost by approximately 6.7% and 10.2%. It did not solve
either episode within 60 seconds. Every state fingerprint, selected candidate,
repair action, and post-repair conflict value matched the uncached execution for
the entire common trace prefix (126 and 95 repairs). Thus static reuse is safe
and useful but insufficient; the next permitted diagnostic is a cheap pre-
analysis activation gate followed by low-conflict/stall fallback. No model,
label, default controller, promotion result, or formal TTF claim changes.

### Registered gated boundary-runtime optimization Quick

The cheap-gate diagnostic first rejected a conflict-count-only phase rule. On
four representative episodes, disabling boundary analysis whenever fewer than
two conflict pairs remained changed `den404d` seed 2 from 20 repairs to 52,
showing that a low-conflict state can still need a topology-boundary action.
The registered `stride-boundary-phase-guard-v2` therefore keeps the frozen V2
ranker and candidate generator unchanged, reuses the static grid, and skips the
expensive dynamic analysis only when one of three joint conditions holds:
fewer than two conflict pairs after at least two non-improving repairs, at
least five consecutive non-improving repairs, or less than five seconds of
wall budget remaining. A map-level gate also requires at least 6% of free
cells to have degree at most two. No map identity, task label, repair outcome,
or PP result is used by either gate.

The revised guard kept all four representative episodes successful: the two
`den020d`/`den404d` checks retained 7/20 repairs, while `hrt002d` and `lak109d`
used 33/46 repairs and rejected 2/9 boundary analyses. A blind cooldown was
not registered because trace inspection found a successful `den404d` episode
with 13 consecutive useful boundary selections. These checks are mechanism
diagnostics, not TTF evidence.

`stride-boundary-runtime-optimization-quick-v1` now repeats the exact same 18
states and 36 paired episodes as the failed boundary Quick. This deliberate
same-cohort reuse isolates runtime logic and is outcome-informed, so it cannot
support promotion, training, an OOD claim, or a formal speed claim. It retains
the original 60-second capped wall-TTF primary metric and all continuation
gates. A pass only permits a separately registered fresh map-disjoint Quick; a
failure retains frozen `v2-full` and stops this optimization route. The frozen
config SHA-256 is
`8a063b9299fff03c0056eb2b0f18a832cf2184fe197b16fc99bf97c3dd54a9cc`.

### Completed gated boundary-runtime optimization Quick

All 36 paired episodes completed with zero execution errors, invalid actions,
semantic mismatches, initial-state mismatches, conflict mismatches, or
fingerprint mismatches. Both frozen V2 and the guarded boundary route solved
17/18 states; the unsolved state was the shared 1000-agent `arena` control.
The guarded route reduced mean repair iterations from 48.50 to 41.67
(14.09%) and reduced mean capped wall TTF from 13.738 to 13.373 seconds
(2.66%). On the 17 common successes, mean wall TTF changed from 11.017 to
10.630 seconds.

The implementation optimization worked as intended but was not sufficient for
the registered speed claim. Mean boundary-analysis time fell from 4.22 seconds
in the previous unconditional Quick to 0.599 seconds here. The gate evaluated
751 decisions, passed 250, rejected 480 on the static map-topology check, and
rejected 21 after a no-progress streak. Mean total controller time was 7.253
seconds versus 7.416 for paired V2, while PP replan time was 2.680 versus
2.830 seconds. The guarded route executed 191 boundary-marked repairs.

All subgroup safeguards passed: capped TTF improved by 3.67% on articulated
states, 17.86% on ultra-bottleneck states, and 0.80% on the low-articulation
control group. The primary continuation gate nevertheless failed because the
overall 2.66% capped-TTF improvement was below the preregistered 5% minimum.
The per-task effects also remained mixed: fewer rounds produced large gains on
some `den020d`, `den404d`, `hrt002d`, and `lak109d` instances, but other paired
instances regressed, and the two 900-agent `lak515d` runs retained 139/122
repairs under both routes.

This is a positive mechanism result but a negative continuation result. It
does not warrant the registered fresh map-disjoint Quick, new-model training,
V2 replacement, or a formal speed claim. Frozen `v2-full` remains the default
and this boundary-runtime optimization route is stopped. Re-analysis reproduced
the report byte-for-byte; its SHA-256 is
`cd033ff8cd052595116d88614474dad631d1ad7cb29e68ad1706db3c36700308`.
The V2 and guarded controller-manifest SHA-256 values are
`9ea205069b7900703bb24f10c21d415fc86ca1307d774124b7ddd7be3edf062c`
and `5ea238f0f37a796112c85b298aa7f4730a501ff70cbde9374bf9bf1de9766ee7`.

### Registered run-to-completion boundary gate ablation

The capped Quick is retained unchanged, but its five-second remaining-time
branch is not used as scientific evidence for the next diagnostic. The
registered `stride-boundary-gate-ablation-rtc-v1` instead measures
reset-inclusive raw wall TTF and runs every episode until feasibility. It has
no scientific wall limit, native time limit, process timeout, repair-iteration
limit, or diagnostic decision cap. An unfinished or interrupted episode has no
raw TTF and prevents selection; it is never converted into a capped score.
The timed route uses the deployment verification profile so the Python shadow
extractor is not charged to solver TTF; exact native-versus-reference feature
equivalence remains covered by the separate audit tests.

The ablation reuses the same outcome-informed 18-state mechanism cohort, so it
cannot promote a model, support an OOD claim, or become training data. Each
state executes a strict rotating triplet under paired deterministic PP replay:
frozen `v2-full`, `stride-boundary-map-gate-v1`, and
`stride-boundary-stall-guard-v1`. The map-gate route uses only the preregistered
6% low-degree-cell map check. The stall-guard route adds exactly one condition:
skip topology-boundary construction after five consecutive repairs without a
conflict reduction. Neither route uses remaining wall time or the earlier
low-conflict joint rule.

All three controllers must solve all 18 states with zero execution, semantic,
or pairing errors before raw TTF is compared. Retaining the stall guard requires
at least 2% lower mean raw TTF than the map-only gate, non-worse mean repair
iterations, at most 10% raw-TTF regression in every topology subgroup, and at
least one executed boundary repair. Passing selects the stall guard for the
next action-preserving feature-cache optimization; failing selects the map-only
gate. This choice does not replace frozen V2. The frozen config SHA-256 is
`ffaad8f216c264759e0cb23a1e5d15df418878b4e8d63a49569874a71075eb8d`.

### Completed run-to-completion boundary gate ablation

All 54 executions completed, covering 18 strictly paired states under all three
controllers with zero execution errors, invalid actions, initial-state
mismatches, semantic mismatches, or fingerprint mismatches. Every controller
solved all 18 states. Mean reset-inclusive raw wall TTF was 13.9051 seconds for
frozen `v2-full`, 14.2713 seconds for the map-only gate, and 13.8851 seconds
for `stride-boundary-stall-guard-v1`. The stall guard therefore improved raw
TTF by 2.71% relative to the map-only gate and passed the registered 2% gate;
it was only 0.14% faster than V2 and is not a replacement result.

Mean repair rounds were 125.72 for V2, 119.44 for the map-only gate, and
118.94 for the stall guard. The stall guard rejected 21 topology analyses
after five non-improving rounds while still executing 191 boundary-selected
repairs. It improved raw TTF relative to the map-only route in every registered
subgroup: 1.54% on articulated, 3.09% on low-articulation control, and 4.89%
on ultra-bottleneck states. Its lower PP time and fewer repairs were largely
offset by controller/topology overhead, explaining why its raw-TTF result was
essentially tied with V2. The stall guard is retained only as the augmented
pool collection route. The reproduced report SHA-256 is
`225de3d0cf0a5795672c0df21645858bebe121c87d00b4e2631d1539cb20f1fd`.

### Completed action-preserving feature-path audit

The audit replayed the first three decisions of all 18 stall-guard episodes,
for 54 real decision samples. It compared the Python reference 124-feature
extractor, Python incremental 124-feature extractor, native dense 124-feature
extractor, and the deployed native dense projection onto V2's 86 required
features. All four paths produced identical rankings and selected actions;
the largest feature difference from the reference was
`1.3322676295501878e-15`, below the registered `1e-12` tolerance.

Median whole-audit times were 5.6547, 6.8508, 0.3020, and 0.2940 seconds,
respectively. The already deployed native 86-feature projection is the fastest
equivalent path: approximately 2.6% faster than native full-124 extraction and
19.2 times faster than the Python reference. Feature extraction accounted for
8.40% of mean raw TTF in the preceding stall-guard experiment. Consequently,
the implementation is retained unchanged; there is no candidate/action change
and no standalone formal speed claim. The audit config SHA-256 is
`9a0a5d36b2bae96d60aaa13459894db449c1e8a1c5fafb4c32bf4d45e476bbc5`.

### Preregistered augmented-pool repairability label

The successor line is independently named `stride-augcontrol-v1`; it is not a
new version of V2, V3, or `stride-quality-v1`. Its label schema is
`lns2.stride.repairability_label.v1`. The frozen candidate pool preserves all
original Target, Collision, and Random requests at sizes 4, 8, and 16, then
adds at most two size-16 `stride-topoboundary-v1` candidates with a four-agent
core. Deduplication may attach topology provenance to an existing base
candidate, but may neither remove nor mutate a base action. The total pool is
capped at 20 actions.

Each state/action pair is repaired under exactly 16 deterministic PP seeds.
The seed for a trial index is paired across all actions in that state, while
the 16 indices are distinct and split into independent halves 0--7 and 8--15.
For action `a` and seed `s`, the current-step score is

`q(a,s) = (conflicts_before - conflicts_after) / max(1, conflicts_before)
           - 0.02 * post_structure_percentile(a,s)`.

The structure term is the mean within-state, within-seed midrank percentile of
post-repair largest-component ratio, conflict-edge density, conflict-event
density, and degree concentration; lower is better. A directed pair is kept
only when one action wins at least 12/16 paired seeds, its absolute mean score
advantage is at least 0.02, and both eight-seed halves have the same mean
direction. All other pairs are explicitly uncertain and excluded. The two
orientations of all retained pairs from one state sum to weight one, preventing
states with many candidate combinations from dominating training.

The label consumes the exact frozen 124-dimensional realized-feature schema.
Repair runtime, TTF, future repair rounds, Cost-to-Go, Receding-Q, and any
controller outcome are forbidden label inputs. Collection is state-resumable,
stores the complete paired product, and verifies base-pool preservation before
running PP. This section registers only the label and collection contract; no
model has yet been trained and no speed or replacement claim is made.

The first registered cohort contains 240 states from equal `official_adaptive`
and frozen-`v2-full` source policies. Sixteen DAO maps are assigned to train
and six disjoint DAO maps to validation before any new candidate repair is
run. State sampling may use only the pre-action map identity, source policy,
decision stage, initial conflicts, agent count, static low-degree ratio, and a
hash tie break. Each effective map must contribute at least eight states; both
sides of the 6% topology threshold must contribute at least 40%, all three
decision stages must occur, and at most two states may come from one episode.

Earlier reset-only qualification found five nominal maps underloaded. Their
agent ladders are therefore re-qualified before collection. Each has one
same-split, checksum-pinned DAO reserve selected from static topology alone;
replacement is allowed only if the primary cannot provide eight conflicting
source states. Candidate repairs, selected actions, runtime, future states,
and TTF cannot influence this resolution. The 12 formal MovingAI/OOD maps
(`random`, `maze`, `room`, `warehouse`, `den312d`, and `lak303d`) remain
excluded from selection, label construction, and training.

If the label pilot passes its integrity gates, the only primary trainable model
is `stride-augcontrol-v1`. It uses the same fixed
`HistGradientBoostingClassifier` capacity and 124-feature/147-pairwise input
construction as the controlled Stage-4 comparison, with no hyperparameter
tuning. Validation is map-held-out. Frozen V2 is rescored on the identical
augmented pool, while base-pool-only and conflict-only-label variants remain
named ablations. The primary model must beat the weighted pairwise-majority
baseline, reduce validation normalized repairability regret versus frozen V2,
retain exact-best and Top-3 rates, and avoid a topology-group regression. Even
an offline pass only permits Shadow evaluation; it cannot replace V2.

### Completed repairability map/load confirmation

The first reset-only sweep covered ten preregistered primary/reserve DAO maps,
60 tasks, and 120 solver-seed jobs. It intentionally tested aggressive loads
and found 14 invalid initializations, all `state contains an empty agent path`.
Those failures were confined to unstable high-load combinations and no
candidate repair, controller outcome, future state, or TTF was read. The sweep
also showed that `arena2` remained almost conflict-free through 1800 agents
and that `den001d` remained weak. The split-preserving replacements were
therefore fixed as `arena2 -> den206d` for train and
`den001d -> den011d` for validation.

A fresh 48-task, 96-job confirmation then used only validated loads. All 96
initializations completed with zero errors and passed dataset, seed-isolation,
and semantic gates. The five effective maps contributed respectively 20
(`brc300d`), 15 (`brc502d`), 11 (`den011d`), 16 (`den204d`), and 16
(`den206d`) nonzero-conflict jobs, all above the minimum of eight. Registered
loads are 900, 1300, 1200, 650, and 700 agents. Candidate and controller
outcomes were not read. The qualification-manifest and qualification-report
SHA-256 values are
`cd648992025f8217e1fa66932f968bf731088e042a626d2c84e27bbbcf300eec`
and `9b0127dbba8dded7cb41da4ab5fa864452de8a2f0cf05cc4ea88f662945e5fc3`.

Auditing all 22 planned maps exposed one additional underload missed by the
original registration: `den005d` produced a nonzero reset on at most one third
of jobs at 500--750 agents. The statically closest unused development map was
`ht_mansion_n`, with low-degree-cell ratio 0.04934 versus 0.05131. A first
1000-agent check was rejected after two empty-path initializations. A fresh
900-agent confirmation then completed 16/16 jobs with zero errors and 16
nonzero states. Median initial conflicts were 97.5 for opposite-exchange and
4.5 for uniform-random OD. This registers the same-train-split replacement
`den005d -> ht_mansion_n`; the failed 1000-agent evidence remains retained but
is not training data. The passed qualification-manifest and report hashes are
`589a4a74ce8ecb0b65faa672e59ce291c78d5fd5f0c4c9d9e2deee20b0a0271b`
and `3d1a191009cf7eb6b0b52756bd4189c94431d0e52e967c5fefc0dfa48cec3283`.

### Frozen repairability source-task cohort

After the three load replacements were fixed, result-blind selection produced
44 source tasks on 22 checksum-pinned MovingAI maps: two tasks per map, 32
train tasks on 16 maps, and 12 validation tasks on six disjoint maps. Twenty
tasks are in the static-topology control group and 24 are boundary-relevant.
Selection used only initial-PP completion/conflict evidence across registered
solver seeds, task identity/variant, load, static low-degree ratio, and the
predeclared research split. No candidate repair, controller action, future
trajectory, runtime, or TTF was read. The effective replacements are exactly
`arena2 -> den206d`, `den001d -> den011d`, and
`den005d -> ht_mansion_n`.

The source runtime registers two solver seeds, both `official_adaptive` and
frozen `v2-full` trajectories, and up to 12 current decisions per episode.
It contains no remaining-time guard or selector time-limit rule; its purpose is
to provide outcome-blind state coverage before the separately paired 16-seed
candidate-repair experiment. Rebuilding the cohort left all three core
artifact hashes unchanged. The cohort config, manifest, report, and summary
SHA-256 values are respectively
`1d96bf649360dcf926f865b71f280d9c29463591ee2e3b6eb57bc67a0214bd21`,
`f2c32aad71393be14128549094d9c69f683c83f611f09fb0e3ad732bb2ab8ef4`,
`a066b0a28de671d502c896459205ec87980eac1e52e4a5e0c92c7c88800da95f`,
and `0e51d0ce19065e620e948425faba49850f6c0527d47efc913af102f3dfbe358d`.
This is a data-source milestone only; it is not a label, model, or speed claim.

### Registered source-state selection and augmented controller training

The two source-policy traces feed a dedicated result-blind selector before any
new candidate is repaired. It selects exactly 120 states from
`official_adaptive` and 120 from frozen `v2-full`, with at most two states per
episode. Each of the 22 effective maps contributes at least eight states and
both source policies; static-topology control and boundary-relevant groups
each contribute at least 40%, states with at least 101 conflicts contribute at
least 10%, and early, middle, and late decisions must all be present. The
selector admits only pre-action identity, conflict load, static topology, and
the deterministic replay prefix. It rejects TTF, repair time, selected action,
candidate outcome, after-state, and future-trajectory fields.

The independent model remains `stride-augcontrol-v1`; its conflict-only
training ablation is `stride-augcontrol-conflict-v1`. Both use the already
registered 124 realized features, 147 pairwise inputs, fixed
`HistGradientBoostingClassifier` capacity, equal state weight, and no
hyperparameter tuning. Six whole validation maps remain excluded from fitting.
Both estimators are exported under their own diagnostic-only bundle names;
only the primary bundle is eligible for Shadow and runtime evaluation.
The primary model is evaluated against frozen V2 on the identical augmented
pool, the same primary model restricted to base candidates, and the
conflict-only-label ablation. The earlier committed gates are preserved: at
least six validation maps, weighted pairwise accuracy at least 0.03 above the
weighted-majority baseline, normalized regret at least 5% relatively or 0.02
absolutely below frozen V2, exact-best and Top-3 noninferiority within 0.01,
and no static-topology group regret degradation above 0.03. A pass authorizes
only action-preserving Shadow; the exported bundle remains diagnostic-only and
cannot replace V2 before paired raw-TTF evidence.

Candidate-state coverage alone cannot satisfy the offline gate. At least half
of the 174 train states and half of the 66 validation states must produce a
stable robust-pair label; all 16 train maps and all six validation maps must
contribute, with at least three labeled states per map. These thresholds were
registered before reading the 16-seed candidate outcomes. They prevent a model
trained on a small, easy subset from passing merely because all selected maps
remain present in the unlabeled candidate aggregate.
Within both train and validation, every observed source-policy, decision-stage,
topology-group, and agent-band subgroup must additionally retain at least 30%
of its selected states and at least five labeled states. This prevents stable
pairs from concentrating only in one controller trajectory or load regime.
The preregistered training-config SHA-256 after this coverage correction is
`ae52f1e58fe2c49f05f78fd9fd69a4a3bc140880d3195bc9666a9435a3a6d4b8`.

### Completed paired source traces and result-blind state selection

The unified source run completed 88 reset-only qualifications, 88
`official_adaptive` episodes, and 88 frozen-`v2-full` episodes over 22 maps,
44 tasks, and solver seeds 1 and 2. All rows had status `ok`; there were zero
timeouts, invalid actions, trace-schema errors, or semantic mismatches. Every
official/V2 row matched the qualification row on initial state fingerprint and
initial conflict count. Official Adaptive solved 84/88 episodes and V2 solved
85/88; the remaining seven rows are valid unsolved algorithm outcomes rather
than collection errors. Qualification, official, V2, collection-summary, and
run-config SHA-256 values are respectively
`431c33fa007eb7e9d1f339302b52477a62b2d0ac8e200c8b1559af426c1cb402`,
`6b6e7b23fbc01d2b868c217233e6eb631c47c43094e3f81828743fa0999ea1f8`,
`4a9aed6d0f5b5fb913d059967ae61a259857245874fd05bc0d4cbbe364ffed92`,
`d9861552d15ce81ae287efd74a1805063751d71f00dea742c2490f95f2684c63`,
and `ea47c5166d272047f2745adea5c6c634e3e624653bb4a1f89132037732df1246`.

The result-blind selector then considered 1,473 active pre-action decisions
and fixed exactly 240 states: 120 per source policy, 80 per decision stage,
105 static-topology controls and 135 boundary-relevant states. Conflict bands
contain 101 states at 1--10 pairs, 85 at 11--100, 52 at 101--500, and two
above 500. All 22 maps occur, the smallest map contribution is nine states,
both policies occur on every map, and no episode contributes more than two.
No candidate outcome, selected action, repair runtime, future trajectory, or
controller TTF field was present. The annotated selection, selection report,
source-selection report, and raw result-blind selection SHA-256 values are
`f523122c62a000274df1250f4bbda9ff7757c73991d26ce33644f4040bfe6bb8`,
`4f8cd24c517bd763718a1f9e240320b40a5624956af4d709154d8d07e1a484e6`,
`d0a36f8f4650b66a91fd1379ffbac88bd0daa9070634322e0b59c3be1299ee70`,
and `f24f7bfb48064791332f046f75db5a4fc9d21227692828dce437bd9b43f4f8eb`.
This selection remains a data-source artifact; it contains no repairability
label and reads no formal OOD map.

### Exact target-path restoration for paired repair trials

The first 16-seed collection attempt was interrupted after nine completed
states because two high-load `v2-full` states exposed a replay-integrity
failure. Re-running reset plus the complete recorded prefix either terminated
before the selected decision or reached a different full fingerprint. This is
an operational wall-clock sensitivity of high-load initial PP, not a candidate
repair outcome, and neither failed state is removed or replaced.

Repairability collection v2 therefore reconstructs the selected pre-action
state directly from the stored trace deltas and stops before reading the target
action or outcome. Each native proposal or PP branch is created with
`reset_paths` from the exact recorded agent paths. Full iteration, low-level,
and wall-clock counters intentionally reset in the native branch, while the
repair-structure fingerprint (map, paths, conflicts, feasibility, and cost)
must match exactly. The registered 124 features continue to use the complete
stored source state, including its true decision index and accumulated search
context. Proposal generation additionally requires an unchanged native state
revision and repair-structure fingerprint. All candidates at a state use the
same deterministic restore seed, and every candidate at a trial index still
uses the same paired PP seed.

The two states that triggered the interruption were restored successfully
under this contract: the `den206d` state retained 225 conflict pairs while its
iteration counter reset from 7 to 0, and the `ht_mansion_n` state retained nine
conflict pairs while its counter reset from 8 to 0. Both repair-structure
fingerprints matched exactly. The interrupted v1 output is preserved as failure
evidence; v2 writes to a new output directory and has a new run fingerprint.
This restoration milestone is not a label, ranking, or TTF result.

After collection completes, the registered read-only audit is:

```bash
python3 scripts/audit_stride_repairability_collection.py \
  --collection build/stride-repairability-collection-v2 \
  --output build/stride-repairability-collection-audit-v1
```

It must pass exact 240-state coverage, trial indices 0--15, deterministic
same-state seed pairing, distinct seeds across trials, the 124-feature schema,
frozen base-family preservation, the two-candidate topology cap, target-path
restore provenance, map-held-out splitting, aggregate/state-row identity, zero
collection errors, and SHA-256 materialization before labels are built.
Label construction requires that passed audit as a positional input and verifies
the audited `repair_trials.jsonl` SHA-256 before reading any row:

```bash
python3 scripts/build_stride_repairability_labels.py \
  --config configs/stride_repairability_label_design.json \
  --trials build/stride-repairability-collection-v2/repair_trials.jsonl \
  --audit build/stride-repairability-collection-audit-v1/repairability_audit_report.json \
  --output build/stride-repairability-labels-v1
```

Training reopens the recorded audit and raw trial source, verifies both stored
SHA-256 values, the passed audit identity, run fingerprint, and audited state
count, and refuses to fit if any provenance item has changed.
The exported feature ranges and bundle training counts use train maps only;
held-out validation candidates cannot influence runtime fallback ranges. The
portable model is then loaded back and its selected action is compared with the
in-memory estimator on every validation state as a separate equivalence gate.

Label construction also emits a descriptive, non-gating
`seed_half_action_stability.jsonl`. It compares trial halves 0--7 and 8--15
using exact winner agreement, Top-3 overlap, comparable-pair ordering agreement,
the full-sample Top-1 score margin, and candidate score dispersion. These
quantify PP-seed action uncertainty without changing the registered robust-pair
rule or consulting runtime and future trajectories.

### Preregistered staged raw-TTF evaluation

Runtime evaluation separates candidate-pool quality from ranking quality with
three strictly paired controllers: original `v2-full`, the frozen V2 ranker on
the augmented base-plus-topology pool (`v2-augmented-pool`), and
`stride-augcontrol-v1` on that identical augmented pool. All speed comparisons
use reset-inclusive `wall_time_to_feasible` under `run-to-completion`; capped
TTF, remaining-time guards, environment time limits, and process timeouts are
not scientific scores and are absent from this protocol. Execution rotates the
three controllers within every task/seed key and requires matching initial
fingerprints and conflict counts.

The order is fixed. An offline pass first permits an action-preserving Shadow
on the six held-out DAO validation maps. Shadow must produce valid decisions
without action override or state-semantic mismatch. It then permits a
development high-load comparison on `maze-128-128-2` at 300 agents and
`room-64-64-8` at 500 agents, two scenarios and four solver seeds per map. The
development gate requires at least 2% lower mean raw TTF than original V2,
non-worse repair rounds and success count, no cohort regression above 10%, and
non-worse raw TTF than the V2 ranker on the same augmented pool.

Only a development pass opens the formal OOD data. The six fixed MovingAI
holdouts are `maze-128-128-10` at 200 agents, `room-64-64-16` at 400,
`random-64-64-20` at 500, `warehouse-20-40-10-2-2` at 500, `den312d` at 300,
and `lak303d` at 500. Each uses random scenarios 4 and 5 and solver seeds 1--3.
The lower maze load prevents the extreme-conflict regime from dominating the
comparison, while the 300--500-agent loads retain substantial pressure on the
other map families. Formal support requires at least 5% lower mean raw TTF
than original V2, non-worse success and repair rounds, no map regression above
10%, at least half of paired episodes faster, and non-worse ranking than frozen
V2 on the identical augmented pool. Even a pass supports a speed claim only;
it does not automatically replace the default V2 controller.
The evaluation, formal-dataset, and formal-runtime config SHA-256 values are
`87fa1d44f61489ee1ad1f5cd9e53e52f49349cd00a09162ea14307e5308fa124`,
`5b0404a3d2bd6898866ceabb30bacbeb1a6678c33441d57c446f5483bf61477c`,
and `23d102c37bb0ed42f05214a6a936b72f85566ab0eef3c3e2a4ae90ba1165b14f`.

## Promotion boundary

The next sequence is paired 16-seed label collection, controlled retraining,
action-preserving Shadow, development high-load raw TTF, and formal OOD raw
TTF. Failure at any gate retains frozen V2 and leaves later holdout data unread.

### Completed augmented-pool repairability pilot and offline gate

The audited collection completed all 240 selected states with zero errors. It
contains 4,385 candidates and 70,160 paired repair trials; all collection gates
passed. The 113 independent Topology Boundary candidates occurred in 98
states. The strict label retained 18,264 of 37,932 candidate pairs and produced
at least one robust pair in 232/240 states. Train and validation pair-state
coverage was 169/174 and 63/66 respectively, with all registered maps and
subgroups passing their coverage thresholds.

PP-seed diagnostics show 56.67% exact winner agreement between the two
eight-seed halves, 82.50% Top-3 overlap, and 90.72% direction agreement among
comparable candidate pairs. Thus coarse candidate quality is much more stable
than a unique winner, supporting robust pair learning rather than a single
Oracle action.

`stride-augcontrol-v1` passed all integrity, coverage, portability, and
pairwise-accuracy gates, but failed the preregistered offline quality gate. On
the six held-out maps its weighted pairwise accuracy was 0.8572 versus a 0.5
majority baseline, but exact-best rate was 0.5000 versus frozen V2's 0.5909 and
normalized repairability regret was 0.1928 versus 0.1876 (2.77% relatively
worse). Top-3 rate improved from 0.6515 to 0.7273. Consequently Shadow remains
forbidden and frozen V2 remains the default.

The conflict-only ablation was materially stronger than the registered
structure-weighted primary model: pairwise accuracy 0.8787, normalized regret
0.1535, exact-best 0.5455, and Top-3 0.7727. It still failed the fixed
exact-best noninferiority gate. Candidate-type diagnostics locate the primary
failure in ranking calibration across the augmented pool: the primary model
improved base-only regret versus V2 but degraded Boundary choices, even though
Boundary was uniquely best in 22.73% of validation states. This result supports
a separately named, train-only calibrated anchored ranker as the next research
step; it does not authorize retuning on these already observed validation maps.
The collection-audit report, Windows label summary, offline training report,
validation predictions, and primary controller manifest SHA-256 values are
respectively `d697cf919805bdcc1e50dad5af434a4e7a8d13f3637e864bfa4c7081ffd2ada2`,
`ba59fda93c145fcdb4b7f29f352d02aa10852193817590824c95e3eeba1fd26e`,
`5cddd7f0c580fc48fe863ff37a4b97f8820daae4a1b33db68b6743933397bff6`,
`c1de46caff61a707da966dd27f0e490c2459458f4bc1943bb0698ac8274e544f`,
and `71f966046c295c3e774ca50116dea543513e4bbc071b482eeabffd2ad672fece`.

### Train-only V2-anchored guard ranker

The separately named `stride-guardrank-v1` follow-up preserves frozen V2 as
the action anchor and permits the conflict-only challenger to replace V2 only
when its symmetric pairwise probability exceeds a candidate-kind threshold.
The base and Topology Boundary thresholds were calibrated without validation,
test, formal-OOD, runtime, or future-trajectory inputs. Four outer map folds
measured generalization; each outer training split used three inner map folds
to choose thresholds. Thus every reported train OOF action was produced by a
ranker and threshold calibration that excluded that action's map.

The nested train-map OOF result did not pass the preregistered regret gate.
Frozen V2's normalized repairability regret of `0.239843` decreased to
`0.228648`, an absolute improvement of `0.011195` and a relative improvement
of `4.6675%`. The fixed gate required at least `5%` relative or `0.02`
absolute improvement. Exact-best improved from `0.4368` to `0.5000`, Top-3
from `0.5920` to `0.6494`, and 18 of 28 changed actions improved while nine
worsened and one tied. Pairwise accuracy, exact-best, Top-3, and both topology
subgroup gates passed; only the regret-improvement gate failed.

The final train-only calibration chose probability thresholds `0.55` for base
candidates and `1.01` for Boundary candidates. Since a probability cannot
exceed one, the latter conservatively disables Boundary overrides and records
that their cross-map confidence is not yet reliable. On the already observed
six-map validation cohort the same frozen model descriptively reduced regret
from `0.187595` to `0.145574`, with 10 better, seven worse, and three tied
changes. That cohort did not select thresholds and cannot authorize promotion.
All runtime bundle selections matched the reference implementation on 174
train and 66 legacy-validation states, but `shadow_eligible` remains false and
frozen V2 remains the default.

The config, training report, offline predictions, and diagnostic controller
manifest SHA-256 values are respectively
`b243f71d226c84f5658f842cf2ff1dce561f27392503617a4f265a9ca7bc7aca`,
`3c22957a5a0a22f876191844380e505b71cfcc87f9e185ca1381b8a76b2976d8`,
`4885e225e410f2524235e190353d67be93f3560c514ce539a26c123a8aa6aa44`,
and `43dad5ad0ef1d1eb4b3368097e6f37b43f0531ef1d5bf5cab445f1161343038d`.

### Registered fresh-map GuardRank training-expansion preflight

The failed `stride-guardrank-v1` gate does not authorize threshold relaxation
or Shadow. Its next step is an outcome-blind load preflight on eight
checksum-pinned DAO maps that did not occur in the current 22-map repairability
cohort or the 12-map formal-OOD registry. A `git grep` audit at predecessor
commit `0a79abb45adccede1a42a8b534ab5a8505a79f80` also found none of the eight
map IDs in the registered configs or this protocol before registration.

The strata are compact high-topology (`lgt101d`, `lak105d`, `orz106d`),
compact mid-topology (`den405d`, `den407d`, `lak103d`), and low-topology
controls (`oth999d`, `isound1`). Their registered low-degree-cell ratios range
from approximately 0.003 to 0.130. Each map uses one new task seed, both
uniform-random and opposite-exchange OD variants, and three map-scaled agent
loads. Two solver seeds give 48 tasks and 96 reset-only jobs.

The preflight stops after initialization (`max_decisions=0` and
`max_repair_iterations=0`). It may read only map topology, load, initial
conflicts, initial PP time, and initial path statistics; candidate repairs,
controller actions, relative TTF, and controller outcomes are forbidden. A
map qualifies at mean initial conflicts of at least 10. At least six maps must
qualify, including two of three high-topology maps and two of three
mid-topology maps; only the two low-topology controls may be underloaded. At
most two tasks per map are then selected against 25/100-conflict targets. A
pass permits a separately registered proposal/state-collection design, not
repair labels, retraining, Shadow, or a speed claim.

The source and reset-runtime config SHA-256 values are respectively
`2de44eea6ba4ed9b5057c7eb8811e2af06a19784348fcf998cecd77149b591d0`
and `91ffea3a5c64f9b69d71393e7cb0b8882d94e13e9dd7c3e33d93971daea85d8d`.

### Completed fresh-map GuardRank training-expansion preflight

The reset-only collection completed all 96 registered jobs with zero errors,
zero timeouts, no candidate repairs, and no controller outcomes. All eight maps
qualified: 3/3 compact high-topology, 3/3 compact mid-topology, and 2/2
low-topology controls. Across all reset rows, initial conflict count had mean
327.75, median 27.5, and range 0--3018; 26 of 96 reset rows were initially
feasible. The analyzer passed every isolation, coverage, group, and forbidden-
outcome gate and selected two tasks per map (16 tasks, 32 paired solver-seed
states) for a proposal-only audit.

The selected mean conflict counts were `den405d` 12.5/126.5, `den407d`
71.5/90.5, `isound1` 617.0/1743.5, `lak103d` 32.0/477.5, `lak105d`
11.0/15.0, `lgt101d` 26.5/103.5, `orz106d` 45.0/312.5, and `oth999d`
10.0/158.0. These are load-qualification statistics, not repair-quality or
TTF results.

The dataset manifest, qualification manifest, qualification report, and
preflight report SHA-256 values are respectively
`1298c20deb2e05dbab756aea4d0c13c9f119b857f0ae8edf0221acb31e1653c7`,
`b42e03e9be57f87141263230998e1e2bbceda1ea1be5846d69d1ea31a279271b`,
`4c6815b112d14676eb6c76b8820115244e4c5252a1415dff868a3939272e3063`,
and `61e147414f433b25865c7861bf22159cc697bb34bfd43f8bc1132cfe9d35cdf4`.

### Registered fresh-map GuardRank candidate coverage

`stride-guardrank-mapcoverage-v1` freezes the existing V2 candidate pool plus
at most two `stride-topoboundary-v1` candidates on each of the 32 selected
initial states. It repeats proposal generation twice to require deterministic
candidate identities and preserves the initial-state fingerprint. Requested
sizes remain 4/8/16, the total candidate count must stay in 12--20, and the
audit measures articulation/low-degree incident coverage, boundary ratio, and
component reach against the best base candidate frontier.

This stage remains proposal-only: PP repair trials, controller actions,
controller outcomes, labels, runtime export, default replacement, formal OOD,
and speed claims are all forbidden. A pass only authorizes registration of a
separate multi-PP-seed repair collection; a failure forbids collecting new
repair labels from this pool. The registered config SHA-256 is
`edcdaa2b59ae8fa93a410df7aa2b77df9f0db0d7ab6c67c001542ad3348b250f`.

### Failed fresh-map GuardRank candidate coverage

The proposal-only audit replayed all 32 selected states without errors. State
fingerprints, solver-seed pairing, requested sizes, repeated proposal
determinism, and the 12--20 candidate cap all passed. It produced 586 total
candidates, of which only 13 were new topology-boundary candidates.

The registered pool failed three gates. Only 8 states contained articulation
events and only 7 contained low-degree events, below the registered 16-state
low-degree minimum. The low-topology controls contained zero relevant topology
events and therefore failed their registered low-degree relevance requirement.
Finally, the boundary candidates improved global event incidence by 0.3640 and
conflict-component reach by 0.2495, but reduced global boundary ratio by
0.1240 relative to the best base frontier, failing the non-negative delta gate.
Their conditional incident coverage remained high (articulation 0.9384,
low-degree 1.0000), as did their conditional boundary ratios (0.7043 and
0.8571). Thus the failure is primarily sparse applicability plus excess
internal closure relative to the base frontier, not nondeterminism or state
corruption.

No repair trials or labels are authorized from this failed 32-state pool. The
next safe diagnostic is an input-only topology-relevance scan over all 96
already qualified reset states. It must determine whether coverage was lost by
the conflict-target task selection or is intrinsically absent on these maps;
low-topology controls must be evaluated as abstention controls if they remain
event-free. The report, state-row, and candidate-row SHA-256 values are
`9db4231f230999a592a457c127d861c7e707c5ff65cccc8a4255512ba79149c6`,
`b111e2ac7ebeb2cad994b8415143de0770b97f1da43da6586545df558e756d60`,
and `1f7fd000acde801a524518ce60f7a0a61a5100d810e493ad36343108a0eadd75`.

### Registered all-state GuardRank topology relevance scan

`stride-guardrank-relevance-v1` replays all 48 tasks and both solver seeds from
the passed reset qualification, giving 96 initial states. It reads only the
map, initial paths, initial conflicts, and static-topology interaction of those
conflict events. Candidate generation, candidate repair, controller actions,
controller outcomes, labels, and speed claims are forbidden.

The failed 32-state selection had any topology relevance in 11/32 states
(0.34375), or 11/24 (0.45833) after excluding the low-topology controls. The
scan attributes the failure to task selection only if the complete high/mid
pool improves that fraction by at least 0.10. A control relevance fraction no
greater than 0.25 confirms that the control maps should measure topology
abstention rather than topology-event coverage. The scan also preregisters an
input-only recommendation of at most four tasks per map: high/mid maps prefer
paired-seed event relevance and event count, while controls prefer zero-
relevance states with substantial initial conflicts.

A valid scan is diagnostic, not an authorization to reuse the failed pool.
Its result determines whether the next separately registered collection should
be topology-enriched or should keep Boundary optional while prioritizing
map-diverse base candidates. The config SHA-256 is
`05a847c730546c6275ef167b0471491f4952cd9c55a9b453c36feb0f8aa360b7`.

### Completed all-state GuardRank topology relevance scan

The input-only scan replayed all 96 states and passed every integrity gate.
It found 33 relevant states (0.34375). Excluding the low-topology controls,
the high/mid relevance fraction was 0.45833, exactly equal to the failed
32-state selection; the preregistered full-minus-selected delta was therefore
0.0 rather than the 0.10 required to attribute undercoverage to task selection.
Both low-topology control maps remained entirely event-free, giving a control
relevance fraction of 0.0 and confirming their role as abstention controls.

This rejects the hypothesis that conflict-target selection caused the sparse
Boundary coverage. Under the current articulation/low-degree definitions,
topology applicability is intrinsically sparse on these fresh maps. Boundary
must remain optional; the next data extension should prioritize repair labels
for map-diverse frozen base candidates and must not reuse the failed 32-state
Boundary pool as though it had passed. No candidates, repairs, actions,
outcomes, or labels were produced by this scan.

The state, task-summary, and input-only recommended-task SHA-256 values are
`88e2425f77a7ae0628ded3b77d436f38c9cec1bb35642077d559d08cbc3ce939`,
`9a1c09cdf577eb55b1de0373831b27333296164c410f75cb42758ed208126050`,
and `3c8755475f7c224c699639622303b8f70d2364482c862142c85b893786e414d8`.

### Registered STRIDE-MapBase v1 repair collection

The next data artifact is separately named `stride-mapbase-v1`. It consumes
the input-only 32-task recommendation, both solver seeds, and excludes the one
initially feasible row, yielding 63 independent active initial states across
all eight fresh maps. Every row is assigned to the training split; none of the
new maps are used to tune a validation threshold or enter formal OOD.

The candidate pool is the frozen V2 base pool only: target, collision, and
random families at requested sizes 4/8/16, with 12--18 deduplicated candidates
per state. Topology Boundary is explicitly forbidden because its fresh-map
coverage gate failed. Each candidate receives trial indices 0--15. A trial
index maps to the same deterministic PP seed for every candidate in the same
state, so candidate comparisons remain paired while exposing PP uncertainty.
Every trial must carry the frozen 124-dimensional realized feature schema.

The collector restores the exact initial repair structure before each branch,
never executes a controller action, and records repair success, conflict
change, structural post-state, native step time, and PP time. Completion is not
sufficient by itself: the registered audit must verify 63 state artifacts,
base-only families, candidate caps, exact 16-seed coverage, paired seeds,
feature names, consolidated JSONL equality, training split, zero errors, and
artifact SHA-256. This remains training data and makes no TTF claim. The config
SHA-256 is
`abc0495fb7e0530a731116d2bf5e0c8d9dcbe76700e040907550ebe535d6c4fb`.

### Registered STRIDE-MapRank v1 successor design

Before inspecting the STRIDE-MapBase collection summary or constructing its
labels, `stride-maprank-design-v1` freezes the separately named successor.
The combined label artifact will be `stride-maprank-labels-v1`, and the model
and controller identity will be `stride-maprank-v1`; neither overwrites frozen
V2, STRIDE-Quality, AugControl, or GuardRank artifacts.

The design expects 240 audited earlier states plus 63 audited MapBase states:
237 training states over 24 maps and 66 legacy-validation states over six
maps. The new eight maps are training-only. The primary label remains the
current-step conflict reduction normalized by conflicts before repair, using
16 paired PP seeds, a 75% paired-win requirement, a 0.02 minimum absolute
effect, consistent direction in both seed halves, and equal total weight per
state. Runtime, TTF, future repair rounds, Cost-to-Go, Receding-Q, controller
outcomes, test data, and formal-OOD data remain forbidden label inputs.

The ranker is a frozen-V2 anchor plus a 124-feature conflict-only challenger.
Calibration remains nested and train-map-only (four outer folds, three inner
folds); the six old validation maps are descriptive and cannot tune the
threshold. The existing offline gates are not relaxed. Only a passing offline
result can proceed through fresh-map Shadow, paired high-load raw TTF, and
paired fresh-map raw run-to-completion TTF. This registration is a design
boundary, not a performance or speed claim. The frozen design SHA-256 is
`54784badfd210019595fdebcc0e06b98564c2c4a4fb8f73c82f60fecd15082cc`.

The outcome-blind selection merge was then materialized before label analysis.
It contains the registered 303 states: 237 training states on 24 maps and 66
legacy-validation states on six disjoint maps. All 63 MapBase states and all
eight new maps are training-only; 47 new states are in the registered
boundary-relevant topology group and 16 are controls. Neither candidate repair
outcomes nor controller outcomes are selection inputs. The combined selection
SHA-256 is
`14644c37c85b16796c053a1a001a8f0e52bb59b07d6a1beba2fa7a0302e263c3`.

The executable training contract is also registered under the distinct
`stride-maprank-v1` controller and `lns2.stride.maprank_strategy.v1` runtime
schema. It reuses the tested V2-anchor guard mechanism without changing the
GuardRank identity or artifacts. Model capacity, four-by-three nested map
folds, threshold grid, 0.06 topology grouping boundary, and all offline gates
match the frozen MapRank design. The runtime loader accepts MapRank only when
the bundle manifest, diagnostic-only status, controller identity, strategy
schema, thresholds, and anchor rankers all match. No label construction or
training result is implied by registering this executable path. The training
config SHA-256 is
`db84f0a5a38d433dc49aa926140a69768f5528c83ae305599e6648b526792760`.

The registered label builder is likewise MapRank-specific at the artifact
boundary. It accepts only the two source paths frozen in the design, requires
their passed audit reports, writes only to `build/stride-maprank-labels-v1`,
and rechecks 303 states, 24/6 train/legacy-validation maps, 16 trials per
candidate, and the runtime/future-input exclusions before emitting a separate
MapRank label report. The underlying conflict-only label formula and row schema
remain unchanged, so the new name means expanded audited data rather than a
post-hoc change to label meaning.

The downstream evaluation is preregistered before MapRank training outcomes.
It has no scientific, environment, or episode-process time limit and uses the
reset-inclusive run-to-completion raw wall TTF clock. Every episode is run in a
strict rotating triplet across frozen V2, V2 with the same augmented candidate
pool, and MapRank, with deterministic PP replay and one worker. A legacy
held-out execution Shadow is only a sanity layer. The registered high-load
development layer uses the two maze-300 and two room-500 tasks with four seeds;
the fresh-map layer uses six untouched maze, room, random, warehouse, den, and
lak cohorts with three seeds. The ranker must beat the same-pool V2 comparator,
not merely benefit from extra candidates, while success, repair iterations,
per-cohort/map regressions, invalid actions, fingerprints, AUC, PP time, and
selector overhead remain explicit gates or required metrics. Execution remains
forbidden unless the offline gate passes. The evaluation config SHA-256 is
`e6ef4a6ff50965b733a313141db22a356ec12511c0ce6154d341169ab0e61430`.

### Completed and audited STRIDE-MapBase v1 collection

The collection completed all 63 registered states with zero errors and no
resumed rows. It produced 18,048 repair trials. Candidate counts were 16 on two
states, 17 on two states, and 18 on 59 states; every candidate was a frozen
target/collision/random base candidate and the Boundary candidate count was
zero. The audit sorts state payloads by state identity, matching the producer's
deterministic consolidated order rather than filesystem hash-name order; this
ordering correction changes no collected row.

All audit gates passed: complete state and artifact coverage, zero collection
errors, base-only families, no topology families, registered candidate caps,
exact candidate-by-trial coverage for indices 0--15, one paired deterministic
PP seed per state/trial index, the exact 124-feature schema, all-train split,
trial count, consolidated JSONL identity, and a nonempty run fingerprint. The
run fingerprint is
`e631303b542e16f70a8d5475f9f865765ced9c90b92e5938946f8e089f46afa0`.
The repair-trial, collection-report, and run-config SHA-256 values are
`34976a38a0a95e11c7c7c4dc5997e27fd4e260636727c8cea2d6e5a56cdeead2`,
`2564ddc07a9827109352062603497f2eba21a9774b0a04ac51380dfb111f8b4f`,
and `aa25e4a27d26f7b42fc5c216e77a5151682d6c12bfe394a4c6f93048aeea0234`.
This is an audited training-data milestone, not a model or TTF result.

### Built STRIDE-MapRank v1 labels

The audited two-source build produced 303 independent states, 5,513 candidate
aggregates, 22,779 robust primary repairability pairs, and 21,033 robust
conflict-only pairs. A total of 295 states have at least one primary robust
pair. The train/legacy-validation map counts are exactly 24/6, and runtime and
future trajectory inputs remain absent.

Relative to the earlier 240-state artifact, MapBase adds 63 states, 1,128
candidates, 4,515 primary robust pairs, and 4,165 conflict-only robust pairs;
all 63 added states have trainable pairs. On the new states alone, seed-half
exact-winner agreement is 0.7619, mean Top-3 overlap is 0.8413, pairwise
direction agreement is 0.9212, and mean candidate-score standard deviation is
0.0510. The combined values are 0.6073, 0.8284, 0.9101, and 0.0690. These are
label-stability diagnostics, not selector or TTF improvements.

The Windows-training-environment candidate aggregate, conflict-only pair, and
label-summary SHA-256 values are
`1a72ab8b0c9df89995ec0cc18f8e9f04ab8bae0261aa6ea0895a540ae688da12`,
`9e3ce3896e61b67e1b5794c22820e91553256a8ff4a3970e7da289fee1b645e7`,
and `fc2222a5c18746c3890f1eb9f60012b78bf94b6cb2368f5cabbfe741ba2d68dd`.
An independent WSL reconstruction has identical feature vectors, means, and
conflict-only pair bytes. Its only candidate-row differences are the final
floating-point bit of two descriptive standard-deviation fields (maximum
absolute difference `1.11e-16`), neither of which is a model input.

### Trained STRIDE-MapRank v1 and passed the offline gate

Nested train-map OOF covers 237 states over 24 maps. MapRank normalized regret
is 0.25687 versus 0.27040 for the frozen V2 anchor: an absolute improvement of
0.01353 and relative improvement of 5.0041%, just above the preregistered 5%
gate. Exact-best rate is 0.4430 versus 0.3924, Top-3 hit rate is 0.6371 versus
0.5654, and pairwise accuracy is 0.8229 versus the 0.5 weighted-majority
baseline. All exact, Top-3, pairwise, regret, and topology-group gates pass.
The boundary-relevant group regresses by 0.00565 normalized regret, below the
allowed 0.03; the control group improves by 0.04026.

The train-calibrated thresholds are 0.60 for base challengers and 1.01 for
Boundary challengers. Thus this bundle changes only sufficiently confident
base-candidate choices and abstains from Boundary overrides, consistent with
the failed fresh-map Boundary coverage audit and base-only MapBase extension.
It changes 64/237 OOF selections: 35 improve, 22 worsen, and seven tie.

The six-map legacy validation remains descriptive only. Its normalized regret
is 0.13134 versus 0.18759 for V2, exact-best is 0.6212 versus 0.5909, and Top-3
is 0.7727 versus 0.6515. Train and legacy runtime bundle equivalence both have
zero selection mismatches. Every integrity gate passes; test and formal-OOD
data were not read. Therefore `stride-maprank-v1` is eligible for fresh
development evaluation, but it is not a default replacement and no TTF claim
has been made.

The training report, controller manifest, and offline-prediction SHA-256 values
are `496144d4a5dc4ca5c47c5f8b4ba7c67a8f669bee19b8479d42138a35ef7543a4`,
`1f367902d21530c531b58859814b28cf48bbfbb861d46c7377867c962e581054`,
and `c14b608e857c2b474d103dd81769678ee9497faf71061753634765412b0ad7bb`.

### Passed the audited STRIDE-MapRank v1 action-preserving Shadow

The legacy held-out Shadow completed all 24 registered episodes after a fresh
run-to-completion qualification. Frozen V2 with the topology-augmented pool was
the only active controller; MapRank was diagnostic-only and could not override
an action. Across 421 paired decisions, MapRank selected a different candidate
209 times, a disagreement fraction of 0.49644. All nine report gates passed:
complete episode coverage, fresh qualification, at least 24 Shadow decisions,
zero execution errors, zero invalid actions, zero semantic mismatches, zero
action overrides, and recorded run and controller-implementation fingerprints.

The runtime boundary now loads diagnostic and active GuardRank-family selectors
through the generic selector contract rather than constructing every diagnostic
bundle as a pairwise-V2 selector. The exact MapRank manifest is registered in
both later runtime configurations. The audited rerun was produced by pushed
commit `6f40051`; its run fingerprint is
`6fa4185f70a2df3cadf6cc2d49c9ff034ec0ffd026b7ecd12ce179e00ffa9f09`,
and its controller-implementation fingerprint is
`ac4d4e40624b01fc435302c574bb39afb81fa392bbd5022bcd554aa571c44608`.

The Shadow report, policy manifest, qualification report, and run-config
SHA-256 values are
`24691648165ff7de3413012118fb23a991c35cf79046e76599fd9e7944945e89`,
`515a2f1ae0bb5efbe955edb91ddc231e95b3c84936806db4e343d735a19773af`,
`acb631288daed0ed0eb9c4cefff35395cad50ce4c3b0e1d2e60215b68a42b5c0`,
and `69447d404f77a1cd7abf7b8c4a6d27d3ceaaf28aa53c91604c5287ef29ce8d4a`.
This proves runtime validity and meaningful action disagreement only; it makes
no TTF improvement claim and leaves V2 as the default. Passing this gate permits
the preregistered paired high-load raw-TTF layer.

### Failed STRIDE-MapRank high-load v2 and registered runtime-only v3 rerun

The first complete high-load run-to-completion evaluation (`high-load-v2`)
passed every integrity gate: 48/48 rotating schedule entries, 16 successful
episodes per controller, matching initial fingerprints and conflicts, the
registered raw-TTF clock, no capped values, and zero execution, action, or
semantic errors. It failed the performance gates. Mean raw wall TTF was
12.785405 seconds for frozen V2, 14.122425 seconds for V2 over the augmented
pool, and 14.277750 seconds for MapRank. The total MapRank method therefore
regressed 11.6723% against V2 and the ranker regressed 1.0998% against the
same-pool comparator, despite reducing mean repair iterations from 21.125 to
12.3125. Maze-300 regressed 2.859% and room-500 regressed 23.586%.

The timing decomposition identifies candidate generation, specifically the
unconditional Topology Boundary analysis, as the dominant avoidable cost.
Mean topology cost per MapRank decision was about 0.4063 seconds on maze-300
and 0.1549 seconds on room-500; model inference was much smaller. The
high-load-v2 report SHA-256 is
`e88a99b0265159dc55de592223da0d3ed1a899951ba8242b54c388699a8ea696`.
This result does not authorize reading the fresh-map raw-TTF layer.

Before running another high-load outcome, `high-load-v3` is registered as an
implementation-only rerun of the same evaluation config (SHA-256
`e6ef4a6ff50965b733a313141db22a356ec12511c0ce6154d341169ab0e61430`).
The 48-entry schedule, datasets, tasks, solver seeds, controller bundles,
candidate definitions, 124 features, model scores, labels, thresholds,
run-to-completion clock, comparison gates, and deterministic PP replay are
frozen. Only two behavior-preserving computations may change: Topology
Boundary candidate scoring may index incident conflict events instead of
rescanning all events, and topology analysis may incrementally update the
temporal conflict index for repaired agents.

Promotion to the fresh-map layer requires both the original high-load
performance gates and a separate exact runtime-equivalence audit against
high-load-v2. That audit must match all 48 episodes' scientific inputs,
candidate pools and scores, selected candidate IDs, explicit actions, PP
seeds, state fingerprints, conflict trajectories, repair iterations, and
final solver state while excluding only timing fields. A mismatch invalidates
the optimization. A performance failure retains frozen V2 and keeps fresh
maps unread; it cannot be hidden by the already observed reduction in repair
rounds.

### Completed high-load v3 and registered incremental-event v4 rerun

`high-load-v3` completed all 48 schedule entries with all 48 episodes
successful, zero execution errors, zero invalid actions, zero semantic
mismatches, matching initial states, and no capped-TTF values. The independent
runtime-equivalence audit matched all 48 episodes and 735 decisions against
`high-load-v2`: scientific inputs, candidate pools and scores, selected
candidate IDs, explicit actions, PP seeds, fingerprints, conflict
trajectories, repair rounds, and final states were exact. Its report and
equivalence-report SHA-256 values are
`d3357f1fd6d7a7cc36305b6afa6dc7f8d74326beb66715187a20dc0f428c58ad`
and
`a7a521cb3ae5cf892d14f30689c905b551f83a9d382e2c66161fede51cd55ca5`.

The optimization materially reduced mean MapRank selector time from 2.6151
to 1.6405 seconds and improved the total regression versus V2 from 11.6723%
to 4.9035%. It nevertheless failed the preregistered total-method gate, which
requires at least a 2% improvement. Mean raw TTF was 10.691592 seconds for
V2, 11.283484 for V2 over the augmented pool, and 11.215851 for MapRank.
MapRank passed the same-pool ranker gate by improving 0.5994%, and repair
iterations remained much lower (12.3125 versus 21.125), but the augmented
pool itself still cost 5.5361% relative to V2. Fresh-map raw TTF therefore
remains forbidden.

The remaining measured bottleneck is topology dynamic-event reconstruction,
not feature extraction or model inference. Before another outcome is read,
`high-load-v4` is registered under the unchanged evaluation config and gates.
The only permitted implementation change is to retain the exact temporal
conflict-event set across repairs: when the path horizon is unchanged, remove
events incident to repaired agents and regenerate only those agents' vertex
and edge events; when the horizon changes, fall back to a complete event
reconstruction. Candidate scoring, ordering, tie-breaking, features, models,
thresholds, controller decisions, PP replay, datasets, and the 48-entry
schedule remain frozen.

The v4 result is valid only if targeted randomized tests match complete event
reconstruction across path and horizon changes, the full test and repository
audits pass, and a post-run exact action-equivalence report matches v3 for all
48 episodes. The original high-load performance gates remain unchanged. Only
an integrity, equivalence, and performance pass may unlock the fresh-map
layer.

### Completed high-load v4 and registered native-event v5 rerun

`high-load-v4` again passed every integrity gate and its exact 48-episode,
735-decision equivalence audit against v3. The raw-TTF and equivalence report
SHA-256 values are
`779a10f9668423de8428aba9bf0b21e1c1bbe0d0dae290b38e6e3a3120cb1492`
and
`b8cbfa545821f40ea39fb46b1ebe1fccdd204a4887c9816593de2b5d52a7fbfb`.
MapRank selector time fell again, from 1.6405 to 1.1190 seconds. Mean raw TTF
was 10.771383 seconds for V2, 11.518072 for the same augmented pool, and
11.013455 for MapRank. The ranker beat its same-pool comparator by 4.3811%,
and room-500 beat V2 by 3.3455%, but maze-300 regressed 6.7604%. Overall
MapRank still regressed 2.2474% rather than meeting the required 2%
improvement, so the performance gate failed and fresh maps remain unread.

The remaining maze cost is the first full Python temporal-event scan. On a
stored maze-300 initial state, the Python topology analysis took about 494 ms
per construction in a local diagnostic, while an exact C++ scan took 17.5 ms;
the corresponding room-500 values were about 138 ms and 5.76 ms. These are
microbenchmark timings, not TTF claims.

`high-load-v5` is therefore registered before its outcomes as another
implementation-only rerun under the unchanged config, schedule, models,
candidate definitions, and gates. The existing native feature module may
expose the full sorted topology event sequence `(time, kind, left, right,
cells)` and the topology cache may consume it. The Python incremental event
index remains the exact fallback when the native function is unavailable.
Random valid-path tests, including horizon changes, must match Python full
reconstruction event-for-event. The environment audit must verify the new
native callable, and v5 must pass the full suite, all original high-load
integrity/performance gates, and exact action equivalence against v4 before
the fresh-map layer can run.

### Completed high-load v5 and registered three-run timing confirmation

`high-load-v5` completed all 48 schedule entries with every episode
successful, zero execution, invalid-action, or semantic errors, matched
initial states, and no capped-TTF values. Its exact action-equivalence audit
against v4 passed all 48 episodes and 735 decisions. The raw-TTF and
equivalence-report SHA-256 values are
`3d2453e68e47bb07d6871d6bc56c4cda76592ced0b2f5db492df924de92f03aa`
and
`52f32e46ad34a7731f3289dd40016bb54c6d82262a1001fb557489a5a73887d6`.

The native event extraction reduced mean MapRank selector time from 1.1190
seconds in v4 to 0.4342 seconds. Mean raw wall TTF was 11.091729 seconds for
frozen V2, 10.334541 seconds for V2 over the same augmented pool, and
10.453622 seconds for MapRank. The complete MapRank method beat V2 by 5.7530%
and reduced mean repair iterations from 21.1250 to 12.3125. Maze-300 improved
5.1445% and room-500 improved 6.4846%. However, MapRank was 1.1523% slower
than the same-pool V2 comparator, so the unchanged ranker-effect gate failed
and this single run does not unlock fresh maps.

The failed same-pool comparison is only 0.1191 seconds in aggregate. The two
controllers differ by about 0.006 seconds of measured selector time, while
their initial-solution and PP wall timings vary by more than the observed
gap. Moreover, v4 and v5 execute identical actions and PP seeds, yet the
same-pool wall-clock comparison changed from a 4.3811% improvement in v4 to a
1.1523% regression in v5. This is direct evidence that a single wall-clock run
is not sufficiently stable for this narrow gate; it is not permission to
relax or replace the gate.

Before reading two additional outcomes, the fixed three-run confirmation is
registered in `configs/stride_maprank_high_load_confirmation.json`, SHA-256
`e5b56b430fb23b925614c774427c860809d513653f4271fd7fc0b6309331eed5`.
It pools the completed v5 run and two new strictly rotated repetitions of the
same 48-entry schedule. All three runs must have identical evaluation and
schedule hashes, the same controller implementation fingerprints, complete
integrity, and exact action equivalence. The pooled 48 episodes per controller
must satisfy every original `high_load_development` performance threshold;
thresholds, tasks, seeds, models, candidate pools, and labels remain frozen.
Only a pooled pass may unlock fresh-map raw TTF. This outcome-informed timing
confirmation improves wall-clock precision but is not a formal speed claim.

### High-load timing confirmation failed the ranker-effect gate

Both added repetitions completed all 48 schedule entries. Pooling all three
runs gives 48 strictly paired episodes per controller. Every integrity gate
passed: all controllers succeeded, the evaluation and schedule hashes are
identical, initial states and conflicts match, the raw-TTF clock is uncapped,
and there are zero execution, invalid-action, or semantic errors. Each new
repetition is exactly equivalent to v5 across 48 controller episodes and 735
decisions, including candidate scores, actions, PP seeds, conflict
trajectories, repair rounds, and final states; all implementation fingerprints
are identical.

Mean raw wall TTF is 11.377811 seconds for frozen V2, 10.761304 seconds for V2
over the augmented pool, and 10.841746 seconds for MapRank. The augmented pool
therefore improves 5.4185% over V2, while the complete MapRank method improves
4.7115% and reduces mean repair iterations from 21.1250 to 12.3125. MapRank is
faster than V2 on 32/48 paired observations. It nevertheless regresses 0.7475%
against the same-pool V2 comparator, an 0.08044-second mean gap, despite a
0.1875-round reduction. The ranker effect is heterogeneous: maze-300 regresses
2.1962%, while room-500 improves 1.0413%. The three individual same-pool
ranker effects are -1.1523%, -1.2486%, and +0.1231%, confirming that the gap is
small but not reliably positive.

Thus four of five performance gates pass, but the unchanged same-pool ranker
gate fails. The confirmation report SHA-256 is
`f8ea8c07f80465182af16b1b087ebb0861fcfaf23598b2fb692fdd41c29859fe`.
Fresh maps remain unread, V2 remains the default, and no formal speed claim is
authorized. The next permitted work is a read-only decision-level diagnosis of
the frozen high-load traces, especially maze-300, before any new model design
or additional outcome run is registered.

The diagnosis is registered before inspecting decision-level differences in
`configs/stride_maprank_high_load_failure_analysis.json`. It uses v5 only as
the canonical trace copy because the two confirmation repetitions are exactly
action-equivalent, and uses all three repetitions only to average episode TTF.
For each of the 16 task/seed episodes it records the common action prefix and
first shared-state override, the two selected candidates' family, size, and
cross-model scores, immediate conflict reduction, repair success, and
deterministic low-level effort, plus eventual repair-round and pooled-TTF
differences. It does not use TTF or future trajectories as a training label,
does not inspect fresh maps, and cannot promote a model. Its sole purpose is to
separate a bad ranking decision from residual wall-clock noise and identify a
bounded next design change.

### Decision-level failure decomposition and paired-seed counterfactual

The registered trace diagnosis passed every integrity gate and read no fresh
map data. Ten of 16 episodes execute identical V2-augmented and MapRank action
sequences; only six contain an override. Three overrides occur on maze-300 and
three on room-500. On room, two first overrides reduce more conflicts and one
ties; two reduce eventual repair rounds and none increase them. On maze, no
first override reduces more conflicts, one ties, and two are worse. The
failure-analysis report SHA-256 is
`f8bcf676321eacf0e49fbfc59bfc078b1fbe75cc31d8c643e71291b6ce2e9659`.

One early maze override dominates the observed regression. On
`maze-128-128-2__random_17__agents_0300`, solver seed 3, MapRank replaces the
V2 `collision:8` candidate with `collision:16` at decision zero. Under the
recorded PP seeds, conflict reduction falls from six to one, low-level
generated nodes rise by 225,093, PP time rises by 0.936 seconds, the episode
needs two extra repair rounds, and its three-run mean TTF rises by 1.578
seconds. By contrast, room overrides occur late and are generally useful.
This localizes the failure to rare override quality on maze rather than the
augmented candidate pool or universal selector overhead.

The recorded runtime repair is only one PP realization per candidate, whereas
the MapRank label was trained from 16 paired PP seeds. Therefore the evidence
does not yet distinguish a wrong robust ranking from unlucky repair
realization. The targeted counterfactual is preregistered in
`configs/stride_maprank_override_counterfactual.json`, SHA-256
`3300198a4914360f10508c81a03c1f1b4a6960720c6c3bd27efd13c0e2cae0c7`.
It takes all and only the six observed first-override states, reconstructs each
pre-action state without reading the target action outcome, and evaluates only
the V2-selected and MapRank-selected neighborhoods under trial indices 0--15
with identical paired PP seeds. The primary comparison is normalized immediate
conflict reduction under the unchanged robust-pair rule (75% paired wins,
0.02 mean effect, and consistent half directions). Repair runtime is
descriptive only; future trajectories, fresh maps, and TTF labels are forbidden.
This is a post-hoc mechanism diagnosis and cannot promote a controller.

### Override counterfactual identifies missing uncertainty abstention

The first collection completed all 192 repairs but wrote the report and
collection identity with their schemas interchanged. That V1 report is rejected
as evidence and preserved for audit. After the schema correction was fully
tested and pushed, V2 reran all repairs in a new directory. Removing only
descriptive PP timing and producer-identity fields, V1 and V2 match on all six
state artifacts with zero scientific mismatches.

V2 passes every integrity gate: six exact reconstructed states, two selected
candidates per state, 16 paired PP seeds per candidate, 192 total trials, no
fresh-map access, no future-trajectory use, and no runtime in the robust label.
Across the six overrides, MapRank's candidate is the robust winner twice,
V2's candidate once, and the pair is uncertain three times. Both robust
MapRank wins are on room-500; the third room override is a robust V2 win. All
three maze-300 overrides are uncertain under the unchanged 75% paired-win,
0.02 effect, and two-half-direction rule.

For the dominant maze failure, `collision:16` averages 0.40625 normalized
conflict reduction versus 0.51042 for V2's `collision:8`, uses 283,620 versus
219,256 mean generated nodes, and progresses on 68.75% versus 93.75% of
paired seeds. But it wins four seeds, loses six, and ties six, so the pair is
not a robust V2 win either. The single runtime seed exposed a real bad tail,
while the 16-seed result says the correct training target is uncertainty and
abstention, not a reversed direction label.

The current MapRank direction model trains only on robust oriented pairs; it
discards uncertain pairs and then treats a high direction score as sufficient
override evidence. It therefore has no learned mechanism for distinguishing a
confident robust preference from confident extrapolation on an unstable pair.
The V2 report, trial, and run-config SHA-256 values are
`65740040d88bf82f5b353b59469f7acdcfc57bf179e4c57bc9a8e3719fdc98a8`,
`b905550131d0d40816eb7f5e79e7259df68c8334e84a5015d34a568b685bf3a6`,
and
`fd5946e8e99b1ab4c68f43d8418a047c7cc0f06de4c135f7a0b6754dafaca7b8`.
These are post-hoc mechanism findings, not training data and not a speed claim.
Fresh maps remain locked.
## 2026-08-04: STRIDE-CertGuard v1 preregistration

The three-replicate MapRank high-load result failed only the ranker-versus-same-pool
gate.  Its paired 16-seed first-override counterfactual then showed that the
direction ranker is not the same thing as an override-confidence estimator:
MapRank training keeps only robust oriented pairs and drops every uncertain pair,
while runtime had no independently learned way to abstain on those dropped cases.

`stride-certguard-v1` is therefore registered as a separately named successor.
It freezes the V2 anchor, MapRank candidate pool, MapRank direction semantics and
deployed direction bundle.  It adds one symmetric binary classifier whose target
is whether an unordered pair has a robust current-step winner under the original
16 paired-PP-seed rule.  Its input is the 124-dimensional absolute candidate
difference plus the 124-dimensional pair mean.  The model may only veto a
MapRank override; it cannot invent an action or bypass V2.

All fitting and certainty-threshold calibration are map-grouped and train-only.
Legacy validation is descriptive.  The already inspected maze300/room500 cohort
is post-hoc development evidence and is forbidden for labels or thresholds.
Fresh-map raw TTF remains locked until nested OOF, runtime equivalence, Shadow and
high-load development gates pass.  Runtime, TTF, repair rounds, future trajectory,
Cost-to-Go, Receding-Q, controller outcome, test and formal OOD inputs remain
forbidden label features.

### CertGuard v1 nested OOF result: rejected before runtime

The checksum-audited label build reproduced all 303 states and 47,471 unordered
pairs from the frozen MapRank sources: 21,033 robust positives and 26,438
uncertain negatives.  No high-load, runtime or future-trajectory field was read.
The label file SHA-256 is
`2cddea6cc7b03a965b9c1aa706768f81610ae6fc02c0b904368347aeda618eca`.

The symmetric uncertainty task itself was learnable under four outer by three
inner map-grouped OOF: weighted ROC-AUC was 0.74519 and weighted accuracy gained
0.11713 over the weighted majority baseline.  This did not make it a safe action
gate.  The nested controller retained 18 of 76 MapRank overrides; seven were
robust in the proposed direction, four were robust in the opposite direction,
and seven were uncertain.  Directionally safe precision was therefore 0.38889,
below the registered 0.50 gate.

CertGuard normalized regret was 0.26343 versus 0.24846 for the same OOF MapRank
direction controller and 0.27040 for frozen V2.  It improved V2 by 2.576%, but
degraded MapRank by 0.01497, exceeding the registered 0.005 limit.  A read-only
threshold sweep confirmed that 0.50 was the best global regret point; thresholds
0.55 through 0.70 reduced useful coverage without meeting safe precision, while
0.75 and above rejected all 76 overrides and collapsed to V2.

The conclusion is label-specific: pair stability is predictable, but it cannot
distinguish a robust correct MapRank direction from a robust reversed direction.
`stride-certguard-v1` is rejected before bundle export, Shadow, high-load TTF or
fresh-map use.  The next separately named design must predict whether the actual
MapRank challenger is robustly better than the V2 anchor, treating uncertainty
and a robust V2 win as explicit negative outcomes.  The completed report SHA-256
is `d15643a5524bbecf7a95edbc2f8dd9cf8f48a38898ac02211e436c59de27f510`.

### STRIDE-OverrideGuard v1 preregistration

After rejecting symmetric CertGuard, `stride-overrideguard-v1` is registered as
a distinct action-conditioned successor before building or inspecting its
labels. Frozen V2 remains the anchor and the deployed MapRank direction model,
thresholds, and candidate pool remain unchanged. The added classifier can only
retain a MapRank override or abstain to the V2 action.

Each training row is one non-anchor candidate against the frozen V2-selected
candidate in the same state. A positive means that candidate is the robust
winner under the unchanged 16 paired-PP-seed, 75% win, 0.02 effect and
two-half-direction rule. Both an uncertain pair and a robust V2-anchor win are
negative. All non-anchor candidates are included and rows from each state sum
to unit weight. The 248-dimensional input preserves direction with 124
candidate-minus-anchor features and adds the 124-dimensional pair mean.

Four outer and three inner map folds are required. Inner threshold selection
must retain at least 20% of MapRank overrides with at least 60% directionally
safe precision while remaining noninferior to V2 on exact-best and Top-3. A
fold with no feasible threshold falls back to V2 and makes the offline gate
fail; thresholds may not be relaxed after observing results. Runtime, TTF,
future rounds, Cost-to-Go, Receding-Q, controller outcomes, legacy validation,
the already observed high-load cohort, test, and formal-OOD data are forbidden
for fitting or calibration.

Wall-clock evaluation follows the user-reported power state; no additional
charger-performance benchmark or preflight is introduced. Every comparator
must still be rerun under the same reported power state and strict rotating
schedule, and timing from unmatched charger or power states may not be pooled.
Consequently, the currently reported slow-charger condition permits label,
offline-model, functional, and semantic work only; it does not permit new
wall-clock evidence. Timing work resumes when the user reports comparable
performance has returned. High load remains post-hoc development evidence and
fresh maps remain the first unseen end-to-end evidence.

### OverrideGuard v1 nested OOF result: rejected before runtime

The checksum-pinned label build covered the registered 303 states and 5,210
non-anchor actions. Only 276 actions (5.30%) robustly beat the V2 anchor; 2,688
were robust anchor wins and 2,246 were uncertain. The action-label SHA-256 is
`fbeacd55ff18966635b27e19714c37e4a8ccea509689a9595f4d903d3f4c675c`.
No runtime, future trajectory, high-load, test, or formal-OOD field was read.

The four-by-three map-grouped action classifier reached weighted ROC-AUC
0.69691, above its 0.65 gate, but its 0.94059 accuracy was 0.00099 below the
weighted majority baseline because robust candidate wins were rare. More
importantly, no registered threshold met the joint action gate in any outer
fold. At 0.50 the global OOF sweep retained 13 of 76 MapRank overrides, seven
of them safe (53.85% precision, 17.11% retention). At 0.55 it retained ten,
seven safe (70.00% precision, 13.16% retention). The required 60% safe
precision and 20% retention were never simultaneous.

The underlying proposal set was itself difficult: only 20 of the 76 MapRank
overrides (26.32%) were robust candidate wins, while 48 were uncertain and
eight were robust V2 wins. The gate therefore had to enrich a low-precision
proposal stream substantially, and the available action features could not do
so without rejecting too much useful coverage.

All folds therefore used the preregistered 1.01 V2 fallback. OverrideGuard
regret was 0.27040, identical to V2 and worse than the same OOF MapRank value
0.24846 by 0.02194. OverrideGuard is rejected before runtime bundle export,
Shadow, high-load raw TTF, or fresh-map evaluation. This is an offline quality
failure independent of the currently reported slow-charger condition; no
wall-clock evidence was collected.

### STRIDE-AnchorChoice v1 preregistration

`stride-anchorchoice-v1` is registered as a distinct post-OverrideGuard
successor before its direct-selection OOF outcomes are computed. It addresses
the measured proposal bottleneck rather than relaxing any label or safety
threshold: the same classifier scores every non-anchor candidate in the frozen
V2-plus-Topology-Boundary pool against the frozen V2 action, then selects the
highest probability only when the registered gate passes. MapRank is retained
as an offline comparator but is no longer allowed to filter the single action
seen by the safety model.

The frozen train labels contain a robust non-anchor opportunity in 77 of 237
states (32.49%), so the registered minimum direct-override coverage is 8% of
all states, while directionally safe precision remains 60%. Calibration also
requires V2 exact-best and Top-3 noninferiority. The final offline gate requires
at least 5% relative normalized-regret improvement over V2, no more than 0.005
absolute regret degradation versus OOF MapRank, and topology-group
noninferiority. Four outer and three inner map-grouped folds remain mandatory.

The label set, candidate pool, V2 anchor, model capacity, 248-dimensional
oriented representation and all source checksums are frozen. Runtime, future
trajectory, high-load, legacy-validation, test and formal-OOD data remain
forbidden for fitting or calibration. The current slow-charger state still
allows only offline and semantic work; no extra charger preflight is added.

### AnchorChoice v1 nested OOF result: rejected before runtime

Directly scanning the full candidate pool improved the useful frontier but did
not satisfy the preregistered joint gate. The action classifier reproduced
weighted ROC-AUC 0.69691. At threshold 0.55 it selected 14 of 237 states,
nine of them robust wins: safe precision was 64.29%, but state coverage was
5.91% rather than the required 8%. Its normalized regret was 0.25713, a 4.91%
relative improvement over V2's 0.27040, just below the registered 5% gate, and
it remained 0.00867 worse than OOF MapRank's 0.24846 versus a 0.005 allowance.

At threshold 0.40 coverage rose to 9.70% and regret was 0.25796, but safe
precision fell to 56.52%. No threshold simultaneously met precision, coverage,
exact-best, Top-3 and regret constraints; all four outer folds were infeasible
and used the registered 1.01 V2 fallback. The reported controller therefore
equals V2 with regret 0.27040 and is rejected before runtime export, Shadow or
TTF. The report SHA-256 is
`56d6a3283ebf4d346dab51054ef17dac90b6d510fa9ffd93bb2863c7eb0927d6`.

The mechanism conclusion is narrower than a general rejection of direct
selection: removing the MapRank proposal filter helped, but a rare binary
robust-win target still did not rank enough safe actions. A subsequent model
must improve the target or representation rather than post-hoc relaxing these
gates.

### STRIDE-MarginChoice v1 preregistration

`stride-marginchoice-v1` is registered as a distinct successor before building
its derived labels or running its OOF model. It keeps the frozen V2 anchor,
candidate pool, 124-dimensional features, 248-dimensional oriented action
representation, 16 paired PP seeds and direct all-candidate scan. It changes
only the learning target and model family; AnchorChoice's failed thresholds are
not relaxed.

For each candidate versus V2, the 16 seed outcomes are retained as two fixed
eight-seed halves. Effects are oriented as candidate minus V2, and the target
is the smaller of the two half means. A consistently useful candidate therefore
has a positive margin, a consistently worse candidate has a negative margin,
and a seed-unstable candidate is penalized by its weaker half. This target keeps
direction and magnitude while remaining a one-step normalized conflict-
reduction label; it contains no runtime or future trajectory.

A fixed-capacity histogram gradient-boosting regressor replaces the rare-event
classifier. Four outer and three inner map folds, 60% safe precision, 8% state
coverage, V2 exact-best and Top-3 noninferiority, 5% relative regret
improvement, MapRank regret tolerance 0.005, topology tolerance 0.03 and a 0.15
weighted OOF margin-correlation gate are preregistered. Legacy validation and
high load remain outside fitting and calibration. No wall-clock stage is
permitted until the user reports that the slow-charger performance condition
has ended; no extra charger benchmark is introduced.

### MarginChoice v1 nested OOF result: rejected before runtime

The derived label build reproduced all 303 states and 5,210 non-anchor actions.
There were 643 actions whose two seed-half mean effects were both positive,
compared with only 276 strict robust candidate wins. The target ranged from
-1.0 to 0.625 with mean -0.22453, and the label SHA-256 is
`5ebe2aff2776fec878bf0765cfc96b3c6a0cd4a7611e13a175da9b184194d240`.

The continuous target was learnable across held-out maps: weighted OOF margin
correlation was 0.46092, and weighted MAE was 0.16635 versus 0.20348 for the
constant-mean baseline. It did not identify strict robust wins. At threshold
-0.01 the selector covered 57 of 237 states but only 17 were strict robust
wins (29.82% safe precision), with normalized regret 0.26330. At threshold
0.05 precision was still only 44.44% at 18 selections. No registered threshold
reached 60% safe precision together with 8% coverage and the quality gates.

All four outer calibrations were infeasible and used the 1.01 V2 fallback, so
the reported regret is V2's 0.27040 rather than the exploratory sweep values.
MarginChoice is rejected before runtime export, Shadow or TTF. The report
SHA-256 is
`d8e5ecd8fe0b262639e3bb1f0e1f66b1f05464474bb09f726c6a12febd956597`.

This separates two questions: the 124-dimensional action representation can
predict a coarse seed-half conflict-reduction margin, but that margin is not a
reliable proxy for the stricter 75%-win, 0.02-effect, two-half robust action.
After CertGuard, OverrideGuard, AnchorChoice and MarginChoice, further target
variants on the same 303 states are stopped to avoid development-set
overfitting. The next model step requires more independent training states and
maps with audited robust-action opportunity; formal fresh maps remain untouched.

### STRIDE-RobustAction data v1 preregistration and static audit

The next line is named `stride-robustaction-data-v1`, with the eventual model
name reserved as `stride-robustaction-v1`; neither name reuses V2, MapRank,
GuardRank, AnchorChoice or MarginChoice artifacts.  The expansion was
registered at parent commit
`63346bd6c9d8b48ed3f42bf55012d604cf93dba3`, before running any new initial PP
or candidate-repair outcome.  It changes the data layer rather than trying a
fifth target on the same 303 states.

Twenty checksum-pinned DAO maps that were absent from the pre-registration
configs and protocol were selected using map identity, static topology and map
family only.  They are disjoint from all 30 current label maps, the 12 formal
OOD maps, and the six registered fresh-evidence map ids.  The groups are:

- high topology (7): `orz200d`, `lak526d`, `orz301d`, `brc201d`, `lak506d`,
  `oth001d`, `ost001d`;
- mid topology (7): `hrt001d`, `lak106d`, `orz101d`, `den900d`, `lgt604d`,
  `oth000d`, `brc200d`;
- low-topology controls (6): `lak507d`, `orz704d`, `den901d`, `lgt600d`,
  `brc999d`, `orz000d`.

The split covers eight map prefixes (`brc`, `den`, `hrt`, `lak`, `lgt`, `orz`,
`ost`, and `oth`).  Low-degree-cell ratio boundaries are frozen at 0.035 and
0.06.  Static audit verified every extracted map and archive member checksum,
every registered obstacle and low-degree ratio, exact 7/7/6 group balance,
and all evidence-isolation gates.  No solver or performance measurement was
run during this audit.  The design SHA-256 is
`6a1a870ea061c233abd570c83ad523b7cbc7e1b4ced46ef8a625128d690519ea` and the
completed static-audit report SHA-256 is
`ad49597d9bd81e93dc14c5f38eb39a6d37292d0bcd8a763b362b70e92c5b3ee9`.

The first deterministic task-generation attempt stopped before PP on
`orz200d` because an odd agent count cannot be deranged by the registered
reverse-and-rotate `opposite_exchange` generator.  This was a generator-domain
error, not an observed load or repair result.  At parent commit `f550c6e`, all
odd registered loads were therefore rounded upward by one and the validator
now requires every load to be even.  No initial-PP, candidate-repair,
controller, or timing outcome was read before this amendment; the failed v1
directory is retained as diagnostic evidence; the corrected generation will
use a new v2 output directory after work resumes.

After power comparability is reported by the user, an outcome-blind initial-PP
qualification may choose two tasks per map near 25 and 100 mean initial
conflicts from the registered agent loads and the two derived task variants.
Candidate-repair outcomes, controller choices, future trajectory, repair time,
and TTF are forbidden selection inputs.  Low-topology maps receive a fourth,
higher registered agent load so underload can be addressed without changing
the rule after observing repair outcomes.

The selected cohort has 20 maps x 2 tasks x 2 solver seeds x 2 source policies
= 160 independent episode units.  At no more than two states per episode it
adds 320 state rows, taking the 303-row corpus to a projected 623 rows.  The
distinction is mandatory: two states from one episode are correlated, and the
16 paired PP repair seeds per candidate reduce label noise but are outcomes,
not additional independent states.  Post-collection gates require all 320
unique state ids, all 160 episode ids, no more than two states per episode,
robust non-anchor opportunities in at least 25% of states, at least 4% robust
positive actions, and positive-opportunity coverage on at least three maps in
each topology group.  Failure registers another outcome-blind cohort; it may
not filter the collected states using their repair outcomes.

Power state remains user-reported only.  No charger benchmark or performance
preflight is added.  Static design, label processing, offline training, and
functional or semantic tests may proceed in the reported slow-power state;
Shadow timing and wall-clock TTF remain deferred until the user reports that
comparable performance has returned.  Timing from different reported power
states may not be pooled.
