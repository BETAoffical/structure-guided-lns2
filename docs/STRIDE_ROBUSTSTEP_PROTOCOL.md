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

## Promotion boundary

The next sequence is design, fresh label confirmation, a small balanced Pilot,
controlled retraining, action-preserving Shadow, and paired TTF Quick. Only a
success-noninferior model with lower mean capped TTF and no material map-family
regression may proceed to formal Stage 5/OOD evaluation.
