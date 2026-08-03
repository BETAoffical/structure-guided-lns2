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

## Promotion boundary

The next sequence is design, fresh label confirmation, a small balanced Pilot,
controlled retraining, action-preserving Shadow, and paired TTF Quick. Only a
success-noninferior model with lower mean capped TTF and no material map-family
regression may proceed to formal Stage 5/OOD evaluation.
