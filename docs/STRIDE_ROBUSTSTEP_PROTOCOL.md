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

## Promotion boundary

The next sequence is design, fresh label confirmation, a small balanced Pilot,
controlled retraining, action-preserving Shadow, and paired TTF Quick. Only a
success-noninferior model with lower mean capped TTF and no material map-family
regression may proceed to formal Stage 5/OOD evaluation.
