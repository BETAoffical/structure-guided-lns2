# STRIDE-MarginalPool v1 root-diagnostic report

## Decision

Stage 1 completed with all integrity gates passing. The 45 repeated/stalled
continuation cases do not show structural-generator collapse under the frozen
Jaccard definition. Every repeat checkpoint contains sufficiently different
alternatives, yet the frozen V2 scorer ranks the repeated structural candidate
first.

The two PP calls preceding every registered third proposal are also genuine
no-ops at the path level. All 90 calls use different PP seeds within each
case, report `step_applied=true` and `replan_success=false`, and change zero
agent paths. There is no hidden waiting, path-cost, delay, occupancy or
bottleneck-order preparation in these two calls.

This establishes the immediate feedback mechanism:

1. a structural candidate wins the augmented V2 ranking;
2. PP cannot repair that neighborhood for two distinct seeds and leaves the
   meaningful path/conflict state unchanged;
3. deterministic generation offers the same candidate again;
4. the memoryless scorer again ranks it above many available alternatives.

It does not yet establish whether one of those alternatives repairs better.
That is the purpose of the preregistered all-candidate paired replay in Stage 2.

## Coverage and integrity

- frozen continuation contrasts: 45/45;
- classification retained only for descriptive reporting: 17 adverse,
  8 beneficial and 20 neutral;
- pre-action checkpoints: 90/90;
- unique state fingerprints: 78;
- candidate rows preserved: 2,788, of which 2,502 are unique
  state/candidate keys;
- registered 124-dimensional features: complete for every checkpoint;
- trace and manifest SHA-256 values: 45/45 matched;
- current third-action outcome and future transitions used: 0;
- new solver or PP calls: 0;
- repeat-run product hashes: identical.

Primary artifact hashes:

- report: `8f21f709f69cba540842cbd11dcfe65c19214c7c55b5d4492d70244d90a2f483`;
- status: `0bde553b4ea039f3d4afc58467c8b16a88d8008f6f57472918b174eaf271eb5c`;
- case manifest: `ea9bc11708cb911c03677e0166b518630b44819ddd9be05414311ca040172956`;
- checkpoint manifest: `bbf03634a2563b3e805cdf99a33071648a08640d82605b6245fe2c16aca3ef81`.

## Candidate-pool result

At every first repeat/stall checkpoint:

- generator-collapse flag: 0/45;
- ranker-lock flag: 45/45;
- selected rank: first in all 45 cases;
- mean sufficiently different alternatives: 29.2 per case;
- mean sufficiently different structural alternatives: 11.6 per case.

The repeated structural winner is not a near-duplicate of the original-pool V2
anchor. Its mean score advantage over that anchor is largest in the adverse
group (`+3.9172`), while its mean agent-set Jaccard with the anchor is only
`0.2052`. This is consistent with an out-of-distribution ranking preference
for broad structural candidates, but candidate scores alone are not repair
quality labels.

## PP response result

The 90 preceding PP calls have one identical status pattern:

```text
step_applied=true
replan_success=false
changed_agent_count=0
```

Within all 45 cases, the two attempted PP seeds are different. Therefore this
specific repetition is not explained by accidentally reusing one PP seed, and
the previous hypothesis that unchanged conflicts might hide useful path
preparation is rejected for this cohort and this two-repair window.

The result still does not prove that PP is globally defective. PP is being
asked to solve a neighborhood chosen by the augmented selector. Stage 2 must
test the other candidates under the same paired seeds before responsibility can
be assigned to ranking, candidate-pool closure or the repair operator.

## Repair-closure evidence

The repeated neighborhood covers substantially less of the persistent
conflict closure in adverse cases:

| TailSwitch class | Cases | Mean missing persistent endpoints | Mean persistent full-edge coverage | Mean truncated components |
| --- | ---: | ---: | ---: | ---: |
| adverse | 17 | 41.65 | 0.3028 | 8.12 |
| beneficial | 8 | 15.13 | 0.5452 | 1.88 |
| neutral | 20 | 7.60 | 0.7800 | 1.35 |

The adverse median closure deficit is 36 agents, compared with 9 for
beneficial and 2 for neutral cases. This is strong descriptive support for the
repair-closure hypothesis: the selected neighborhood often touches conflicts
without jointly containing the agents needed to resolve the persistent
component. It is not sufficient as a causal rule: two adverse cases have zero
registered deficit, and classification is based on the later continuation.

## Next step

Stage 2 will restore both frozen checkpoints and repair every candidate already
present in the complete pool with 16 strictly paired PP seeds. The logical
matrix contains 2,788 checkpoint/candidate rows, or 2,502 unique
state/candidate keys after identity-safe deduplication, for 40,032 unique PP
trials.

The decision rules remain:

- if another existing candidate is stably better, the primary defect is
  ranking;
- if all existing candidates fail similarly, the old pool lacks an effective
  repair-closure action;
- if the repeated winner is the stable one-step best, run the frozen two-step
  sequence test before changing the ranking objective;
- if every candidate and sequence remains ineffective, stop selector-only
  work and audit PP/operator capability.

No controller, model or TTF claim is promoted from Stage 1.
