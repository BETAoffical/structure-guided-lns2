# Research Roadmap

## Scoped contribution

The completed evidence supports a narrower contribution than the original plan:

> During InitLNS first-feasible repair, rank concrete agent neighborhoods produced
> by the official Target/Collision/Random generators using the current dynamic
> conflict state, selected-agent path statistics and local topology.

The official neighborhood generators and PP+SIPPS repair remain unchanged. The
frozen policy is confirmed on unseen maps from the three development layout
families and across solver seeds 1, 2 and 3. Static map/OD/density context did not
show reliable incremental value. No RL policy or autoregressive agent-subset
generator was trained, and the project does not claim to be the first learned
neighborhood method, variable-size LNS, or RL extension of LNS2.

The standard MovingAI test increased success from 123/144 to 131/144 and improved
fixed-budget conflict AUC by 4.105%, with positive map bootstrap evidence and no
average regression on any active map or layout family. It nevertheless missed the
preregistered 5% primary gate, so strict cross-layout generalization is not
confirmed. See `docs/INITLNS_RESEARCH_REPORT_ZH.md` for the frozen evidence ledger.

## Historical evidence chain

Before collecting more data or training RL, the project asks whether static
context adds measurable predictive value beyond the action, seed, and dynamic
repair state. `scripts/run_context_audit.py` constructs Horizon-4 Pareto
preferences from the existing counterfactual collection and trains three fixed-
seed pairwise GBDT ablations:

1. action and seed features;
2. action, seed, and dynamic state;
3. dynamic state plus map, static OD, density, and flow context.

The 2026-07-14 audit used 7,344 candidate outcomes from 128 Train and 72
Validation states. Train and Validation contained no shared map or task instance.
The full-context model improved Pareto top-1 hit rate over the dynamic model by
4.17 percentage points, below the 5-point gate, and worsened mean conflict-AUC
regret by 2.28% instead of improving it by 5%. The paired bootstrap did not show
significant degradation, but the overall offline gate failed.

Consequently, Validation closed-loop evaluation, the planned 31,104/62,208-
outcome expansions, semantic v3 data, and RL training were paused. This result
did not prove context useless; it said the Pilot v2 representation, Adaptive-
state distribution, and first GBDT protocol did not establish the required
incremental value. See `docs/CONTEXT_AUDIT.md`.

A pre-registered secondary diagnostic then merged the old Train/Validation into
a development set, encoded neighborhood size categorically, added a direct
Pareto-membership learner, excluded runtime from the primary Pareto relation,
used map-grouped cross-validation, and compared real context with 500 task-level
context permutations. Labels, dynamic-state ranking, and oracle heterogeneity
passed their prerequisites. Static context did not: its top-1 gain was at the
6.2 percentile of the permutation null and its AUC-regret reduction at the 59.2
percentile, both below the required 95 percentile. The primary model also retained
a 91% single-size concentration against a 90% limit.

The secondary gate is therefore **FAIL**. Under the registered stopping rule, the
new 12/6-map confirmation data and maximum 23,328 outcomes are not generated.
The current evidence supports only oracle action heterogeneity, not incremental
predictive value from the hand-crafted static context. See
`docs/CONTEXT_SECONDARY_AUDIT.md`.

The subsequent local-representation audit also failed its registered recovery
gates. Neither pre-generation local features nor realized-neighborhood visibility
produced a stable map-grouped improvement, although realized context had some
non-robust point-estimate signal. A bounded MovingAI mechanism probe then found
real immediate action and realized-neighborhood diversity, but solver seeds 0 and
1 reproduced identical states. After pooling them as four action trials per unique
state, action identity explained 39.2% of conflict variation, below the 50% gate;
map and density permutation percentiles were also below 95%. The registered next
step is an action-trial stability confirmation on unique states, not contextual
ranking or RL. See `docs/MOVINGAI_MECHANISM_PROBE.md`.

The subsequent quality audit showed why that confirmation cannot simply duplicate
the existing collection. The 1,368 rows contain only 12 independent states and four
trials per candidate; split-half action rankings are unstable, actual neighborhoods
have low overlap, and one scenario per map confounds topology with task realization.
The corrected v2 mechanism probe therefore uses three MovingAI scenarios, one solver
seed for state acquisition, eight action trials, and bounded initial-conflict sources.
Independent maps per layout family are still required before any transfer claim.
See `docs/MOVINGAI_PROBE_QUALITY.md`.

The v2 partial confirmation then recovered 35 independent states and 7,776 outcomes
with eight trials per candidate. Candidate-rank stability improved to 0.684, but
top-1 overlap, realized-neighborhood stability, density alignment, and compute-aware
label agreement remained weak. One low-conflict 600-agent warehouse episode did not
complete within a clean 20-minute window and is recorded as a partial-run limitation;
runtime sensitivity is invalid because earlier host timeouts left overlapping WSL
collectors. The next useful increase is independent layout-family
replication and balanced scenario/density coverage, not more trials on these states.

The bounded follow-up was therefore the independent-layout confirmation in
`docs/INDEPENDENT_LAYOUT_PROBE.md`. It uses two new maps per seen layout family and a
complete balanced/bottleneck by 80/100 static-task design. Qualification, exact
paired tests, Holm correction, and eight action trials must pass before the paused
12/6-map confirmation or any learned policy is allowed to resume.

That confirmation is now complete: 23 states and 6,480 outcomes were collected with
no errors. Split-rank Spearman reached 0.638, but action eta-squared was 0.404,
Pareto-family Jaccard was 0.432, and no layout/OD/density test survived Holm
correction. The registered decision is to stop expansion and redefine the action
surface. Because realized neighborhoods had Jaccard 0.428, the next admissible
mechanism study is candidate-neighborhood generation followed by ranking; the
12/6-map dataset, supervised policy, and RL remain inactive.

The active mechanism gate is now `docs/REALIZED_NEIGHBORHOOD_PROBE.md`. It selects
representative agent sets without reading repair outcomes, then evaluates each
fixed set with independent PP-order seeds. Only a stable improvement over the
nominal action labels permits a realized-neighborhood ranking audit; otherwise the
repair order must become part of the action definition.

The gate completed with 23 states, 412 explicit candidates, 3,296 outcomes, and no
errors. Realized-neighborhood eta-squared rose from 0.404 to 0.595, rank Spearman
was 0.803, and exact Pareto/best-candidate Jaccard was 0.518/0.547. All registered
criteria passed. The next active stage is therefore a learned ranking audit over
these concrete candidate sets, beginning with dynamic and realized features;
static map/OD/density context remains an ablation and RL remains paused.

The ranking audit is now complete. It aggregated all eight PP-order trials for each
of the 412 explicit neighborhoods and used six leave-one-map-out folds. Adding the
actual agent-set representation to dynamic proposal features raised Pareto top-1
from 13.0% to 43.5% and reduced mean conflict regret by 55.6%; all six held-out maps
were no worse and the map-bootstrap intervals were positive. This passes the
registered realized-ranking gate and is the first strong learned-ranking signal in
the current project.

The result does not restore the original static transfer claim. Static context
added only 4.35 top-1 percentage points, while real-context performance reached the
92.0 and 74.2 percentiles of the task-context permutation null. The registered
decision is to advance dynamic-state plus realized-neighborhood ranking and shrink
the handcrafted map/OD/density claim. RL remains paused. The next permitted gate is
an independent-map ranking confirmation plus a proposal-only candidate-generation
interface; see `docs/REALIZED_NEIGHBORHOOD_RANKING_AUDIT.md`.

The natural-distribution confirmation then retained zero-conflict and high-conflict
tasks without resampling. On 12 fresh maps, the frozen realized-neighborhood ranker
raised top-1 from 19.5% to 43.9% and reduced conflict regret by 33.8%. A separate
six-map closed-loop test reduced fixed-budget conflict AUC by 57.3%, and the
12-map, three-seed confirmation reduced it by 52.5% while preserving 144/144
successes. These experiments confirm dynamic realized-neighborhood control within
the three registered layout families.

Policy-visited aggregation, objective alignment, greater GBDT capacity, neural
agent/graph encoders and deterministic graph features did not pass their Train or
Validation gates. Horizon-4 outcomes contained oracle opportunity but were not
split-half stable. PP repair order materially changed outcomes, but the contextual
order selector also failed. These stopping results prevent RL warm-start or further
model tuning on the inspected data.

Finally, the frozen v1 policy was evaluated without retraining on 12 untouched
MovingAI maps from five layout families. It produced broad positive evidence but
missed the registered 5% AUC gate at 4.105%. The registered decision is therefore
to consolidate results rather than modify the model against these OOD outcomes.

## Baseline taxonomy

- **Official MAPF-LNS2 Adaptive and fixed Target/Collision/Random:** isolate the
  InitLNS high-level choice while keeping the low-level solver identical.
- **GPBS:** an independent end-to-end feasibility solver and high-level search
  baseline. It is not represented as an LNS2 destroy heuristic.
- **LNS2+RL:** an end-to-end comparison with a learned low-level repair policy,
  not evidence about this project's high-level action selection alone.
- **DiffLNS:** a paper-level learned-initialization comparison when no public
  implementation is available.
- **BALANCE, ADDRESS, and NNS:** useful policy ideas and code references. Any
  version applied to InitLNS must be labeled an adapted baseline because the
  original method did not directly evaluate this exact InitLNS control problem.

## Frozen boundary and possible follow-up

The current experiment sequence is closed. Existing MovingAI labels may not be
used to retune the frozen model or lower the registered gate. The next independent
research question, if opened, should be outcome-blind candidate pruning and batch
scoring: preserve the frozen v1 candidate choice or conflict trajectory while
reducing proposal, feature and pairwise-inference cost. A new spatiotemporal model
or RL study requires new preregistered data and cannot rewrite the current OOD
decision.

Primary metrics remain success, time-to-feasible, conflict AUC, SIPPS calls/nodes
and runtime. Sum of costs is secondary because the research target ends at the
first feasible solution.

## Public code provenance

- [MAPF-LNS2](https://github.com/Jiaoyang-Li/MAPF-LNS2): pinned active solver core.
- [GPBS](https://github.com/shchan13/GPBS): pinned independent feasibility baseline.
- [BALANCE](https://github.com/thomyphan/anytime-mapf): online heuristic/size bandit reference.
- [ADDRESS](https://github.com/JimyZ13/ADDRESS): delayed-agent seed policy reference.
- [NNS](https://github.com/mit-wu-lab/mapf_neural_neighborhood_search): candidate-ranking reference.
- [LNS2+RL](https://github.com/marmotlab/LNS2-RL): learned low-level repair baseline.
- [Unified LNS benchmark](https://github.com/ChristinaTan0704/mapf-lns-benchmark): evaluation reference only; no unlicensed implementation is copied.

No author release was found for DROP-LNS or DiffLNS during the review, so they
remain paper-level comparisons unless provenance changes.
