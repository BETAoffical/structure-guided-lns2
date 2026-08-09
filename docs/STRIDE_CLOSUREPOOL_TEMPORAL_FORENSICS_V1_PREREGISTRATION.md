# ClosurePool Temporal Forensics v1 Preregistration

## Question

The first ClosurePool diagnostic found strong post-repair evidence of long-tail
harm, but simple pre-action conflict-boundary and component-coverage measures
did not separate harmful from beneficial Maze actions. This supplement asks a
narrower question: does a structural neighborhood cut a map-defined temporal
dependency chain through a low-degree corridor, or contain an opposing queue
that PP can repeatedly recreate?

## Fixed cohort

The rule-selection cohort consists of every historical divergent comparison
whose registered group is `maze100`: ten controller comparisons over five
task/solver keys and two task seeds. It contains three adverse/severe and seven
beneficial/tied comparisons. Outcomes are used only for retrospective
contrasts. No comparison is removed after seeing the new features.

The two known-regression comparisons are reconstructed separately and are
illustrative only. They cannot select a metric, rule, or threshold. The small
Room regression is outside this Maze-specific mechanism study and remains in
the parent report.

## Corridor and temporal dependency definition

A corridor cell is a free grid cell with static four-neighbor degree at most
two. A corridor segment is a connected component of the subgraph induced by
those cells. This definition is derived only from the map and has no learned or
outcome-dependent parameter.

For each agent, a segment visit is a maximal contiguous run of its current path
inside one segment. Two agents have a temporal corridor dependency when visits
to the same segment have intersecting closed time intervals. It is an opposing
dependency when the two visits traverse at least one shared grid edge in
opposite directions. There is no tunable time window.

For the V2 and challenger neighborhoods at their identical first-divergence
state, the analysis records:

- internal, boundary, and external dependency edges;
- selected-to-unselected dependency-cut ratio and outside queue agents;
- opposing internal and boundary dependencies;
- coverage of touched dependency components;
- low-degree segment length, visit duration, and waiting mass;
- changes in dependency and opposing-dependency edges after the paired PP
  repair;
- exact cell coordinates and segment context for every previously registered
  persistent pair in the Maze cohort.

## Decision boundary

The primary contrast is historical adverse/severe minus historical
beneficial/tied, reported overall and by controller. A candidate closure rule
may be designed only after human review finds an interpretable pre-action
mechanism that is directionally consistent across at least two independent
historical adverse map/task groups and is not contradicted by the retained
controls. Because this supplement contains only one map group, it cannot by
itself satisfy that requirement; at most it can define a hypothesis for a
separate cross-map confirmation.

The analysis cannot train a model, search a threshold, change candidate
generation, run a controller, or claim TTF improvement.
