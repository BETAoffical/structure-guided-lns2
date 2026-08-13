# STRIDE ShellBudget Audit v1 execution amendment

The first complete report was rejected before acceptance by its own
size-balance integrity gate. The registered reducer prioritizes the least
represented nominal size and the frozen readiness gate permits a maximum
selected size-count spread of one. The initial executor nevertheless continued
filling the nominal budget after one size bucket had no remaining exact-distinct
candidate, producing a spread of two in some states.

The executor now stops adding candidates when every remaining addition would
violate the registered spread. Because a budget is a maximum rather than a
required count, this is the direct implementation of the frozen rule. If the
entire exact-deduplicated state pool is already no larger than the budget, all
candidates remain preserved and the state is exempt from a compression-balance
check; dropping a candidate in that case would contradict the registered
full-pool preservation behavior.

No budget, threshold, ordering criterion, state, candidate, PP outcome, or
bounded-continuation label changed. The rejected report is overwritten only
after the corrected integrity check passes.
