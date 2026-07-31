#include "CBSNode.h"
#include "Conflict.h"
#include "ECBSNode.h"
#include "GCBSNode.h"
#include "PBS.h"
#include "SingleAgentSolver.h"
#include "SpaceTimeAStar.h"
#include "WeightedSampling.h"
#include "structure_guided/native_semantics.hpp"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void require(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <typename Comparator, typename Node>
void requireStrictComparator(
    const Comparator& comparator,
    const Node* first,
    const Node* second,
    const char* message)
{
    require(!comparator(first, first), message);
    require(!comparator(second, second), message);
    require(
        !(comparator(first, second) && comparator(second, first)),
        message
    );
}

void requireWeightedSamplingBoundaries()
{
    const std::vector<int> weights{2, 1, 3};
    require(
        corrected_native::weightedIndexForDraw(weights, 0) == 0,
        "weighted draw 0 did not select the first bucket"
    );
    require(
        corrected_native::weightedIndexForDraw(weights, 1) == 0,
        "the first weighted bucket lost its final draw"
    );
    require(
        corrected_native::weightedIndexForDraw(weights, 2) == 1,
        "the second weighted bucket did not start at its 0-based boundary"
    );
    require(
        corrected_native::weightedIndexForDraw(weights, 3) == 2,
        "the final weighted bucket did not start at its 0-based boundary"
    );
    require(
        corrected_native::weightedIndexForDraw(weights, 5) == 2,
        "the final weighted bucket lost its final draw"
    );
    bool rejected = false;
    try
    {
        corrected_native::weightedIndexForDraw(weights, 6);
    }
    catch (const std::out_of_range&)
    {
        rejected = true;
    }
    require(rejected, "an out-of-range weighted draw was accepted");
}

void requireStrictHeapOrderings()
{
    LLNode low_level_a(1, 2, 3, nullptr, 2, 0);
    LLNode low_level_b(2, 2, 3, nullptr, 2, 0);
    requireStrictComparator(
        LLNode::compare_node(),
        &low_level_a,
        &low_level_b,
        "low-level OPEN comparator is not a strict ordering"
    );
    requireStrictComparator(
        LLNode::secondary_compare_node(),
        &low_level_a,
        &low_level_b,
        "low-level FOCAL comparator is not a strict ordering"
    );

    CBSNode cbs_a;
    CBSNode cbs_b;
    cbs_a.time_generated = 1;
    cbs_b.time_generated = 2;
    requireStrictComparator(
        CBSNode::compare_node_by_f(),
        &cbs_a,
        &cbs_b,
        "CBS cleanup comparator is not a strict ordering"
    );
    requireStrictComparator(
        CBSNode::compare_node_by_d(),
        &cbs_a,
        &cbs_b,
        "CBS focal comparator is not a strict ordering"
    );
    requireStrictComparator(
        CBSNode::compare_node_by_inadmissible_f(),
        &cbs_a,
        &cbs_b,
        "CBS open comparator is not a strict ordering"
    );

    ECBSNode ecbs_a;
    ECBSNode ecbs_b;
    ecbs_a.time_generated = 1;
    ecbs_b.time_generated = 2;
    requireStrictComparator(
        ECBSNode::compare_node_by_f(),
        &ecbs_a,
        &ecbs_b,
        "ECBS cleanup comparator is not a strict ordering"
    );
    requireStrictComparator(
        ECBSNode::compare_node_by_d(),
        &ecbs_a,
        &ecbs_b,
        "ECBS focal comparator is not a strict ordering"
    );
    requireStrictComparator(
        ECBSNode::compare_node_by_inadmissible_f(),
        &ecbs_a,
        &ecbs_b,
        "ECBS open comparator is not a strict ordering"
    );

    GCBSNode gcbs_a;
    GCBSNode gcbs_b;
    gcbs_a.time_generated = 1;
    gcbs_b.time_generated = 2;
    requireStrictComparator(
        GCBSNode::compare_node_by_d(),
        &gcbs_a,
        &gcbs_b,
        "GCBS comparator is not a strict ordering"
    );

    PBSNode pbs_a;
    PBSNode pbs_b;
    pbs_a.time_generated = 1;
    pbs_b.time_generated = 2;
    requireStrictComparator(
        PBSNode::compare_node(),
        &pbs_a,
        &pbs_b,
        "PBS comparator is not a strict ordering"
    );

    Conflict conflict_a;
    conflict_a.a1 = 1;
    conflict_a.a2 = 2;
    conflict_a.type = conflict_type::STANDARD;
    conflict_a.priority = conflict_priority::CARDINAL;
    Conflict conflict_b = conflict_a;
    require(
        !(conflict_a < conflict_a),
        "conflict ordering is not irreflexive"
    );
    require(
        !(conflict_a < conflict_b) && !(conflict_b < conflict_a),
        "equal conflicts do not compare equivalently"
    );
}

void requireTargetCountingAvoidsAllForeignGoals()
{
    Instance instance(TARGET_COUNT_TEST_MAP, TARGET_COUNT_TEST_SCEN, 3);
    SpaceTimeAStar solver(instance, 0);
    std::vector<int> goal_table(
        static_cast<std::size_t>(instance.map_size), -1
    );
    const std::vector<int> goals = instance.getGoals();
    for (std::size_t agent = 0; agent < goals.size(); agent++)
        goal_table[goals[agent]] = static_cast<int>(agent);

    std::set<int> targets;
    solver.findMinimumSetofColldingTargets(goal_table, targets);
    require(
        targets.empty(),
        "target-minimizing path used a shorter route through foreign goals"
    );
}
}

int main()
{
    require(
        std::string(structure_guided::kNativeSemanticsSchema) ==
            "lns2.corrected_native.v1",
        "corrected-native semantics schema changed unexpectedly"
    );
    requireWeightedSamplingBoundaries();
    requireStrictHeapOrderings();
    requireTargetCountingAvoidsAllForeignGoals();
    return 0;
}
