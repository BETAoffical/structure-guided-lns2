#pragma once

#include "common.h"

#include <cstdint>

enum class RepairActionMode
{
    OFFICIAL,
    SEED,
    EXPLICIT_NEIGHBORHOOD,
    // Reserved for deterministic offline trace replay. Unlike the normal
    // explicit controller action, this mode may reproduce a recorded random
    // neighborhood that did not happen to touch a conflict.
    REPLAY_NEIGHBORHOOD
};

enum class RepairHeuristic
{
    ADAPTIVE,
    TARGET,
    COLLISION,
    RANDOM
};

struct RepairAction
{
    RepairActionMode mode = RepairActionMode::OFFICIAL;
    RepairHeuristic heuristic = RepairHeuristic::ADAPTIVE;
    int seed_agent = -1;
    int neighborhood_size = 0;
    int random_seed = -1;
    // Optional seed applied immediately before PP.  This is deliberately
    // separate from random_seed because official neighborhood generation and
    // PP both consume the process-global C RNG.  Trace replay can reproduce
    // the recorded neighborhood without repeating the former, so it needs an
    // independent PP seed to reproduce low-level tie breaking exactly.
    int pp_random_seed = -1;
    // Opt-in causal instrumentation. Disabled by default so normal solver
    // timing and PP acceptance semantics do not pay diagnostic-loop overhead.
    bool collect_pp_diagnostics = false;
    vector<int> agents;
    vector<int> repair_order;
};

struct RepairProposal
{
    RepairAction requested_action;
    RepairHeuristic applied_heuristic = RepairHeuristic::COLLISION;
    vector<int> neighborhood;
    bool action_valid = false;
    bool generated = false;
};

struct RepairAgentState
{
    int id = -1;
    int start = -1;
    int goal = -1;
    int path_cost = -1;
    int shortest_path_cost = -1;
    int delay = -1;
    int conflict_degree = 0;
    vector<int> path;
};

struct RepairState
{
    bool initialized = false;
    bool initial_solution_complete = false;
    bool feasible = false;
    bool done = false;
    int iteration = 0;
    int rows = 0;
    int cols = 0;
    int sum_of_costs = 0;
    int num_of_colliding_pairs = 0;
    double runtime = 0;
    uint64_t low_level_expanded = 0;
    uint64_t low_level_generated = 0;
    uint64_t low_level_reopened = 0;
    uint64_t low_level_runs = 0;
    vector<unsigned char> obstacles;
    vector<pair<int, int>> conflict_edges;
    vector<RepairAgentState> agents;
};

enum class PPFailureReason
{
    NOT_RUN,
    NONE,
    CONFLICT_BOUND_EXCEEDED,
    TIME_LIMIT
};

struct PPAgentDiagnostic
{
    int agent_id = -1;
    int order_index = -1;
    int path_cost_before = -1;
    int path_cost_after = -1;
    int low_level_collision_count = 0;
    int cumulative_conflict_pair_count = 0;
    bool path_changed = false;
    bool inserted_into_path_table = false;
    vector<pair<int, int>> new_conflict_pairs;
    vector<int> internal_blocker_agents;
    vector<int> external_blocker_agents;
};

struct RepairTransition
{
    RepairAction requested_action;
    RepairHeuristic applied_heuristic = RepairHeuristic::COLLISION;
    vector<int> neighborhood;
    vector<int> repair_order;
    bool action_valid = true;
    bool generated = false;
    bool replan_success = false;
    PPFailureReason pp_failure_reason = PPFailureReason::NOT_RUN;
    int pp_attempted_agent_count = 0;
    int pp_inserted_agent_count = 0;
    int pp_failed_agent = -1;
    int pp_failed_order_index = -1;
    int pp_old_conflict_pair_count = 0;
    int pp_attempt_conflict_pair_count = 0;
    bool pp_rolled_back = false;
    vector<PPAgentDiagnostic> pp_agent_diagnostics;
    int applied_pp_random_seed = -1;
    int iteration = 0;
    int conflicts_before = 0;
    int conflicts_after = 0;
    int sum_of_costs_before = 0;
    int sum_of_costs_after = 0;
    double runtime_before = 0;
    double runtime_after = 0;
    // Diagnostic-only wall-clock timings. These fields partition the native
    // repair step without changing neighborhood generation or replanning.
    double native_step_seconds = 0;
    double neighborhood_generation_seconds = 0;
    double replan_seconds = 0;
    double pp_replan_seconds = 0;
    double state_snapshot_seconds = 0;
    double repair_bookkeeping_seconds = 0;
    double native_residual_seconds = 0;
};

class NeighborhoodPolicy
{
public:
    virtual ~NeighborhoodPolicy() = default;
    virtual RepairAction choose(const RepairState& state) = 0;
};

class RepairObserver
{
public:
    virtual ~RepairObserver() = default;
    virtual void onInitialState(const RepairState&) {}
    virtual void onTransition(const RepairState&, const RepairTransition&, const RepairState&) {}
    virtual void onFinish(const RepairState&, bool) {}
};

inline const char* repairActionModeName(RepairActionMode mode)
{
    switch (mode)
    {
        case RepairActionMode::OFFICIAL: return "official";
        case RepairActionMode::SEED: return "seed";
        case RepairActionMode::EXPLICIT_NEIGHBORHOOD: return "explicit_neighborhood";
        case RepairActionMode::REPLAY_NEIGHBORHOOD: return "replay_neighborhood";
    }
    return "unknown";
}

inline const char* repairHeuristicName(RepairHeuristic heuristic)
{
    switch (heuristic)
    {
        case RepairHeuristic::ADAPTIVE: return "adaptive";
        case RepairHeuristic::TARGET: return "target";
        case RepairHeuristic::COLLISION: return "collision";
        case RepairHeuristic::RANDOM: return "random";
    }
    return "unknown";
}

inline const char* ppFailureReasonName(PPFailureReason reason)
{
    switch (reason)
    {
        case PPFailureReason::NOT_RUN: return "not_run";
        case PPFailureReason::NONE: return "none";
        case PPFailureReason::CONFLICT_BOUND_EXCEEDED: return "conflict_bound_exceeded";
        case PPFailureReason::TIME_LIMIT: return "time_limit";
    }
    return "unknown";
}
