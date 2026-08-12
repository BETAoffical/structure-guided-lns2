#include "structure_guided/jsonl_observer.hpp"

#include <stdexcept>

JsonlRepairObserver::JsonlRepairObserver(const std::string& path) :
    path(path), output(path)
{
    if (!output)
        throw std::runtime_error("failed to open repair trace: " + path);
}

void JsonlRepairObserver::flushChecked()
{
    output.flush();
    if (!output)
        throw std::runtime_error("failed to write repair trace: " + path);
}

void JsonlRepairObserver::writeIntArray(std::ostream& stream, const vector<int>& values)
{
    stream << '[';
    for (size_t index = 0; index < values.size(); index++)
    {
        if (index > 0)
            stream << ',';
        stream << values[index];
    }
    stream << ']';
}

void JsonlRepairObserver::writePairArray(
    std::ostream& stream, const vector<pair<int, int>>& values)
{
    stream << '[';
    for (size_t index = 0; index < values.size(); index++)
    {
        if (index > 0)
            stream << ',';
        stream << '[' << values[index].first << ',' << values[index].second << ']';
    }
    stream << ']';
}

void JsonlRepairObserver::writePPDiagnostics(const RepairTransition& transition)
{
    output << "{\"schema\":\"lns2.pp_repair_diagnostic.v1\""
           << ",\"failure_reason\":\""
           << ppFailureReasonName(transition.pp_failure_reason) << '"'
           << ",\"attempted_agent_count\":"
           << transition.pp_attempted_agent_count
           << ",\"inserted_agent_count\":"
           << transition.pp_inserted_agent_count
           << ",\"failed_agent\":" << transition.pp_failed_agent
           << ",\"failed_order_index\":"
           << transition.pp_failed_order_index
           << ",\"old_conflict_pair_count\":"
           << transition.pp_old_conflict_pair_count
           << ",\"attempt_conflict_pair_count\":"
           << transition.pp_attempt_conflict_pair_count
           << ",\"rolled_back\":"
           << (transition.pp_rolled_back ? "true" : "false")
           << ",\"agents\":[";
    for (size_t index = 0; index < transition.pp_agent_diagnostics.size(); index++)
    {
        if (index > 0)
            output << ',';
        const auto& row = transition.pp_agent_diagnostics[index];
        output << "{\"agent_id\":" << row.agent_id
               << ",\"order_index\":" << row.order_index
               << ",\"path_cost_before\":" << row.path_cost_before
               << ",\"path_cost_after\":" << row.path_cost_after
               << ",\"low_level_collision_count\":"
               << row.low_level_collision_count
               << ",\"cumulative_conflict_pair_count\":"
               << row.cumulative_conflict_pair_count
               << ",\"path_changed\":"
               << (row.path_changed ? "true" : "false")
               << ",\"inserted_into_path_table\":"
               << (row.inserted_into_path_table ? "true" : "false")
               << ",\"new_conflict_pairs\":";
        writePairArray(output, row.new_conflict_pairs);
        output << ",\"internal_blocker_agents\":";
        writeIntArray(output, row.internal_blocker_agents);
        output << ",\"external_blocker_agents\":";
        writeIntArray(output, row.external_blocker_agents);
        output << '}';
    }
    output << "]}";
}

void JsonlRepairObserver::writeAction(const RepairAction& action)
{
    output << "{\"mode\":\"" << repairActionModeName(action.mode)
           << "\",\"heuristic\":\"" << repairHeuristicName(action.heuristic)
           << "\",\"seed_agent\":" << action.seed_agent
           << ",\"neighborhood_size\":" << action.neighborhood_size
           << ",\"random_seed\":" << action.random_seed
           << ",\"pp_random_seed\":" << action.pp_random_seed
           << ",\"collect_pp_diagnostics\":"
           << (action.collect_pp_diagnostics ? "true" : "false")
           << ",\"agents\":";
    writeIntArray(output, action.agents);
    output << ",\"repair_order\":";
    writeIntArray(output, action.repair_order);
    output << '}';
}

void JsonlRepairObserver::writeState(const RepairState& state)
{
    output << "{\"initialized\":" << (state.initialized ? "true" : "false")
           << ",\"initial_solution_complete\":"
           << (state.initial_solution_complete ? "true" : "false")
           << ",\"feasible\":" << (state.feasible ? "true" : "false")
           << ",\"done\":" << (state.done ? "true" : "false")
           << ",\"iteration\":" << state.iteration
           << ",\"rows\":" << state.rows
           << ",\"cols\":" << state.cols
           << ",\"sum_of_costs\":" << state.sum_of_costs
           << ",\"num_of_colliding_pairs\":" << state.num_of_colliding_pairs
           << ",\"runtime\":" << state.runtime
           << ",\"low_level\":{\"expanded\":" << state.low_level_expanded
           << ",\"generated\":" << state.low_level_generated
           << ",\"reopened\":" << state.low_level_reopened
           << ",\"runs\":" << state.low_level_runs << '}';

    output << ",\"obstacles\":[";
    for (size_t index = 0; index < state.obstacles.size(); index++)
    {
        if (index > 0)
            output << ',';
        output << (int)state.obstacles[index];
    }
    output << "],\"conflict_edges\":[";
    for (size_t index = 0; index < state.conflict_edges.size(); index++)
    {
        if (index > 0)
            output << ',';
        output << '[' << state.conflict_edges[index].first << ','
               << state.conflict_edges[index].second << ']';
    }
    output << "],\"agents\":[";
    for (size_t index = 0; index < state.agents.size(); index++)
    {
        if (index > 0)
            output << ',';
        const auto& agent = state.agents[index];
        output << "{\"id\":" << agent.id
               << ",\"start\":" << agent.start
               << ",\"goal\":" << agent.goal
               << ",\"path_cost\":" << agent.path_cost
               << ",\"shortest_path_cost\":" << agent.shortest_path_cost
               << ",\"delay\":" << agent.delay
               << ",\"conflict_degree\":" << agent.conflict_degree
               << ",\"path\":";
        writeIntArray(output, agent.path);
        output << '}';
    }
    output << "]}";
}

void JsonlRepairObserver::onInitialState(const RepairState& state)
{
    output << "{\"schema_version\":1,\"event\":\"initial\",\"state\":";
    writeState(state);
    output << "}\n";
    flushChecked();
}

void JsonlRepairObserver::onTransition(const RepairState& before,
                                       const RepairTransition& transition,
                                       const RepairState& after)
{
    output << "{\"schema_version\":1,\"event\":\"transition\",\"action\":";
    writeAction(transition.requested_action);
    output << ",\"native_timing_schema\":\"lns2.repair_timing.v2\""
           << ",\"applied_heuristic\":\""
           << repairHeuristicName(transition.applied_heuristic)
           << "\",\"action_valid\":" << (transition.action_valid ? "true" : "false")
           << ",\"generated\":" << (transition.generated ? "true" : "false")
           << ",\"replan_success\":" << (transition.replan_success ? "true" : "false")
           << ",\"neighborhood\":";
    writeIntArray(output, transition.neighborhood);
    output << ",\"repair_order\":";
    writeIntArray(output, transition.repair_order);
    output << ",\"pp_diagnostic\":";
    writePPDiagnostics(transition);
    output << ",\"metrics\":{\"iteration\":" << transition.iteration
           << ",\"conflicts_before\":" << transition.conflicts_before
           << ",\"conflicts_after\":" << transition.conflicts_after
           << ",\"sum_of_costs_before\":" << transition.sum_of_costs_before
           << ",\"sum_of_costs_after\":" << transition.sum_of_costs_after
           << ",\"runtime_before\":" << transition.runtime_before
           << ",\"runtime_after\":" << transition.runtime_after
           << ",\"step_runtime\":" << transition.native_step_seconds
           << ",\"episode_runtime_delta_seconds\":"
           << transition.runtime_after - transition.runtime_before
           << ",\"applied_pp_random_seed\":"
           << transition.applied_pp_random_seed
           << ",\"native_step_seconds\":" << transition.native_step_seconds
           << ",\"native_neighborhood_generation_seconds\":"
           << transition.neighborhood_generation_seconds
           << ",\"native_replan_seconds\":" << transition.replan_seconds
           << ",\"pp_replan_seconds\":" << transition.pp_replan_seconds
           << ",\"native_state_snapshot_seconds\":"
           << transition.state_snapshot_seconds
           << ",\"native_repair_bookkeeping_seconds\":"
           << transition.repair_bookkeeping_seconds
           << ",\"native_residual_seconds\":"
           << transition.native_residual_seconds << '}'
           << ",\"before\":";
    writeState(before);
    output << ",\"after\":";
    writeState(after);
    output << "}\n";
    flushChecked();
}

void JsonlRepairObserver::onFinish(const RepairState& state, bool success)
{
    output << "{\"schema_version\":1,\"event\":\"finish\",\"success\":"
           << (success ? "true" : "false") << ",\"state\":";
    writeState(state);
    output << "}\n";
    flushChecked();
}
