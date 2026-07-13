#include "structure_guided/jsonl_observer.hpp"

#include <stdexcept>

JsonlRepairObserver::JsonlRepairObserver(const std::string& path) : output(path)
{
    if (!output)
        throw std::runtime_error("failed to open repair trace: " + path);
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

void JsonlRepairObserver::writeAction(const RepairAction& action)
{
    output << "{\"mode\":\"" << repairActionModeName(action.mode)
           << "\",\"heuristic\":\"" << repairHeuristicName(action.heuristic)
           << "\",\"seed_agent\":" << action.seed_agent
           << ",\"neighborhood_size\":" << action.neighborhood_size
           << ",\"agents\":";
    writeIntArray(output, action.agents);
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
    output.flush();
}

void JsonlRepairObserver::onTransition(const RepairState& before,
                                       const RepairTransition& transition,
                                       const RepairState& after)
{
    output << "{\"schema_version\":1,\"event\":\"transition\",\"action\":";
    writeAction(transition.requested_action);
    output << ",\"applied_heuristic\":\""
           << repairHeuristicName(transition.applied_heuristic)
           << "\",\"action_valid\":" << (transition.action_valid ? "true" : "false")
           << ",\"generated\":" << (transition.generated ? "true" : "false")
           << ",\"replan_success\":" << (transition.replan_success ? "true" : "false")
           << ",\"neighborhood\":";
    writeIntArray(output, transition.neighborhood);
    output << ",\"metrics\":{\"iteration\":" << transition.iteration
           << ",\"conflicts_before\":" << transition.conflicts_before
           << ",\"conflicts_after\":" << transition.conflicts_after
           << ",\"sum_of_costs_before\":" << transition.sum_of_costs_before
           << ",\"sum_of_costs_after\":" << transition.sum_of_costs_after
           << ",\"runtime_before\":" << transition.runtime_before
           << ",\"runtime_after\":" << transition.runtime_after << '}'
           << ",\"before\":";
    writeState(before);
    output << ",\"after\":";
    writeState(after);
    output << "}\n";
    output.flush();
}

void JsonlRepairObserver::onFinish(const RepairState& state, bool success)
{
    output << "{\"schema_version\":1,\"event\":\"finish\",\"success\":"
           << (success ? "true" : "false") << ",\"state\":";
    writeState(state);
    output << "}\n";
    output.flush();
}
