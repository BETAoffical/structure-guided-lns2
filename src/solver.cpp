#include "lns2/solver.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace lns2 {
namespace {

Location position_at(const Path& path, int timestep) {
    return path[std::min(timestep, static_cast<int>(path.size()) - 1)];
}

struct SearchNode {
    Location location;
    int timestep;
    int conflicts;
    int heuristic;
    int parent;
};

struct QueueEntry {
    int conflicts;
    int estimated_length;
    int heuristic;
    int node;

    bool operator>(const QueueEntry& other) const {
        return std::tie(conflicts, estimated_length, heuristic, node) >
               std::tie(
                   other.conflicts,
                   other.estimated_length,
                   other.heuristic,
                   other.node);
    }
};

int transition_conflicts(
    Location current,
    Location next,
    int timestep,
    const Paths& fixed_paths) {
    int conflicts = 0;
    for (const auto& path : fixed_paths) {
        if (path.empty()) {
            continue;
        }
        if (position_at(path, timestep) == next) {
            ++conflicts;
        }
        if (timestep > 0 &&
            position_at(path, timestep - 1) == next &&
            position_at(path, timestep) == current &&
            current != next) {
            ++conflicts;
        }
    }
    return conflicts;
}

int future_goal_conflicts(
    Location goal,
    int arrival,
    int future_limit,
    const Paths& fixed_paths) {
    int conflicts = 0;
    for (const auto& path : fixed_paths) {
        if (path.empty()) {
            continue;
        }
        for (int t = arrival + 1; t <= future_limit; ++t) {
            if (position_at(path, t) == goal) {
                ++conflicts;
            }
        }
    }
    return conflicts;
}

}  // namespace

Solver::Solver(
    const Instance& instance,
    SolverOptions options,
    GuidanceCallback guidance)
    : instance_(instance),
      options_(options),
      random_(options.seed),
      guidance_(std::move(guidance)) {
    options_.neighborhood_size =
        std::max(1, std::min(
                        options_.neighborhood_size,
                        static_cast<int>(instance_.agents().size())));
    options_.max_iterations = std::max(0, options_.max_iterations);
    options_.time_limit_ms = std::max(1, options_.time_limit_ms);
}

Path Solver::plan_agent(int agent_id, const Paths& fixed_paths) {
    const auto& agent = instance_.agents()[agent_id];
    const int static_distance =
        instance_.shortest_distance(agent.start, agent.goal);
    if (static_distance < 0) {
        return {};
    }

    int reference_horizon = 0;
    for (const auto& path : fixed_paths) {
        reference_horizon =
            std::max(reference_horizon, static_cast<int>(path.size()) - 1);
    }
    const int horizon = std::max(
        static_distance + instance_.size(),
        reference_horizon + instance_.size());

    std::vector<int> distance_to_goal(instance_.size(), -1);
    std::queue<Location> distance_open;
    distance_to_goal[agent.goal] = 0;
    distance_open.push(agent.goal);
    while (!distance_open.empty()) {
        const auto current = distance_open.front();
        distance_open.pop();
        for (const auto next : instance_.neighbors_with_wait(current)) {
            if (next == current || distance_to_goal[next] >= 0) {
                continue;
            }
            distance_to_goal[next] = distance_to_goal[current] + 1;
            distance_open.push(next);
        }
    }

    std::vector<SearchNode> nodes;
    nodes.push_back(
        {agent.start, 0, 0, distance_to_goal[agent.start], -1});
    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>>
        open;
    open.push({0, distance_to_goal[agent.start],
               distance_to_goal[agent.start], 0});

    const int state_count = (horizon + 1) * instance_.size();
    std::vector<int> best_conflicts(
        state_count, std::numeric_limits<int>::max());
    best_conflicts[agent.start] = 0;

    int best_goal_node = -1;
    int best_goal_conflicts = std::numeric_limits<int>::max();
    int best_goal_time = std::numeric_limits<int>::max();

    while (!open.empty() && !timed_out()) {
        const auto entry = open.top();
        open.pop();
        const auto current = nodes[entry.node];
        const int state_index =
            current.timestep * instance_.size() + current.location;
        if (current.conflicts != best_conflicts[state_index]) {
            continue;
        }
        if (current.conflicts > best_goal_conflicts) {
            break;
        }

        if (current.location == agent.goal) {
            const int total_conflicts =
                current.conflicts +
                future_goal_conflicts(
                    agent.goal,
                    current.timestep,
                    reference_horizon + instance_.size(),
                    fixed_paths);
            if (std::tie(total_conflicts, current.timestep) <
                std::tie(best_goal_conflicts, best_goal_time)) {
                best_goal_conflicts = total_conflicts;
                best_goal_time = current.timestep;
                best_goal_node = entry.node;
                if (total_conflicts == 0) {
                    break;
                }
            }
        }

        if (current.timestep >= horizon) {
            continue;
        }
        for (const auto next :
             instance_.neighbors_with_wait(current.location)) {
            const int next_time = current.timestep + 1;
            const int next_conflicts =
                current.conflicts +
                transition_conflicts(
                    current.location, next, next_time, fixed_paths);
            const int next_index =
                next_time * instance_.size() + next;
            if (next_conflicts >= best_conflicts[next_index]) {
                continue;
            }
            best_conflicts[next_index] = next_conflicts;
            const int node_index = static_cast<int>(nodes.size());
            nodes.push_back(
                {next, next_time, next_conflicts,
                 distance_to_goal[next], entry.node});
            open.push(
                {next_conflicts,
                 next_time + distance_to_goal[next],
                 distance_to_goal[next],
                 node_index});
        }
    }

    if (best_goal_node < 0) {
        return {};
    }
    Path path;
    for (int node = best_goal_node; node >= 0; node = nodes[node].parent) {
        path.push_back(nodes[node].location);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

Paths Solver::initial_solution() {
    const int agent_count = static_cast<int>(instance_.agents().size());
    std::vector<int> order(agent_count);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), random_);

    Paths paths(agent_count);
    Paths fixed;
    for (const int agent : order) {
        paths[agent] = plan_agent(agent, fixed);
        if (paths[agent].empty()) {
            return {};
        }
        fixed.push_back(paths[agent]);
    }
    return paths;
}

std::vector<ConflictEvent> Solver::conflict_events(const Paths& paths) {
    std::vector<ConflictEvent> result;
    for (int first = 0; first < static_cast<int>(paths.size()); ++first) {
        for (int second = first + 1;
             second < static_cast<int>(paths.size());
             ++second) {
            if (paths[first].empty() || paths[second].empty()) {
                continue;
            }
            const int horizon = std::max(
                static_cast<int>(paths[first].size()),
                static_cast<int>(paths[second].size()));
            bool conflict = false;
            for (int t = 0; t < horizon; ++t) {
                if (position_at(paths[first], t) ==
                    position_at(paths[second], t)) {
                    result.push_back(
                        {first,
                         second,
                         t,
                         ConflictKind::Vertex,
                         {position_at(paths[first], t)}});
                    conflict = true;
                    break;
                }
                if (t > 0 &&
                    position_at(paths[first], t - 1) ==
                        position_at(paths[second], t) &&
                    position_at(paths[first], t) ==
                        position_at(paths[second], t - 1)) {
                    result.push_back(
                        {first,
                         second,
                         t,
                         ConflictKind::EdgeSwap,
                         {position_at(paths[first], t - 1),
                          position_at(paths[first], t)}});
                    conflict = true;
                    break;
                }
            }
            (void)conflict;
        }
    }
    return result;
}

std::vector<std::pair<int, int>> Solver::conflicting_pairs(
    const Paths& paths) {
    std::vector<std::pair<int, int>> result;
    for (const auto& event : conflict_events(paths)) {
        result.emplace_back(event.first_agent, event.second_agent);
    }
    return result;
}

Solver::NeighborhoodSelection Solver::select_neighborhood(
    const std::vector<std::pair<int, int>>& conflicts) {
    const int agent_count = static_cast<int>(instance_.agents().size());
    std::vector<std::vector<int>> graph(agent_count);
    for (const auto& [first, second] : conflicts) {
        graph[first].push_back(second);
        graph[second].push_back(first);
    }

    std::uniform_int_distribution<int> conflict_choice(
        0, static_cast<int>(conflicts.size()) - 1);
    const auto seed_pair = conflicts[conflict_choice(random_)];
    std::vector<int> neighborhood;
    std::queue<int> open;
    std::vector<bool> selected(agent_count, false);
    const int seed =
        std::uniform_int_distribution<int>(0, 1)(random_) == 0
            ? seed_pair.first
            : seed_pair.second;
    selected[seed] = true;
    open.push(seed);

    while (!open.empty() &&
           static_cast<int>(neighborhood.size()) <
               options_.neighborhood_size) {
        const int current = open.front();
        open.pop();
        neighborhood.push_back(current);
        std::shuffle(
            graph[current].begin(), graph[current].end(), random_);
        for (const int next : graph[current]) {
            if (!selected[next]) {
                selected[next] = true;
                open.push(next);
            }
        }
    }

    std::vector<int> remaining(agent_count);
    std::iota(remaining.begin(), remaining.end(), 0);
    std::shuffle(remaining.begin(), remaining.end(), random_);
    std::stable_sort(
        remaining.begin(), remaining.end(),
        [&](int left, int right) {
            return graph[left].size() > graph[right].size();
        });
    for (const int agent : remaining) {
        if (static_cast<int>(neighborhood.size()) >=
            options_.neighborhood_size) {
            break;
        }
        if (!selected[agent]) {
            selected[agent] = true;
            neighborhood.push_back(agent);
        }
    }
    return {seed_pair, std::move(neighborhood)};
}

Paths Solver::replan_neighborhood(
    const Paths& current,
    const std::vector<int>& neighborhood) {
    Paths candidate = current;
    std::vector<bool> selected(current.size(), false);
    for (const int agent : neighborhood) {
        selected[agent] = true;
        candidate[agent].clear();
    }

    Paths fixed;
    for (int agent = 0; agent < static_cast<int>(candidate.size()); ++agent) {
        if (!selected[agent]) {
            fixed.push_back(candidate[agent]);
        }
    }

    auto order = neighborhood;
    std::shuffle(order.begin(), order.end(), random_);
    for (const int agent : order) {
        candidate[agent] = plan_agent(agent, fixed);
        if (candidate[agent].empty()) {
            return {};
        }
        fixed.push_back(candidate[agent]);
    }
    return candidate;
}

SolveResult Solver::solve() {
    const auto start_time = std::chrono::steady_clock::now();
    deadline_ = start_time +
                std::chrono::milliseconds(options_.time_limit_ms);

    SolveResult result;
    result.paths = initial_solution();
    if (result.paths.empty()) {
        result.metrics.runtime_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start_time)
                .count();
        result.metrics.search_runtime_ms = result.metrics.runtime_ms;
        return result;
    }

    auto current_events = conflict_events(result.paths);
    std::vector<std::pair<int, int>> conflicts;
    for (const auto& event : current_events) {
        conflicts.emplace_back(event.first_agent, event.second_agent);
    }
    result.metrics.initial_conflicting_pairs =
        static_cast<int>(conflicts.size());

    while (!conflicts.empty() &&
           result.metrics.iterations < options_.max_iterations &&
           !timed_out()) {
        ++result.metrics.iterations;
        auto selection = select_neighborhood(conflicts);
        IterationTrace trace;
        trace.iteration = result.metrics.iterations;
        trace.seed_conflict = selection.seed_conflict;
        trace.baseline_neighborhood = selection.agents;
        if (guidance_) {
            trace.guidance_requested = true;
            ++result.metrics.guidance_requests;
            const auto guidance_start =
                std::chrono::steady_clock::now();
            GuidanceResponse response;
            try {
                response = guidance_(
                    {trace.iteration,
                     selection.seed_conflict,
                     selection.agents,
                     current_events,
                     result.paths});
            } catch (const std::exception& error) {
                response.fallback_reason =
                    std::string("callback_error:") + error.what();
            } catch (...) {
                response.fallback_reason = "callback_error";
            }
            const auto guidance_duration =
                std::chrono::steady_clock::now() - guidance_start;
            deadline_ += guidance_duration;
            trace.guidance_runtime_ms =
                std::chrono::duration<double, std::milli>(
                    guidance_duration)
                    .count();
            result.metrics.guidance_runtime_ms +=
                trace.guidance_runtime_ms;
            trace.guidance_out_of_distribution =
                response.out_of_distribution;
            trace.guidance_effective_probability =
                response.effective_probability;
            trace.guidance_nearest_distance =
                response.nearest_distance;
            trace.guidance_fallback_reason =
                response.fallback_reason;

            bool valid = response.use_guidance;
            const int expected_size = options_.neighborhood_size;
            if (valid &&
                static_cast<int>(response.agents.size()) != expected_size) {
                valid = false;
                trace.guidance_fallback_reason = "wrong_size";
            }
            std::vector<bool> seen(instance_.agents().size(), false);
            for (const int agent : response.agents) {
                if (agent < 0 ||
                    agent >= static_cast<int>(seen.size()) ||
                    seen[agent]) {
                    valid = false;
                    trace.guidance_fallback_reason = "invalid_agents";
                    break;
                }
                seen[agent] = true;
            }
            if (valid &&
                (!seen[selection.seed_conflict.first] ||
                 !seen[selection.seed_conflict.second])) {
                valid = false;
                trace.guidance_fallback_reason = "seed_pair_missing";
            }
            if (valid) {
                selection.agents = std::move(response.agents);
                trace.guidance_used = true;
                ++result.metrics.guidance_used;
            } else {
                if (trace.guidance_fallback_reason.empty()) {
                    trace.guidance_fallback_reason =
                        response.out_of_distribution
                            ? "out_of_distribution"
                            : "guidance_declined";
                }
                ++result.metrics.guidance_fallbacks;
            }
        }
        trace.neighborhood = selection.agents;
        trace.conflict_events_before = current_events;
        for (const int agent : selection.agents) {
            trace.neighborhood_paths_before.push_back(
                result.paths[agent]
            );
        }
        trace.conflicting_pairs_before =
            static_cast<int>(conflicts.size());
        trace.sum_of_costs_before = path_cost(result.paths);
        const auto replan_start = std::chrono::steady_clock::now();
        auto candidate =
            replan_neighborhood(result.paths, selection.agents);
        trace.replan_runtime_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - replan_start)
                .count();
        if (candidate.empty()) {
            result.trace.push_back(std::move(trace));
            continue;
        }
        trace.candidate_valid = true;
        auto candidate_events = conflict_events(candidate);
        trace.conflict_events_after = candidate_events;
        for (const int agent : selection.agents) {
            trace.neighborhood_paths_after.push_back(candidate[agent]);
        }
        std::vector<std::pair<int, int>> candidate_conflicts;
        for (const auto& event : candidate_events) {
            candidate_conflicts.emplace_back(
                event.first_agent, event.second_agent
            );
        }
        trace.conflicting_pairs_after =
            static_cast<int>(candidate_conflicts.size());
        trace.sum_of_costs_after = path_cost(candidate);
        const auto current_score =
            std::make_pair(
                static_cast<int>(conflicts.size()),
                trace.sum_of_costs_before);
        const auto candidate_score =
            std::make_pair(
                static_cast<int>(candidate_conflicts.size()),
                trace.sum_of_costs_after);
        if (candidate_score <= current_score) {
            trace.accepted = true;
            result.paths = std::move(candidate);
            conflicts = std::move(candidate_conflicts);
            current_events = std::move(candidate_events);
            ++result.metrics.accepted_iterations;
        }
        result.trace.push_back(std::move(trace));
    }

    result.metrics.final_conflicting_pairs =
        static_cast<int>(conflicts.size());
    result.metrics.success =
        conflicts.empty() && validate(instance_, result.paths);
    result.metrics.makespan = makespan(result.paths);
    result.metrics.sum_of_costs = path_cost(result.paths);
    result.metrics.runtime_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_time)
                .count();
    result.metrics.search_runtime_ms =
        result.metrics.runtime_ms -
        result.metrics.guidance_runtime_ms;
    return result;
}

bool Solver::validate(
    const Instance& instance,
    const Paths& paths,
    std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    if (paths.size() != instance.agents().size()) {
        return fail("path count does not match agent count");
    }
    for (int agent = 0; agent < static_cast<int>(paths.size()); ++agent) {
        const auto& path = paths[agent];
        if (path.empty()) {
            return fail("empty path for agent " + std::to_string(agent));
        }
        if (path.front() != instance.agents()[agent].start ||
            path.back() != instance.agents()[agent].goal) {
            return fail(
                "wrong start or goal for agent " + std::to_string(agent));
        }
        for (int t = 0; t < static_cast<int>(path.size()); ++t) {
            if (!instance.traversable(path[t])) {
                return fail(
                    "blocked location in path for agent " +
                    std::to_string(agent));
            }
            if (t > 0) {
                const auto [r1, c1] = instance.coordinate(path[t - 1]);
                const auto [r2, c2] = instance.coordinate(path[t]);
                if (std::abs(r1 - r2) + std::abs(c1 - c2) > 1) {
                    return fail(
                        "non-adjacent move for agent " +
                        std::to_string(agent));
                }
            }
        }
    }
    if (!conflicting_pairs(paths).empty()) {
        return fail("solution contains agent conflicts");
    }
    return true;
}

int Solver::path_cost(const Paths& paths) const {
    int cost = 0;
    for (const auto& path : paths) {
        cost += std::max(0, static_cast<int>(path.size()) - 1);
    }
    return cost;
}

int Solver::makespan(const Paths& paths) const {
    int value = 0;
    for (const auto& path : paths) {
        value = std::max(value, static_cast<int>(path.size()) - 1);
    }
    return value;
}

bool Solver::timed_out() const {
    return std::chrono::steady_clock::now() >= deadline_;
}

void write_paths(
    const std::string& path,
    const Instance& instance,
    const Paths& paths) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write paths: " + path);
    }
    for (int agent = 0; agent < static_cast<int>(paths.size()); ++agent) {
        output << "agent " << agent << ':';
        for (const auto location : paths[agent]) {
            const auto [row, col] = instance.coordinate(location);
            output << " (" << row << ',' << col << ')';
        }
        output << '\n';
    }
}

void write_trace_jsonl(
    const std::string& path,
    const Instance& instance,
    const SolverOptions& options,
    const SolveResult& result) {
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    std::ofstream output(destination);
    if (!output) {
        throw std::runtime_error("cannot write trace: " + path);
    }
    output << std::boolalpha << std::fixed << std::setprecision(3);
    const int schema_version =
        result.metrics.guidance_requests > 0 ? 3 : 2;
    const auto write_int_array = [&](const std::vector<int>& values) {
        output << '[';
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index > 0) {
                output << ',';
            }
            output << values[index];
        }
        output << ']';
    };
    const auto write_json_string = [&](const std::string& value) {
        output << '"';
        for (const char character : value) {
            if (character == '"' || character == '\\') {
                output << '\\';
            }
            output << character;
        }
        output << '"';
    };
    const auto write_path_json = [&](const Path& value) {
        output << '[';
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index > 0) {
                output << ',';
            }
            const auto [row, col] = instance.coordinate(value[index]);
            output << '[' << row << ',' << col << ']';
        }
        output << ']';
    };
    const auto write_conflict_events =
        [&](const std::vector<ConflictEvent>& events) {
            output << '[';
            for (std::size_t index = 0; index < events.size(); ++index) {
                if (index > 0) {
                    output << ',';
                }
                const auto& event = events[index];
                output << "{\"agents\":[" << event.first_agent << ','
                       << event.second_agent << ']'
                       << ",\"timestep\":" << event.timestep
                       << ",\"type\":\""
                       << (event.kind == ConflictKind::Vertex
                               ? "vertex"
                               : "edge_swap")
                       << "\",\"cells\":[";
                for (std::size_t cell_index = 0;
                     cell_index < event.cells.size();
                     ++cell_index) {
                    if (cell_index > 0) {
                        output << ',';
                    }
                    const auto [row, col] =
                        instance.coordinate(event.cells[cell_index]);
                    output << '[' << row << ',' << col << ']';
                }
                output << "]}";
            }
            output << ']';
        };
    const auto write_neighborhood_paths =
        [&](const std::vector<int>& agents, const Paths& paths) {
            output << '[';
            for (std::size_t index = 0; index < paths.size(); ++index) {
                if (index > 0) {
                    output << ',';
                }
                output << "{\"agent\":" << agents[index]
                       << ",\"path\":";
                write_path_json(paths[index]);
                output << '}';
            }
            output << ']';
        };
    for (const auto& trace : result.trace) {
        output << "{\"schema_version\":" << schema_version
               << ",\"event_type\":\"iteration\""
               << ",\"solver_seed\":" << options.seed
               << ",\"iteration\":" << trace.iteration
               << ",\"seed_conflict\":[" << trace.seed_conflict.first
               << ',' << trace.seed_conflict.second << ']'
               << ",\"baseline_neighborhood\":";
        write_int_array(trace.baseline_neighborhood);
        output << ",\"neighborhood\":";
        write_int_array(trace.neighborhood);
        output
               << ",\"conflicting_pairs_before\":"
               << trace.conflicting_pairs_before
               << ",\"conflicting_pairs_after\":"
               << trace.conflicting_pairs_after
               << ",\"sum_of_costs_before\":"
               << trace.sum_of_costs_before
               << ",\"sum_of_costs_after\":"
               << trace.sum_of_costs_after
               << ",\"candidate_valid\":" << trace.candidate_valid
               << ",\"accepted\":" << trace.accepted
               << ",\"replan_runtime_ms\":"
               << trace.replan_runtime_ms
               << ",\"guidance_requested\":"
               << trace.guidance_requested
               << ",\"guidance_used\":" << trace.guidance_used
               << ",\"guidance_out_of_distribution\":"
               << trace.guidance_out_of_distribution
               << ",\"guidance_effective_probability\":"
               << trace.guidance_effective_probability
               << ",\"guidance_nearest_distance\":"
               << trace.guidance_nearest_distance
               << ",\"guidance_runtime_ms\":"
               << trace.guidance_runtime_ms
               << ",\"guidance_fallback_reason\":";
        write_json_string(trace.guidance_fallback_reason);
        output << ",\"conflict_events_before\":";
        write_conflict_events(trace.conflict_events_before);
        output << ",\"conflict_events_after\":";
        write_conflict_events(trace.conflict_events_after);
        output << ",\"neighborhood_paths_before\":";
        write_neighborhood_paths(
            trace.neighborhood, trace.neighborhood_paths_before
        );
        output << ",\"neighborhood_paths_after\":";
        write_neighborhood_paths(
            trace.neighborhood, trace.neighborhood_paths_after
        );
        output << "}\n";
    }
    const auto& metrics = result.metrics;
    output << "{\"schema_version\":" << schema_version
           << ",\"event_type\":\"summary\""
           << ",\"solver_seed\":" << options.seed
           << ",\"success\":" << metrics.success
           << ",\"initial_conflicting_pairs\":"
           << metrics.initial_conflicting_pairs
           << ",\"final_conflicting_pairs\":"
           << metrics.final_conflicting_pairs
           << ",\"iterations\":" << metrics.iterations
           << ",\"accepted_iterations\":"
           << metrics.accepted_iterations
           << ",\"makespan\":" << metrics.makespan
           << ",\"sum_of_costs\":" << metrics.sum_of_costs
           << ",\"runtime_ms\":" << metrics.runtime_ms
           << ",\"search_runtime_ms\":"
           << metrics.search_runtime_ms
           << ",\"guidance_runtime_ms\":"
           << metrics.guidance_runtime_ms
           << ",\"guidance_requests\":"
           << metrics.guidance_requests
           << ",\"guidance_used\":" << metrics.guidance_used
           << ",\"guidance_fallbacks\":"
           << metrics.guidance_fallbacks << "}\n";
}

}  // namespace lns2
