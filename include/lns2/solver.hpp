#pragma once

#include "lns2/instance.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace lns2 {

using Path = std::vector<Location>;
using Paths = std::vector<Path>;

struct SolverOptions {
    std::uint32_t seed = 1234;
    int neighborhood_size = 8;
    int max_iterations = 1000;
    int time_limit_ms = 5000;
};

struct SolverMetrics {
    bool success = false;
    int initial_conflicting_pairs = 0;
    int final_conflicting_pairs = 0;
    int iterations = 0;
    int accepted_iterations = 0;
    int makespan = 0;
    int sum_of_costs = 0;
    double runtime_ms = 0.0;
    double search_runtime_ms = 0.0;
    double guidance_runtime_ms = 0.0;
    int guidance_requests = 0;
    int guidance_used = 0;
    int guidance_fallbacks = 0;
};

enum class ConflictKind {
    Vertex,
    EdgeSwap,
};

struct ConflictEvent {
    int first_agent = -1;
    int second_agent = -1;
    int timestep = -1;
    ConflictKind kind = ConflictKind::Vertex;
    std::vector<Location> cells;

    bool operator==(const ConflictEvent& other) const {
        return first_agent == other.first_agent &&
               second_agent == other.second_agent &&
               timestep == other.timestep && kind == other.kind &&
               cells == other.cells;
    }
};

struct IterationTrace {
    int iteration = 0;
    std::pair<int, int> seed_conflict = {-1, -1};
    std::vector<int> baseline_neighborhood;
    std::vector<int> neighborhood;
    int conflicting_pairs_before = 0;
    int conflicting_pairs_after = -1;
    int sum_of_costs_before = 0;
    int sum_of_costs_after = -1;
    bool candidate_valid = false;
    bool accepted = false;
    double replan_runtime_ms = 0.0;
    std::vector<ConflictEvent> conflict_events_before;
    std::vector<ConflictEvent> conflict_events_after;
    Paths neighborhood_paths_before;
    Paths neighborhood_paths_after;
    bool guidance_requested = false;
    bool guidance_used = false;
    bool guidance_out_of_distribution = false;
    double guidance_effective_probability = -1.0;
    double guidance_nearest_distance = -1.0;
    double guidance_runtime_ms = 0.0;
    std::string guidance_fallback_reason;
};

struct SolveResult {
    Paths paths;
    SolverMetrics metrics;
    std::vector<IterationTrace> trace;
};

struct GuidanceRequest {
    int iteration = 0;
    std::pair<int, int> seed_conflict = {-1, -1};
    std::vector<int> baseline_neighborhood;
    std::vector<ConflictEvent> conflict_events;
    Paths paths;
};

struct GuidanceResponse {
    bool use_guidance = false;
    bool out_of_distribution = false;
    double effective_probability = -1.0;
    double nearest_distance = -1.0;
    std::vector<int> agents;
    std::string fallback_reason;
};

using GuidanceCallback =
    std::function<GuidanceResponse(const GuidanceRequest&)>;

class Solver {
public:
    Solver(
        const Instance& instance,
        SolverOptions options,
        GuidanceCallback guidance = {});

    SolveResult solve();
    static std::vector<ConflictEvent> conflict_events(const Paths& paths);
    static std::vector<std::pair<int, int>> conflicting_pairs(const Paths& paths);
    static bool validate(
        const Instance& instance,
        const Paths& paths,
        std::string* error = nullptr);

private:
    struct NeighborhoodSelection {
        std::pair<int, int> seed_conflict;
        std::vector<int> agents;
    };

    Path plan_agent(int agent_id, const Paths& fixed_paths);
    Paths initial_solution();
    NeighborhoodSelection select_neighborhood(
        const std::vector<std::pair<int, int>>& conflicts);
    Paths replan_neighborhood(
        const Paths& current,
        const std::vector<int>& neighborhood);

    int path_cost(const Paths& paths) const;
    int makespan(const Paths& paths) const;
    bool timed_out() const;

    const Instance& instance_;
    SolverOptions options_;
    std::mt19937 random_;
    std::chrono::steady_clock::time_point deadline_;
    GuidanceCallback guidance_;
};

void write_paths(
    const std::string& path,
    const Instance& instance,
    const Paths& paths);
void write_trace_jsonl(
    const std::string& path,
    const Instance& instance,
    const SolverOptions& options,
    const SolveResult& result);

}  // namespace lns2
