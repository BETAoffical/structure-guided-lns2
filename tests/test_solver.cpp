#include "lns2/instance.hpp"
#include "lns2/solver.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void run_case(
    const std::string& name,
    int neighborhood,
    int max_iterations,
    bool require_lns = false,
    std::uint32_t seed = 1234) {
    const auto path =
        std::string(TEST_DATA_DIR) + "/" + name + ".mapf";
    const auto instance = lns2::Instance::load(path);
    lns2::SolverOptions options;
    options.seed = seed;
    options.neighborhood_size = neighborhood;
    options.max_iterations = max_iterations;
    options.time_limit_ms = 3000;

    lns2::SolveResult result;
    if (require_lns) {
        for (std::uint32_t candidate_seed = seed;
             candidate_seed < seed + 32;
             ++candidate_seed) {
            options.seed = candidate_seed;
            lns2::Solver solver(instance, options);
            result = solver.solve();
            if (result.metrics.success &&
                result.metrics.initial_conflicting_pairs > 0 &&
                result.metrics.accepted_iterations > 0) {
                break;
            }
        }
    } else {
        lns2::Solver solver(instance, options);
        result = solver.solve();
    }
    std::string error;
    require(result.metrics.success, name + " was not solved");
    require(
        lns2::Solver::validate(instance, result.paths, &error),
        name + " validation failed: " + error);
    require(
        result.metrics.final_conflicting_pairs == 0,
        name + " retained conflicts");
    if (require_lns) {
        require(
            result.metrics.initial_conflicting_pairs > 0,
            name + " did not start with a conflict");
        require(
            result.metrics.iterations > 0 &&
                result.metrics.accepted_iterations > 0,
            name + " did not exercise LNS repair");
        require(
            static_cast<int>(result.trace.size()) ==
                result.metrics.iterations,
            name + " trace count does not match iteration count");
        require(
            !result.trace.front().neighborhood.empty(),
            name + " trace omitted the neighborhood");
        require(
            result.trace.front().seed_conflict.first >= 0 &&
                result.trace.front().seed_conflict.second >= 0,
            name + " trace omitted its seed conflict");
        for (const auto& trace : result.trace) {
            require(
                static_cast<int>(trace.conflict_events_before.size()) ==
                    trace.conflicting_pairs_before,
                name + " trace conflict events do not match pair count");
            require(
                trace.neighborhood_paths_before.size() ==
                    trace.neighborhood.size(),
                name + " trace omitted a pre-repair path");
            for (std::size_t index = 0;
                 index < trace.neighborhood.size();
                 ++index) {
                const int agent = trace.neighborhood[index];
                const auto& before =
                    trace.neighborhood_paths_before[index];
                require(
                    !before.empty() &&
                        before.front() ==
                            instance.agents()[agent].start &&
                        before.back() == instance.agents()[agent].goal,
                    name + " trace contains an invalid pre-repair path");
            }
            if (trace.candidate_valid) {
                require(
                    static_cast<int>(
                        trace.conflict_events_after.size()
                    ) == trace.conflicting_pairs_after,
                    name + " candidate conflicts do not match events");
                require(
                    trace.neighborhood_paths_after.size() ==
                        trace.neighborhood.size(),
                    name + " trace omitted a candidate path");
            } else {
                require(
                    trace.neighborhood_paths_after.empty(),
                    name + " invalid candidate retained paths");
            }
        }
    }

    lns2::Solver repeated_solver(instance, options);
    const auto repeated = repeated_solver.solve();
    require(
        repeated.paths == result.paths,
        name + " is not deterministic for a fixed seed");
    require(
        repeated.trace.size() == result.trace.size(),
        name + " trace length is not deterministic");
    for (std::size_t index = 0; index < result.trace.size(); ++index) {
        const auto& first = result.trace[index];
        const auto& second = repeated.trace[index];
        require(
            first.iteration == second.iteration &&
                first.seed_conflict == second.seed_conflict &&
                first.neighborhood == second.neighborhood &&
                first.conflicting_pairs_before ==
                    second.conflicting_pairs_before &&
                first.conflicting_pairs_after ==
                    second.conflicting_pairs_after &&
                first.sum_of_costs_before ==
                    second.sum_of_costs_before &&
                first.sum_of_costs_after ==
                    second.sum_of_costs_after &&
                first.candidate_valid == second.candidate_valid &&
                first.accepted == second.accepted &&
                first.conflict_events_before ==
                    second.conflict_events_before &&
                first.conflict_events_after ==
                    second.conflict_events_after &&
                first.neighborhood_paths_before ==
                    second.neighborhood_paths_before &&
                first.neighborhood_paths_after ==
                    second.neighborhood_paths_after,
            name + " trace decisions are not deterministic");
    }
}

void test_guidance_boundary() {
    const auto path =
        std::string(TEST_DATA_DIR) + "/lns_required.mapf";
    const auto instance = lns2::Instance::load(path);
    lns2::SolverOptions options;
    options.seed = 1;
    options.neighborhood_size = 2;
    options.max_iterations = 100;
    options.time_limit_ms = 3000;

    lns2::Solver guided(
        instance,
        options,
        [](const lns2::GuidanceRequest& request) {
            lns2::GuidanceResponse response;
            response.use_guidance = true;
            response.effective_probability = 0.9;
            response.nearest_distance = 0.1;
            response.agents = request.baseline_neighborhood;
            return response;
        });
    const auto guided_result = guided.solve();
    require(guided_result.metrics.success, "guided boundary did not solve");
    require(
        guided_result.metrics.guidance_requests > 0 &&
            guided_result.metrics.guidance_used ==
                guided_result.metrics.guidance_requests &&
            guided_result.metrics.guidance_fallbacks == 0,
        "valid guidance was not used");
    require(
        guided_result.trace.front().guidance_requested &&
            guided_result.trace.front().guidance_used,
        "guided trace omitted guidance status");

    lns2::Solver invalid(
        instance,
        options,
        [](const lns2::GuidanceRequest& request) {
            lns2::GuidanceResponse response;
            response.use_guidance = true;
            response.agents = {
                request.seed_conflict.first,
                request.seed_conflict.first};
            return response;
        });
    const auto fallback_result = invalid.solve();
    require(fallback_result.metrics.success, "fallback boundary did not solve");
    require(
        fallback_result.metrics.guidance_requests > 0 &&
            fallback_result.metrics.guidance_used == 0 &&
            fallback_result.metrics.guidance_fallbacks ==
                fallback_result.metrics.guidance_requests,
        "invalid guidance did not fall back");
    require(
        fallback_result.trace.front().neighborhood ==
            fallback_result.trace.front().baseline_neighborhood,
        "fallback changed the baseline neighborhood");
}

}  // namespace

int main() {
    try {
        run_case("crossing", 2, 100);
        run_case("lns_required", 2, 100, true, 1);
        run_case("warehouse_small", 6, 500);
        test_guidance_boundary();
        std::cout << "all tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
