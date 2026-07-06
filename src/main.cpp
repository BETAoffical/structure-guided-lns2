#include "lns2/instance.hpp"
#include "lns2/solver.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

int parse_int(const char* value, const std::string& option) {
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error("invalid value for " + option);
    }
}

void usage(const char* executable) {
    std::cerr
        << "Usage: " << executable
        << " --instance FILE [--seed N] [--neighborhood N]"
           " [--iterations N] [--time-limit-ms N] [--paths FILE]"
           " [--trace FILE] [--guidance-stdio]\n";
}

lns2::GuidanceResponse request_guidance(
    const lns2::Instance& instance,
    const lns2::GuidanceRequest& request) {
    std::cout << "GUIDANCE_REQUEST {\"iteration\":"
              << request.iteration << ",\"seed_conflict\":["
              << request.seed_conflict.first << ','
              << request.seed_conflict.second
              << "],\"baseline_neighborhood\":[";
    for (std::size_t index = 0;
         index < request.baseline_neighborhood.size();
         ++index) {
        if (index > 0) {
            std::cout << ',';
        }
        std::cout << request.baseline_neighborhood[index];
    }
    std::cout << "],\"conflict_events\":[";
    for (std::size_t index = 0;
         index < request.conflict_events.size();
         ++index) {
        if (index > 0) {
            std::cout << ',';
        }
        const auto& event = request.conflict_events[index];
        std::cout << "{\"agents\":[" << event.first_agent << ','
                  << event.second_agent << "],\"timestep\":"
                  << event.timestep << ",\"type\":\""
                  << (event.kind == lns2::ConflictKind::Vertex
                          ? "vertex"
                          : "edge_swap")
                  << "\",\"cells\":[";
        for (std::size_t cell_index = 0;
             cell_index < event.cells.size();
             ++cell_index) {
            if (cell_index > 0) {
                std::cout << ',';
            }
            const auto [row, col] =
                instance.coordinate(event.cells[cell_index]);
            std::cout << '[' << row << ',' << col << ']';
        }
        std::cout << "]}";
    }
    std::cout << "],\"paths\":[";
    for (std::size_t agent = 0; agent < request.paths.size(); ++agent) {
        if (agent > 0) {
            std::cout << ',';
        }
        std::cout << '[';
        for (std::size_t index = 0;
             index < request.paths[agent].size();
             ++index) {
            if (index > 0) {
                std::cout << ',';
            }
            const auto [row, col] =
                instance.coordinate(request.paths[agent][index]);
            std::cout << '[' << row << ',' << col << ']';
        }
        std::cout << ']';
    }
    std::cout << "]}" << std::endl;

    std::string line;
    if (!std::getline(std::cin, line)) {
        return {false, false, -1.0, -1.0, {}, "guidance_eof"};
    }
    std::istringstream response(line);
    std::string tag;
    response >> tag;
    if (tag != "GUIDANCE") {
        return {false, false, -1.0, -1.0, {}, "invalid_response"};
    }
    int use_guidance = 0;
    int out_of_distribution = 0;
    std::string reason;
    lns2::GuidanceResponse result;
    if (!(response >> use_guidance
          >> result.effective_probability
          >> result.nearest_distance
          >> out_of_distribution
          >> reason)) {
        result.fallback_reason = "invalid_response";
        return result;
    }
    result.use_guidance = use_guidance != 0;
    result.out_of_distribution = out_of_distribution != 0;
    if (reason != "-") {
        result.fallback_reason = reason;
    }
    int agent = -1;
    while (response >> agent) {
        result.agents.push_back(agent);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    std::string instance_path;
    std::string output_path;
    std::string trace_path;
    bool guidance_stdio = false;
    lns2::SolverOptions options;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--instance" && i + 1 < argc) {
                instance_path = argv[++i];
            } else if (option == "--seed" && i + 1 < argc) {
                options.seed = static_cast<std::uint32_t>(
                    parse_int(argv[++i], option));
            } else if (option == "--neighborhood" && i + 1 < argc) {
                options.neighborhood_size = parse_int(argv[++i], option);
            } else if (option == "--iterations" && i + 1 < argc) {
                options.max_iterations = parse_int(argv[++i], option);
            } else if (option == "--time-limit-ms" && i + 1 < argc) {
                options.time_limit_ms = parse_int(argv[++i], option);
            } else if (option == "--paths" && i + 1 < argc) {
                output_path = argv[++i];
            } else if (option == "--trace" && i + 1 < argc) {
                trace_path = argv[++i];
            } else if (option == "--guidance-stdio") {
                guidance_stdio = true;
            } else {
                usage(argv[0]);
                return 2;
            }
        }
        if (instance_path.empty()) {
            usage(argv[0]);
            return 2;
        }

        const auto instance = lns2::Instance::load(instance_path);
        lns2::GuidanceCallback guidance;
        if (guidance_stdio) {
            guidance = [&](const lns2::GuidanceRequest& request) {
                return request_guidance(instance, request);
            };
        }
        lns2::Solver solver(instance, options, guidance);
        const auto result = solver.solve();

        std::cout << std::boolalpha << std::fixed << std::setprecision(3);
        if (guidance_stdio) {
            std::cout << "RESULT {"
                      << "\"success\":" << result.metrics.success
                      << ",\"initial_conflicting_pairs\":"
                      << result.metrics.initial_conflicting_pairs
                      << ",\"final_conflicting_pairs\":"
                      << result.metrics.final_conflicting_pairs
                      << ",\"iterations\":"
                      << result.metrics.iterations
                      << ",\"accepted_iterations\":"
                      << result.metrics.accepted_iterations
                      << ",\"makespan\":" << result.metrics.makespan
                      << ",\"sum_of_costs\":"
                      << result.metrics.sum_of_costs
                      << ",\"runtime_ms\":"
                      << result.metrics.runtime_ms
                      << ",\"search_runtime_ms\":"
                      << result.metrics.search_runtime_ms
                      << ",\"guidance_runtime_ms\":"
                      << result.metrics.guidance_runtime_ms
                      << ",\"guidance_requests\":"
                      << result.metrics.guidance_requests
                      << ",\"guidance_used\":"
                      << result.metrics.guidance_used
                      << ",\"guidance_fallbacks\":"
                      << result.metrics.guidance_fallbacks
                      << "}\n";
        } else {
            std::cout << "{\n"
                  << "  \"success\": " << result.metrics.success << ",\n"
                  << "  \"initial_conflicting_pairs\": "
                  << result.metrics.initial_conflicting_pairs << ",\n"
                  << "  \"final_conflicting_pairs\": "
                  << result.metrics.final_conflicting_pairs << ",\n"
                  << "  \"iterations\": " << result.metrics.iterations
                  << ",\n"
                  << "  \"accepted_iterations\": "
                  << result.metrics.accepted_iterations << ",\n"
                  << "  \"makespan\": " << result.metrics.makespan << ",\n"
                  << "  \"sum_of_costs\": "
                  << result.metrics.sum_of_costs << ",\n"
                  << "  \"runtime_ms\": " << result.metrics.runtime_ms
                  << ",\n"
                  << "  \"search_runtime_ms\": "
                  << result.metrics.search_runtime_ms << ",\n"
                  << "  \"guidance_runtime_ms\": "
                  << result.metrics.guidance_runtime_ms << ",\n"
                  << "  \"guidance_requests\": "
                  << result.metrics.guidance_requests << ",\n"
                  << "  \"guidance_used\": "
                  << result.metrics.guidance_used << ",\n"
                  << "  \"guidance_fallbacks\": "
                  << result.metrics.guidance_fallbacks
                  << "\n}\n";
        }

        if (!output_path.empty()) {
            lns2::write_paths(
                output_path, instance, result.paths);
        }
        if (!trace_path.empty()) {
            lns2::write_trace_jsonl(
                trace_path, instance, options, result);
        }
        return result.metrics.success ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
