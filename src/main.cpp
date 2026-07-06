#include "lns2/instance.hpp"
#include "lns2/solver.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
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
           " [--trace FILE]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string instance_path;
    std::string output_path;
    std::string trace_path;
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
        lns2::Solver solver(instance, options);
        const auto result = solver.solve();

        std::cout << std::boolalpha << std::fixed << std::setprecision(3)
                  << "{\n"
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
                  << "\n}\n";

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
