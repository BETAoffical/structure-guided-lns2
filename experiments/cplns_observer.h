#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>

// Output-only instrumentation for a separately built, single-solver reference.
namespace cplns_observer {
inline thread_local std::ofstream output;
inline thread_local std::chrono::steady_clock::time_point origin;
inline thread_local unsigned decisions = 0;
inline constexpr unsigned limit = 128;

inline void begin() {
    const char* path = std::getenv("CPLNS_OBSERVER_PATH");
    if (!path) return;
    origin = std::chrono::steady_clock::now();
    output.open(path);
    if (!output) throw std::runtime_error("cannot open observer output");
    output << std::setprecision(17);
}

inline double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - origin).count();
}

template<class Agents, class IDs>
void state(const char* event, const Agents& agents, const IDs& selected,
           int restart, int iteration, int conflicts, int cost,
           int accepted = -1, int old_pairs = -1, int new_pairs = -1,
           double temperature = -1, bool force = false) {
    if (!output.is_open() || (!force && decisions > limit)) return;
    const double elapsed = seconds();
    output << "{\"event\":\"" << event << "\",\"seconds\":" << elapsed
           << ",\"restart\":" << restart << ",\"iteration\":" << iteration
           << ",\"conflicts\":" << conflicts << ",\"cost\":" << cost
           << ",\"accepted\":" << accepted << ",\"old_pairs\":" << old_pairs
           << ",\"new_pairs\":" << new_pairs << ",\"temperature\":" << temperature
           << ",\"decisions_seen\":" << decisions << ",\"selected\":[";
    bool comma = false;
    for (auto id : selected) { if (comma) output << ','; output << id; comma = true; }
    output << "],\"agents\":[";
    comma = false;
    for (const auto& agent : agents) {
        if (comma) output << ',';
        comma = true;
        output << "{\"id\":" << agent.id << ",\"start\":"
               << agent.path_planner->start_location << ",\"goal\":"
               << agent.path_planner->goal_location << ",\"path\":[";
        bool separator = false;
        for (const auto& entry : agent.path) {
            if (separator) output << ',';
            output << entry.location;
            separator = true;
        }
        output << "]}";
    }
    output << "]}\n";
    output.flush();
    if (!output) throw std::runtime_error("observer write failed");
}

template<class IDs>
void order(const IDs& ids, int restart, int iteration, double temperature) {
    if (!output.is_open()) return;
    ++decisions;
    if (decisions > limit) return;
    output << "{\"event\":\"pp_order\",\"seconds\":" << seconds()
           << ",\"restart\":" << restart << ",\"iteration\":" << iteration
           << ",\"temperature\":" << temperature << ",\"order\":[";
    bool comma = false;
    for (auto id : ids) { if (comma) output << ','; output << id; comma = true; }
    output << "]}\n";
    output.flush();
    if (!output) throw std::runtime_error("observer write failed");
}
} // namespace cplns_observer
