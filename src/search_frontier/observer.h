#pragma once
#include <array>
#include <memory>
#include <vector>
#include "SIPP.h"

namespace frontier_observer {
using Event = std::array<int, 12>;
struct Capture {
    std::vector<Event> events;
    size_t limit, offered = 0, enqueued = 0, popped = 0, virtual_goals = 0;
    int goal_conflicts = -1, goal_cost = -1;
    explicit Capture(size_t count) : limit(count) { events.reserve(count); }
};
extern thread_local std::unique_ptr<Capture> active;
inline void record(int kind, const SIPPNode* node) {
    if (!active) return;
    if (kind == 1) ++active->enqueued; else ++active->popped;
    if (node->is_goal) { ++active->virtual_goals; return; }
    if (!node->parent || node->num_of_conflicts <= node->parent->num_of_conflicts) return;
    ++active->offered;
    if (active->events.size() >= active->limit) return;
    active->events.push_back({{kind, node->parent->location, node->parent->timestep,
        node->location, node->timestep, node->g_val, node->h_val,
        node->num_of_conflicts, node->parent->num_of_conflicts,
        node->collision_v, node->wait_at_goal, node->is_goal}});
}
inline void finish(const LLNode* node) {
    if (active) { active->goal_conflicts = node->num_of_conflicts; active->goal_cost = node->timestep; }
}
}
