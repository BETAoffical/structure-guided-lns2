#include <cstdlib>
#include <new>
#include <vector>
#include "cplns_observer_noalloc.h"

static bool forbid_allocations = false;
void* operator new(std::size_t n) {
    if (forbid_allocations) _exit(91);
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

struct Entry { int location; };
struct Planner { int start_location, goal_location; };
struct Agent { int id; Planner* path_planner; std::vector<Entry> path; };

int main() {
    Planner planner{0, 1};
    std::vector<Agent> agents{{7, &planner, {}}};
    for (int i = 0; i < 12000; ++i) agents[0].path.push_back({i % 2});
    std::vector<int> selected{7};
    // Build all inputs first, then reject any C++ heap allocation by the observer.
    forbid_allocations = true;
    cplns_observer::begin();
    cplns_observer::state("initial_pp", agents, selected, -1, 0, 0, 11999);
    for (int i = 0; i < 129; ++i) {
        cplns_observer::order(selected, 0, i + 2, 1000.0 / (i + 1));
        cplns_observer::state("step", agents, selected, 0, i + 2, 0, 11999, 1, 0, 0, 0.99);
    }
    cplns_observer::state("final", agents, selected, -1, -1, 0, 11999, -1, -1, -1, -1, true);
    forbid_allocations = false;
    return 0;
}
