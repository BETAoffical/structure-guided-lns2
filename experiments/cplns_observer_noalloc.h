#pragma once

#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

// POD TLS and stack buffers only: Target search must not see new heap allocations.
namespace cplns_observer {
inline thread_local int descriptor = -1;
inline thread_local timespec origin{};
inline thread_local unsigned decisions = 0;
inline constexpr unsigned limit = 128;

inline void begin() {
    const char* path = std::getenv("CPLNS_OBSERVER_PATH");
    if (!path) return;
    if (clock_gettime(CLOCK_MONOTONIC, &origin) != 0) _exit(86);
    descriptor = ::open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (descriptor < 0) _exit(86);
}

inline double seconds() {
    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) _exit(86);
    return double(now.tv_sec - origin.tv_sec) + double(now.tv_nsec - origin.tv_nsec) * 1e-9;
}

struct Buffer {
    char data[8192];
    unsigned used = 0;

    void flush() {
        unsigned offset = 0;
        while (offset < used) {
            const ssize_t n = ::write(descriptor, data + offset, used - offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) _exit(86);
            offset += static_cast<unsigned>(n);
        }
        used = 0;
    }
    void text(const char* s) {
        while (*s) {
            if (used == sizeof(data)) flush();
            data[used++] = *s++;
        }
    }
    template<class T> void number(T value) {
        char local[128];
        auto result = std::to_chars(local, local + sizeof(local) - 1, value);
        if (result.ec != std::errc{}) _exit(86);
        *result.ptr = '\0';
        text(local);
    }
    template<class T> void field(const char* key, T value) {
        text(",\""); text(key); text("\":"); number(value);
    }
};

template<class Agents, class IDs>
void state(const char* event, const Agents& agents, const IDs& selected,
           int restart, int iteration, int conflicts, int cost,
           int accepted = -1, int old_pairs = -1, int new_pairs = -1,
           double temperature = -1, bool force = false) {
    if (descriptor < 0 || (!force && decisions > limit)) return;
    const double elapsed = seconds();
    Buffer b;
    b.text("{\"event\":\""); b.text(event); b.text("\"");
    b.field("seconds", elapsed); b.field("restart", restart); b.field("iteration", iteration);
    b.field("conflicts", conflicts); b.field("cost", cost); b.field("accepted", accepted);
    b.field("old_pairs", old_pairs); b.field("new_pairs", new_pairs);
    b.field("temperature", temperature); b.field("decisions_seen", decisions);
    b.text(",\"selected\":[");
    bool comma = false;
    for (auto id : selected) { if (comma) b.text(","); b.number(id); comma = true; }
    b.text("],\"agents\":[");
    comma = false;
    for (const auto& agent : agents) {
        if (comma) b.text(",");
        comma = true;
        b.text("{\"id\":"); b.number(agent.id);
        b.field("start", agent.path_planner->start_location);
        b.field("goal", agent.path_planner->goal_location);
        b.text(",\"path\":[");
        bool separator = false;
        for (const auto& entry : agent.path) {
            if (separator) b.text(",");
            b.number(entry.location);
            separator = true;
        }
        b.text("]}");
    }
    b.text("]}\n");
    b.flush();
    if (force) {
        if (::close(descriptor) != 0) _exit(86);
        descriptor = -1;
    }
}

template<class IDs>
void order(const IDs& ids, int restart, int iteration, double temperature) {
    if (descriptor < 0) return;
    ++decisions;
    if (decisions > limit) return;
    Buffer b;
    b.text("{\"event\":\"pp_order\"");
    b.field("seconds", seconds()); b.field("restart", restart); b.field("iteration", iteration);
    b.field("temperature", temperature); b.text(",\"order\":[");
    bool comma = false;
    for (auto id : ids) { if (comma) b.text(","); b.number(id); comma = true; }
    b.text("]}\n"); b.flush();
}
} // namespace cplns_observer
