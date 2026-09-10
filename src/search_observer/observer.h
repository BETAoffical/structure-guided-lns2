#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <vector>
#include "ReservationTable.h"

namespace occupancy_observer {
using Event = std::array<int, 10>;
struct Capture {
    std::vector<Event> events;
    size_t limit;
    size_t offered = 0;
    size_t queries = 0;
    explicit Capture(size_t count) : limit(count) { events.reserve(count); }
};
extern thread_local std::unique_ptr<Capture> active;
inline void query() { if (active) ++active->queries; }
inline void record(const Event& event) {
    if (!active || (!event[8] && !event[9])) return;
    ++active->offered;
    if (active->events.size() < active->limit) active->events.push_back(event);
}
}

class ObservedReservationTable : public ReservationTable {
public:
    using ReservationTable::ReservationTable;
    list<tuple<int, int, int, bool, bool>> get_safe_intervals(int from, int to, int lower, int upper) {
        auto result = ReservationTable::get_safe_intervals(from, to, lower, upper);
        occupancy_observer::query();
        for (const auto& row : result)
            occupancy_observer::record({{0, from, to, lower, upper, get<1>(row), get<0>(row),
                                          get<2>(row), get<3>(row), get<4>(row)}});
        return result;
    }
    Interval get_first_safe_interval(size_t location) {
        auto result = ReservationTable::get_first_safe_interval(location);
        occupancy_observer::query();
        occupancy_observer::record({{1, static_cast<int>(location), static_cast<int>(location),
            get<0>(result), get<1>(result), get<0>(result), get<1>(result), get<1>(result), get<2>(result), 0}});
        return result;
    }
    bool find_safe_interval(Interval& interval, size_t location, int time) {
        bool found = ReservationTable::find_safe_interval(interval, location, time);
        occupancy_observer::query();
        if (found)
            occupancy_observer::record({{2, static_cast<int>(location), static_cast<int>(location),
                time, get<1>(interval), get<0>(interval), get<1>(interval), get<1>(interval), get<2>(interval), 0}});
        return found;
    }
};
