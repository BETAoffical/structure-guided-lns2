#pragma once

#include <string>
#include <utility>
#include <vector>

namespace lns2 {

using Location = int;

struct Agent {
    Location start = -1;
    Location goal = -1;
};

class Instance {
public:
    static Instance load(const std::string& path);

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return rows_ * cols_; }
    const std::vector<Agent>& agents() const { return agents_; }

    bool traversable(Location location) const;
    Location location(int row, int col) const;
    std::pair<int, int> coordinate(Location location) const;
    std::vector<Location> neighbors_with_wait(Location location) const;
    int shortest_distance(Location start, Location goal) const;

private:
    int rows_ = 0;
    int cols_ = 0;
    std::vector<bool> blocked_;
    std::vector<Agent> agents_;
};

}  // namespace lns2
