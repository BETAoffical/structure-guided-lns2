#include "lns2/instance.hpp"

#include <fstream>
#include <queue>
#include <stdexcept>

namespace lns2 {

Instance Instance::load(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open instance: " + path);
    }

    Instance instance;
    input >> instance.rows_ >> instance.cols_;
    if (instance.rows_ <= 0 || instance.cols_ <= 0) {
        throw std::runtime_error("invalid grid dimensions");
    }

    instance.blocked_.assign(instance.size(), false);
    std::string row;
    for (int r = 0; r < instance.rows_; ++r) {
        input >> row;
        if (static_cast<int>(row.size()) != instance.cols_) {
            throw std::runtime_error("invalid grid row");
        }
        for (int c = 0; c < instance.cols_; ++c) {
            if (row[c] != '.' && row[c] != '@') {
                throw std::runtime_error("grid supports only '.' and '@'");
            }
            instance.blocked_[instance.location(r, c)] = row[c] == '@';
        }
    }

    int agent_count = 0;
    input >> agent_count;
    if (agent_count <= 0) {
        throw std::runtime_error("instance must contain at least one agent");
    }

    std::vector<bool> starts(instance.size(), false);
    std::vector<bool> goals(instance.size(), false);
    for (int i = 0; i < agent_count; ++i) {
        int sr = 0;
        int sc = 0;
        int gr = 0;
        int gc = 0;
        input >> sr >> sc >> gr >> gc;
        const auto start = instance.location(sr, sc);
        const auto goal = instance.location(gr, gc);
        if (!instance.traversable(start) || !instance.traversable(goal)) {
            throw std::runtime_error("agent start or goal is blocked");
        }
        if (starts[start] || goals[goal]) {
            throw std::runtime_error("agent starts and goals must be unique");
        }
        starts[start] = true;
        goals[goal] = true;
        instance.agents_.push_back({start, goal});
    }

    return instance;
}

bool Instance::traversable(Location location) const {
    return location >= 0 && location < size() && !blocked_[location];
}

Location Instance::location(int row, int col) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::runtime_error("coordinate outside grid");
    }
    return row * cols_ + col;
}

std::pair<int, int> Instance::coordinate(Location location) const {
    return {location / cols_, location % cols_};
}

std::vector<Location> Instance::neighbors_with_wait(Location current) const {
    const auto [row, col] = coordinate(current);
    const int dr[] = {0, 0, 1, 0, -1};
    const int dc[] = {0, 1, 0, -1, 0};
    std::vector<Location> result;
    result.reserve(5);
    for (int i = 0; i < 5; ++i) {
        const int next_row = row + dr[i];
        const int next_col = col + dc[i];
        if (next_row < 0 || next_row >= rows_ ||
            next_col < 0 || next_col >= cols_) {
            continue;
        }
        const auto next = next_row * cols_ + next_col;
        if (traversable(next)) {
            result.push_back(next);
        }
    }
    return result;
}

int Instance::shortest_distance(Location start, Location goal) const {
    std::vector<int> distance(size(), -1);
    std::queue<Location> open;
    distance[start] = 0;
    open.push(start);

    while (!open.empty()) {
        const auto current = open.front();
        open.pop();
        if (current == goal) {
            return distance[current];
        }
        for (const auto next : neighbors_with_wait(current)) {
            if (next == current || distance[next] >= 0) {
                continue;
            }
            distance[next] = distance[current] + 1;
            open.push(next);
        }
    }
    return -1;
}

}  // namespace lns2
