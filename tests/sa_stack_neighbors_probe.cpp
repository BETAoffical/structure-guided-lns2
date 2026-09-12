#include "Instance.h"
#include <chrono>
#include <cstdint>
#include <stdexcept>

struct Neighbors { int values[4]; int size = 0; };

__attribute__((noinline)) Neighbors stackNeighbors(const Instance& instance, int current)
{
    Neighbors result;
    const int candidates[4] = {current + 1, current - 1,
        current + instance.num_of_cols, current - instance.num_of_cols};
    for (int next : candidates)
        if (instance.validMove(current, next)) result.values[result.size++] = next;
    return result;
}

volatile uint64_t sink = 0;
double measure(const Instance& instance, const vector<int>& cells, bool stack)
{
    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 200000; ++i)
    {
        const int current = cells[i % cells.size()];
        if (stack)
        {
            auto neighbors = stackNeighbors(instance, current);
            for (int j = 0; j < neighbors.size; ++j) checksum += neighbors.values[j];
        }
        else
            for (int next : instance.getNeighbors(current)) checksum += next;
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    sink = checksum;
    return elapsed;
}

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    Instance instance(argv[1], argv[2], 1);
    vector<int> cells;
    for (int current = 0; current < instance.map_size; ++current)
    {
        if (instance.isObstacle(current)) continue;
        cells.push_back(current);
        const auto reference = instance.getNeighbors(current);
        const auto candidate = stackNeighbors(instance, current);
        if (reference.size() != static_cast<size_t>(candidate.size)) return 3;
        int i = 0;
        for (int next : reference) if (next != candidate.values[i++]) return 4;
    }
    if (cells.empty()) return 5;
    // Warm both paths, then alternate the measured order without RNG calls.
    measure(instance, cells, false);
    measure(instance, cells, true);
    for (int repeat = 0; repeat < 4; ++repeat)
    {
        double a, b;
        uint64_t ca, cb;
        if (repeat % 2 == 0)
        {
            a = measure(instance, cells, false); ca = sink;
            b = measure(instance, cells, true); cb = sink;
        }
        else
        {
            b = measure(instance, cells, true); cb = sink;
            a = measure(instance, cells, false); ca = sink;
        }
        if (ca != cb) return 6;
        cout << "RESULT " << cells.size() << " " << a << " " << b << "\n";
    }
}
