#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace corrected_native
{
inline int weightedIndexForDraw(const std::vector<int>& weights, int draw)
{
    if (draw < 0)
        throw std::invalid_argument("weighted draw must be non-negative");

    std::int64_t cumulative = 0;
    for (std::size_t index = 0; index < weights.size(); index++)
    {
        if (weights[index] < 0)
            throw std::invalid_argument("weighted sampling requires non-negative weights");
        cumulative += weights[index];
        if (draw < cumulative)
            return static_cast<int>(index);
    }
    throw std::out_of_range("weighted draw is outside the total weight");
}

inline int uniformZeroBasedDraw(int upper_bound)
{
    if (upper_bound <= 0)
        throw std::invalid_argument("random upper bound must be positive");

    // Reject the incomplete final bucket instead of using rand() % upper_bound
    // directly. This keeps every 0-based draw equally likely.
    const std::uint64_t random_range =
        static_cast<std::uint64_t>(RAND_MAX) + 1;
    if (static_cast<std::uint64_t>(upper_bound) > random_range)
        throw std::invalid_argument("random upper bound exceeds the RNG range");
    const std::uint64_t acceptance_limit =
        random_range -
        random_range % static_cast<std::uint64_t>(upper_bound);
    std::uint64_t value = 0;
    do
    {
        value = static_cast<std::uint64_t>(rand());
    }
    while (value >= acceptance_limit);
    return static_cast<int>(
        value % static_cast<std::uint64_t>(upper_bound)
    );
}

inline int sampleWeightedIndex(const std::vector<int>& weights)
{
    std::int64_t total = 0;
    for (int weight : weights)
    {
        if (weight < 0)
            throw std::invalid_argument("weighted sampling requires non-negative weights");
        total += weight;
    }
    if (total <= 0 || total > RAND_MAX)
        throw std::invalid_argument("total sampling weight is outside the RNG range");
    return weightedIndexForDraw(
        weights, uniformZeroBasedDraw(static_cast<int>(total))
    );
}
}
