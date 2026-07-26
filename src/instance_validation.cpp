#include "structure_guided/instance_validation.hpp"

#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
struct ValidatedMap
{
    bool moving_ai = false;
    int rows = 0;
    int cols = 0;
    std::vector<unsigned char> obstacles;
};

void stripTrailingCarriageReturn(std::string& line)
{
    if (!line.empty() && line.back() == '\r')
        line.pop_back();
}

int parseStrictInt(const std::string& value, const std::string& field)
{
    try
    {
        std::size_t consumed = 0;
        const long parsed = std::stol(value, &consumed);
        if (consumed != value.size() ||
            parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max())
            throw std::out_of_range(field);
        return static_cast<int>(parsed);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument(
            "invalid integer for " + field + ": " + value
        );
    }
}

std::vector<std::string> splitOn(
    const std::string& line, char delimiter, bool discard_empty)
{
    std::istringstream stream(line);
    std::vector<std::string> values;
    std::string value;
    while (std::getline(stream, value, delimiter))
    {
        if (!discard_empty || !value.empty())
            values.push_back(value);
    }
    return values;
}

ValidatedMap validateMapFile(const std::string& path)
{
    std::ifstream input(path);
    if (!input)
        throw std::invalid_argument(
            "map_path does not exist or is not readable: " + path
        );

    ValidatedMap map;
    std::string line;
    if (!std::getline(input, line) || line.empty())
        throw std::invalid_argument("map file is empty: " + path);
    stripTrailingCarriageReturn(line);
    if (line.empty())
        throw std::invalid_argument("map file is empty: " + path);
    map.moving_ai = line.front() == 't';
    if (map.moving_ai)
    {
        std::string height_line;
        std::string width_line;
        std::string marker_line;
        if (!std::getline(input, height_line) ||
            !std::getline(input, width_line) ||
            !std::getline(input, marker_line))
            throw std::invalid_argument("map header is truncated: " + path);
        stripTrailingCarriageReturn(height_line);
        stripTrailingCarriageReturn(width_line);
        stripTrailingCarriageReturn(marker_line);
        // Instance::loadMap() splits these two lines only on literal spaces.
        // Discard repeated spaces like boost::char_separator, but reject tabs
        // that the upstream parser would otherwise dereference unsafely.
        const auto height = splitOn(height_line, ' ', true);
        const auto width = splitOn(width_line, ' ', true);
        const auto marker = splitOn(marker_line, ' ', true);
        if (height.size() != 2 || height[0] != "height" ||
            width.size() != 2 || width[0] != "width" ||
            marker.size() != 1 || marker[0] != "map")
            throw std::invalid_argument("map header is malformed: " + path);
        map.rows = parseStrictInt(height[1], "map height");
        map.cols = parseStrictInt(width[1], "map width");
    }
    else
    {
        const auto dimensions = splitOn(line, ',', false);
        if (dimensions.size() < 2)
            throw std::invalid_argument(
                "custom map header is malformed: " + path
            );
        map.rows = parseStrictInt(dimensions[0], "map rows");
        map.cols = parseStrictInt(dimensions[1], "map columns");
    }
    if (map.rows <= 0 || map.cols <= 0 ||
        static_cast<long long>(map.rows) * map.cols >
            std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "map dimensions must be positive and fit in an int"
        );

    map.obstacles.reserve(
        static_cast<std::size_t>(map.rows) * map.cols
    );
    for (int row = 0; row < map.rows; row++)
    {
        if (!std::getline(input, line))
            throw std::invalid_argument(
                "map grid is truncated at row " + std::to_string(row)
            );
        stripTrailingCarriageReturn(line);
        if (line.size() < static_cast<std::size_t>(map.cols))
            throw std::invalid_argument(
                "map grid row " + std::to_string(row) +
                " is shorter than its width"
            );
        for (int col = 0; col < map.cols; col++)
            map.obstacles.push_back(line[col] == '.' ? 0 : 1);
    }
    return map;
}

void validateScenarioCell(const ValidatedMap& map, int row, int col,
                          const std::string& label)
{
    if (row < 0 || row >= map.rows || col < 0 || col >= map.cols)
        throw std::invalid_argument(label + " is outside the map");
    if (map.obstacles[
            static_cast<std::size_t>(row) * map.cols + col
        ])
        throw std::invalid_argument(label + " is an obstacle");
}

void validateScenarioFile(const std::string& path, int agent_count,
                          const ValidatedMap& map)
{
    std::ifstream input(path);
    if (!input)
        throw std::invalid_argument(
            "scenario_path does not exist or is not readable: " + path
        );
    std::string line;
    if (!std::getline(input, line))
        throw std::invalid_argument("scenario file is empty: " + path);
    stripTrailingCarriageReturn(line);

    if (map.moving_ai)
    {
        for (int agent = 0; agent < agent_count; agent++)
        {
            if (!std::getline(input, line) || line.empty())
                throw std::invalid_argument(
                    "scenario file contains fewer than " +
                    std::to_string(agent_count) + " agents"
                );
            stripTrailingCarriageReturn(line);
            const auto values = splitOn(line, '\t', false);
            if (values.size() < 8)
                throw std::invalid_argument(
                    "scenario row " + std::to_string(agent) + " is malformed"
                );
            const int declared_cols =
                parseStrictInt(values[2], "scenario map width");
            const int declared_rows =
                parseStrictInt(values[3], "scenario map height");
            if (declared_rows != map.rows || declared_cols != map.cols)
                throw std::invalid_argument(
                    "scenario row " + std::to_string(agent) +
                    " has map dimensions that do not match map_path"
                );
            const int start_col = parseStrictInt(values[4], "start column");
            const int start_row = parseStrictInt(values[5], "start row");
            const int goal_col = parseStrictInt(values[6], "goal column");
            const int goal_row = parseStrictInt(values[7], "goal row");
            validateScenarioCell(
                map, start_row, start_col,
                "scenario row " + std::to_string(agent) + " start"
            );
            validateScenarioCell(
                map, goal_row, goal_col,
                "scenario row " + std::to_string(agent) + " goal"
            );
        }
        return;
    }

    const auto header = splitOn(line, ',', false);
    if (header.empty())
        throw std::invalid_argument(
            "custom scenario header is malformed: " + path
        );
    const int declared_agents =
        parseStrictInt(header.front(), "custom scenario agent count");
    if (declared_agents != agent_count)
        throw std::invalid_argument(
            "custom scenario agent count does not match agent_count"
        );
    for (int agent = 0; agent < agent_count; agent++)
    {
        if (!std::getline(input, line) || line.empty())
            throw std::invalid_argument(
                "scenario file contains fewer than " +
                std::to_string(agent_count) + " agents"
            );
        stripTrailingCarriageReturn(line);
        const auto values = splitOn(line, ',', false);
        if (values.size() < 4)
            throw std::invalid_argument(
                "custom scenario row " + std::to_string(agent) +
                " is malformed"
            );
        const int start_row = parseStrictInt(values[0], "start row");
        const int start_col = parseStrictInt(values[1], "start column");
        const int goal_row = parseStrictInt(values[2], "goal row");
        const int goal_col = parseStrictInt(values[3], "goal column");
        validateScenarioCell(
            map, start_row, start_col,
            "scenario row " + std::to_string(agent) + " start"
        );
        validateScenarioCell(
            map, goal_row, goal_col,
            "scenario row " + std::to_string(agent) + " goal"
        );
    }
}
}

void structure_guided::validateInstanceFiles(
    const std::string& map_path,
    const std::string& scenario_path,
    int agent_count)
{
    if (agent_count <= 0)
        throw std::invalid_argument("agent_count must be greater than zero");
    const ValidatedMap map = validateMapFile(map_path);
    validateScenarioFile(scenario_path, agent_count, map);
}
