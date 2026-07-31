#pragma once

#include <string>

namespace structure_guided
{
int resolveInstanceAgentCount(const std::string& map_path,
                              const std::string& scenario_path,
                              int requested_agent_count);

void validateInstanceFiles(const std::string& map_path,
                           const std::string& scenario_path,
                           int agent_count);
}
