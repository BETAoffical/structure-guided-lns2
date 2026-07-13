#pragma once

#include "RepairPolicy.h"

#include <fstream>
#include <string>

class JsonlRepairObserver : public RepairObserver
{
public:
    explicit JsonlRepairObserver(const std::string& path);

    void onInitialState(const RepairState& state) override;
    void onTransition(const RepairState& before, const RepairTransition& transition,
                      const RepairState& after) override;
    void onFinish(const RepairState& state, bool success) override;

private:
    std::ofstream output;

    void writeState(const RepairState& state);
    void writeAction(const RepairAction& action);
    static void writeIntArray(std::ostream& stream, const vector<int>& values);
};
