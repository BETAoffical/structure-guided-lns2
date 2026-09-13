// Expected-failure diagnostic, deliberately not registered as a passing CTest.
#include "ConstraintTable.h"
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    const std::string mode(argv[1]);
    Path external;
    for (int location : {0, 3, 4, 7, 8}) external.emplace_back(location);
    PathTableWC table(9, 1);
    table.insertPath(0, external);
    ConstraintTable ct(3, 9, nullptr, &table);
    Path local;
    for (int location : {1, 2}) local.emplace_back(location);
    int query = 4, expected = 2;
    if (mode == "empty_local_external_hit") ct.insert2CAT(local);
    else if (mode == "local_only")
    {
        ct.insert2CAT(local);
        query = 2;
        expected = 1;
    }
    else if (mode == "mixed_populated")
    {
        local.clear();
        for (int location : {1, 1, 1, 4, 5}) local.emplace_back(location);
        ct.insert2CAT(local);
        expected = 3;
    }
    else if (mode == "no_hit")
    {
        ct.insert2CAT(local);
        query = 6;
        expected = -1;
    }
    else if (mode != "external_table_only") return 2;
    std::cout << "mode=" << mode << " query=" << query << " expected=" << expected << std::endl;
    const int actual = ct.getLastCollisionTimestep(query);
    std::cout << "actual=" << actual << std::endl;
    return actual == expected ? 0 : 3;
}
