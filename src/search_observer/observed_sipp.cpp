// Compile the frozen implementation with an observation-only table facade.
// Do not change SIPP node layouts, comparator code, returned intervals, or RNG.
#include "SIPP.h"
#include "observer.h"
#define ReservationTable ObservedReservationTable
#include "../../third_party/mapf_lns2/src/SIPP.cpp"
#undef ReservationTable
