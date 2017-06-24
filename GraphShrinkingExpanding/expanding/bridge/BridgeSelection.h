#pragma once

#include <unordered_map>
#include <random>
#include "../SampledGraphVersionStruct.h"

class BridgeSelection {
public:
	virtual int select_bridges(Sampled_Graph_Version* graph) = 0;
};