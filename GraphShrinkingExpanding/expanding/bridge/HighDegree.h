#pragma once

#include <algorithm>
#include "BridgeSelection.h"
#include "../SampledGraphVersionStruct.h"

class HighDegree : public BridgeSelection {
public:
	int select_bridges(Sampled_Graph_Version* graph);
	int get_random_high_degree_node(Sampled_Graph_Version* graph); // TODO: make private?
};