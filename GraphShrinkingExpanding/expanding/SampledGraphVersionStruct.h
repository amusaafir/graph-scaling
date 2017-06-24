#pragma once

#include <vector>
#include "../sampling/EdgeStruct.h"

typedef struct Sampled_Graph_Version {
	std::vector<Edge> edges;
	std::vector<int> high_degree_nodes;
	char label;
} Sampled_Graph_Version;