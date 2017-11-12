#pragma once

#include <random>
#include <unordered_map>
#include <algorithm>
#include "../SampledGraphVersionStruct.h"
#include "../BridgeEdgeStruct.h"
#include "../bridge/BridgeSelection.h"

class Topology {
private:
	int _AMOUNT_INTERCONNECTIONS;
	BridgeSelection* _BRIDGE_SELECTION;
	bool _FORCE_UNDIRECTED_BRIDGES;
public:
	Topology();
	Topology(int AMOUNT_INTERCONNECTIONS, BridgeSelection* BRIDGE_SELECTION, bool FORCE_UNDIRECTED_BRIDGES);
	~Topology();
	virtual void link(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges) = 0;
	void add_edge_interconnection_between_graphs(Sampled_Graph_Version* graph_a, Sampled_Graph_Version* graph_b, std::vector<Bridge_Edge>& bridge_edges);
};