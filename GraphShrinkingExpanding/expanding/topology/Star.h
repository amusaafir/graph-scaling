#pragma once

#include "Topology.h"

class Star : public Topology {
public:
	Star(int AMOUNT_INTERCONNECTIONS, BridgeSelection* BRIDGE_SELECTION, bool FORCE_UNDIRECTED_BRIDGES) : Topology(AMOUNT_INTERCONNECTIONS, BRIDGE_SELECTION, FORCE_UNDIRECTED_BRIDGES) {}
	virtual void link(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges);
};