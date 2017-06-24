#include "Topology.h"

Topology::Topology() {
}

Topology::Topology(int AMOUNT_INTERCONNECTIONS, BridgeSelection* BRIDGE_SELECTION, bool FORCE_UNDIRECTED_BRIDGES) {
	_AMOUNT_INTERCONNECTIONS = AMOUNT_INTERCONNECTIONS;
	_BRIDGE_SELECTION = BRIDGE_SELECTION;
	_FORCE_UNDIRECTED_BRIDGES = FORCE_UNDIRECTED_BRIDGES;
}

void Topology::add_edge_interconnection_between_graphs(Sampled_Graph_Version* graph_a, Sampled_Graph_Version* graph_b, std::vector<Bridge_Edge>& bridge_edges) {
	for (int i = 0; i < _AMOUNT_INTERCONNECTIONS; i++) {
		int vertex_a = _BRIDGE_SELECTION->select_bridges(graph_a);
		int vertex_b = _BRIDGE_SELECTION->select_bridges(graph_b);

		// Add edge
		Bridge_Edge bridge_edge;
		sprintf(bridge_edge.source, "%c%d", graph_a->label, vertex_a);
		sprintf(bridge_edge.destination, "%c%d", graph_b->label, vertex_b);
		bridge_edges.push_back(bridge_edge);
		//printf("\nBridge selection - Selected: (%s, %s)", bridge_edge.source, bridge_edge.destination);

		if (_FORCE_UNDIRECTED_BRIDGES) {
			Bridge_Edge bridge_edge_undirected;
			sprintf(bridge_edge_undirected.source, "%c%d", graph_b->label, vertex_b);
			sprintf(bridge_edge_undirected.destination, "%c%d", graph_a->label, vertex_a);
			bridge_edges.push_back(bridge_edge_undirected);
			//printf("\nBridge selection (undirected) - Selected: (%s, %s)", bridge_edge_undirected.source, bridge_edge_undirected.destination);
		}
	}
}

Topology::~Topology() {
	delete(_BRIDGE_SELECTION);
}