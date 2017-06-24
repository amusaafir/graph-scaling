#include "FullyConnected.h"

void FullyConnected::link(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges) {
	printf("\nCalling FullyConnected!");

	for (int x = 0; x < amount_of_sampled_graphs; x++) {
		Sampled_Graph_Version current_graph = sampled_graph_version_list[x];

		for (int y = 0; y < amount_of_sampled_graphs; y++) {
			if (x == y) { // Don't link the current graph to itself
				continue;
			}

			add_edge_interconnection_between_graphs(&(sampled_graph_version_list[x]), &(sampled_graph_version_list[y]), bridge_edges);
		}
	}
}