#include "Ring.h"

void Ring::link(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges) {
	printf("\nCalling RingTopology!");

	for (int i = 0; i < amount_of_sampled_graphs; i++) {
		if (i == (amount_of_sampled_graphs - 1)) { // We're at the last sampled graph, so connect it back to the first one in the list
			add_edge_interconnection_between_graphs(&(sampled_graph_version_list[i]), &(sampled_graph_version_list[0]), bridge_edges);
			break;
		}

		add_edge_interconnection_between_graphs(&(sampled_graph_version_list[i]), &(sampled_graph_version_list[i + 1]), bridge_edges);
	}
}