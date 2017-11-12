#include "Chain.h"

void Chain::link(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges) {
	printf("\nCalling ChainTopology!");

	for (int i = 0; i < (amount_of_sampled_graphs - 1); i++) {
		add_edge_interconnection_between_graphs(&(sampled_graph_version_list[i]), &(sampled_graph_version_list[i + 1]), bridge_edges);
	}
}