#include "Star.h"

void Star::link(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges) {
	printf("\nCalling StarTopology!");

	Sampled_Graph_Version center_graph = sampled_graph_version_list[0]; // First sampled version will be the graph in the center

	for (int i = 1; i < amount_of_sampled_graphs; i++) { // Skip the center graph 
		add_edge_interconnection_between_graphs(&(sampled_graph_version_list[i]), &center_graph, bridge_edges);
	}
}