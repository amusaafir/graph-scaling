#include "RandomBridge.h"

int RandomBridge::select_bridges(Sampled_Graph_Version* graph) {
	// TODO: Move to add_edge_interconnection_between_graphs
	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_int_distribution<int> range_edges(0, ((*graph).edges.size()) - 1);
	int random_edge_index = range_edges(engine);

	// Return source or destination (50/50 chance)
	std::random_device destination_or_source_seeder;
	std::mt19937 engine_source_or_destination(destination_or_source_seeder());
	std::uniform_int_distribution<int> range_destination_source(0, 1);
	int destination_or_source = range_destination_source(engine_source_or_destination);

	if (destination_or_source == 0) {
		return (*graph).edges[random_edge_index].source;
	}
	else {
		return (*graph).edges[random_edge_index].destination;
	}
}