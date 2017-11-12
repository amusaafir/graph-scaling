#include "HighDegree.h"

int HighDegree::select_bridges(Sampled_Graph_Version* graph) {
	if (graph->high_degree_nodes.size() > 0) { // There already exists some high degree nodes here, so just select them randomly for instance.
		return get_random_high_degree_node(graph);
	}
	else { // Collect high degree nodes and add them to the current graph
		   // Map all vertices onto a map along with their degree
		std::unordered_map<int, int> node_degree;

		for (auto &edge : graph->edges) {
			++node_degree[edge.source];
			++node_degree[edge.destination];
		}

		// Convert the map to a vector
		std::vector<std::pair<int, int>> node_degree_vect(node_degree.begin(), node_degree.end());

		// Sort the vector (ascending, high degree nodes are on top)
		std::sort(node_degree_vect.begin(), node_degree_vect.end(), [](const std::pair<int, int> &left, const std::pair<int, int> &right) {
			return left.second > right.second;
		});

		// Collect only the nodes (half of the total nodes) that have a high degree
		for (int i = 0; i < node_degree_vect.size() / 2; i++) {
			graph->high_degree_nodes.push_back(node_degree_vect[i].first);
		}

		return get_random_high_degree_node(graph);
	}
}

int HighDegree::get_random_high_degree_node(Sampled_Graph_Version* graph) {
	std::random_device seeder;
	std::mt19937 engine(seeder());

	std::uniform_int_distribution<int> range_edges(0, (graph->high_degree_nodes.size() - 1));
	int random_vertex_index = range_edges(engine);

	return graph->high_degree_nodes[random_vertex_index];
}