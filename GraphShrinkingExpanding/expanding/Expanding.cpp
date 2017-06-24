#include "Expanding.h"

Expanding::Expanding(GraphIO* graph_io) {
	_graph_io = graph_io;
	_sampler = new Sampling(_graph_io);
}

void Expanding::collect_expanding_parameters(char* argv[]) {
	// Factor
	SCALING_FACTOR = atof(argv[4]);
	printf("\nFactor: %f", SCALING_FACTOR);

	// Fraction
	SAMPLING_FRACTION = atof(argv[5]);
	printf("\nFraction per sample: %f", SAMPLING_FRACTION); // TODO: Residu
	
	// Topology
	char* topology = argv[6];
	
	printf("\nTopology: ");

	if (strcmp(topology, "star") == 0) {
		SELECTED_TOPOLOGY = STAR;
		printf("%s", "star");
	}
	else if (strcmp(topology, "chain") == 0) {
		SELECTED_TOPOLOGY = CHAIN;
		printf("%s", "chain");
	}
	else if (strcmp(topology, "circle") == 0) {
		SELECTED_TOPOLOGY = CIRCLE;
		printf("%s", "circle");
	}
	else if (strcmp(topology, "mesh") == 0) {
		SELECTED_TOPOLOGY = MESH;
		printf("%s", "mesh");
	}
	else {
		printf("\nGiven topology type is undefined.");
		exit(1);
	}

	// Bridge
	char* bridge = argv[7];
	if (strcmp(bridge, "high_degree") == 0) {
		SELECTED_BRIDGE_NODE_SELECTION = HIGH_DEGREE_NODES;
		printf("\nBridge: %s", "high degree");
	}
	else if (strcmp(bridge, "random") == 0) {
		SELECTED_BRIDGE_NODE_SELECTION = RANDOM_NODES;
		printf("\nBridge: %s", "random");
	}
	else {
		printf("\nGiven bridge type is undefined.");
		exit(1);
	}

	//  Interconnection
	sscanf(argv[8], "%d", &AMOUNT_INTERCONNECTIONS);
	printf("\nAmount of interconnection: %d", AMOUNT_INTERCONNECTIONS);

	// Force undirected (TODO: Should be optional)
	char* force_undirected = argv[9];
	if (strcmp(force_undirected, "undirected") == 0) {
		FORCE_UNDIRECTED_BRIDGES = true;
		printf("\nUndirected bridges added.");
	}
}

void Expanding::expand_graph(char* input_path, char* output_path) {
	std::vector<int> source_vertices;
	std::vector<int> destination_vertices;
	COO_List* coo_list = _graph_io->load_graph_from_edge_list_file_to_coo(source_vertices, destination_vertices, input_path);
	CSR_List* csr_list = _graph_io->convert_coo_to_csr_format(coo_list->source, coo_list->destination);

	const int amount_of_sampled_graphs = SCALING_FACTOR / SAMPLING_FRACTION;
	printf("Amount of sampled graph versions: %d", amount_of_sampled_graphs);

	Sampled_Vertices** sampled_vertices_per_graph = (Sampled_Vertices**)malloc(sizeof(Sampled_Vertices)*amount_of_sampled_graphs);

	int** d_size_collected_edges = (int**)malloc(sizeof(int*)*amount_of_sampled_graphs);
	Edge** d_edge_data_expanding = (Edge**)malloc(sizeof(Edge*)*amount_of_sampled_graphs);

	Sampled_Graph_Version* sampled_graph_version_list = new Sampled_Graph_Version[amount_of_sampled_graphs];
	char current_label = 'a';

	int* d_offsets;
	int* d_indices;
	gpuErrchk(cudaMalloc((void**)&d_offsets, sizeof(int) * (_graph_io->SIZE_VERTICES + 1)));
	gpuErrchk(cudaMalloc((void**)&d_indices, sizeof(int) * _graph_io->SIZE_EDGES));

	gpuErrchk(cudaMemcpyToSymbol(&D_SIZE_EDGES, &(_graph_io->SIZE_EDGES), sizeof(int), 0, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(&D_SIZE_VERTICES, &(_graph_io->SIZE_VERTICES), sizeof(int), 0, cudaMemcpyHostToDevice));

	_sampler->SAMPLING_FRACTION = SAMPLING_FRACTION;

	for (int i = 0; i < amount_of_sampled_graphs; i++) {
		sampled_vertices_per_graph[i] = _sampler->perform_edge_based_node_sampling_step(coo_list->source, coo_list->destination);
		printf("\nCollected %d vertices.", sampled_vertices_per_graph[i]->sampled_vertices_size);

		gpuErrchk(cudaMemcpy(d_indices, csr_list->indices, _graph_io->SIZE_EDGES * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_offsets, csr_list->offsets, sizeof(int) * (_graph_io->SIZE_VERTICES + 1), cudaMemcpyHostToDevice));

		int* d_sampled_vertices;
		gpuErrchk(cudaMalloc((void**)&d_sampled_vertices, sizeof(int) * _graph_io->SIZE_VERTICES));
		gpuErrchk(cudaMemcpy(d_sampled_vertices, sampled_vertices_per_graph[i]->vertices, sizeof(int) * (_graph_io->SIZE_VERTICES), cudaMemcpyHostToDevice));
		
		int* h_size_edges = 0;
		gpuErrchk(cudaMalloc((void**)&d_size_collected_edges[i], sizeof(int)));
		gpuErrchk(cudaMemcpy(d_size_collected_edges[i], &h_size_edges, sizeof(int), cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc((void**)&d_edge_data_expanding[i], sizeof(Edge) * _graph_io->SIZE_EDGES));

		cudaDeviceSynchronize(); // This can be deleted - double check

		printf("\nRunning kernel (induction step) with block size %d and thread size %d:", get_block_size(), get_thread_size());
		perform_induction_step_expanding(get_block_size(), get_thread_size(), d_sampled_vertices, d_offsets, d_indices, d_edge_data_expanding[i], d_size_collected_edges[i]);
		//perform_induction_step_expanding <<<get_block_size(), get_thread_size()>>>(d_sampled_vertices, d_offsets, d_indices, d_edge_data_expanding[i], d_size_collected_edges[i]);

		// Edge size
		int h_size_edges_result;
		gpuErrchk(cudaMemcpy(&h_size_edges_result, d_size_collected_edges[i], sizeof(int), cudaMemcpyDeviceToHost));

		// Edges
		printf("\nh_size_edges: %d", h_size_edges_result);
		Sampled_Graph_Version* sampled_graph_version = new Sampled_Graph_Version();
		(*sampled_graph_version).edges.resize(h_size_edges_result);

		gpuErrchk(cudaMemcpy(&sampled_graph_version->edges[0], d_edge_data_expanding[i], sizeof(Edge)*(h_size_edges_result), cudaMemcpyDeviceToHost));

		// Label
		sampled_graph_version->label = current_label++;

		// Copy data to the sampled version list
		sampled_graph_version_list[i] = (*sampled_graph_version);

		// Cleanup
		delete(sampled_graph_version);

		cudaFree(d_sampled_vertices);
		cudaFree(d_edge_data_expanding[i]);
		cudaFree(d_size_collected_edges);
		free(sampled_vertices_per_graph[i]->vertices);
		free(sampled_vertices_per_graph[i]);
	}

	cudaFree(d_offsets);
	cudaFree(d_indices);
	free(sampled_vertices_per_graph);
	free(coo_list);
	free(csr_list->indices);
	free(csr_list->offsets);
	free(csr_list);

	// For each sampled graph version, copy the data back to the host
	std::vector<Bridge_Edge> bridge_edges;

	switch (SELECTED_TOPOLOGY) {
	case STAR:
		link_using_star_topology(sampled_graph_version_list, amount_of_sampled_graphs, bridge_edges);
		break;
	case CHAIN:
		link_using_line_topology(sampled_graph_version_list, amount_of_sampled_graphs, bridge_edges);
		break;
	case CIRCLE:
		link_using_circle_topology(sampled_graph_version_list, amount_of_sampled_graphs, bridge_edges);
		break;
	case MESH:
		link_using_mesh_topology(sampled_graph_version_list, amount_of_sampled_graphs, bridge_edges);
	}

	printf("\nConnected by adding a total of %d bridge edges.", bridge_edges.size());

	_graph_io->write_expanded_output_to_file(sampled_graph_version_list, amount_of_sampled_graphs, bridge_edges, output_path);

	// Cleanup
	delete[] sampled_graph_version_list;
}

int Expanding::get_thread_size() {
	return ((_graph_io->SIZE_VERTICES + 1) > MAX_THREADS) ? MAX_THREADS : _graph_io->SIZE_VERTICES;
}
int Expanding::get_block_size() {
	return ((_graph_io->SIZE_VERTICES + 1) > MAX_THREADS) ? ((_graph_io->SIZE_VERTICES / MAX_THREADS) + 1) : 1;
}

void Expanding::link_using_star_topology(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges) {
	Sampled_Graph_Version center_graph = sampled_graph_version_list[0]; // First sampled version will be the graph in the center

	for (int i = 1; i < amount_of_sampled_graphs; i++) { // Skip the center graph 
		add_edge_interconnection_between_graphs(&(sampled_graph_version_list[i]), &center_graph, bridge_edges);
	}

}

void Expanding::link_using_line_topology(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges) {
	for (int i = 0; i < (amount_of_sampled_graphs - 1); i++) {
		add_edge_interconnection_between_graphs(&(sampled_graph_version_list[i]), &(sampled_graph_version_list[i + 1]), bridge_edges);
	}
}

void Expanding::link_using_circle_topology(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges) {
	for (int i = 0; i < amount_of_sampled_graphs; i++) {
		if (i == (amount_of_sampled_graphs - 1)) { // We're at the last sampled graph, so connect it back to the first one in the list
			add_edge_interconnection_between_graphs(&(sampled_graph_version_list[i]), &(sampled_graph_version_list[0]), bridge_edges);
			break;
		}

		add_edge_interconnection_between_graphs(&(sampled_graph_version_list[i]), &(sampled_graph_version_list[i + 1]), bridge_edges);
	}
}

void Expanding::link_using_mesh_topology(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges) {
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

void Expanding::add_edge_interconnection_between_graphs(Sampled_Graph_Version* graph_a, Sampled_Graph_Version* graph_b, std::vector<Bridge_Edge>& bridge_edges) {
	for (int i = 0; i < AMOUNT_INTERCONNECTIONS; i++) {
		int vertex_a = get_node_bridge_vertex(graph_a);
		int vertex_b = get_node_bridge_vertex(graph_b);

		// Add edge
		Bridge_Edge bridge_edge;
		sprintf(bridge_edge.source, "%c%d", graph_a->label, vertex_a);
		sprintf(bridge_edge.destination, "%c%d", graph_b->label, vertex_b);
		bridge_edges.push_back(bridge_edge);
		//printf("\nBridge selection - Selected: (%s, %s)", bridge_edge.source, bridge_edge.destination);

		if (FORCE_UNDIRECTED_BRIDGES) {
			Bridge_Edge bridge_edge_undirected;
			sprintf(bridge_edge_undirected.source, "%c%d", graph_b->label, vertex_b);
			sprintf(bridge_edge_undirected.destination, "%c%d", graph_a->label, vertex_a);
			bridge_edges.push_back(bridge_edge_undirected);
			//printf("\nBridge selection (undirected) - Selected: (%s, %s)", bridge_edge_undirected.source, bridge_edge_undirected.destination);
		}
	}
}

int Expanding::select_random_bridge_vertex(Sampled_Graph_Version* graph) {
	// TODO: Move to add_edge_interconnection_between_graphs
	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_int_distribution<int> range_edges(0, ((*graph).edges.size()) - 1);
	int random_edge_index = range_edges(engine);

	// 50:50 return source or destination
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

int Expanding::select_high_degree_node_bridge_vertex(Sampled_Graph_Version* graph) {
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

int Expanding::get_random_high_degree_node(Sampled_Graph_Version* graph) {
	std::random_device seeder;
	std::mt19937 engine(seeder());

	std::uniform_int_distribution<int> range_edges(0, (graph->high_degree_nodes.size() - 1));
	int random_vertex_index = range_edges(engine);

	return graph->high_degree_nodes[random_vertex_index];
}

int Expanding::get_node_bridge_vertex(Sampled_Graph_Version* graph) {
	switch (SELECTED_BRIDGE_NODE_SELECTION) {
	case RANDOM_NODES:
		return select_random_bridge_vertex(graph);
	case HIGH_DEGREE_NODES:
		return select_high_degree_node_bridge_vertex(graph);
	}
}

Expanding::~Expanding() {
	delete(_sampler);
}