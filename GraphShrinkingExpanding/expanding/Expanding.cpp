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
	
	// Bridge
	char* bridge = argv[7];
	if (strcmp(bridge, "high_degree") == 0) {
		//SELECTED_BRIDGE_NODE_SELECTION = HIGH_DEGREE_NODES;
		_bridge_selection = new HighDegree();
		printf("\nBridge: %s", "high degree");
	}
	else if (strcmp(bridge, "random") == 0) {
		//SELECTED_BRIDGE_NODE_SELECTION = RANDOM_NODES;
		_bridge_selection = new RandomBridge();
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

	// Topology
	char* topology = argv[6];

	printf("\nTopology: ");

	if (strcmp(topology, "star") == 0) {
		//SELECTED_TOPOLOGY = STAR;
		_topology = new Star(AMOUNT_INTERCONNECTIONS, _bridge_selection, FORCE_UNDIRECTED_BRIDGES);
		printf("%s", "star");
	}
	else if (strcmp(topology, "chain") == 0) {
		//SELECTED_TOPOLOGY = CHAIN;
		_topology = new Chain(AMOUNT_INTERCONNECTIONS, _bridge_selection, FORCE_UNDIRECTED_BRIDGES);
		printf("%s", "chain");
	}
	else if (strcmp(topology, "circle") == 0) {
		//SELECTED_TOPOLOGY = CIRCLE;
		_topology = new Ring(AMOUNT_INTERCONNECTIONS, _bridge_selection, FORCE_UNDIRECTED_BRIDGES);
		printf("%s", "circle");
	}
	else if (strcmp(topology, "mesh") == 0) {
		//SELECTED_TOPOLOGY = MESH;
		_topology = new FullyConnected(AMOUNT_INTERCONNECTIONS, _bridge_selection, FORCE_UNDIRECTED_BRIDGES);
		printf("%s", "mesh");
	}
	else {
		printf("\nGiven topology type is undefined.");
		exit(1);
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

	// Topology, bridges and interconnections
	std::vector<Bridge_Edge> bridge_edges;
	_topology->link(sampled_graph_version_list, amount_of_sampled_graphs, bridge_edges);
	printf("\nConnected by adding a total of %d bridge edges.", bridge_edges.size());

	// Write expanded graph to output file
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

// Only for debugging purposes
void Expanding::set_topology(Topology* topology) {
	_topology = topology;
}

Expanding::~Expanding() {
	delete(_sampler);
	delete(_topology);
}