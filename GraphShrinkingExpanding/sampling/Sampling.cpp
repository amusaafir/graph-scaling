#include "Sampling.h"

void Sampling::collect_sampling_parameters(char* argv[]) {
	float fraction = atof(argv[4]);
	SAMPLING_FRACTION = fraction;
	printf("\nSample fraction: %f", fraction);
}

void Sampling::sample_graph(char* input_path, char* output_path) {
	std::vector<int> source_vertices;
	std::vector<int> destination_vertices;

	// Convert edge list to COO
	COO_List* coo_list = graph_io.load_graph_from_edge_list_file_to_coo(source_vertices, destination_vertices, input_path);
	
	// Convert the COO graph into a CSR format (for the in-memory GPU representation) 
	CSR_List* csr_list = graph_io.convert_coo_to_csr_format(coo_list->source, coo_list->destination);
	
	// Edge based Node Sampling Step
	Sampled_Vertices* sampled_vertices = perform_edge_based_node_sampling_step(coo_list->source, coo_list->destination);
	printf("\nCollected %d vertices.", sampled_vertices->sampled_vertices_size);

	// Induction step (TODO: re-use device memory from CSR conversion)
	int* d_offsets;
	int* d_indices;
	gpuErrchk(cudaMalloc((void**)&d_offsets, sizeof(int)*(graph_io.SIZE_VERTICES + 1)));
	gpuErrchk(cudaMalloc((void**)&d_indices, sizeof(int)*graph_io.SIZE_EDGES));
	gpuErrchk(cudaMemcpy(d_indices, csr_list->indices, graph_io.SIZE_EDGES * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_offsets, csr_list->offsets, sizeof(int)*(graph_io.SIZE_VERTICES + 1), cudaMemcpyHostToDevice));
	
	int* d_sampled_vertices;
	gpuErrchk(cudaMalloc((void**)&d_sampled_vertices, sizeof(int)*graph_io.SIZE_VERTICES));
	gpuErrchk(cudaMemcpy(d_sampled_vertices, sampled_vertices->vertices, sizeof(int)*graph_io.SIZE_VERTICES, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpyToSymbol(&D_SIZE_EDGES, &(graph_io.SIZE_EDGES), sizeof(int), 0, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(&D_SIZE_VERTICES, &(graph_io.SIZE_VERTICES), sizeof(int), 0, cudaMemcpyHostToDevice));
	
	Edge* d_edge_data;
	gpuErrchk(cudaMalloc((void**)&d_edge_data, sizeof(Edge)*graph_io.SIZE_EDGES));

	printf("\nRunning kernel (induction step) with block size %d and thread size %d:", get_block_size(), get_thread_size());
	//perform_induction_step <<<get_block_size(), get_thread_size() >> >(d_sampled_vertices, d_offsets, d_indices, d_edge_data);
	
	perform_induction_step(get_block_size(), get_thread_size(), d_sampled_vertices, d_offsets, d_indices, d_edge_data);
	
	int h_edge_count;
	gpuErrchk(cudaMemcpyFromSymbol(&h_edge_count, &d_edge_count, sizeof(int)));
	if (h_edge_count >= graph_io.SIZE_EDGES + 1) {
		printf("overflow error\n"); return;
	}
	printf("\nAmount of edges collected: %d", h_edge_count);
	std::vector<Edge> results(h_edge_count);
	gpuErrchk(cudaMemcpy(&(results[0]), d_edge_data, h_edge_count * sizeof(Edge), cudaMemcpyDeviceToHost));
	
	graph_io.write_output_to_file(results, output_path);
	cudaFree(d_offsets);
	cudaFree(d_indices);
	cudaFree(d_sampled_vertices);

	// Cleanup
	free(sampled_vertices->vertices);
	free(sampled_vertices);
	free(coo_list);
	free(csr_list->indices);
	free(csr_list->offsets);
	free(csr_list);
}

Sampled_Vertices* Sampling::perform_edge_based_node_sampling_step(int* source_vertices, int* target_vertices) {
	printf("\nPerforming edge based node sampling step.\n");
	Sampled_Vertices* sampled_vertices = (Sampled_Vertices*)malloc(sizeof(Sampled_Vertices));
	int amount_total_sampled_vertices = calculate_node_sampled_size();
	std::random_device seeder;
	std::mt19937 engine(seeder());
	sampled_vertices->vertices = (int*)calloc(graph_io.SIZE_VERTICES, sizeof(int));
	int collected_amount = 0;
	while (collected_amount < amount_total_sampled_vertices) {
		// Pick a random vertex u
		std::uniform_int_distribution<int> range_edges(0, (graph_io.SIZE_EDGES - 1)); // Don't select the last element in the offset
		int random_edge_index = range_edges(engine);

		// Insert u, v (TODO: extract to method per vertex)
		if (!sampled_vertices->vertices[source_vertices[random_edge_index]]) {
			sampled_vertices->vertices[source_vertices[random_edge_index]] = 1;
			//print_debug_log("\nCollected vertex:", source_vertices[random_edge_index]);
			//printf("\nCollected vertex: %d", source_vertices[random_edge_index]);
			collected_amount++;
		}
		if (!sampled_vertices->vertices[target_vertices[random_edge_index]]) {
			sampled_vertices->vertices[target_vertices[random_edge_index]] = 1;
			//print_debug_log("\nCollected vertex:", target_vertices[random_edge_index]);
			//printf("\nCollected vertex: %d", target_vertices[random_edge_index]);
			collected_amount++;
		}
	}
	sampled_vertices->sampled_vertices_size = collected_amount;
	printf("\nDone with node sampling step..");
	return sampled_vertices;
}

int Sampling::calculate_node_sampled_size() {
	return int(graph_io.SIZE_VERTICES * SAMPLING_FRACTION);
}

int Sampling::get_thread_size() {
	return ((graph_io.SIZE_VERTICES + 1) > MAX_THREADS) ? MAX_THREADS : graph_io.SIZE_VERTICES;
}
int Sampling::get_block_size() {
	return ((graph_io.SIZE_VERTICES + 1) > MAX_THREADS) ? ((graph_io.SIZE_VERTICES / MAX_THREADS) + 1) : 1;
}