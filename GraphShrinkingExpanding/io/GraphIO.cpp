#include "GraphIO.h"

COO_List* GraphIO::load_graph_from_edge_list_file_to_coo(std::vector<int>& source_vertices, std::vector<int>& destination_vertices, char* file_path) {
	printf("\nLoading graph file from: %s", file_path);
	FILE* file = fopen(file_path, "r");
	char line[256];
	int current_coordinate = 0;
	if (IS_INPUT_FILE_COO) { // Saves many 'if' ticks inside the while loop - If the input file is already a COO, simply add the coordinates the vectors.
		std::unordered_set<int> vertices;

		while (fgets(line, sizeof(line), file)) {
			if (line[0] == '#' || line[0] == '\n') {
				//print_debug_log("\nEscaped a comment.");
				continue;
			}
			// Save source and target vertex (temp)
			int source_vertex;
			int target_vertex;
			sscanf(line, "%d%d\t", &source_vertex, &target_vertex);
			// Add vertices to the source and target arrays, forming an edge accordingly
			source_vertices.push_back(source_vertex);
			destination_vertices.push_back(target_vertex);
			vertices.insert(source_vertex);
			vertices.insert(target_vertex);
		}
		SIZE_VERTICES = vertices.size();
		SIZE_EDGES = source_vertices.size();
		printf("\nTotal amount of vertices: %zd", SIZE_VERTICES);
		printf("\nTotal amount of edges: %zd", SIZE_EDGES);
	}
	else {
		std::unordered_map<int, int> map_from_edge_to_coordinate;
		while (fgets(line, sizeof(line), file)) {
			if (line[0] == '#' || line[0] == '\n') {
				//print_debug_log("\nEscaped a comment.");
				continue;
			}
			// Save source and target vertex (temp)
			int source_vertex;
			int target_vertex;
			sscanf(line, "%d%d\t", &source_vertex, &target_vertex);
			// Add vertices to the source and target arrays, forming an edge accordingly
			current_coordinate = add_vertex_as_coordinate(source_vertices, map_from_edge_to_coordinate, source_vertex, current_coordinate);
			current_coordinate = add_vertex_as_coordinate(destination_vertices, map_from_edge_to_coordinate, target_vertex, current_coordinate);
		}
		SIZE_VERTICES = map_from_edge_to_coordinate.size();
		SIZE_EDGES = source_vertices.size();
		printf("\nTotal amount of vertices: %zd", SIZE_VERTICES);
		printf("\nTotal amount of edges: %zd", SIZE_EDGES);
	}
	COO_List* coo_list = (COO_List*)malloc(sizeof(COO_List));
	source_vertices.reserve(source_vertices.size());
	destination_vertices.reserve(destination_vertices.size());
	coo_list->source = &source_vertices[0];
	coo_list->destination = &destination_vertices[0];
	if (source_vertices.size() != destination_vertices.size()) {
		printf("\nThe size of the source vertices does not equal the destination vertices.");
		exit(1);
	}
	bool SAVE_INPUT_FILE_AS_COO = false;
	if (SAVE_INPUT_FILE_AS_COO) {
		save_input_file_as_coo(source_vertices, destination_vertices, "C:\\Users\\AJ\\Desktop\\new_datasets\\coo\\none.txt");
	}
	// Print edges
	/*for (int i = 0; i < source_vertices.size(); i++) {
	printf("\n(%d, %d)", coo_list->source[i], coo_list->destination[i]);
	}*/
	fclose(file);
	return coo_list;
}

int GraphIO::add_vertex_as_coordinate(std::vector<int>& vertices_type, std::unordered_map<int, int>& map_from_edge_to_coordinate, int vertex, int coordinate) {
	if (map_from_edge_to_coordinate.count(vertex)) {
		vertices_type.push_back(map_from_edge_to_coordinate.at(vertex));
		return coordinate;
	}
	else {
		map_from_edge_to_coordinate[vertex] = coordinate;
		vertices_type.push_back(coordinate);
		coordinate++;
		return coordinate;
	}
}

void GraphIO::save_input_file_as_coo(std::vector<int>& source_vertices, std::vector<int>& destination_vertices, char* save_path) {
	printf("\nWriting results to output file.");
	char* file_path = save_path;
	FILE *output_file = fopen(file_path, "w");
	if (output_file == NULL) {
		printf("\nError writing results to output file.");
		exit(1);
	}
	for (int i = 0; i < source_vertices.size(); i++) {
		fprintf(output_file, "%d\t%d\n", source_vertices[i], destination_vertices[i]);
	}
	fclose(output_file);
}

CSR_List* GraphIO::convert_coo_to_csr_format(int* source_vertices, int* target_vertices) {
	printf("\nConverting COO to CSR format.");
	CSR_List* csr_list = (CSR_List*)malloc(sizeof(CSR_List));
	csr_list->offsets = (int*)malloc((SIZE_VERTICES + 1) * sizeof(int));
	csr_list->indices = (int*)malloc(SIZE_EDGES * sizeof(int));
	
	// First setup the COO format from the input (source_vertices and target_vertices array)
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCreate(&handle);
	nvgraphCreateGraphDescr(handle, &graph);
	nvgraphCOOTopology32I_t cooTopology = (nvgraphCOOTopology32I_t)malloc(sizeof(struct nvgraphCOOTopology32I_st));
	cooTopology->nedges = SIZE_EDGES;
	cooTopology->nvertices = SIZE_VERTICES;
	cooTopology->tag = NVGRAPH_UNSORTED;
	gpuErrchk(cudaMalloc((void**)&cooTopology->source_indices, SIZE_EDGES * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&cooTopology->destination_indices, SIZE_EDGES * sizeof(int)));
	gpuErrchk(cudaMemcpy(cooTopology->source_indices, source_vertices, SIZE_EDGES * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cooTopology->destination_indices, target_vertices, SIZE_EDGES * sizeof(int), cudaMemcpyHostToDevice));
	
	// Edge data (allocated, but not used)
	cudaDataType_t data_type = CUDA_R_32F;
	float* d_edge_data;
	float* d_destination_edge_data;
	gpuErrchk(cudaMalloc((void**)&d_edge_data, sizeof(float) * SIZE_EDGES)); // Note: only allocate this for 1 float since we don't have any data yet
	gpuErrchk(cudaMalloc((void**)&d_destination_edge_data, sizeof(float) * SIZE_EDGES)); // Note: only allocate this for 1 float since we don't have any data yet
	
	nvgraphCSRTopology32I_t csrTopology = (nvgraphCSRTopology32I_t)malloc(sizeof(struct nvgraphCSRTopology32I_st));
	int **d_indices = &(csrTopology->destination_indices);
	int **d_offsets = &(csrTopology->source_offsets);
	gpuErrchk(cudaMalloc((void**)d_indices, SIZE_EDGES * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)d_offsets, (SIZE_VERTICES + 1) * sizeof(int)));

	check(nvgraphConvertTopology(handle, NVGRAPH_COO_32, cooTopology, d_edge_data, &data_type, NVGRAPH_CSR_32, csrTopology, d_destination_edge_data));
	gpuErrchk(cudaPeekAtLastError());
	
	// Copy data to the host (without edge data)
	gpuErrchk(cudaMemcpy(csr_list->indices, *d_indices, SIZE_EDGES * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(csr_list->offsets, *d_offsets, (SIZE_VERTICES + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	
	// Clean up (Data allocated on device and both topologies, since we only want to work with indices and offsets for now)
	cudaFree(d_indices);
	cudaFree(d_offsets);
	cudaFree(d_edge_data);
	cudaFree(d_destination_edge_data);
	cudaFree(cooTopology->destination_indices);
	cudaFree(cooTopology->source_indices);
	free(cooTopology);
	free(csrTopology);
	return csr_list;
}

void GraphIO::check(nvgraphStatus_t status) {
	if (status == NVGRAPH_STATUS_NOT_INITIALIZED) {
		printf("\nError converting to CSR: %d - NVGRAPH_STATUS_NOT_INITIALIZED", status);
		exit(0);
	}
	else if (status == NVGRAPH_STATUS_ALLOC_FAILED) {
		printf("\nError converting to CSR: %d - NVGRAPH_STATUS_ALLOC_FAILED", status);
		exit(0);
	}
	else if (status == NVGRAPH_STATUS_INVALID_VALUE) {
		printf("\nError converting to CSR: %d - NVGRAPH_STATUS_INVALID_VALUE", status);
		exit(0);
	}
	else if (status == NVGRAPH_STATUS_INTERNAL_ERROR) {
		printf("\nError converting to CSR: %d - NVGRAPH_STATUS_INTERNAL_ERROR", status);
		exit(0);
	}
	else if (status == NVGRAPH_STATUS_SUCCESS) {
		printf("\nConverted to CSR successfully (statuscode %d).\n", status);
	}
	else {
		printf("\nSome other error occurred while trying to convert to CSR.");
		exit(0);
	}
}

void GraphIO::write_output_to_file(std::vector<Edge>& results, char* output_path) {
	printf("\nWriting results to output file.");

	char* file_path = output_path;
	FILE *output_file = fopen(file_path, "w");
	if (output_file == NULL) {
		printf("\nError writing results to output file.");
		exit(1);
	}
	for (int i = 0; i < results.size(); i++) {
		fprintf(output_file, "%d\t%d\n", results[i].source, results[i].destination);
	}
	fclose(output_file);
}

void GraphIO::write_expanded_output_to_file(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges, char* ouput_path) {
	printf("\nWriting results to output file.");

	char* file_path = ouput_path;
	FILE *output_file = fopen(file_path, "w");

	if (output_file == NULL) {
		printf("\nError writing results to output file.");
		exit(1);
	}

	// Write sampled graph versions
	for (int i = 0; i < amount_of_sampled_graphs; i++) {
		for (int p = 0; p < sampled_graph_version_list[i].edges.size(); p++) {
			fprintf(output_file, "\n%c%d\t%c%d", sampled_graph_version_list[i].label, sampled_graph_version_list[i].edges[p].source, sampled_graph_version_list[i].label, sampled_graph_version_list[i].edges[p].destination);
		}
	}

	for (int i = 0; i < bridge_edges.size(); i++) {
		fprintf(output_file, "\n%s\t%s", bridge_edges[i].source, bridge_edges[i].destination);
	}

	fclose(output_file);
}