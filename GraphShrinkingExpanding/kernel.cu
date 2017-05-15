/*
NOTE: Run in VS using x64 platform.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
#include <string.h>
#include <nvgraph.h>

#define SIZE_VERTICES 3
#define SIZE_EDGES 5
#define DEBUG_MODE false

void load_graph_from_edge_list_file(int*, int*, char*);
void print_debug(char*);
void print_debug(char*, int);
void print_edge_list(int*, int*);
void convert_coo_to_csr_format(int*, int*);

int main() {
	
	int* source_vertices;
	int* target_vertices;
	char* file_path = "C:\\Users\\AJ\\Documents\\example_graph.txt";

	source_vertices = (int*) malloc(sizeof(int) * SIZE_EDGES);
	target_vertices = (int*) malloc(sizeof(int) * SIZE_EDGES);

	// Read an input graph into a COO format.
	load_graph_from_edge_list_file(source_vertices, target_vertices, file_path);
	
	// print_edge_list(source_vertices, target_vertices);

	// Convert the COO graph into a CSR format
	convert_coo_to_csr_format(source_vertices, target_vertices);

	// Cleanup
	free(source_vertices);
	free(target_vertices);

	return 0;
}

void check(nvgraphStatus_t status) {
	if (status != NVGRAPH_STATUS_SUCCESS) {
		printf("%d", nvgraphStatusGetString(status));
		system("PAUSE");
		exit(0);
	}
}

/*
Using nvGraph for conversion
Modified from: github.com/bmass02/nvGraphExample
*/
void convert_coo_to_csr_format(int* source_vertices, int* target_vertices) {
	printf("\nConvert COO to CSR format");

	// First setup the COO format from the input (source_vertices and target_vertices array)
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCreate(&handle);
	nvgraphCreateGraphDescr(handle, &graph);

	//nvgraphGraphDescr graph;
	nvgraphCOOTopology32I_t cooTopology = (nvgraphCOOTopology32I_t) malloc(sizeof(struct nvgraphCOOTopology32I_st));
	cooTopology->nedges = SIZE_EDGES;
	cooTopology->nvertices = SIZE_VERTICES;
	cooTopology->tag = NVGRAPH_UNSORTED;

	cudaMalloc((void**)&cooTopology->source_indices, SIZE_EDGES * sizeof(int));
	cudaMalloc((void**)&cooTopology->destination_indices, SIZE_EDGES * sizeof(int));

	cudaMemcpy(cooTopology->source_indices, source_vertices, SIZE_EDGES * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cooTopology->destination_indices, target_vertices, SIZE_EDGES * sizeof(int), cudaMemcpyHostToDevice);
	
	/* ==== EDGE DATA  ====*/
	// Add edge data(1 for all weights for now) (maybe this is redundant? like with destination edge data)
	cudaDataType_t data_type = CUDA_R_32F;
	float *h_edge_data = (float*)malloc(sizeof(float)*SIZE_EDGES);
	for (int i = 0; i < SIZE_EDGES; i++) {
		h_edge_data[i] = 1.0f;
	}

	// Add edge data on device
	float* d_edge_data;
	float* d_destination_edge_data;
	cudaMalloc((void**)&d_edge_data, sizeof(float) * SIZE_EDGES);
	cudaMalloc((void**)&d_destination_edge_data, sizeof(float) * SIZE_EDGES);

	cudaMemcpy(d_edge_data, h_edge_data, sizeof(float) * SIZE_EDGES, cudaMemcpyHostToDevice);
	/*=====================*/

	// Convert COO to a CSR format
	int *indices_h, *offsets_h, **indices_d, **offsets_d;
	nvgraphCSRTopology32I_t csrTopology = (nvgraphCSRTopology32I_t)malloc(sizeof(struct nvgraphCSRTopology32I_st));
	indices_d = &(csrTopology->destination_indices);
	offsets_d = &(csrTopology->source_offsets);

	cudaMalloc((void**)(indices_d), SIZE_EDGES * sizeof(int));
	cudaMalloc((void**)(offsets_d), (SIZE_VERTICES + 1) * sizeof(int));
	indices_h = (int*)malloc(SIZE_EDGES * sizeof(int));
	offsets_h = (int*)malloc((SIZE_VERTICES + 1) * sizeof(int));

	check(nvgraphConvertTopology(handle, NVGRAPH_COO_32, cooTopology, d_edge_data, &data_type, NVGRAPH_CSR_32, csrTopology, d_destination_edge_data));

	// Copy data to the host
	cudaMemcpy(indices_h, *indices_d, SIZE_EDGES * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(offsets_h, *offsets_d, (SIZE_VERTICES + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(edge_data_h, dst_edge_data_d, nedges * sizeof(float), cudaMemcpyDeviceToHost);

	printf("\mINDICES:\n");
	for (int i = 0; i < SIZE_EDGES;i++) {
		printf("%d, ", indices_h[i]);
	}

	printf("\nOFFSETS:\n");
	for (int i = 0; i < SIZE_VERTICES+1; i++) {
		printf("%d, ", offsets_h[i]);
	}

	cudaFree(indices_d);
	cudaFree(offsets_d);
	cudaFree(d_edge_data);
	cudaFree(d_destination_edge_data);
	cudaFree(cooTopology->destination_indices);
	cudaFree(cooTopology->source_indices);
	free(cooTopology);
}

/*
NOTE: Only reads integer vertices for now (through the 'sscanf' function) and obvious input vertices arrays
*/
void load_graph_from_edge_list_file(int* source_vertices, int* target_vertices, char* file_path) {
	printf("\nLoading graph file: %s", file_path);

	FILE* file = fopen(file_path, "r");
	char line[256];

	int edge_index = 0;

	while (fgets(line, sizeof(line), file)) {
		if (line[0]=='#') {
			print_debug("\nEscaped a comment.");
			continue;
		}

		// Save source and target vertex (temp)
		int source_vertex;
		int target_vertex;

		sscanf(line, "%d%d\t", &source_vertex, &target_vertex);

		// Add vertices to the source and target arrays, forming an edge accordingly
		source_vertices[edge_index] = source_vertex;
		target_vertices[edge_index] = target_vertex;

		// Increment edge index to add any new edge
		edge_index++;
		
		// Debug: Print source and target vertex
		print_debug("\nAdded start vertex:", source_vertex);
		print_debug("\nAdded end vertex:", target_vertex);
	}
	
	fclose(file);
}

void print_edge_list(int* source_vertices, int* end_vertices) {
	for (int i = 0; i < SIZE_EDGES; i++) {
		printf("\n%d, %d", source_vertices[i], end_vertices[i]);
	}
}

void print_debug(char* message) {
	if (DEBUG_MODE)
		printf("%s", message);
}

void print_debug(char* message, int value) {
	if (DEBUG_MODE)
		printf("%s %d", message, value);
}