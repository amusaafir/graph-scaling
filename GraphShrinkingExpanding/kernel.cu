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
void print_coo(int*, int*);
void print_csr(int*, int*);
void convert_coo_to_csr_format(int*, int*, int*, int*);
void check(nvgraphStatus_t);

__global__
void loopThroughVertexNeighbours(int* offsets, int* indices, int MAX_VERTEX_SIZE) {
	int neighbor_index_start_offset = blockIdx.x * blockDim.x + threadIdx.x;
	int neighbor_index_end_offset = blockIdx.x * blockDim.x + threadIdx.x + 1;

	printf("\nVertex: %d, start: %d, end: %d", neighbor_index_start_offset, offsets[neighbor_index_start_offset], offsets[neighbor_index_end_offset]);

	for (int n = offsets[neighbor_index_start_offset]; n < offsets[neighbor_index_end_offset]; n++) {
		printf("\n\tVertex: %d has Neighbor: %d", blockIdx.x * blockDim.x + threadIdx.x, indices[n]);
	}
}

int main() {
	int* source_vertices;
	int* target_vertices;
	char* file_path = "C:\\Users\\AJ\\Documents\\example_graph.txt";
	
	source_vertices = (int*) malloc(sizeof(int) * SIZE_EDGES);
	target_vertices = (int*) malloc(sizeof(int) * SIZE_EDGES);

	// Read an input graph into a COO format.
	load_graph_from_edge_list_file(source_vertices, target_vertices, file_path);
	
	// print_coo(source_vertices, target_vertices);

	// Convert the COO graph into a CSR format
	int* h_indices;
	int* h_offsets;
	h_indices = (int*)malloc(SIZE_EDGES * sizeof(int));
	h_offsets = (int*)malloc((SIZE_VERTICES + 1) * sizeof(int));
	convert_coo_to_csr_format(source_vertices, target_vertices, h_indices, h_offsets);
	
	print_csr(h_indices, h_offsets);

	// Cleanup
	free(source_vertices);
	free(target_vertices);

	free(h_indices);
	free(h_offsets);

	return 0;
}

/*
Fast conversion to CSR - Using nvGraph for conversion
Modified from: github.com/bmass02/nvGraphExample
*/
void convert_coo_to_csr_format(int* source_vertices, int* target_vertices, int* h_indices, int* h_offsets) {
	printf("\nConvert COO to CSR format");

	// First setup the COO format from the input (source_vertices and target_vertices array)
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCreate(&handle);
	nvgraphCreateGraphDescr(handle, &graph);
	nvgraphCOOTopology32I_t cooTopology = (nvgraphCOOTopology32I_t) malloc(sizeof(struct nvgraphCOOTopology32I_st));
	cooTopology->nedges = SIZE_EDGES;
	cooTopology->nvertices = SIZE_VERTICES;
	cooTopology->tag = NVGRAPH_UNSORTED;

	cudaMalloc((void**)&cooTopology->source_indices, SIZE_EDGES * sizeof(int));
	cudaMalloc((void**)&cooTopology->destination_indices, SIZE_EDGES * sizeof(int));

	cudaMemcpy(cooTopology->source_indices, source_vertices, SIZE_EDGES * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cooTopology->destination_indices, target_vertices, SIZE_EDGES * sizeof(int), cudaMemcpyHostToDevice);
	
	// Edge data (allocated, but not used)
	cudaDataType_t data_type = CUDA_R_32F;
	float* d_edge_data;
	float* d_destination_edge_data;
	cudaMalloc((void**)&d_edge_data, sizeof(float) * SIZE_EDGES); // Note, only allocate this for 1 float
	cudaMalloc((void**)&d_destination_edge_data, sizeof(float) * SIZE_EDGES); // Note, only allocate this for 1 float

	// Convert COO to a CSR format
	int **d_indices, **d_offsets;
	nvgraphCSRTopology32I_t csrTopology = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSRTopology32I_st));
	d_indices = &(csrTopology->destination_indices);
	d_offsets = &(csrTopology->source_offsets);

	cudaMalloc((void**) d_indices, SIZE_EDGES * sizeof(int));
	cudaMalloc((void**) d_offsets, (SIZE_VERTICES + 1) * sizeof(int));

	check(nvgraphConvertTopology(handle, NVGRAPH_COO_32, cooTopology, d_edge_data, &data_type, NVGRAPH_CSR_32, csrTopology, d_destination_edge_data));

	// Copy data to the host (without edge data)
	cudaMemcpy(h_indices, *d_indices, SIZE_EDGES * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_offsets, *d_offsets, (SIZE_VERTICES + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	
	loopThroughVertexNeighbours<<<1, SIZE_VERTICES>>>(*d_offsets, *d_indices, SIZE_VERTICES);

	// Clean up (Data allocated on device and both topologies, since we only want to work with indices and offsets for now)
	cudaFree(d_indices);
	cudaFree(d_offsets);
	cudaFree(d_edge_data);
	cudaFree(d_destination_edge_data);
	cudaFree(cooTopology->destination_indices);
	cudaFree(cooTopology->source_indices);
	free(cooTopology); 
	free(csrTopology);
	
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

void print_coo(int* source_vertices, int* end_vertices) {
	for (int i = 0; i < SIZE_EDGES; i++) {
		printf("\n%d, %d", source_vertices[i], end_vertices[i]);
	}
}

void print_csr(int* h_indices, int* h_offsets) {
	printf("\nRow Offsets (Vertex Table):\n");
	for (int i = 0; i < SIZE_VERTICES + 1; i++) {
		printf("%d, ", h_offsets[i]);
	}

	printf("\nColumn Indices (Edge Table):\n");
	for (int i = 0; i < SIZE_EDGES; i++) {
		printf("%d, ", h_indices[i]);
	}
}

void check(nvgraphStatus_t status) {
	if (status != NVGRAPH_STATUS_SUCCESS) {
		printf("ERROR : %d\n", status);
		exit(0);
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