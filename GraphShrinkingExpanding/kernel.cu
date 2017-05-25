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
#include "device_functions.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <map>

#define SIZE_VERTICES 281903 //20
#define SIZE_EDGES 2312497 //72 NOTE IF YOU AUTOMATE THIS, MAKE SURE TO CHECK WHETHER THE EDGE DATA DEVICE ARRAY STILL WORKS.
#define ENABLE_DEBUG_LOG false

typedef struct Sampled_Vertices;
typedef struct COO_List;
typedef struct Edge;
void load_graph_from_edge_list_file(int*, int*, char*);
COO_List* load_graph_from_edge_list_file_to_coo(std::vector<int>&, std::vector<int>&, char*);
int add_vertex_as_coordinate(std::vector<int>&, std::unordered_map<int, int>&, int, int);
int get_thread_size();
int calculate_node_sampled_size(float);
int get_block_size();
Sampled_Vertices* perform_edge_based_node_sampling_step(int*, int*, float);
void print_debug_log(char*);
void print_debug_log(char*, int);
void print_coo(int*, int*);
void print_csr(int*, int*);
void convert_coo_to_csr_format(int*, int*, int*, int*);
void write_output_to_file(std::vector<Edge>&, char* output_path);
void check(nvgraphStatus_t);

typedef struct COO_List {
	int* source;
	int* destination;
} COO_List;

typedef struct Sampled_Vertices {
	int* vertices;
	int sampled_vertices_size;
} Sampled_Vertices;

typedef struct Edge {
	int source, destination;
} Edge;

__device__ Edge edge_data[SIZE_EDGES];
__device__ int d_edge_count = 0;

__device__ int push_edge(Edge &edge) {
	int edge_index = atomicAdd(&d_edge_count, 1);
	if (edge_index < SIZE_EDGES) {
		edge_data[edge_index] = edge;
		return edge_index;
	} else {
		printf("Maximum edge size threshold reached.");
		return -1;
	}
}

__global__
void perform_induction_step(int* sampled_vertices, int* offsets, int* indices) {
	int neighbor_index_start_offset = blockIdx.x * blockDim.x + threadIdx.x;
	int neighbor_index_end_offset = neighbor_index_start_offset + 1;

	for (int n = offsets[neighbor_index_start_offset]; n < offsets[neighbor_index_end_offset]; n++) {
		bool found_vertex_u = sampled_vertices[neighbor_index_start_offset] != -1;
		bool found_vertex_v = sampled_vertices[indices[n]] != -1;

		if (found_vertex_u && found_vertex_v) {
			//printf("\nAdd edge: (%d,%d).", neighbor_index_start_offset, indices[n]);
			Edge edge;
			edge.source = neighbor_index_start_offset;
			edge.destination = indices[n];
			push_edge(edge);
		}

		found_vertex_u = false;
		found_vertex_v = false;
	}
}

/*
TODO: Allocate the memory on the GPU only when you need it, after collecting the edge-based node step.
*/
int main() {
	//char* file_path = "C:\\Users\\AJ\\Documents\\example_graph.txt";
	//char* file_path = "C:\\Users\\AJ\\Desktop\\nvgraphtest\\nvGraphExample-master\\nvGraphExample\\web-Stanford.txt";
	char* file_path = "C:\\Users\\AJ\\Desktop\\nvgraphtest\\nvGraphExample-master\\nvGraphExample\\web-Stanford_large.txt";
	//char* file_path = "C:\\Users\\AJ\\Desktop\\edge_list_example.txt";
	std::vector<int> source_vertices;
	std::vector<int> destination_vertices;
	COO_List* coo_list = load_graph_from_edge_list_file_to_coo(source_vertices, destination_vertices, file_path);

	size_t print_size = (sizeof(int) * SIZE_EDGES) + (3000 * sizeof(int));
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, print_size);

	// print_coo(source_vertices, target_vertices);

	// Convert the COO graph into a CSR format for the in memory GPU representation
	int* h_offsets = (int*)malloc((SIZE_VERTICES + 1) * sizeof(int));
	int* h_indices = (int*)malloc(SIZE_EDGES * sizeof(int));

	convert_coo_to_csr_format(coo_list->source, coo_list->destination, h_offsets, h_indices);

	//print_csr(h_offsets, h_indices);

	// Edge based Node Sampling Step
	Sampled_Vertices* sampled_vertices = perform_edge_based_node_sampling_step(coo_list->source, coo_list->destination, 0.5);
	printf("\nCollected %d vertices.", sampled_vertices->sampled_vertices_size);

	// Induction step (TODO: re-use device memory from CSR conversion)
	int* d_offsets;
	int* d_indices;
	cudaMalloc((void**)&d_offsets, sizeof(int)*(SIZE_VERTICES + 1));
	cudaMalloc((void**)&d_indices, sizeof(int)*SIZE_EDGES);
	cudaMemcpy(d_indices, h_indices, SIZE_EDGES * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_offsets, h_offsets, sizeof(int)*(SIZE_VERTICES + 1), cudaMemcpyHostToDevice);

	int* d_sampled_vertices;
	cudaMalloc((void**)&d_sampled_vertices, sizeof(int)*SIZE_VERTICES);
	cudaMemcpy(d_sampled_vertices, sampled_vertices->vertices, sizeof(int)*(SIZE_VERTICES), cudaMemcpyHostToDevice);

	printf("\nRunning kernel (induction step) with block size %d and thread size %d:", get_block_size(), get_thread_size());
	perform_induction_step<<<get_block_size(), get_thread_size()>>>(d_sampled_vertices, d_offsets, d_indices);
	
	int h_edge_count;
	cudaMemcpyFromSymbol(&h_edge_count, d_edge_count, sizeof(int));
	if (h_edge_count >= SIZE_EDGES + 1) {
		printf("overflow error\n"); return 1; 
	}

	printf("\nAmount of edges collected: %d", h_edge_count);
	std::vector<Edge> results(h_edge_count);
	cudaMemcpyFromSymbol(&(results[0]), edge_data, h_edge_count * sizeof(Edge));

	write_output_to_file(results, "C:\\Users\\AJ\\Desktop\\output_test\\outputexample.txt");

	cudaFree(d_offsets);
	cudaFree(d_indices);
	cudaFree(d_sampled_vertices);

	// Cleanup
	free(sampled_vertices->vertices);
	free(sampled_vertices);
	free(coo_list);
	free(h_indices);
	free(h_offsets);
	return 0;
}

/*
Fast conversion to CSR - Using nvGraph for conversion
Modified from: github.com/bmass02/nvGraphExample
*/
void convert_coo_to_csr_format(int* source_vertices, int* target_vertices, int* h_offsets, int* h_indices) {
	printf("\nConverting COO to CSR format.");

	// First setup the COO format from the input (source_vertices and target_vertices array)
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCreate(&handle);
	nvgraphCreateGraphDescr(handle, &graph);
	nvgraphCOOTopology32I_t cooTopology = (nvgraphCOOTopology32I_t)malloc(sizeof(struct nvgraphCOOTopology32I_st));
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
	cudaMalloc((void**)&d_edge_data, sizeof(float) * SIZE_EDGES); // Note: only allocate this for 1 float since we don't have any data yet
	cudaMalloc((void**)&d_destination_edge_data, sizeof(float) * SIZE_EDGES); // Note: only allocate this for 1 float since we don't have any data yet

	// Convert COO to a CSR format
	nvgraphCSRTopology32I_t csrTopology = (nvgraphCSRTopology32I_t)malloc(sizeof(struct nvgraphCSRTopology32I_st));
	int **d_indices = &(csrTopology->destination_indices);
	int **d_offsets = &(csrTopology->source_offsets);

	cudaMalloc((void**)d_indices, SIZE_EDGES * sizeof(int));
	cudaMalloc((void**)d_offsets, (SIZE_VERTICES + 1) * sizeof(int));

	check(nvgraphConvertTopology(handle, NVGRAPH_COO_32, cooTopology, d_edge_data, &data_type, NVGRAPH_CSR_32, csrTopology, d_destination_edge_data));

	// Copy data to the host (without edge data)
	cudaMemcpy(h_indices, *d_indices, SIZE_EDGES * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_offsets, *d_offsets, (SIZE_VERTICES + 1) * sizeof(int), cudaMemcpyDeviceToHost);

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

int get_thread_size() {
	return ((SIZE_VERTICES + 1) > 1024) ? 1024 : SIZE_VERTICES;
}

int get_block_size() {
	return ((SIZE_VERTICES + 1) > 1024) ? ((SIZE_VERTICES / 1024) + 1) : 1;
}

int calculate_node_sampled_size(float fraction) {
	return int(SIZE_VERTICES * fraction);
}

/*
NOTE: Only reads integer vertices for now (through the 'sscanf' function) and obvious input vertices arrays
*/
void load_graph_from_edge_list_file(int* source_vertices, int* target_vertices, char* file_path) {
	printf("\nLoading graph file from: %s", file_path);

	FILE* file = fopen(file_path, "r");
	char line[256];
	int edge_index = 0;

	while (fgets(line, sizeof(line), file)) {
		if (line[0] == '#') {
			//print_debug_log("\nEscaped a comment.");
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

		//print_debug_log("\nAdded start vertex:", source_vertex);
		//print_debug_log("\nAdded end vertex:", target_vertex);
	}

	fclose(file);
}

COO_List* load_graph_from_edge_list_file_to_coo(std::vector<int>& source_vertices, std::vector<int>& destination_vertices, char* file_path) {
	printf("\nLoading graph file from: %s", file_path);

	std::unordered_map<int, int> map_from_edge_to_coordinate;

	FILE* file = fopen(file_path, "r");
	
	char line[256];
	int edge_index = 0;

	int current_coordinate = 0;

	while (fgets(line, sizeof(line), file)) {
		if (line[0] == '#') {
			//print_debug_log("\nEscaped a comment.");
			continue;
		}

		// Save source and target vertex (temp)
		int source_vertex;
		int target_vertex;

		sscanf(line, "%d%d\t", &source_vertex, &target_vertex);

		// Add vertices to the source and target arrays, forming an edge accordingly
		
		// You definitely need to make a function out of this one for the destination nodes, since those might be different
		current_coordinate = add_vertex_as_coordinate(source_vertices, map_from_edge_to_coordinate, source_vertex, current_coordinate);
		current_coordinate = add_vertex_as_coordinate(destination_vertices, map_from_edge_to_coordinate, target_vertex, current_coordinate);
	}

	COO_List* coo_list = (COO_List*)malloc(sizeof(COO_List));

	source_vertices.reserve(source_vertices.size());
	destination_vertices.reserve(destination_vertices.size());
	coo_list->source = &source_vertices[0];
	coo_list->destination = &destination_vertices[0];

	printf("\nTotal amount of vertices: %d", map_from_edge_to_coordinate.size());
	printf("\Total amount of edges: %d", source_vertices.size());

	// Print edges
	/*for (int i = 0; i < source_vertices.size(); i++) {
		printf("\n(%d, %d)", coo_list->source[i], coo_list->destination[i]);
	}*/

	return coo_list;

	fclose(file);
}

int add_vertex_as_coordinate(std::vector<int>& vertices_type, std::unordered_map<int, int>& map_from_edge_to_coordinate, int vertex, int coordinate) {
	if (map_from_edge_to_coordinate.count(vertex)) {
		vertices_type.push_back(map_from_edge_to_coordinate.at(vertex));

		return coordinate;
	} else {
		map_from_edge_to_coordinate[vertex] = coordinate;
		vertices_type.push_back(coordinate);
		coordinate++;

		return coordinate;
	}

	return 0;
}

Sampled_Vertices* perform_edge_based_node_sampling_step(int* source_vertices, int* target_vertices, float fraction) {
	printf("\nPerforming edge based node sampling step.\n");

	Sampled_Vertices* sampled_vertices = (Sampled_Vertices*) malloc(sizeof(Sampled_Vertices));

	int amount_total_sampled_vertices = calculate_node_sampled_size(fraction);

	std::random_device seeder;
	std::mt19937 engine(seeder());

	sampled_vertices->vertices = (int*) calloc(SIZE_VERTICES, sizeof(int));
	int collected_amount = 0;

	// TODO: memcpy
	for (int x = 0; x < SIZE_VERTICES; x++) {
		sampled_vertices->vertices[x] = -1;
	}

	
	while (collected_amount < amount_total_sampled_vertices) {
		// Pick a random vertex u
		std::uniform_int_distribution<int> range_edges(0, (SIZE_EDGES-1)); // Don't select the last element in the offset
		int random_edge_index = range_edges(engine);

		// Insert u, v 
		if (sampled_vertices->vertices[source_vertices[random_edge_index]] == -1) {
			sampled_vertices->vertices[source_vertices[random_edge_index]] = source_vertices[random_edge_index];
			print_debug_log("\nCollected vertex:", source_vertices[random_edge_index]);
			collected_amount++;
		}
		if (sampled_vertices->vertices[target_vertices[random_edge_index]] == -1) {
			sampled_vertices->vertices[target_vertices[random_edge_index]] = target_vertices[random_edge_index];
			print_debug_log("\nCollected vertex:", target_vertices[random_edge_index]);
			collected_amount++;
		}
	}

	sampled_vertices->sampled_vertices_size = collected_amount;

	return sampled_vertices;
}

void write_output_to_file(std::vector<Edge>& results, char* ouput_path) {
	char* file_path = ouput_path;
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

void print_coo(int* source_vertices, int* end_vertices) {
	for (int i = 0; i < SIZE_EDGES; i++) {
		printf("\n%d, %d", source_vertices[i], end_vertices[i]);
	}
}

void print_csr(int* h_offsets, int* h_indices) {
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

void print_debug_log(char* message) {
	if (ENABLE_DEBUG_LOG)
		printf("%s", message);
}

void print_debug_log(char* message, int value) {
	if (ENABLE_DEBUG_LOG)
		printf("%s %d", message, value);
}