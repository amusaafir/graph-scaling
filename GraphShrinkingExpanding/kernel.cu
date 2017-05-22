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
#include <unordered_set>
#include <random>

#define SIZE_VERTICES 281903 //20
#define SIZE_EDGES 2312497 //72 NOTE IF YOU AUTOMATE THIS, MAKE SURE TO CHECK WHETHER THE EDGE DATA DEVICE ARRAY STILL WORKS.
#define DEBUG_MODE false

void load_graph_from_edge_list_file(int*, int*, char*);
int get_thread_size();
int calculate_node_sampled_size(float);
int get_block_size();
typedef struct Sampled_Vertices;
Sampled_Vertices* perform_edge_based_node_sampling_step(int*, int*, float);
void print_debug(char*);
void print_debug(char*, int);
void print_coo(int*, int*);
void print_csr(int*, int*);
void convert_coo_to_csr_format(int*, int*, int*, int*);
void check(nvgraphStatus_t);

typedef struct Sampled_Vertices {
	int* vertices;
	int sampled_vertices_size;
} Sampled_Vertices;

/*
__global__
void loopThroughVertexNeighbours(int* offsets, int* indices, int MAX_VERTEX_SIZE, int* total_num_edges) {
int neighbor_index_start_offset = blockIdx.x * blockDim.x + threadIdx.x;
int neighbor_index_end_offset = blockIdx.x * blockDim.x + threadIdx.x + 1;

//printf("\nThread %d: between: %d, %d", neighbor_index_start_offset, offsets[neighbor_index_start_offset], offsets[neighbor_index_end_offset]);

for (int n = offsets[neighbor_index_start_offset]; n < offsets[neighbor_index_end_offset]; n++) {
printf("\nEdge: (%d, %d)", neighbor_index_start_offset, indices[n]);
atomicAdd(total_num_edges, 1);
}
}
*/

typedef struct {
	int source, destination;
} Edge;

__device__ Edge edge_data[SIZE_EDGES];
__device__ int dev_count = 0;

__device__ int my_push_back(Edge & mt) {
	int insert_pt = atomicAdd(&dev_count, 1);
	if (insert_pt < SIZE_EDGES) {
		edge_data[insert_pt] = mt;
		return insert_pt;
	}
	else {
		return -1;
	}
}

__global__
void perform_induction_step(int* sampled_vertices, int* sampled_vertices_size, int* offsets, int* indices, unsigned long long* total_num_edges) {
	int neighbor_index_start_offset = blockIdx.x * blockDim.x + threadIdx.x;
	int neighbor_index_end_offset = blockIdx.x * blockDim.x + threadIdx.x + 1;
	
	//printf("\nSize: %d", *sampled_vertices_size);
	
	for (int n = offsets[neighbor_index_start_offset]; n < offsets[neighbor_index_end_offset]; n++) {
		
		//printf("\nEdge: (%d, %d)", neighbor_index_start_offset, indices[n]);
		
		//printf("\nAdd edge: (%d,%d).", neighbor_index_start_offset, indices[n]);
		
		bool found_vertex_u = false;
		bool found_vertex_v = false;

		for (int i = 0; i < 6000; i++) {
			if (neighbor_index_start_offset == sampled_vertices[i]) {
				found_vertex_u = true;
			}

			if (indices[n] == sampled_vertices[i]) {
				found_vertex_v = true;
			}
		}
		
		if (found_vertex_u && found_vertex_v) {
			//printf("\nAdd edge: (%d,%d).", neighbor_index_start_offset, indices[n]);
			Edge edge;
			edge.source = neighbor_index_start_offset;
			edge.destination = indices[n];
			my_push_back(edge);
			atomicAdd(total_num_edges, 1);
		}
		found_vertex_u = false;
		found_vertex_v = false;
	}
}

/*
TODO: Allocate the memory on the GPU only when you need it, after collecting the edge-based node step.
*/
int main() {
	int* source_vertices;
	int* target_vertices;
	//char* file_path = "C:\\Users\\AJ\\Documents\\example_graph.txt";
	//char* file_path = "C:\\Users\\AJ\\Desktop\\nvgraphtest\\nvGraphExample-master\\nvGraphExample\\web-Stanford.txt";
	char* file_path = "C:\\Users\\AJ\\Desktop\\nvgraphtest\\nvGraphExample-master\\nvGraphExample\\web-Stanford_large.txt";

	size_t print_size = (sizeof(int) * SIZE_EDGES) + (3000 * sizeof(int));
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, print_size);

	source_vertices = (int*)malloc(sizeof(int) * SIZE_EDGES);
	target_vertices = (int*)malloc(sizeof(int) * SIZE_EDGES);

	// Read an input graph into a COO format.
	load_graph_from_edge_list_file(source_vertices, target_vertices, file_path);

	// print_coo(source_vertices, target_vertices);

	// Convert the COO graph into a CSR format for the in memory GPU representation
	int* h_offsets = (int*)malloc((SIZE_VERTICES + 1) * sizeof(int));
	int* h_indices = (int*)malloc(SIZE_EDGES * sizeof(int));

	convert_coo_to_csr_format(source_vertices, target_vertices, h_offsets, h_indices);

	//print_csr(h_offsets, h_indices);

	// Edge based Node Sampling Step
	Sampled_Vertices* sampled_vertices = perform_edge_based_node_sampling_step(source_vertices, target_vertices, 0.5);
	printf("\nCollected %d vertices.", sampled_vertices->sampled_vertices_size);

	// Induction step (TODO: re-use device memory from CSR conversion)
	int* d_offsets;
	int* d_indices;
	cudaMalloc((void**)&d_offsets, sizeof(int)*(SIZE_VERTICES + 1));
	cudaMalloc((void**)&d_indices, sizeof(int)*SIZE_EDGES);
	cudaMemcpy(d_indices, h_indices, SIZE_EDGES * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_offsets, h_offsets, sizeof(int)*(SIZE_VERTICES + 1), cudaMemcpyHostToDevice);

	int* d_sampled_vertices;
	cudaMalloc((void**)&d_sampled_vertices, sizeof(int)*sampled_vertices->sampled_vertices_size);
	cudaMemcpy(d_sampled_vertices, sampled_vertices->vertices, sizeof(int)*(sampled_vertices->sampled_vertices_size), cudaMemcpyHostToDevice);

	int* d_sampled_vertices_size;
	cudaMalloc(&d_sampled_vertices_size, sizeof(int));
	cudaMemcpy(d_sampled_vertices_size, &sampled_vertices->sampled_vertices_size, sizeof(int), cudaMemcpyHostToDevice);

	printf("\nRunning with block size %d and thread size %d:", get_block_size(), get_thread_size());
	unsigned long long* init_total_edges = (unsigned long long*)malloc(sizeof(unsigned long long));
	*init_total_edges = 0;
	unsigned long long* d_init_total_edges;
	cudaMalloc(&d_init_total_edges, sizeof(unsigned long long));
	cudaMemcpy(d_init_total_edges, init_total_edges, sizeof(unsigned long long), cudaMemcpyHostToDevice);
	perform_induction_step <<<get_block_size(), get_thread_size()>>>(d_sampled_vertices, d_sampled_vertices_size, d_offsets, d_indices, d_init_total_edges);
	unsigned long long* result_total_edges = (unsigned long long*)malloc(sizeof(unsigned long long));
	cudaMemcpy(result_total_edges, d_init_total_edges, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	printf("\nTotal selected edges: %llu\n", (*result_total_edges));

	int dsize;
	cudaMemcpyFromSymbol(&dsize, dev_count, sizeof(int));
	if (dsize >= SIZE_EDGES+1) { printf("overflow error\n"); return 1; }
	printf("d_size: %d", dsize);
	/*std::vector<Edge> results(dsize);
	cudaMemcpyFromSymbol(&(results[0]), edge_data, dsize * sizeof(Edge));
	printf("\nWOOHOO: %d", dsize);
	printf("\nTest: (%d, %d)", results[0].source, results[0].destination);
	*/
	cudaFree(d_offsets);
	cudaFree(d_indices);
	cudaFree(d_sampled_vertices_size);
	cudaFree(d_sampled_vertices);

	// Cleanup
	free(sampled_vertices->vertices);
	free(sampled_vertices);
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
void convert_coo_to_csr_format(int* source_vertices, int* target_vertices, int* h_offsets, int* h_indices) {
	printf("\nConvert COO to CSR format");

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

	/*int* total_num_init = (int*) malloc(sizeof(int));
	*total_num_init = 0;
	printf("hi: %d", *total_num_init);

	int* d_total_num_edges;
	cudaMalloc(&d_total_num_edges, sizeof(int));
	cudaMemcpy(d_total_num_edges, total_num_init, sizeof(int), cudaMemcpyHostToDevice);

	printf("\nRunning with a block_size of %d and thread_size of %d.", get_block_size(), get_thread_size());
	loopThroughVertexNeighbours<<<get_block_size(), get_thread_size()>>>(*d_offsets, *d_indices, SIZE_VERTICES, d_total_num_edges);

	int* result = (int*)malloc(sizeof(int));
	cudaMemcpy(result, d_total_num_edges, sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nTotal edges: %d", (*result));*/

	// Clean up (Data allocated on device and both topologies, since we only want to work with indices and offsets for now)
	//cudaFree(d_total_num_edges);
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
	printf("\nLoading graph file: %s", file_path);

	FILE* file = fopen(file_path, "r");
	char line[256];
	int edge_index = 0;

	while (fgets(line, sizeof(line), file)) {
		if (line[0] == '#') {
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

/*
This approach has no bias towards high degree nodes, which is not the good way to go if we want to use TIES.
If we still want to use CSR as the representation about outside the memory of the GPU, we would have a problem with selecting random edges
in constant time (it will become O=log n for every edge).
Sampled_Vertices* perform_edge_based_node_sampling_step(int* h_offsets, int* h_indices, float fraction) {
Sampled_Vertices* sampled_vertices = (Sampled_Vertices*) malloc(sizeof(Sampled_Vertices));

int amount_total_sampled_vertices = calculate_node_sampled_size(fraction);

std::random_device seeder;
std::mt19937 engine(seeder());

std::unordered_set<int> sampled_vertices_set = {};
while (sampled_vertices_set.size() < amount_total_sampled_vertices) {
// Pick a random vertex u
std::uniform_int_distribution<int> range_vertex_u(0, (SIZE_VERTICES - 1)); // Don't select the last element in the offset
int random_vertex_u = range_vertex_u(engine);

// Pick random vertex v
int start_offset = h_offsets[random_vertex_u];
int end_offset = h_offsets[random_vertex_u + 1];
std::uniform_int_distribution<int> range_vertex_v(start_offset, end_offset-1);
int random_vertex_v = h_indices[range_vertex_v(engine)];

// Insert u, v
sampled_vertices_set.insert(random_vertex_u);
sampled_vertices_set.insert(random_vertex_v);
}

printf("\nCollected %d vertices.", sampled_vertices_set.size());

for (std::unordered_set<int>::iterator itr = sampled_vertices_set.begin(); itr != sampled_vertices_set.end(); ++itr) {
printf("%d,",*itr);
}

return sampled_vertices;
}
*/

Sampled_Vertices* perform_edge_based_node_sampling_step(int* source_vertices, int* target_vertices, float fraction) {
	Sampled_Vertices* sampled_vertices = (Sampled_Vertices*)malloc(sizeof(Sampled_Vertices));

	int amount_total_sampled_vertices = calculate_node_sampled_size(fraction);

	std::random_device seeder;
	std::mt19937 engine(seeder());

	std::unordered_set<int> sampled_vertices_set = {};
	while (sampled_vertices_set.size() <= amount_total_sampled_vertices) {
		// Pick a random vertex u
		std::uniform_int_distribution<int> range_edges(0, (SIZE_EDGES - 1)); // Don't select the last element in the offset
		int random_edge_index = range_edges(engine);

		// Insert u, v 
		sampled_vertices_set.insert(source_vertices[random_edge_index]);
		sampled_vertices_set.insert(source_vertices[random_edge_index]);
	}

	// Copy elements back to a normal array
	sampled_vertices->vertices = (int*)malloc(sizeof(int)*sampled_vertices_set.size());
	int i = 0;
	for (std::unordered_set<int>::iterator itr = sampled_vertices_set.begin(); itr != sampled_vertices_set.end(); ++itr) {
		sampled_vertices->vertices[i] = *itr;
		//printf("\nCollected vertex: %d", sampled_vertices->vertices[i]);
		i++;
	}
	sampled_vertices->sampled_vertices_size = sampled_vertices_set.size();

	sampled_vertices_set = std::unordered_set<int>();

	return sampled_vertices;
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

void print_debug(char* message) {
	if (DEBUG_MODE)
		printf("%s", message);
}

void print_debug(char* message, int value) {
	if (DEBUG_MODE)
		printf("%s %d", message, value);
}