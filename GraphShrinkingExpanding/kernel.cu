/*
NOTE: Run in VS using x64 platform.

TODO:

SHRINKING:
- Look into edge based vs CSR based device.
- Refactor code (multiple files)
- Ignore spaces while reading file!
- Performance improvement loading a graph
- Count vertices/edges automatically (Can be done later, use globals, need to allocate memory on gpu though in runtime)
- Load graph should be a separate method

EXPANDING:
- Force undirected (edge interconnection)
- Decrease size of char in Bridge_Edge
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

//#define SIZE_VERTICES 281903
//#define SIZE_EDGES 2312497

//#define SIZE_VERTICES 1632803
//#define SIZE_EDGES 30622564 

#define SIZE_VERTICES 6
#define SIZE_EDGES 5

//#define SIZE_VERTICES 4039
//#define SIZE_EDGES 88234
#define MAX_THREADS 1024
#define DEFAULT_EXPANDING_SAMPLE_SIZE 0.5
#define ENABLE_DEBUG_LOG false

typedef struct Sampled_Vertices sampled_vertices;
typedef struct COO_List coo_list;
typedef struct Edge edge;
typedef struct Sampled_Graph_Version;
typedef struct Bridge_Edge;
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
void sample_graph(char*, char*, float);
void convert_coo_to_csr_format(int*, int*, int*, int*);
void expand_graph(char*, char*, float);
void link_using_star_topology(Sampled_Graph_Version*, int, std::vector<Bridge_Edge>&);
void add_edge_interconnection_between_graphs(int, Sampled_Graph_Version*, Sampled_Graph_Version*, std::vector<Bridge_Edge>&);
int select_random_bridge_vertex(Sampled_Graph_Version*);
void write_expanded_output_to_file(Sampled_Graph_Version*, int, std::vector<Bridge_Edge>&, char*);
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

typedef struct Sampled_Graph_Version {
	std::vector<Edge> edges;
	char label;
} Sampled_Graph_Version;

typedef struct Bridge_Edge {
	char source[20];
	char destination[20];
} Bridge_Edge;

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
		if (sampled_vertices[neighbor_index_start_offset] && sampled_vertices[indices[n]]) {
			//printf("\nAdd edge: (%d,%d).", neighbor_index_start_offset, indices[n]);
			Edge edge;
			edge.source = neighbor_index_start_offset;
			edge.destination = indices[n];
			push_edge(edge);
		}
	}
}

//__device__ Edge edge_data_expanding[SIZE_EDGES];
//__device__ int d_edge_count_expanding = 0;

__device__ int push_edge_expanding(Edge &edge, Edge* edge_data_expanding, int* d_edge_count_expanding) {
	int edge_index = atomicAdd(d_edge_count_expanding, 1);
	if (edge_index < SIZE_EDGES) {
		edge_data_expanding[edge_index] = edge;
		return edge_index;
	}
	else {
		printf("Maximum edge size threshold reached.");
		return -1;
	}
}

__global__
void perform_induction_step_expanding(int* sampled_vertices, int* offsets, int* indices, Edge* edge_data_expanding, int* d_edge_count_expanding) {
	int neighbor_index_start_offset = blockIdx.x * blockDim.x + threadIdx.x;
	int neighbor_index_end_offset = neighbor_index_start_offset + 1;

	for (int n = offsets[neighbor_index_start_offset]; n < offsets[neighbor_index_end_offset]; n++) {
		if (sampled_vertices[neighbor_index_start_offset] && sampled_vertices[indices[n]]) {
			//printf("\nAdd edge: (%d,%d).", neighbor_index_start_offset, indices[n]);
			Edge edge;
			edge.source = neighbor_index_start_offset;
			edge.destination = indices[n];
			push_edge_expanding(edge, edge_data_expanding, d_edge_count_expanding);
		}
	}
}

/*
TODO: Allocate the memory on the GPU only when you need it, after collecting the edge-based node step.
*/
int main() {
	//char* input_path = "C:\\Users\\AJ\\Documents\\example_graph.txt";
	//char* input_path = "C:\\Users\\AJ\\Desktop\\nvgraphtest\\nvGraphExample-master\\nvGraphExample\\web-Stanford.txt";
	//char* input_path = "C:\\Users\\AJ\\Desktop\\nvgraphtest\\nvGraphExample-master\\nvGraphExample\\web-Stanford_large.txt";
	char* input_path = "C:\\Users\\AJ\\Desktop\\edge_list_example.txt";
	//char* input_path = "C:\\Users\\AJ\\Desktop\\roadnet.txt";
	//char* input_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\facebook_graph.txt";
	//char* input_path = "C:\\Users\\AJ\\Desktop\\output_test\\social\\soc-pokec-relationships.txt";
	//char* input_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\roadNet-PA.txt";
	//char* input_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\soc-pokec-relationships.txt";

	char* output_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\output\\debug_small_graph.txt";

	expand_graph(input_path, output_path, 3);

	//sample_graph(input_path, output_path, 0.5);

	return 0;
}

void sample_graph(char* input_path, char* output_path, float fraction) {
	std::vector<int> source_vertices;
	std::vector<int> destination_vertices;
	COO_List* coo_list = load_graph_from_edge_list_file_to_coo(source_vertices, destination_vertices, input_path);

	// print_coo(source_vertices, target_vertices);

	// Convert the COO graph into a CSR format for the in memory GPU representation
	int* h_offsets = (int*)malloc((SIZE_VERTICES + 1) * sizeof(int));
	int* h_indices = (int*)malloc(SIZE_EDGES * sizeof(int));

	convert_coo_to_csr_format(coo_list->source, coo_list->destination, h_offsets, h_indices);

	//print_csr(h_offsets, h_indices);

	// Edge based Node Sampling Step
	Sampled_Vertices* sampled_vertices = perform_edge_based_node_sampling_step(coo_list->source, coo_list->destination, fraction);
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
	perform_induction_step <<<get_block_size(), get_thread_size() >> >(d_sampled_vertices, d_offsets, d_indices);

	int h_edge_count;
	cudaMemcpyFromSymbol(&h_edge_count, d_edge_count, sizeof(int));
	if (h_edge_count >= SIZE_EDGES + 1) {
		printf("overflow error\n"); return;
	}

	printf("\nAmount of edges collected: %d", h_edge_count);
	std::vector<Edge> results(h_edge_count);
	cudaMemcpyFromSymbol(&(results[0]), edge_data, h_edge_count * sizeof(Edge));

	write_output_to_file(results, output_path);

	cudaFree(d_offsets);
	cudaFree(d_indices);
	cudaFree(d_sampled_vertices);

	// Cleanup
	free(sampled_vertices->vertices);
	free(sampled_vertices);
	free(coo_list);
	free(h_indices);
	free(h_offsets);
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
	return ((SIZE_VERTICES + 1) > MAX_THREADS) ? MAX_THREADS : SIZE_VERTICES;
}

int get_block_size() {
	return ((SIZE_VERTICES + 1) > MAX_THREADS) ? ((SIZE_VERTICES / MAX_THREADS) + 1) : 1;
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
		current_coordinate = add_vertex_as_coordinate(source_vertices, map_from_edge_to_coordinate, source_vertex, current_coordinate);
		current_coordinate = add_vertex_as_coordinate(destination_vertices, map_from_edge_to_coordinate, target_vertex, current_coordinate);
	}

	COO_List* coo_list = (COO_List*)malloc(sizeof(COO_List));

	source_vertices.reserve(source_vertices.size());
	destination_vertices.reserve(destination_vertices.size());
	coo_list->source = &source_vertices[0];
	coo_list->destination = &destination_vertices[0];

	printf("\nTotal amount of vertices: %zd", map_from_edge_to_coordinate.size());
	printf("\nTotal amount of edges: %zd", source_vertices.size());

	// Print edges
	/*for (int i = 0; i < source_vertices.size(); i++) {
		printf("\n(%d, %d)", coo_list->source[i], coo_list->destination[i]);
	}*/

	fclose(file);

	return coo_list;
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
}

Sampled_Vertices* perform_edge_based_node_sampling_step(int* source_vertices, int* target_vertices, float fraction) {
	printf("\nPerforming edge based node sampling step.\n");

	Sampled_Vertices* sampled_vertices = (Sampled_Vertices*) malloc(sizeof(Sampled_Vertices));

	int amount_total_sampled_vertices = calculate_node_sampled_size(fraction);

	std::random_device seeder;
	std::mt19937 engine(seeder());

	sampled_vertices->vertices = (int*) calloc(SIZE_VERTICES, sizeof(int));
	int collected_amount = 0;

	while (collected_amount < amount_total_sampled_vertices) {
		// Pick a random vertex u
		std::uniform_int_distribution<int> range_edges(0, (SIZE_EDGES-1)); // Don't select the last element in the offset
		int random_edge_index = range_edges(engine);

		// Insert u, v (TODO: extract to method per vertex)
		if (!sampled_vertices->vertices[source_vertices[random_edge_index]]) {
			sampled_vertices->vertices[source_vertices[random_edge_index]] = 1;
			print_debug_log("\nCollected vertex:", source_vertices[random_edge_index]);
			collected_amount++;
		}
		if (!sampled_vertices->vertices[target_vertices[random_edge_index]]) {
			sampled_vertices->vertices[target_vertices[random_edge_index]] = 1;
			print_debug_log("\nCollected vertex:", target_vertices[random_edge_index]);
			collected_amount++;
		}
	}

	sampled_vertices->sampled_vertices_size = collected_amount;

	return sampled_vertices;
}


/*
=======================================================================================
Expanding code
=======================================================================================
*/

void expand_graph(char* input_path, char* output_path, float scaling_factor) {
	std::vector<int> source_vertices;
	std::vector<int> destination_vertices;
	COO_List* coo_list = load_graph_from_edge_list_file_to_coo(source_vertices, destination_vertices, input_path);

	// Convert the COO graph into a CSR format for the in memory GPU representation
	int* h_offsets = (int*)malloc((SIZE_VERTICES + 1) * sizeof(int));
	int* h_indices = (int*)malloc(SIZE_EDGES * sizeof(int));

	convert_coo_to_csr_format(coo_list->source, coo_list->destination, h_offsets, h_indices);

	const int amount_of_sampled_graphs = scaling_factor / DEFAULT_EXPANDING_SAMPLE_SIZE;

	printf("Amount of sampled graphs: %d", amount_of_sampled_graphs);

	Sampled_Vertices** sampled_vertices_per_graph = (Sampled_Vertices**) malloc(sizeof(Sampled_Vertices)*amount_of_sampled_graphs);
	
	int** d_size_edges = (int**) malloc(sizeof(int*)*amount_of_sampled_graphs);
	Edge** d_edge_data_expanding = (Edge**) malloc(sizeof(Edge*)*amount_of_sampled_graphs);

	Sampled_Graph_Version* sampled_graph_version_list = new Sampled_Graph_Version[amount_of_sampled_graphs];
	char current_label = 'a';

	for (int i = 0; i < amount_of_sampled_graphs; i++) {
		sampled_vertices_per_graph[i] = perform_edge_based_node_sampling_step(coo_list->source, coo_list->destination, DEFAULT_EXPANDING_SAMPLE_SIZE);
		printf("\nCollected %d vertices.", sampled_vertices_per_graph[i]->sampled_vertices_size);
		printf("\nDone with node sampling step..");
		// Induction step (TODO: re-use device memory from CSR conversion)
		int* d_offsets;
		int* d_indices;
		cudaMalloc((void**)&d_offsets, sizeof(int)*(SIZE_VERTICES + 1));
		cudaMalloc((void**)&d_indices, sizeof(int)*SIZE_EDGES);
		cudaMemcpy(d_indices, h_indices, SIZE_EDGES * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_offsets, h_offsets, sizeof(int)*(SIZE_VERTICES + 1), cudaMemcpyHostToDevice);

		int* d_sampled_vertices;
		cudaMalloc((void**)&d_sampled_vertices, sizeof(int)*SIZE_VERTICES);
		cudaMemcpy(d_sampled_vertices, sampled_vertices_per_graph[i]->vertices, sizeof(int)*(SIZE_VERTICES), cudaMemcpyHostToDevice);

		int* h_size_edges = 0;
		cudaMalloc((void**)&d_size_edges[i], sizeof(int));
		cudaMemcpy(d_size_edges[i], &h_size_edges, sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&d_edge_data_expanding[i], sizeof(Edge)*SIZE_EDGES);
		
		cudaDeviceSynchronize();

		printf("\nRunning kernel (induction step) with block size %d and thread size %d:", get_block_size(), get_thread_size());
		perform_induction_step_expanding <<<get_block_size(), get_thread_size()>>>(d_sampled_vertices, d_offsets, d_indices, d_edge_data_expanding[i], d_size_edges[i]);
		
		// Edge size
		int h_size_edges_result;
		cudaMemcpy(&h_size_edges_result, d_size_edges[i], sizeof(int), cudaMemcpyDeviceToHost);

		// Edges
		printf("\nh_size_edges: %d", h_size_edges_result);
		Sampled_Graph_Version* sampled_graph_version = new Sampled_Graph_Version();
		(*sampled_graph_version).edges.resize(h_size_edges_result);

		cudaMemcpy(&sampled_graph_version->edges[0], d_edge_data_expanding[i], sizeof(Edge)*(h_size_edges_result), cudaMemcpyDeviceToHost);

		// Label
		sampled_graph_version->label = current_label++;

		// Copy data to the sampled version list
		sampled_graph_version_list[i] = (*sampled_graph_version);

		// Cleanup
		delete(sampled_graph_version);
		
		cudaFree(d_sampled_vertices);
		cudaFree(d_offsets);
		cudaFree(d_indices);
		free(sampled_vertices_per_graph[i]->vertices);
		free(sampled_vertices_per_graph[i]);
	}

	free(sampled_vertices_per_graph);
	free(coo_list);
	free(h_indices);
	free(h_offsets);
	
	printf("\nAfter test: %d", sampled_graph_version_list[0].edges.size());

	// For each sampled graph version, copy the data back to the host
	std::vector<Bridge_Edge> bridge_edges;
	link_using_star_topology(sampled_graph_version_list, amount_of_sampled_graphs, bridge_edges);
	
	write_expanded_output_to_file(sampled_graph_version_list, amount_of_sampled_graphs, bridge_edges, output_path);

	// Cleanup
	delete[] sampled_graph_version_list;
	cudaFree(d_edge_data_expanding); // Perhaps these cuda allocations can be freed in the for loop..
	cudaFree(d_size_edges);
}

void link_using_star_topology(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges) {
	/*printf("\nAfter size now 0: %d with label: %c", sampled_graph_version_list[0].edges.size(), sampled_graph_version_list[0].label);
	printf("\nIs there an actual edge here: (%d, %d)", sampled_graph_version_list[0].edges[0].source, sampled_graph_version_list[0].edges[0].destination);
	printf("\nAfter size now 1: %d with label: %c", sampled_graph_version_list[1].edges.size(), sampled_graph_version_list[1].label);
	printf("\nAfter size now 2: %d with label: %c", sampled_graph_version_list[2].edges.size(), sampled_graph_version_list[2].label);
	printf("\nAfter size now 3: %d with label: %c", sampled_graph_version_list[3].edges.size(), sampled_graph_version_list[3].label);*/
	
	// First sampled version will be the graph in the center
	Sampled_Graph_Version center_graph = sampled_graph_version_list[0];
	
	int amount_of_edge_interconnections = 1;
	for (int i = 1; i < amount_of_sampled_graphs; i++) { // Skip the center graph 
		add_edge_interconnection_between_graphs(amount_of_edge_interconnections, &(sampled_graph_version_list[i]), &center_graph, bridge_edges);
	}

	printf("\nCollected a total of %d bridge edges.", bridge_edges.size());
}

/*
-> Probably parallelizable.
-> if(amount_of_edge_interconnections<1) = fraction of the edges/nodes?
*/
void add_edge_interconnection_between_graphs(int amount_of_edge_interconnections, Sampled_Graph_Version* graph_a, Sampled_Graph_Version* graph_b, std::vector<Bridge_Edge>& bridge_edges) {
	printf("\n============================");
	for (int i = 0; i < amount_of_edge_interconnections; i++) {
		int vertex_a = select_random_bridge_vertex(graph_a);
		int vertex_b = select_random_bridge_vertex(graph_b);
		
		// TODO: Extract function
		// Add edge
		Bridge_Edge bridge_edge;
		sprintf(bridge_edge.source, "%c%d", graph_a->label, vertex_a);
		sprintf(bridge_edge.destination, "%c%d", graph_b->label, vertex_b);

		bridge_edges.push_back(bridge_edge);
		printf("\nBridge selection - Selected: (%s, %s)", bridge_edge.source, bridge_edge.destination);
	}
}

// TODO: Add parameter (e.g. Random/high-degree nodes/low-degree nodes)
int select_random_bridge_vertex(Sampled_Graph_Version* graph) {
	// TODO: Move to add_edge_interconnection_between_graphs
	std::random_device seeder;
	std::mt19937 engine(seeder());
	std::uniform_int_distribution<int> range_edges(0, ((*graph).edges.size()) - 1);
	int random_edge_index = range_edges(engine);

	return (*graph).edges[random_edge_index].destination; // Select destination vertex (perhaps make this 50:50?)
}

void write_expanded_output_to_file(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges, char* ouput_path) {
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