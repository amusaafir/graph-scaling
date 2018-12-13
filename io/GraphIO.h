#pragma once

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <nvgraph.h>
#include <algorithm>
#include <bits/stdc++.h>
#include <mpi.h>
#include "COOStruct.h"
#include "../kernel/CudaErrCheck.h"
#include "../sampling/SampledVerticesStruct.h"

class GraphIO {
public:
	bool IS_INPUT_FILE_COO = false;
	int SIZE_VERTICES;
	int SIZE_EDGES;
	int NODE_SIZE_VERTICES;
	int NODE_VIRTUAL_SIZE_VERTICES;
	int NODE_SIZE_EDGES;
	int NODE_START_VERTEX;
	int NODE_REACHABLE_VERTICES;
	int BIGGEST_N_VERTICES;
	uint64_t GPU_MEM_SIZE;
	uint64_t CPU_MEM_SIZE;

	void collect_sampling_parameters(char* argv[]);
	COO_List* read_and_distribute_graph(std::vector<int>& source_vertices, std::vector<int>& destination_vertices, char* file_path, std::vector<int>& start_vertex_per_node, int mpi_size, int mpi_id);
	void read_graph_size(FILE* file);
	int get_num_edges(int mpi_size, int mpi_id);
	int read_n_edges_rounded(FILE* file, int n_edges, std::vector<int>& source_vertices, std::vector<int>& destination_vertices, bool store, int target_mpi_id);
	int get_n_reachable_vertices(std::vector<int>& source_vertices, std::vector<int>& destination_vertices);
	void assert_fits_mem(int n_edges, int n_vertices, int biggest_n_vertices, int mpi_id);
	int get_biggest_n_vertices(std::vector<int>& start_vertex_per_node, int mpi_size);
	int get_n_vertices(std::vector<int>& start_vertex_per_node, int node_id, int mpi_size);
	int get_n_extra_virtual_vertices(int mpi_id, int mpi_size, int n_vertices_with_only_incoming_edges);
	void write_output_to_file(std::vector<int>& results, COO_List * coo_list, Sampled_Vertices* sampled_vertices, char* output_path, int mpi_size, int mpi_id);
	void write_to_file_sorted(std::vector<int>& source_vertices, std::vector<int>& destination_vertices, char* file_path);
};