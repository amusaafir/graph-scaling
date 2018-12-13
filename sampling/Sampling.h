#pragma once

#include <stdio.h>
#include <math.h>
#include <vector>
#include <random>
#include <set>
#include <time.h>
#include "../io/GraphIO.h"
#include "SampledVerticesStruct.h"
#include "../kernel/CudaErrCheck.h"
#include "../kernel/kernel.h"

#define CONDITION_EVEN 0
#define CONDITION_ODD 1
#define CONDITION_BIGGER 2
#define CONDITION_SMALLER 3

class Sampling {
private:
	GraphIO* _graph_io;
public:
	Sampling(GraphIO* graph_io);
	int MAX_THREADS = 1024;
	float SAMPLING_FRACTION;
	void collect_sampling_parameters(char* argv[]);
	void sample_graph(char* input_path, char* output_path, int mpi_size, int mpi_id);
	int get_vertex_location(std::vector<int>& start_vertex_per_node, int vertex);
	Sampled_Vertices* perform_node_sampling_step(int* source_vertices, int* target_vertices, std::vector<int>& start_vertex_per_node, int mpi_size, int mpi_id);
	int update_remote_samples(std::vector< std::set<int> >& remote_sampled_vertices, Sampled_Vertices* sampled_vertices, int mpi_size, int mpi_id);
	int update_local_sample(std::vector< int >& vertices, Sampled_Vertices* sampled_vertices);
	Sampled_Vertices* perform_controlled_node_sampling_step(int condition, int threshold, int mpi_id);
	bool even(int vertex_id);
	int calculate_node_sampled_size();
	void perform_distributed_induction_step(COO_List* coo_list, Sampled_Vertices* sampled_vertices, std::vector<int>& source_vertices, std::vector<int>& destination_vertices, std::vector<int>& start_vertex_per_node, std::vector<int>& results, int mpi_size, int mpi_id);
	int get_thread_size();
	int get_block_size();
};