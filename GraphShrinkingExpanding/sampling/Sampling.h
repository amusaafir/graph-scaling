#pragma once

#include <stdio.h>
#include <math.h>
#include <vector>
#include <random>
#include "../io/GraphIO.h"
#include "SampledVerticesStruct.h"
#include "../kernel/CudaErrCheck.h"
#include "../kernel/kernel.h"

class Sampling {
private:
	GraphIO* _graph_io;
public:
	Sampling(GraphIO* graph_io);
	int MAX_THREADS = 1024;
	float SAMPLING_FRACTION;
	void collect_sampling_parameters(char* argv[]);
	void sample_graph(char* input_path, char* output_path);
	Sampled_Vertices* perform_edge_based_node_sampling_step(int*, int*);
	int calculate_node_sampled_size();
	int get_thread_size();
	int get_block_size();
};