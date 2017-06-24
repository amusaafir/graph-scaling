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
public:
	int MAX_THREADS = 1024;
	GraphIO graph_io;
	float SAMPLING_FRACTION;
	void collect_sampling_parameters(char* argv[]);
	void sample_graph(char* input_path, char* output_path);
	Sampled_Vertices* perform_edge_based_node_sampling_step(int*, int*);
	int calculate_node_sampled_size();
	int get_thread_size();
	int get_block_size();
};