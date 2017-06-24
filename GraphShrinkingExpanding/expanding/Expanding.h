#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "../io/COOStruct.h"
#include "../io/CSRStruct.h"
#include "../io/GraphIO.h"
#include "../kernel/kernel.h"
#include "../kernel/CudaErrCheck.h"
#include "../sampling/SampledVerticesStruct.h"
#include "../sampling/Sampling.h"
#include "SampledGraphVersionStruct.h"
#include "BridgeEdgeStruct.h"
#include "topology/Star.h"
#include "topology/Chain.h"
#include "topology/Ring.h"
#include "topology/FullyConnected.h"
#include "bridge/HighDegree.h"
#include "bridge/RandomBridge.h"

typedef enum Bridge_Node_Selection { HIGH_DEGREE_NODES, RANDOM_NODES } Bridge_Node_Selection;

class Expanding {
private:
	GraphIO* _graph_io;
	Sampling* _sampler;
	BridgeSelection* _bridge_selection;
	Topology* _topology;
public:
	Expanding(GraphIO* graph_io);
	~Expanding();
	int MAX_THREADS = 1024;
	bool FORCE_UNDIRECTED_BRIDGES = false;
	float SAMPLING_FRACTION;
	float SCALING_FACTOR;
	int AMOUNT_INTERCONNECTIONS;
	Bridge_Node_Selection SELECTED_BRIDGE_NODE_SELECTION;
	void expand_graph(char* input_path, char* output_path);
	void collect_expanding_parameters(char* argv[]);
	int get_thread_size();
	int get_block_size();
	void set_topology(Topology* topology);
};