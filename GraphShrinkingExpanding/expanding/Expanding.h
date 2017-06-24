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
#include <algorithm>

typedef enum Bridge_Node_Selection { HIGH_DEGREE_NODES, RANDOM_NODES } Bridge_Node_Selection;
typedef enum Topology { STAR, CHAIN, CIRCLE, MESH };

class Expanding {
private:
	GraphIO* _graph_io;
	Sampling* _sampler;
public:
	Expanding(GraphIO* graph_io);
	~Expanding();
	int MAX_THREADS = 1024;
	bool FORCE_UNDIRECTED_BRIDGES = false;
	float SAMPLING_FRACTION;
	float SCALING_FACTOR;
	int AMOUNT_INTERCONNECTIONS;
	Bridge_Node_Selection SELECTED_BRIDGE_NODE_SELECTION;
	Topology SELECTED_TOPOLOGY;
	void expand_graph(char* input_path, char* output_path);
	void collect_expanding_parameters(char* argv[]);
	int get_thread_size();
	int get_block_size();
	void link_using_star_topology(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges);
	void link_using_line_topology(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges);
	void link_using_circle_topology(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges);
	void link_using_mesh_topology(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges);
	void add_edge_interconnection_between_graphs(Sampled_Graph_Version* graph_a, Sampled_Graph_Version* graph_b, std::vector<Bridge_Edge>& bridge_edges);
	int select_random_bridge_vertex(Sampled_Graph_Version* graph);
	int select_high_degree_node_bridge_vertex(Sampled_Graph_Version* graph);
	int get_random_high_degree_node(Sampled_Graph_Version* graph);
	int get_node_bridge_vertex(Sampled_Graph_Version* graph);
};