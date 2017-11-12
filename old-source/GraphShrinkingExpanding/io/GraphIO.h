#pragma once

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <nvgraph.h>
#include "COOStruct.h"
#include "CSRStruct.h"
#include "../kernel/CudaErrCheck.h"
#include "../sampling/EdgeStruct.h"
#include "../expanding/SampledGraphVersionStruct.h"
#include "../expanding/BridgeEdgeStruct.h"

class GraphIO {
public:
	bool IS_INPUT_FILE_COO = false;
	int SIZE_VERTICES;
	int SIZE_EDGES;
	COO_List* load_graph_from_edge_list_file_to_coo(std::vector<int>& source_vertices, std::vector<int>& destination_vertices, char* file_path);
	int add_vertex_as_coordinate(std::vector<int>& vertices_type, std::unordered_map<int, int>& map_from_edge_to_coordinate, int vertex, int coordinate);
	void save_input_file_as_coo(std::vector<int>& source_vertices, std::vector<int>& destination_vertices, char* save_path);
	CSR_List* convert_coo_to_csr_format(int*, int*);
	void check(nvgraphStatus_t status);
	void write_output_to_file(std::vector<Edge>& results, char* output_path);
	void write_expanded_output_to_file(Sampled_Graph_Version* sampled_graph_version_list, int amount_of_sampled_graphs, std::vector<Bridge_Edge>& bridge_edges, char* ouput_path);
};