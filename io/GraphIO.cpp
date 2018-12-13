#include "GraphIO.h"


void GraphIO::collect_sampling_parameters(char* argv[]) {
	CPU_MEM_SIZE = strtoull(argv[4], NULL, 10);
	GPU_MEM_SIZE = strtoull(argv[5], NULL, 10);
	// printf("CPU memory size: % " PRIu64 " bytes\nGPU memory size: % " PRIu64 " bytes.\n",
		// CPU_MEM_SIZE, GPU_MEM_SIZE);
}

COO_List* GraphIO::read_and_distribute_graph(std::vector<int>& source_vertices, std::vector<int>& destination_vertices, char* file_path, std::vector<int>& start_vertex_per_node, int mpi_size, int mpi_id) {
	int n_edges, n_vertices, n_vertices_with_only_incoming_edges;

	if (mpi_id == 0) {
		long file_position;
		int last_vertex;
		int current_first_vertex = 0, diff_n_edges = 0;
	
		printf("Loading graph file from: %s\n", file_path);
		FILE* file = fopen(file_path, "r");

		/* Read graph size. */
		read_graph_size(file);

		/* Save position to come back here later. */
		file_position = ftell(file);

		/* Read and distribute edges of all nodes. */
		for (int i = 0; i < mpi_size; i++) {

			/* Determine number of vertices and read edges. */
			n_edges = get_num_edges(mpi_size, i);

			/* Subtract what was already given to previous node due to
			 * 'rounding'. */
			n_edges -= diff_n_edges;

			assert_fits_mem(n_edges, 0, 0, mpi_id);

			// printf("Reading %d edges for node %d\n", n_edges, i);
			last_vertex = read_n_edges_rounded(file, n_edges, source_vertices, destination_vertices, true, i);
			n_vertices = last_vertex - current_first_vertex + 1;

			/* Store how many edges were read more than intended (for
			 * 'rounding' to source vertices). */
			diff_n_edges = source_vertices.size() - n_edges;

			/* Give all vertices without outgoing edges to the last node (for
			 * sampling). */
			if (i == mpi_size - 1) {
				n_vertices_with_only_incoming_edges = SIZE_VERTICES - (current_first_vertex + n_vertices);
				// printf("Stretched node %d's number of vertices from %d to %d, only incoming %d\n", i, n_vertices, SIZE_VERTICES - current_first_vertex, n_vertices_with_only_incoming_edges);
				n_vertices = SIZE_VERTICES - current_first_vertex;
			}

			/* Make sure n_edges/n_vertices fit in the node's memory. */
			assert_fits_mem(n_edges, n_vertices, 0, mpi_id);

			/* Send edges to node. */
			if (i != mpi_id) {
				MPI_Send(&source_vertices[0], source_vertices.size(), MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&destination_vertices[0], destination_vertices.size(), MPI_INT, i, 0, MPI_COMM_WORLD);
			}

			/* Empty vectors. */
			source_vertices.resize(0);
			destination_vertices.resize(0);

			/* Update current first index after it has been sent to the current
			 * node. */
			start_vertex_per_node.push_back(current_first_vertex);
			current_first_vertex = last_vertex + 1;
		}

		/* Rewind file and read (and store) edges for node 0. */
		fseek(file, file_position, SEEK_SET);

		current_first_vertex = 0;
		n_edges = get_num_edges(mpi_size, mpi_id);
		last_vertex = read_n_edges_rounded(file, n_edges, source_vertices, destination_vertices, true, mpi_id);
		n_vertices = last_vertex - current_first_vertex + 1;
		n_edges = source_vertices.size();

		/* If 0 is the only node, give all nodes without outgoing edges to it
		 * as well (it is the 'last' node). */
		if (mpi_size == 1) {
			n_vertices_with_only_incoming_edges = SIZE_VERTICES - (current_first_vertex + n_vertices);
			// printf("Stretched node %d's number of vertices from %d to %d, only incoming %d\n", mpi_id, n_vertices, SIZE_VERTICES - current_first_vertex, n_vertices_with_only_incoming_edges);
			n_vertices = SIZE_VERTICES - current_first_vertex;
		}

		fclose(file);

		/* Send size of graph to all nodes. */
		std::vector<int> graph_size;
		graph_size.push_back(SIZE_EDGES);
		graph_size.push_back(SIZE_VERTICES);
		graph_size.push_back(n_vertices_with_only_incoming_edges);

		/* Broadcast number of vertices and edges. */
		MPI_Bcast(graph_size.data(), graph_size.size(), MPI_INT, 0, MPI_COMM_WORLD);

		/* Broadcast start vertices to indicate how vertices are distributed. */
		MPI_Bcast(start_vertex_per_node.data(), mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
	}
	else {
		/* Get number of edges to receive. */
		MPI_Status status;
		MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_INT, &n_edges);

		/* Allocate space. */
		source_vertices.resize(n_edges);
		destination_vertices.resize(n_edges);

		/* Receive source and destination vertices. */
		MPI_Recv(source_vertices.data(), n_edges, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(destination_vertices.data(), n_edges, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* Receive graph size data. */
		std::vector<int> graph_size;
		graph_size.resize(3);

		MPI_Bcast(graph_size.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
		
		SIZE_EDGES = graph_size[0];
		SIZE_VERTICES = graph_size[1];
		n_vertices_with_only_incoming_edges = graph_size[2];

		start_vertex_per_node.resize(mpi_size);

		/* Receive start vertices to indicate how vertices are distributed. */
		MPI_Bcast(start_vertex_per_node.data(), mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
	}

	/* Set node specific data. */
	NODE_SIZE_EDGES = n_edges;
	NODE_SIZE_VERTICES = get_n_vertices(start_vertex_per_node, mpi_id, mpi_size);
	NODE_START_VERTEX = start_vertex_per_node[mpi_id];
	NODE_REACHABLE_VERTICES = get_n_reachable_vertices(source_vertices, destination_vertices);
	BIGGEST_N_VERTICES = get_biggest_n_vertices(start_vertex_per_node, mpi_size);

	/* Make sure n_edges/n_vertices fit in the node's memory. */
	assert_fits_mem(NODE_SIZE_EDGES, NODE_SIZE_VERTICES, BIGGEST_N_VERTICES, mpi_id);

	/* There is an (almost) equal amount of edges in every node, and likely a
	 * similar amount of vertices if not counting the vertices with only
	 * incoming edges. Virtually 'distribute' incoming-only vertices over the
	 * nodes. This, together with the number of vertices that a node is
	 * responsible for that do (possibly) have outgoing edges, makes up the
	 * virtual number of vertices of the node. This number is used to calculate
	 * the number of vertices with that need to be sampled. This works
	 * because every (non-last) node can sample those incoming-only
	 * vertices as a remote vertex (on the last node).
	 */
	if (mpi_id == mpi_size - 1) {
		int n_outgoing_vertices = (SIZE_VERTICES -
			n_vertices_with_only_incoming_edges) - NODE_START_VERTEX;
		
		NODE_VIRTUAL_SIZE_VERTICES = n_outgoing_vertices +
			get_n_extra_virtual_vertices(mpi_id, mpi_size, n_vertices_with_only_incoming_edges);
	}
	else {
		NODE_VIRTUAL_SIZE_VERTICES = NODE_SIZE_VERTICES +
			get_n_extra_virtual_vertices(mpi_id, mpi_size, n_vertices_with_only_incoming_edges);
	}

	printf("Node %d: %d edges, %d vertices, first vertex: %d, biggest n vertices: %d, virtual n vertices: %d, reachable n vertices: %d\n",
		mpi_id, NODE_SIZE_EDGES, NODE_SIZE_VERTICES, NODE_START_VERTEX, BIGGEST_N_VERTICES, NODE_VIRTUAL_SIZE_VERTICES, NODE_REACHABLE_VERTICES);
	
	/* Set coo_list source and destination. */
	COO_List* coo_list = (COO_List*)malloc(sizeof(COO_List));
	coo_list->source = &source_vertices[0];
	coo_list->destination = &destination_vertices[0];

	return coo_list;
}

/* Reads the size of the graph from file (number of vertices and edges). */
void GraphIO::read_graph_size(FILE* file) {
	int num_vertices, num_edges;
	char line[256];

	while (fgets(line, sizeof(line), file)) {
		if (line[0] == '#' || line[0] == '\n') {
			//print_debug_log("\nEscaped a comment.");
			continue;
		}
		if (line[0] == '!') {
			sscanf(line, "!%d%d\t", &num_vertices, &num_edges);
			break;
		}
		else {
			printf("File does not start with graph size.");
			exit(1);
		}
	}

	SIZE_VERTICES = num_vertices;
	SIZE_EDGES = num_edges;
	printf("Total amount of vertices: %d\n", SIZE_VERTICES);
	printf("Total amount of edges: %d\n", SIZE_EDGES);
}

/* Calculate the number of edges that the node with the given mpi_id has to
 * process if the edges are distributed evenly.
 * Note: only works if SIZE_EDGES is set to the total number of
 * vertices in the graph (that is, after the call to distribute_grap for slave
 * nodes. */
int GraphIO::get_num_edges(int mpi_size, int mpi_id) {
	int n_edges = SIZE_EDGES / mpi_size;
	int remainder = SIZE_EDGES % mpi_size;
	
	if (mpi_id < remainder)
		n_edges++;

	return n_edges;
}

/* Starts reading edges from file at the current position of the file pointer,
 * and reads all edges that have a source vertex <= last_vertex. The file
 * pointer is left right after the last read vertex.
 */
int GraphIO::read_n_edges_rounded(FILE* file, int n_edges, std::vector<int>& source_vertices, std::vector<int>& destination_vertices, bool store, int target_mpi_id) {
	int temp_end_vertex, source_vertex, target_vertex;
	char line[256];
	long file_position = ftell(file);
	int n_edges_done = 0;


	/* Read n edges */
	while (n_edges_done < n_edges) {
		if (!fgets(line, sizeof(line), file))
			break;

		if (line[0] == '#' || line[0] == '\n') {
			//print_debug_log("\nEscaped a comment.");
			continue;
		}

		sscanf(line, "%d%d\t", &source_vertex, &target_vertex);

		if (store) {
			source_vertices.push_back(source_vertex);
			destination_vertices.push_back(target_vertex);
		}

		n_edges_done++;
	}

	temp_end_vertex = source_vertex;

	/* Save file position. */
	file_position = ftell(file);

	/* Round to next vertex. */
	while (fgets(line, sizeof(line), file)) {
		if (line[0] == '#' || line[0] == '\n') {
			//print_debug_log("\nEscaped a comment.");
			continue;
		}

		sscanf(line, "%d%d\t", &source_vertex, &target_vertex);

		/* Stop when we hit a new source vertex. Rewind file pointer to the
		 * previous line. */
		if (source_vertex != temp_end_vertex) {
			fseek(file, file_position, SEEK_SET);
			break;
		}

		/* Make sure the edges will still fit in memory. */
		n_edges_done++;
		assert_fits_mem(n_edges_done, 0, 0, target_mpi_id);

		if (store) {
			source_vertices.push_back(source_vertex);
			destination_vertices.push_back(target_vertex);
		}

		file_position = ftell(file);
	}

	return temp_end_vertex;
}

int GraphIO::get_n_reachable_vertices(std::vector<int>& source_vertices,
	std::vector<int>& destination_vertices) {

	// std::unordered_set<int> reachable_vertices;
	// int first_vertex = source_vertices[0];
	// int last_vertex = source_vertices[source_vertices.size() - 1];
	// int n_reachable_vertices = 0;
	// int prev_vertex = -1;
	int vertex;
	// std::vector<int>::iterator result;

	/* Count number of distinct source vertices. source_vertices is sorted. */
	// for (unsigned int i = 0; i < source_vertices.size(); i++) {
	// 	vertex = source_vertices[i];
	// 	if (vertex != prev_vertex) {
	// 		n_reachable_vertices++;
	// 		prev_vertex = vertex;
	// 	}
	// }

	// for (unsigned int i = 0; i < destination_vertices.size(); i++) {
	// 	vertex = destination_vertices[i];

	// 	if (vertex < first_vertex || vertex > last_vertex) {
	// 		reachable_vertices.insert(vertex);
	// 	}
	// 	else {
	// 		 Check if this vertex is in source vertices. Add as reachable if not. 
	// 		result = lower_bound(source_vertices.begin(), source_vertices.end(), vertex);

	// 		if (*result != vertex) {
	// 			reachable_vertices.insert(vertex);
	// 		}
	// 	}
	// }

	std::unordered_set<int> sources2(source_vertices.begin(), source_vertices.end());

	for (unsigned int i = 0; i < destination_vertices.size(); i++) {
		vertex = destination_vertices[i];
		sources2.insert(vertex);
	}

	return sources2.size();
	// return n_reachable_vertices + reachable_vertices.size();
}

void GraphIO::assert_fits_mem(int n_edges, int n_vertices,
	int biggest_n_vertices, int mpi_id) {
	
	uint64_t mem_needed = 3 * n_edges + 2 * n_vertices + 2 * biggest_n_vertices;
	mem_needed *= 4;

	/* Make sure n_edges/n_vertices fit in the node's memory. */

	/* Maximum CPU memory usage:
	 * edges * 4 (source vertices + destination
	 * vertices + results/counting reachable (destination) vertices /remote
	 * vertices to include in sample)
	 * + biggest_n_vertices (communication during induction)
	 * + biggest_n_vertices (for conversion to vector when communicating
	 * remote updates).
	 * + n_vertices (for local sample). 
	 * + n_vertices (for counting reachable (source) vertices). */
	if (mem_needed > CPU_MEM_SIZE) {
		printf("Node %d's edges/vertices may not fit CPU memory.\n", mpi_id);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
	
	/* Maximum GPU memory usage:
 	 * edges * 3 (source vertices + destination vertices + results)
	 * + biggest_n_vertices (sample present in GPU memory). */
	mem_needed = 3 * n_edges + biggest_n_vertices;
	mem_needed *= 4;

	if (mem_needed > GPU_MEM_SIZE) {
		printf("Node %d's edges/vertices may not fit GPU memory.\n", mpi_id);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
}

int GraphIO::get_biggest_n_vertices(std::vector<int>& start_vertex_per_node, int mpi_size) {
	int biggest_n_vertices = 0;
	int n_vertices;

	for (int i = 0; i < mpi_size; i++) {
		n_vertices = get_n_vertices(start_vertex_per_node, i, mpi_size);

		if (n_vertices > biggest_n_vertices)
			biggest_n_vertices = n_vertices;
	}

	return biggest_n_vertices;
}

int GraphIO::get_n_vertices(std::vector<int>& start_vertex_per_node, int node_id, int mpi_size) {
	if (node_id + 1 == mpi_size) {
		return SIZE_VERTICES - start_vertex_per_node[node_id];
	}
	else {
		return start_vertex_per_node[node_id + 1] - start_vertex_per_node[node_id];
	}
}

int GraphIO::get_n_extra_virtual_vertices(int mpi_id, int mpi_size, int n_vertices_with_only_incoming_edges) {
	int virtual_n_vertices = n_vertices_with_only_incoming_edges / mpi_size;
	int remainder = n_vertices_with_only_incoming_edges % mpi_size;

	if (mpi_id < remainder)
		virtual_n_vertices += 1;

	return virtual_n_vertices;
}

void GraphIO::write_output_to_file(std::vector<int>& results, COO_List * coo_list, Sampled_Vertices* sampled_vertices, char* output_path, int mpi_size, int mpi_id) {
	FILE *output_file;

	int n_edges = 0;
	int total_n_vertices, total_n_edges;

	if (mpi_id == 0)
		printf("Writing output to file.\n");

	for (int i = 0; i < mpi_size; i++) {
		MPI_Barrier(MPI_COMM_WORLD);
		
		if (mpi_id == i) {
			char* file_path = output_path;

			if (mpi_id == 0)
				output_file = fopen(file_path, "w");
			else
				output_file = fopen(file_path, "a");

			if (output_file == NULL) {
				printf("\nError writing results to output file.");
				exit(1);
			}
			for (unsigned int i = 0; i < results.size(); i++) {
				if (results[i] == -2) {
					fprintf(output_file, "%d\t%d\n", coo_list->source[i], coo_list->destination[i]);
					n_edges++;
				}
			}
			fclose(output_file);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	/* Collect graph size data at node 0. */
	MPI_Reduce(&sampled_vertices->sampled_vertices_size, &total_n_vertices, 1,
		MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Reduce(&n_edges, &total_n_edges, 1, MPI_INT, MPI_SUM, 0,
		MPI_COMM_WORLD);

	// /* Debug code: */
	// for (int i = 0; i < mpi_size; i++) {
	// 	MPI_Barrier(MPI_COMM_WORLD);

	// 	if (mpi_id == i) {
	// 		if (mpi_id == 0)
	// 			output_file = fopen("datasets/output/sample.txt", "w");
	// 		else
	// 			output_file = fopen("datasets/output/sample.txt", "a");

	// 		if (output_file == NULL) {
	// 			printf("\nError writing results to output file.");
	// 			exit(1);
	// 		}
	// 		for (int i = 0; i < NODE_SIZE_VERTICES; i++) {
	// 			if (sampled_vertices->vertices[i])
	// 				fprintf(output_file, "%d\n", i + NODE_START_VERTEX);
	// 		}
	// 		fclose(output_file);
	// 	}

	// 	MPI_Barrier(MPI_COMM_WORLD);
	// }

	if (mpi_id == 0) {
		printf("Done writing output to file.\n");
		printf("Output graph size: %d vertices, %d edges\n", total_n_vertices, total_n_edges);
	}
}

/* Debug function. */
void GraphIO::write_to_file_sorted(std::vector<int>& source_vertices, std::vector<int>& destination_vertices, char* file_path) {
	std::vector< std::pair <int,int> > pairs;

   /* Enter src, dst pairs as in vector of pairs. */
    for (unsigned int i = 0; i < source_vertices.size(); i++)
        pairs.push_back(std::make_pair(source_vertices[i], destination_vertices[i]));

    /* Sort. */
    std::sort(pairs.begin(), pairs.end());

	printf("Writing sorted graph to file:\n");
	FILE *output_file = fopen(file_path, "w");

	if (output_file == NULL) {
		printf("\nError writing results to output file.");
		exit(1);
	}

    /* Write sorted pairs to file. */
	for (unsigned int i = 0; i < source_vertices.size(); i++) {
		fprintf(output_file, "%d\t%d\n", pairs[i].first, pairs[i].second);
	}

	fclose(output_file);
	
	printf("Done writing sorted graph to file:\n");
}
