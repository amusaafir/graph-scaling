#include "Sampling.h"

Sampling::Sampling(GraphIO* graph_io) {
	_graph_io = graph_io;
}

void Sampling::collect_sampling_parameters(char* argv[]) {
	float fraction = atof(argv[3]);
	SAMPLING_FRACTION = fraction;
	printf("Sample fraction: %f\n", fraction);
}

void Sampling::sample_graph(char* input_path, char* output_path, int mpi_size, int mpi_id) {
	COO_List* coo_list;
	std::vector<int> start_vertex_per_node;
	std::vector<int> source_vertices;
	std::vector<int> destination_vertices;
	std::vector<int> results;

	coo_list = _graph_io->read_and_distribute_graph(source_vertices, destination_vertices, input_path, start_vertex_per_node, mpi_size, mpi_id);
	// _graph_io->write_to_file_sorted(source_vertices, destination_vertices, (char*)"/var/scratch/dreuning/sorted.txt");


	/* Edge based Node Sampling Step */
	Sampled_Vertices* sampled_vertices = perform_node_sampling_step(coo_list->source, coo_list->destination, start_vertex_per_node, mpi_size, mpi_id);
	// Sampled_Vertices* sampled_vertices = perform_controlled_node_sampling_step(CONDITION_EVEN, 0, mpi_id);

	// printf("Process %d collected %d vertices.\n", mpi_id, sampled_vertices->sampled_vertices_size);

	// if (mpi_id == 0)
		// printf("Running induction step.\n");

	perform_distributed_induction_step(coo_list, sampled_vertices, source_vertices, destination_vertices, start_vertex_per_node, results, mpi_size, mpi_id);

	_graph_io->write_output_to_file(results, coo_list, sampled_vertices,
		output_path, mpi_size, mpi_id);

	/* Cleanup */
	free(sampled_vertices->vertices);
	free(sampled_vertices);
	free(coo_list);
}

int Sampling::get_vertex_location(std::vector<int>& start_vertex_per_node,
	int vertex) {
	unsigned int node;

	for (node = 0; node < start_vertex_per_node.size(); node++) {
		if (vertex < start_vertex_per_node[node])
			break;
	}

	return node - 1;
}

Sampled_Vertices* Sampling::perform_node_sampling_step(int* source_vertices,
	int* target_vertices, std::vector<int>& start_vertex_per_node,
	int mpi_size, int mpi_id) {
	
	// if (mpi_id == 0)
		// printf("Performing edge based node sampling step.\n");

	Sampled_Vertices* sampled_vertices = (Sampled_Vertices*)malloc(sizeof(Sampled_Vertices));
	int amount_total_sampled_vertices = calculate_node_sampled_size();

	/* Make sure the amount of vertices to sample is not bigger than the amount
	 * of reachable vertices of this node. */
	if (amount_total_sampled_vertices > _graph_io->NODE_REACHABLE_VERTICES) {
		printf("Amount of vertices to sample (%d) is bigger than the number of reachable vertices (%d) on node %d\n",
			amount_total_sampled_vertices, _graph_io->NODE_REACHABLE_VERTICES,
			mpi_id);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	std::vector< std::set <int> > remote_sampled_vertices(mpi_size);
	std::set <int>::iterator it;

	std::random_device seeder;
	std::mt19937 engine(seeder());
	sampled_vertices->vertices = (int*)calloc(_graph_io->NODE_SIZE_VERTICES, sizeof(int));

	int local_collected_amount = 0, remote_collected_amount = 0,
		random_edge_index = 0, destination_vertex = 0, node = 0,
		source_vertex = 0;

	std::uniform_int_distribution<int> range_edges(0, (_graph_io->NODE_SIZE_EDGES - 1));

	while (local_collected_amount + remote_collected_amount < amount_total_sampled_vertices) {
		/* Pick a random vertex u. */
		random_edge_index = range_edges(engine);

		/* Set randomly picked vertex as 'in sample' if it was not already. */

		source_vertex = source_vertices[random_edge_index] - _graph_io->NODE_START_VERTEX;

		if (!sampled_vertices->vertices[source_vertex]) {
			sampled_vertices->vertices[source_vertex] = 1;
			local_collected_amount++;
		}

		destination_vertex = target_vertices[random_edge_index];

		if (destination_vertex >= _graph_io->NODE_START_VERTEX
			&& destination_vertex < _graph_io->NODE_START_VERTEX + _graph_io->NODE_SIZE_VERTICES) {

			/* Destination vertex is also in our sample/range of vertices,
			 * so sample it. */
			destination_vertex -= _graph_io->NODE_START_VERTEX;
			
			if (!sampled_vertices->vertices[destination_vertex]) {
				sampled_vertices->vertices[destination_vertex] = 1;
				local_collected_amount++;
			}
		}
		else {
			/* Put vertex in a list of vertices to be sent to a remote node.
			 * Assume it is successfully added to the sample there (and it was
			 * not already sampled by another node).
			 */
			node = get_vertex_location(start_vertex_per_node, destination_vertex);
			if (remote_sampled_vertices[node].find(destination_vertex) == remote_sampled_vertices[node].end()) {
				remote_sampled_vertices[node].insert(destination_vertex);
				remote_collected_amount++;
			}
			else {
				/* The vertex is already in there. */
			}
		}
	}

	sampled_vertices->sampled_vertices_size = local_collected_amount;

	sampled_vertices->sampled_vertices_size += update_remote_samples(
		remote_sampled_vertices, sampled_vertices, mpi_size, mpi_id);

	// printf("Node %d is done with node sampling step.. %d %d\n", mpi_id,
	// 	remote_collected_amount, local_collected_amount);

	return sampled_vertices;
}

int Sampling::update_remote_samples(
	std::vector< std::set<int> >& remote_sampled_vertices,
	Sampled_Vertices* sampled_vertices, int mpi_size, int mpi_id) {
	
	// printf("Updating remote samples.\n");
	
	/* Send and receive vertices to be sampled, update if not already in sample. */

	MPI_Status status;
	int n_vertices;
	int extra_included = 0;
	int total2 = 0;

	for (int i = 0; i < mpi_size; i++) {
		if (i == mpi_id) {
			for (int j = 0; j < mpi_size; j++) {
				if (j != mpi_id) {
					/* Convert to vector. */
					std::vector <int> vertices(remote_sampled_vertices[j].begin(),
						remote_sampled_vertices[j].end());

					MPI_Send(vertices.data(), vertices.size(), MPI_INT, j, 0,
						MPI_COMM_WORLD);
				}
			}
		}
		else {

			MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
			MPI_Get_count(&status, MPI_INT, &n_vertices);

			/* Allocate space. */
			std::vector <int> vertices;
			vertices.resize(n_vertices);

			/* Receive vertices to be updated. */
			MPI_Recv(vertices.data(), n_vertices, MPI_INT, i, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			total2 += n_vertices;
			extra_included += update_local_sample(vertices, sampled_vertices);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	// printf("Node %d locally updated %d extra vertices (out of %d).\n",
	// 	mpi_id, extra_included, total2);

	return extra_included;
}

/* Updates local sample acording to remotely picked edges. Returns the number
 * of vertices added to the sample (and were thus not already included before.
 */
int Sampling::update_local_sample(std::vector< int >& vertices,
	Sampled_Vertices* sampled_vertices) {
	
	int source_vertex;
	int locally_updated_amount = 0;

	for (unsigned int i = 0; i < vertices.size(); i++) {
		source_vertex = vertices[i] - _graph_io->NODE_START_VERTEX;

		if (!sampled_vertices->vertices[source_vertex]) {
			sampled_vertices->vertices[source_vertex] = 1;
			locally_updated_amount++;
		}
	}
	return locally_updated_amount;
}

Sampled_Vertices* Sampling::perform_controlled_node_sampling_step(int condition, int threshold, int mpi_id) {

	// if (mpi_id == 0)
		// printf("Performing edge based node sampling step in controlled environment.\n");

	int local_collected_amount = 0;
	Sampled_Vertices* sampled_vertices = (Sampled_Vertices*)malloc(sizeof(Sampled_Vertices));
	sampled_vertices->vertices = (int*)calloc(_graph_io->NODE_SIZE_VERTICES, sizeof(int));

	if (condition == CONDITION_EVEN) {
		for (int i = 0; i < _graph_io->NODE_SIZE_VERTICES; i++) {
			if (even(i + _graph_io->NODE_START_VERTEX)) {
				sampled_vertices->vertices[i] = 1;
				local_collected_amount++;
			}
		}
	}
	else if (condition == CONDITION_ODD) {
		for (int i = 0; i < _graph_io->NODE_SIZE_VERTICES; i++) {
			if (!even(i + _graph_io->NODE_START_VERTEX)) {
				sampled_vertices->vertices[i] = 1;
				local_collected_amount++;
			}
		}
	}
	else if (condition == CONDITION_BIGGER) {
		for (int i = 0; i < _graph_io->NODE_SIZE_VERTICES; i++) {
			if (i + _graph_io->NODE_START_VERTEX > threshold) {
				sampled_vertices->vertices[i] = 1;
				local_collected_amount++;
			}
		}
	}
	else if (condition == CONDITION_SMALLER) {
		for (int i = 0; i < _graph_io->NODE_SIZE_VERTICES; i++) {
			if (i + _graph_io->NODE_START_VERTEX < threshold) {
				sampled_vertices->vertices[i] = 1;
				local_collected_amount++;
			}
		}
	}

	sampled_vertices->sampled_vertices_size = local_collected_amount;

	printf("Node %d is done with node sampling step.. %d\n", mpi_id, local_collected_amount);

	return sampled_vertices;
}

bool Sampling::even(int vertex_id) {
	if (vertex_id % 2 == 0)
		return true;
	else
		return false;
}

int Sampling::calculate_node_sampled_size() {
	return int(_graph_io->NODE_VIRTUAL_SIZE_VERTICES * SAMPLING_FRACTION);
}

void Sampling::perform_distributed_induction_step(COO_List* coo_list,
	Sampled_Vertices* sampled_vertices,
	std::vector<int>& source_vertices,
	std::vector<int>& destination_vertices,
	std::vector<int>& start_vertex_per_node,
	std::vector<int>& results,
	int mpi_size, int mpi_id) {
	
	int recv_size, start_vertex, end_vertex;
	std::vector<int> new_sample;

	/* Allocate and copy to GPU memory. */
	int *d_sources;
	int *d_destinations;
	int* d_sampled_vertices;
	int *d_results;

	gpuErrchk(cudaMalloc((void**)&d_sources, sizeof(int) * (_graph_io->NODE_SIZE_EDGES)));
	gpuErrchk(cudaMalloc((void**)&d_destinations, sizeof(int) * _graph_io->NODE_SIZE_EDGES));
	gpuErrchk(cudaMemcpy(d_sources, coo_list->source, sizeof(int) * _graph_io->NODE_SIZE_EDGES, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_destinations, coo_list->destination, sizeof(int) * _graph_io->NODE_SIZE_EDGES, cudaMemcpyHostToDevice));

	/* Make sure this space is big enough for samples of other nodes as well. */
	gpuErrchk(cudaMalloc((void**)&d_sampled_vertices, sizeof(int) * _graph_io->BIGGEST_N_VERTICES));
	gpuErrchk(cudaMemcpy(d_sampled_vertices, sampled_vertices->vertices, sizeof(int) * _graph_io->NODE_SIZE_VERTICES, cudaMemcpyHostToDevice));

	int node_end_vertex = _graph_io->NODE_START_VERTEX + _graph_io->NODE_SIZE_VERTICES - 1;
	gpuErrchk(cudaMemcpyToSymbol(&D_NODE_START_VERTEX, &(_graph_io->NODE_START_VERTEX), sizeof(int), 0, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(&D_NODE_SIZE_EDGES, &(_graph_io->NODE_SIZE_EDGES), sizeof(int), 0, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(&D_NODE_END_VERTEX, &node_end_vertex, sizeof(int), 0, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc((void**)&d_results, sizeof(int) * (_graph_io->NODE_SIZE_EDGES)));

	perform_induction_step(get_block_size(), get_thread_size(),
		d_sampled_vertices, d_sources, d_destinations, d_results, mpi_id);

	for (int i = 0; i < mpi_size; i++) {
		if (i == mpi_id) {

			/* Send bookkeeping info to all nodes. */
			std::vector<int> info;
			info.push_back(_graph_io->NODE_SIZE_VERTICES);
			info.push_back(_graph_io->NODE_START_VERTEX);

			/* Broadcast number of vertices and edges. */
			MPI_Bcast(info.data(), 2, MPI_INT, i, MPI_COMM_WORLD);

			/* And the vertices/sample itself. */
			MPI_Bcast(sampled_vertices->vertices, _graph_io->NODE_SIZE_VERTICES,
				MPI_INT, i, MPI_COMM_WORLD);
		}
		else {

			/* Receive bookkeeping data. */
			std::vector<int> info;
			info.resize(2);

			MPI_Bcast(info.data(), 2, MPI_INT, i, MPI_COMM_WORLD);

			recv_size = info[0];
			start_vertex = info[1];
			end_vertex = start_vertex + recv_size - 1;
			new_sample.resize(recv_size);

			/* Receive data from other node. */
			MPI_Bcast(new_sample.data(), recv_size, MPI_INT, i, MPI_COMM_WORLD);

			/* Make sure previous kernel call finishes before copying new
			 * sample data. */
			cudaDeviceSynchronize();

			/* Copy new sample */
			gpuErrchk(cudaMemcpy(d_sampled_vertices, new_sample.data(),
				sizeof(int) * recv_size, cudaMemcpyHostToDevice));

			/* Launch kernel. */
			perform_induction_step2(get_block_size(), get_thread_size(),
				d_sampled_vertices, d_destinations, d_results, i, start_vertex,
				end_vertex);
		}
	}

	/* Copy result to CPU memory. */
	results.resize(_graph_io->NODE_SIZE_EDGES);
	gpuErrchk(cudaMemcpy(&(results[0]), d_results, sizeof(int) * (_graph_io->NODE_SIZE_EDGES), cudaMemcpyDeviceToHost));

	/* Cleanup */
	cudaFree(d_sources);
	cudaFree(d_destinations);
	cudaFree(d_sampled_vertices);
	cudaFree(d_results);
}

int Sampling::get_thread_size() {
	return (_graph_io->NODE_SIZE_EDGES > MAX_THREADS) ? MAX_THREADS : _graph_io->NODE_SIZE_EDGES;
}
int Sampling::get_block_size() {
	return (_graph_io->NODE_SIZE_EDGES > MAX_THREADS) ? ((_graph_io->NODE_SIZE_EDGES / MAX_THREADS) + 1) : 1;
}
