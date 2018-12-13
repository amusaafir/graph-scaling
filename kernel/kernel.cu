#include "kernel.h"

__constant__ int D_NODE_START_VERTEX;
__constant__ int D_NODE_SIZE_EDGES;
__constant__ int D_NODE_END_VERTEX;


void perform_induction_step(int block_size, int thread_size, int *d_sampled_vertices, int *d_sources, int* d_destinations, int* d_results, int mpi_id) {
	perform_induction_step <<<block_size, thread_size >>>(d_sampled_vertices, d_sources, d_destinations, d_results, mpi_id);
}

void perform_induction_step2(int block_size, int thread_size, int *d_sampled_vertices, int* d_destinations, int* d_results, int curr_node, int curr_node_start_vertex, int curr_node_end_vertex) {
	perform_induction_step2 <<<block_size, thread_size >>>(d_sampled_vertices, d_destinations, d_results, curr_node, curr_node_start_vertex, curr_node_end_vertex);
}

__global__ void perform_induction_step(int *d_sampled_vertices, int *d_sources, int* d_destinations, int* d_results, int mpi_id) {
	int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
	int source, destination;

	/* Skip out of range indices. */
	if (edge_index < D_NODE_SIZE_EDGES) {
	
		/* Exclude edges of which the source is not in the sample. */
		source = d_sources[edge_index] - D_NODE_START_VERTEX;

		if (!d_sampled_vertices[source]) {
			d_results[edge_index] = EXCLUDED;
		}
		else {
			/* If destination is in own vertices range, we can make a decision
			 * based on the local sample. */
			destination = d_destinations[edge_index];
			
			if (destination >= D_NODE_START_VERTEX
				&& destination <= D_NODE_END_VERTEX) {
				
				destination -= D_NODE_START_VERTEX;
				if (d_sampled_vertices[destination]) {
					/* Both source and destination are in local sample, include
					 * edge. */
					d_results[edge_index] = INCLUDED;
				}
				else {
					/* Destination not sampled. Don't include edge.  */
					d_results[edge_index] = EXCLUDED;
				}
			}
			else {
				/* Destination is in a different node's vertex range.
				 * Put the destination in the results array so it will be
				 * updates in one of the next rounds. */
				d_results[edge_index] = destination;
			}
		}
	}
}

__global__ void perform_induction_step2(int *d_sampled_vertices, int* d_destinations, int* d_results, int curr_node, int curr_node_start_vertex, int curr_node_end_vertex) {
	int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
	int destination;

	if (edge_index < D_NODE_SIZE_EDGES) {

		destination = d_results[edge_index];
		/* Check if result depends on me. Because dst/EXCLUDED/INCLUDED was put
		 * in the result, only 1 read is needed, and no calculation of on which
		 * node the vertex resides. */
		if (destination >= curr_node_start_vertex
			&& destination <= curr_node_end_vertex) {

			/* If destination is the current local sample, we can make a
			 * decision. */
			destination -= curr_node_start_vertex;

			if (d_sampled_vertices[destination]) {
				/* Both source and destination are in local sample, include
				 * edge. */
				d_results[edge_index] = INCLUDED;
			}
			else {
				/* Destination not sampled. Don't include edge. */
				d_results[edge_index] = EXCLUDED;
			}	
		}
	}
}
