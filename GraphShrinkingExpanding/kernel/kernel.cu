#include "kernel.h"

__device__ int d_edge_count = 0;
__constant__ int D_SIZE_EDGES;
__constant__ int D_SIZE_VERTICES;

void perform_induction_step(int block_size, int thread_size, int *d_sampled_vertices, int *d_offsets, int* d_indices, Edge* d_edge_data) {
	perform_induction_step <<<block_size, thread_size >>>(d_sampled_vertices, d_offsets, d_indices, d_edge_data);
}

void perform_induction_step_expanding(int block_size, int thread_size, int* d_sampled_vertices, int* d_offsets, int* d_indices, Edge* d_edge_data_expanding, int* d_edge_count_expanding) {
	perform_induction_step_expanding<<<block_size, thread_size >>>(d_sampled_vertices, d_offsets, d_indices, d_edge_data_expanding, d_edge_count_expanding);
}

__device__ int push_edge(Edge &edge, Edge* d_edge_data) {
	int edge_index = atomicAdd(&d_edge_count, 1);
	if (edge_index < D_SIZE_EDGES) {
		d_edge_data[edge_index] = edge;
		return edge_index;
	}
	else {
		printf("Maximum edge size threshold reached: %d", D_SIZE_EDGES);
		return -1;
	}
}

__global__ void perform_induction_step(int* sampled_vertices, int* offsets, int* indices, Edge* d_edge_data) {
	int neighbor_index_start_offset = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (neighbor_index_start_offset < D_SIZE_VERTICES) {
		int neighbor_index_end_offset = neighbor_index_start_offset + 1;

		for (int n = offsets[neighbor_index_start_offset]; n < offsets[neighbor_index_end_offset]; n++) {
			if (sampled_vertices[neighbor_index_start_offset] && sampled_vertices[indices[n]]) {
				//printf("\nAdd edge: (%d,%d).", neighbor_index_start_offset, indices[n]);
				Edge edge;
				edge.source = neighbor_index_start_offset;
				edge.destination = indices[n];
				push_edge(edge, d_edge_data);
			}
		}
	}
}

__device__ int push_edge_expanding(Edge &edge, Edge* edge_data_expanding, int* d_edge_count_expanding) {
	int edge_index = atomicAdd(d_edge_count_expanding, 1);
	if (edge_index < D_SIZE_EDGES) {
		edge_data_expanding[edge_index] = edge;
		return edge_index;
	}
	else {
		printf("Maximum edge size threshold reached.");
		return -1;
	}
}

__global__ void perform_induction_step_expanding(int* sampled_vertices, int* offsets, int* indices, Edge* edge_data_expanding, int* d_edge_count_expanding) {
	int neighbor_index_start_offset = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (neighbor_index_start_offset < D_SIZE_VERTICES) {
		int neighbor_index_end_offset = neighbor_index_start_offset + 1;

		for (int n = offsets[neighbor_index_start_offset]; n < offsets[neighbor_index_end_offset]; n++) {
			if (sampled_vertices[neighbor_index_start_offset] && sampled_vertices[indices[n]]) {
				//printf("\nAdd edge: (%d,%d).", neighbor_index_start_offset, indices[n]);
				Edge edge;
				edge.source = neighbor_index_start_offset;
				edge.destination = indices[n];
				push_edge_expanding(edge, edge_data_expanding, d_edge_count_expanding);
			}
		}
	}
}