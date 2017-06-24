#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "device_functions.h"
#include <curand.h>
#include <curand_kernel.h>
#include "../sampling/EdgeStruct.h"
#include "CudaErrCheck.h"

extern  __device__ int d_edge_count;
extern  __constant__ int D_SIZE_EDGES;
extern __constant__ int D_SIZE_VERTICES;

extern void perform_induction_step(int block_size, int thread_size, int *d_sampled_vertices, int *d_offsets, int* d_indices, Edge* d_edge_data);
extern void perform_induction_step_expanding(int block_size, int thread_size, int* d_sampled_vertices, int* d_offsets, int* d_indices, Edge* d_edge_data_expanding, int* d_edge_count_expanding);

__device__ int push_edge(Edge &edge, Edge* d_edge_data);
__global__ void perform_induction_step(int* sampled_vertices, int* offsets, int* indices, Edge* d_edge_data);
__device__ int push_edge_expanding(Edge &edge, Edge* edge_data_expanding, int* d_edge_count_expanding);
__global__ void perform_induction_step_expanding(int* sampled_vertices, int* offsets, int* indices, Edge* edge_data_expanding, int* d_edge_count_expanding);