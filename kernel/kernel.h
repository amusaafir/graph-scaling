#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "device_functions.h"
#include <curand.h>
#include <curand_kernel.h>
#include "CudaErrCheck.h"

#define EXCLUDED -1
#define INCLUDED -2

extern __constant__ int D_NODE_START_VERTEX;
extern  __constant__ int D_NODE_SIZE_EDGES;
extern __constant__ int D_NODE_END_VERTEX;

extern void perform_induction_step(int block_size, int thread_size, int *d_sampled_vertices, int *d_sources, int* d_destinations, int* d_results, int mpi_id);
extern void perform_induction_step2(int block_size, int thread_size, int *d_sampled_vertices, int* d_destinations, int* d_results, int curr_node, int curr_node_start_vertex, int curr_node_end_vertex);

__global__ void perform_induction_step(int *d_sampled_vertices, int *d_sources, int* d_destinations, int* d_results, int mpi_id);
__global__ void perform_induction_step2(int *d_sampled_vertices, int* d_destinations, int* d_results, int curr_node, int curr_node_start_vertex, int curr_node_end_vertex);
