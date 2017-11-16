cmake_minimum_required(VERSION 3.8)
project(graph_scaling_tool)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES Main.cpp graph/Graph.cpp graph/Graph.h graph/Edge.cpp graph/Edge.h loader/GraphLoader.cpp loader/GraphLoader.h scaling/ScalingManager.cpp scaling/ScalingManager.h scaling/scale-down/Sampling.h scaling/scale-down/TIES.cpp scaling/scale-down/TIES.h scaling/scale-down/Sampling.cpp scaling/scale-up/ScaleUp.cpp scaling/scale-up/ScaleUp.h)
add_executable(graph_scaling_tool ${SOURCE_FILES})
