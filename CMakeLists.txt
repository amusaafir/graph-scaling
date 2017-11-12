cmake_minimum_required(VERSION 3.8)
project(graph_scaling_tool)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES Main.cpp graph/Graph.cpp graph/Graph.h graph/Edge.cpp graph/Edge.h loader/GraphLoader.cpp loader/GraphLoader.h Scaler.cpp Scaler.h)
add_executable(graph_scaling_tool ${SOURCE_FILES})
