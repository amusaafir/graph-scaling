cmake_minimum_required(VERSION 3.8)
project(graph_scaling_tool)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES Main.cpp graph/Graph.cpp graph/Graph.h graph/Edge.cpp graph/Edge.h loader/GraphLoader.cpp loader/GraphLoader.h scaling/Scaling.cpp scaling/Scaling.h scaling/scale-down/Sampling.h scaling/scale-down/TIES.cpp scaling/scale-down/TIES.h scaling/scale-down/Sampling.cpp scaling/scale-up/ScaleUp.cpp scaling/scale-up/ScaleUp.h scaling/scale-up/ScaleUpSamplesInfo.cpp scaling/scale-up/ScaleUpSamplesInfo.h scaling/scale-up/identifier/IdentifierTracker.cpp scaling/scale-up/identifier/IdentifierTracker.h scaling/scale-up/topology/Topology.cpp scaling/scale-up/topology/Topology.h scaling/scale-up/bridge/Random.cpp scaling/scale-up/bridge/Random.h scaling/scale-up/bridge/HighDegree.cpp scaling/scale-up/bridge/HighDegree.h scaling/scale-up/bridge/Bridge.cpp scaling/scale-up/bridge/Bridge.h plan/ScalePlan.cpp plan/ScalePlan.h)
add_executable(graph_scaling_tool ${SOURCE_FILES})
