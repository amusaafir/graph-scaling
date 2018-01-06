cmake_minimum_required(VERSION 3.8)
project(graph_scaling_tool)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES Main.cpp graph/Graph.cpp graph/Graph.h graph/Edge.h io/GraphLoader.cpp io/GraphLoader.h scaling/Scaling.cpp scaling/Scaling.h scaling/scale-down/Sampling.h scaling/scale-down/TIES.cpp scaling/scale-down/TIES.h scaling/scale-down/Sampling.cpp scaling/scale-up/ScaleUp.cpp scaling/scale-up/ScaleUp.h scaling/scale-up/ScaleUpSamplesInfo.cpp scaling/scale-up/ScaleUpSamplesInfo.h scaling/scale-up/IdentifierTracker.cpp scaling/scale-up/IdentifierTracker.h scaling/scale-up/topology/Topology.cpp scaling/scale-up/topology/Topology.h scaling/scale-up/bridge/RandomBridge.cpp scaling/scale-up/bridge/RandomBridge.h scaling/scale-up/bridge/HighDegreeBridge.cpp scaling/scale-up/bridge/HighDegreeBridge.h scaling/scale-up/bridge/Bridge.cpp scaling/scale-up/bridge/Bridge.h plan/ScalePlan.cpp plan/ScalePlan.h scaling/scale-up/topology/StarTopology.cpp scaling/scale-up/topology/StarTopology.h io/WriteScaledUpGraph.cpp io/WriteScaledUpGraph.h scaling/scale-up/topology/RingTopology.cpp scaling/scale-up/topology/RingTopology.h)
add_executable(graph_scaling_tool ${SOURCE_FILES})