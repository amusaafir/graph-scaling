cmake_minimum_required(VERSION 2.8)
project(graph_scaling_tool)

option(BUILD_TEST "Build the tests (test_graph_scaling_tool)" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_CXX_STANDARD 11)

include_directories(lib/snap/glib-core)
include_directories(lib/snap/snap-core)
include_directories(lib/snap/snap-adv)

set(SOURCE_FILES graph/Graph.cpp graph/Graph.h graph/Edge.h io/GraphLoader.cpp io/GraphLoader.h scaling/Scaling.cpp scaling/Scaling.h scaling/scale-down/Sampling.h scaling/scale-down/TIES.cpp scaling/scale-down/TIES.h scaling/scale-down/Sampling.cpp scaling/scale-up/ScaleUp.cpp scaling/scale-up/ScaleUp.h scaling/scale-up/ScalingUpConfig.cpp scaling/scale-up/ScalingUpConfig.h scaling/scale-up/IdentifierTracker.cpp scaling/scale-up/IdentifierTracker.h scaling/scale-up/topology/Topology.cpp scaling/scale-up/topology/Topology.h scaling/scale-up/bridge/RandomBridge.cpp scaling/scale-up/bridge/RandomBridge.h scaling/scale-up/bridge/HighDegreeBridge.cpp scaling/scale-up/bridge/HighDegreeBridge.h scaling/scale-up/bridge/Bridge.cpp scaling/scale-up/bridge/Bridge.h scaling/scale-up/topology/StarTopology.cpp scaling/scale-up/topology/StarTopology.h io/WriteScaledUpGraph.cpp io/WriteScaledUpGraph.h scaling/scale-up/topology/RingTopology.cpp scaling/scale-up/topology/RingTopology.h scaling/scale-up/topology/ChainTopology.cpp scaling/scale-up/topology/ChainTopology.h scaling/scale-up/topology/FullyConnectedTopology.cpp scaling/scale-up/topology/FullyConnectedTopology.h io/user-input/UserInput.cpp io/user-input/UserInput.h io/user-input/UserInputCMD.cpp io/user-input/UserInputCMD.h io/user-input/UserInputPrompt.cpp io/user-input/UserInputPrompt.h io/WriteSampledGraph.cpp io/WriteSampledGraph.h scaling/scale-down/RandomNode.cpp scaling/scale-down/RandomNode.h io/WriteGraph.cpp io/WriteGraph.h scaling/scale-up/auto-tuner/Autotuner.cpp scaling/scale-up/auto-tuner/Autotuner.h scaling/scale-up/auto-tuner/GraphAnalyser.cpp scaling/scale-up/auto-tuner/GraphAnalyser.h scaling/scale-down/RandomEdge.cpp scaling/scale-down/RandomEdge.h)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

SET(OBJS ${CMAKE_CURRENT_SOURCE_DIR}/lib/snap/snap-core/Snap.o)

add_executable(graph_scaling_tool ${OBJS} Main.cpp ${SOURCE_FILES})

if(BUILD_TEST)
    add_subdirectory(lib/googletest)
    include_directories(lib/googletest/googletest/include)
    include_directories(lib/googletest/googlemock/include)

    set(TEST_FILES tests/TestHighDegreeBridge.cpp tests/TestInit.cpp tests/TestScalingUpConfig.cpp)
    add_executable(test_graph_scaling_tool ${OBJS} ${SOURCE_FILES} ${TEST_FILES})

    target_link_libraries(test_graph_scaling_tool gtest gtest_main)
endif()