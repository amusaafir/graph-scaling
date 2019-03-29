cmake_minimum_required(VERSION 2.8)
project(graph_scaling_tool)

option(BUILD_TEST "Build the tests (test_graph_scaling_tool)" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_CXX_STANDARD 11)


set(SOURCE_FILES graph/Graph.cpp graph/Graph.h graph/Edge.h io/GraphLoader.cpp io/GraphLoader.h scaling/Scaling.cpp scaling/Scaling.h scaling/scale-down/Sampling.h scaling/scale-down/TIES.cpp scaling/scale-down/TIES.h scaling/scale-down/Sampling.cpp scaling/scale-up/ScaleUp.cpp scaling/scale-up/ScaleUp.h scaling/scale-up/ScalingUpConfig.cpp scaling/scale-up/ScalingUpConfig.h scaling/scale-up/IdentifierTracker.cpp scaling/scale-up/IdentifierTracker.h scaling/scale-up/topology/Topology.cpp scaling/scale-up/topology/Topology.h scaling/scale-up/bridge/RandomBridge.cpp scaling/scale-up/bridge/RandomBridge.h scaling/scale-up/bridge/HighDegreeBridge.cpp scaling/scale-up/bridge/HighDegreeBridge.h scaling/scale-up/bridge/Bridge.cpp scaling/scale-up/bridge/Bridge.h scaling/scale-up/topology/StarTopology.cpp scaling/scale-up/topology/StarTopology.h io/WriteScaledUpGraph.cpp io/WriteScaledUpGraph.h scaling/scale-up/topology/RingTopology.cpp scaling/scale-up/topology/RingTopology.h scaling/scale-up/topology/ChainTopology.cpp scaling/scale-up/topology/ChainTopology.h scaling/scale-up/topology/FullyConnectedTopology.cpp scaling/scale-up/topology/FullyConnectedTopology.h io/user-input/UserInput.cpp io/user-input/UserInput.h io/user-input/UserInputCMD.cpp io/user-input/UserInputCMD.h io/user-input/UserInputPrompt.cpp io/user-input/UserInputPrompt.h io/WriteSampledGraph.cpp io/WriteSampledGraph.h scaling/scale-down/RandomNode.cpp scaling/scale-down/RandomNode.h io/WriteGraph.cpp io/WriteGraph.h)
add_executable(graph_scaling_tool Main.cpp ${SOURCE_FILES})

if(BUILD_TEST)
    add_subdirectory(lib/googletest-master)
    include_directories(lib/googletest-master/googletest/include)
    include_directories(lib/googletest-master/googlemock/include)

    set(TEST_FILES tests/TestHighDegreeBridge.cpp tests/TestInit.cpp)
    add_executable(test_graph_scaling_tool ${SOURCE_FILES} ${TEST_FILES})

    target_link_libraries(test_graph_scaling_tool gtest gtest_main)
endif()