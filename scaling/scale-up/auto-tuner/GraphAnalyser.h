//
// Created by Ahmed on 30-3-19.
//

#ifndef GRAPH_SCALING_TOOL_GRAPHANALYSER_H
#define GRAPH_SCALING_TOOL_GRAPHANALYSER_H

#include <iostream>
#include <unordered_map>
#include "Snap.h"
#include "../../../graph/Graph.h"

class GraphAnalyser {
    const int DIAMETER_TEST_NODES = 50;
    PUNGraph graph;
public:
    GraphAnalyser();

    void loadGraph(std::vector<Graph*> samples, std::vector<Edge<std::string>> bridges);

    int calculateDiameter();

    bool deleteGraph();
};


#endif //GRAPH_SCALING_TOOL_GRAPHANALYSER_H
