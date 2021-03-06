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
    PUNGraph graph;
public:
    GraphAnalyser();

    void loadGraph(std::vector<Graph*> samples, std::vector<Edge<std::string>> bridges);
};


#endif //GRAPH_SCALING_TOOL_GRAPHANALYSER_H
