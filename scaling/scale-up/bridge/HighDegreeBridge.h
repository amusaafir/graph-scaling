//
// Created by Ahmed on 12-12-17.
//

#ifndef GRAPH_SCALING_TOOL_HIGHDEGREE_H
#define GRAPH_SCALING_TOOL_HIGHDEGREE_H

#include <iostream>
#include <unordered_map>
#include <algorithm>
#include "Bridge.h"

class HighDegreeBridge : public Bridge {
private:
    int getRandomHighDegreeVertex(Graph* graph);
    std::random_device seed;

    void collectHighDegreeVertices(Graph *graph);
public:
    HighDegreeBridge(int numberOfInterconnections, bool forceUndirectedEdges);
    void addBridgesBetweenGraphs(Graph *sourceGraph, Graph *targetGraph, std::vector<Edge<std::string>>& bridges);
    std::string getName();
};


#endif //GRAPH_SCALING_TOOL_HIGHDEGREE_H
