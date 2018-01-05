//
// Created by Ahmed on 12-12-17.
//

#ifndef GRAPH_SCALING_TOOL_RANDOM_H
#define GRAPH_SCALING_TOOL_RANDOM_H

#include "Bridge.h"
#include <random>

class RandomBridge : public Bridge {
private:
    int getRandomVertexFromGraph(Graph* graph);

public:
    RandomBridge(int numberOfInterconnections, bool forceUndirectedEdges);
    void addBridgesBetweenGraphs(Graph *left, Graph *right, std::vector<Edge<std::string>*>& bridges);
    std::string getName();
};


#endif //GRAPH_SCALING_TOOL_RANDOM_H
