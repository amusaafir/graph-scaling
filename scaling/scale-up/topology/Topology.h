//
// Created by Ahmed on 12-12-17.
//

#ifndef GRAPH_SCALING_TOOL_TOPOLOGY_H
#define GRAPH_SCALING_TOOL_TOPOLOGY_H

#include <iostream>
#include "../../../graph/Graph.h"
#include "../bridge/Bridge.h"

class Topology {
protected:
    Bridge* bridge;
public:
    Topology(Bridge* bridge);
    Bridge* getBridge();
    virtual std::vector<Edge<std::string>> getBridgeEdges(std::vector<Graph*> samples) = 0;
    virtual std::string getName() = 0;
};

#endif //GRAPH_SCALING_TOOL_TOPOLOGY_H
