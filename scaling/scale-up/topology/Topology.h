//
// Created by Ahmed on 12-12-17.
//

#ifndef GRAPH_SCALING_TOOL_TOPOLOGY_H
#define GRAPH_SCALING_TOOL_TOPOLOGY_H

#include "../../../graph/Graph.h"
#include "../bridge/Bridge.h"

class Topology {
protected:
    Bridge* bridge;
public:
    Topology(Bridge* bridge);
    virtual std::vector<Edge<std::string>*> getBridgeEdges(std::vector<Graph*> samples) = 0;
};

#endif //GRAPH_SCALING_TOOL_TOPOLOGY_H
