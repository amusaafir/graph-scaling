//
// Created by Ahmed on 12-12-17.
//

#ifndef GRAPH_SCALING_TOOL_TOPOLOGY_H
#define GRAPH_SCALING_TOOL_TOPOLOGY_H

#include "../../../graph/Graph.h"
#include "../bridge/Bridge.h"

class Topology {
private:
    std::vector<Graph*> samples;
    Bridge* bridge;
public:
    Topology(std::vector<Graph*> samples, Bridge* bridge);
    virtual void getBridgeEdges() = 0;
};

#endif //GRAPH_SCALING_TOOL_TOPOLOGY_H
