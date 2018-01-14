//
// Created by Ahmed on 6-1-18.
//

#ifndef GRAPH_SCALING_TOOL_CHAINTOPOLOGY_H
#define GRAPH_SCALING_TOOL_CHAINTOPOLOGY_H


#include "Topology.h"

class ChainTopology : public Topology {
public:
    ChainTopology(Bridge* bridge);
    std::vector<Edge<std::string>> getBridgeEdges(std::vector<Graph*> samples);
    std::string getName();
};


#endif //GRAPH_SCALING_TOOL_CHAINTOPOLOGY_H
