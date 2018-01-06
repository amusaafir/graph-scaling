//
// Created by Ahmed on 6-1-18.
//

#ifndef GRAPH_SCALING_TOOL_MESHTOPOLOGY_H
#define GRAPH_SCALING_TOOL_MESHTOPOLOGY_H


#include "Topology.h"

class FullyConnectedTopology : public Topology {
    FullyConnectedTopology(Bridge* bridge);
    std::vector<Edge<std::string>> getBridgeEdges(std::vector<Graph*> samples);
    std::string getName();
};


#endif //GRAPH_SCALING_TOOL_MESHTOPOLOGY_H
