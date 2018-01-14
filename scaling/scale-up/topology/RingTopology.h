//
// Created by Ahmed on 6-1-18.
//

#ifndef GRAPH_SCALING_TOOL_RINGTOPOLOGY_H
#define GRAPH_SCALING_TOOL_RINGTOPOLOGY_H


#include "Topology.h"

class RingTopology : public Topology {
public:
    RingTopology(Bridge* bridge);
    std::vector<Edge<std::string>> getBridgeEdges(std::vector<Graph*> samples);
    std::string getName();
};


#endif //GRAPH_SCALING_TOOL_RINGTOPOLOGY_H
