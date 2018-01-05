//
// Created by Ahmed on 3-1-18.
//

#ifndef GRAPH_SCALING_TOOL_STAR_H
#define GRAPH_SCALING_TOOL_STAR_H

#include "Topology.h"

class StarTopology : public Topology {
public:
    StarTopology(Bridge* bridge);
    std::vector<Edge<std::string>*> getBridgeEdges(std::vector<Graph*> samples);
    std::string getName();
};


#endif //GRAPH_SCALING_TOOL_STAR_H
