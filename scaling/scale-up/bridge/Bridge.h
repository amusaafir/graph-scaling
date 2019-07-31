//
// Created by Ahmed on 12-12-17.
//

#ifndef GRAPH_SCALING_TOOL_BRIDGE_H
#define GRAPH_SCALING_TOOL_BRIDGE_H

#include "../../../graph/Graph.h"

class Bridge {
protected:
    long long numberOfInterconnections;
    bool addDirectedBridges;
public:
    Bridge(long long numberOfInterconnections, bool addDirectedBridges);
    virtual void addBridgesBetweenGraphs(Graph *sourceGraph, Graph *targetGraph, std::vector<Edge<std::string>>& bridges) = 0;
    virtual std::string getName() = 0;

    long long getNumberOfInterconnections();

    void setNumberOfInterconnections(long long numberOfInterConnections);
};


#endif //GRAPH_SCALING_TOOL_BRIDGE_H
