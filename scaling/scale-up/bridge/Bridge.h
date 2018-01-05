//
// Created by Ahmed on 12-12-17.
//

#ifndef GRAPH_SCALING_TOOL_BRIDGE_H
#define GRAPH_SCALING_TOOL_BRIDGE_H

#include "../../../graph/Graph.h"

class Bridge {
protected:
    int numberOfInterconnections;
    bool addDirectedBridges;
public:
    Bridge(int numberOfInterconnections, bool addDirectedBridges);
    virtual void addBridgesBetweenGraphs(Graph *left, Graph *right, std::vector<Edge<std::string>*>& bridges) = 0;
    virtual std::string getName() = 0;
};


#endif //GRAPH_SCALING_TOOL_BRIDGE_H
