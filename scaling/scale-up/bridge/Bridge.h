//
// Created by Ahmed on 12-12-17.
//

#ifndef GRAPH_SCALING_TOOL_BRIDGE_H
#define GRAPH_SCALING_TOOL_BRIDGE_H

#include "../../../graph/Graph.h"

class Bridge {
protected:
    Graph* left;
    Graph* right;
    int numberOfInterconnections;
    bool forceUndirectedEdges;
public:
    Bridge(Graph* left, Graph* right, int numberOfInterconnections, bool forceUndirectedEdges);
    virtual void addBridges() = 0;
};


#endif //GRAPH_SCALING_TOOL_BRIDGE_H
