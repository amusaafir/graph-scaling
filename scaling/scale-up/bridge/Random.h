//
// Created by Ahmed on 12-12-17.
//

#ifndef GRAPH_SCALING_TOOL_RANDOM_H
#define GRAPH_SCALING_TOOL_RANDOM_H

#include "Bridge.h"

class Random : public Bridge {
public:
    Random(int numberOfInterconnections, bool forceUndirectedEdges);
    void addBridgesBetweenGraphs(Graph *left, Graph *right);
};


#endif //GRAPH_SCALING_TOOL_RANDOM_H
