//
// Created by Ahmed on 12-12-17.
//

#ifndef GRAPH_SCALING_TOOL_RANDOM_H
#define GRAPH_SCALING_TOOL_RANDOM_H

#include "Bridge.h"

class Random : public Bridge {
public:
    Random(Graph *left, Graph *right, int numberOfInterconnections, bool forceUndirectedEdges);
    void addBridges();
};


#endif //GRAPH_SCALING_TOOL_RANDOM_H
