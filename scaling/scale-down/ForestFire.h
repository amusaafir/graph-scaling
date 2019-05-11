//
// Created by Ahmed on 10-5-19.
//

#ifndef GRAPH_SCALING_TOOL_FORESTFIRE_H
#define GRAPH_SCALING_TOOL_FORESTFIRE_H


#include "Sampling.h"

class ForestFire : public Sampling {
private:
    long long sourceVertex;
public:
    ForestFire(Graph* graph, long long sourceVertex) : Sampling(graph, "ForestFire") {
        this->sourceVertex = sourceVertex;
    };

    Graph* sample(float fraction);
};


#endif //GRAPH_SCALING_TOOL_FORESTFIRE_H
