//
// Created by Ahmed on 30-3-19.
//

#ifndef GRAPH_SCALING_TOOL_AUTOTUNER_H
#define GRAPH_SCALING_TOOL_AUTOTUNER_H


#include "../../../graph/Graph.h"

class Autotuner {
private:
    int iterations; // Number of attempts to find a proper scaled up match
public:
    Autotuner(Graph* originalGraph, std::vector<Graph*> &samples);
};


#endif //GRAPH_SCALING_TOOL_AUTOTUNER_H
