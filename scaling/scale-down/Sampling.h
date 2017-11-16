//
// Created by Ahmed on 12-11-17.
//

#ifndef GRAPH_SCALING_TOOL_SAMPLING_H
#define GRAPH_SCALING_TOOL_SAMPLING_H

#include <iostream>
#include <random>
#include <string>
#include "../../graph/Graph.h"

class Sampling {
protected:
    Graph* graph;
public:
    Sampling(Graph* graph, std::string samplingAlgorithmName);
    virtual void sample(float fraction) = 0;
    int getRequiredVerticesFraction(float fraction);
    int getRequiredEdgesFraction(float fraction);
    int getRandomIntBetweenRange(int min, int max);
};


#endif //GRAPH_SCALING_TOOL_SAMPLING_H
