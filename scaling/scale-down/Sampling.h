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

    virtual Graph* sample(float fraction) = 0;

    int getNumberOfVerticesFromFraction(float fraction);

    int getNumberOfEdgesFromFraction(float fraction);

    int getRandomIntBetweenRange(int min, int max);

    void run(float fraction, std::string outputPath);
};


#endif //GRAPH_SCALING_TOOL_SAMPLING_H
