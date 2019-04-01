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
    std::random_device seed;
public:
    Sampling(Graph* graph, std::string samplingAlgorithmName);

    virtual Graph* sample(float fraction) = 0;

    long long getNumberOfVerticesFromFraction(float fraction);

    long long getNumberOfEdgesFromFraction(float fraction);

    long long getRandomIntBetweenRange(long long min, long long max);

    void run(float fraction, std::string outputPath);

    Graph* getFullGraphCopy();
};


#endif //GRAPH_SCALING_TOOL_SAMPLING_H
