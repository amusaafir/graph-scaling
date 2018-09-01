/*
- Created by Ahmed on 12-11-17.

Implementation of Total Induced Edge Sampling
*/

#ifndef GRAPH_SCALING_TOOL_TIES_H
#define GRAPH_SCALING_TOOL_TIES_H

#include <iostream>
#include "Sampling.h"

class TIES : public Sampling {
private:
    void edgeBasedNodeSamplingStep(std::unordered_set<long long> &sampledVertices, float fraction);
    void inductionStep(std::unordered_set<long long> &sampledVertices, std::vector<Edge<long long>> &sampledEdges);
    bool isVertexSampled(long long vertex, std::unordered_set<long long> &sampledVertices);
    Edge<long long> getRandomEdge();

public:
    TIES(Graph* graph) : Sampling(graph, "TIES") { };
    Graph* sample(float fraction);
};


#endif //GRAPH_SCALING_TOOL_TIES_H
