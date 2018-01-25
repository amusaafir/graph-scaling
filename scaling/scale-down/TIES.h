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
    void edgeBasedNodeSamplingStep(std::unordered_set<int> &sampledVertices, float fraction);
    void inductionStep(std::unordered_set<int> &sampledVertices, std::vector<Edge<int>> &sampledEdges);
    bool isVertexSampled(int vertex, std::unordered_set<int> &sampledVertices);
    Edge<int> getRandomEdge();

public:
    TIES(Graph* graph) : Sampling(graph, "TIES") { };
    Graph* sample(float fraction);
};


#endif //GRAPH_SCALING_TOOL_TIES_H
