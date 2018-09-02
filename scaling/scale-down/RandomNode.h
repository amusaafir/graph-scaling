//
// Created by Ahmed on 2-9-18.
//

#ifndef GRAPH_SCALING_TOOL_RANDOMNODE_H
#define GRAPH_SCALING_TOOL_RANDOMNODE_H


#include "Sampling.h"

class RandomNode : public Sampling {
private:
    void samplingStep(std::unordered_set<long long> &sampledVertices, float fraction);
    void inductionStep(std::unordered_set<long long> &sampledVertices, std::vector<Edge<long long>> &sampledEdges);
    bool isVertexSampled(long long vertex, std::unordered_set<long long> &sampledVertices);
    long long getRandomNode();
    std::vector<long long> nodes;

public:
    RandomNode(Graph* graph) : Sampling(graph, "RandomNode") {
        nodes.insert(nodes.end(), graph->getVertices().begin(), graph->getVertices().end());
    };
    Graph* sample(float fraction);
};


#endif //GRAPH_SCALING_TOOL_RANDOMNODE_H
