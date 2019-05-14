//
// Created by Ahmed on 3-4-19.
//

#ifndef GRAPH_SCALING_TOOL_RANDOMEDGE_H
#define GRAPH_SCALING_TOOL_RANDOMEDGE_H


#include <chrono>
#include "Sampling.h"
#include <unordered_map>
#include <set>
#include <assert.h>

class RandomEdge : public Sampling {
private:
    bool isInputGraphInBothDirections;
    void edgeSamplingStep(std::unordered_set<long long>& samplesVertices, std::vector<Edge<long long>>& sampledEdges, float fraction);

public:
    RandomEdge(Graph* graph, bool isInputGraphInBothDirections) : Sampling(graph, "RandomEdge") {
        this->isInputGraphInBothDirections = isInputGraphInBothDirections;
    };
    Graph* sample(float fraction);
};


#endif //GRAPH_SCALING_TOOL_RANDOMEDGE_H
