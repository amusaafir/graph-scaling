//
// Created by Ahmed on 3-4-19.
//

#ifndef GRAPH_SCALING_TOOL_RANDOMEDGE_H
#define GRAPH_SCALING_TOOL_RANDOMEDGE_H


#include "Sampling.h"
#include <unordered_map>

class RandomEdge : public Sampling {
private:
    void edgeSamplingStep(std::unordered_set<long long>& samplesVertices, std::vector<Edge<long long>>& sampledEdges, float fraction);

public:
    RandomEdge(Graph* graph) : Sampling(graph, "RandomEdge") { };
    Graph* sample(float fraction);
};


#endif //GRAPH_SCALING_TOOL_RANDOMEDGE_H
