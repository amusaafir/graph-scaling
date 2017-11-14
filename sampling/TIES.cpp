//
// Created by Ahmed on 12-11-17.
//

#include "TIES.h"

void TIES::sample(float fraction) {
    this->graph = graph;

    std::unordered_set<int> sampledVertices;
    performEdgeBasedNodeSamplingStep(sampledVertices, fraction);

    std::vector<Edge> sampledEdges;
    performInductionStep(sampledVertices, sampledEdges);
}

void TIES::performEdgeBasedNodeSamplingStep(std::unordered_set<int>& sampledVertices, float fraction) {
    std::cout << "Performing Edge-based Node Sampling step." << std::endl;

    int requiredNumberOfVertices = getRequiredVerticesFraction(fraction);

    while (sampledVertices.size() < requiredNumberOfVertices) {
        Edge edge = getRandomEdge();
        sampledVertices.insert(edge.getSource());
        sampledVertices.insert(edge.getTarget());
    }

    std::cout << "Finished performing Edge-based Node Sampling step: "
            "collected " << sampledVertices.size() << " sampled vertices." << std::endl;
}

std::vector<Edge> TIES::performInductionStep(std::unordered_set<int>& sampledVertices, std::vector<Edge>& sampledEdges) {
    std::cout << "Performing total induction step." << std::endl;

    for (int i = 0; i < graph->getEdges().size(); i++) {
        Edge edge = graph->getEdges()[i];

        if (isVertexInSampledVertices(edge.getSource(), sampledVertices)
            && isVertexInSampledVertices(edge.getTarget(), sampledVertices)) {
            sampledEdges.push_back(edge);
        }
    }

    std::cout << "Finished performing total induction step: collected " << sampledEdges.size() << " edges." << std::endl;

    return sampledEdges;
}

bool TIES::isVertexInSampledVertices(int vertex, std::unordered_set<int> &sampledVertices) {
    return sampledVertices.find(vertex) != sampledVertices.end();
}

Edge TIES::getRandomEdge() {
    return graph->getEdges()[getRandomIntBetweenRange(0, graph->getEdges().size())];
}