//
// Created by Ahmed on 12-11-17.
//

#include "TIES.h"

/**
 * Sampling orchestrator function for TIES.
 * @param fraction: from the total amount of vertices.
 */
Graph* TIES::sample(float fraction) {
    std::unordered_set<int> sampledVertices;
    executeEdgeBasedNodeSamplingStep(sampledVertices, fraction);

    std::vector<Edge> sampledEdges;
    executeInductionStep(sampledVertices, sampledEdges);

    Graph* sampledGraph = new Graph();
    sampledGraph->setVertices(sampledVertices);
    sampledGraph->setEdges(sampledEdges);

    return sampledGraph;
}

/**
 * Edge-based node sampling step: select end vertices from random edges in the graph up until a preferred amount of
 * (sampled) vertices is reached.
 * @param sampledVertices - set to hold the sampled vertices, which are collected in this function.
 * @param fraction - from the total amount of vertices.
 */
void TIES::executeEdgeBasedNodeSamplingStep(std::unordered_set<int> &sampledVertices, float fraction) {
    std::cout << "Performing Edge-based Node Sampling step." << std::endl;

    int requiredNumberOfVertices = getNumberOfVerticesFromFraction(fraction);

    while (sampledVertices.size() < requiredNumberOfVertices) {
        Edge edge = getRandomEdge();
        sampledVertices.insert(edge.getSource());
        sampledVertices.insert(edge.getTarget());
    }

    std::cout << "Finished performing Edge-based Node Sampling step: "
            "collected " << sampledVertices.size() << " sampled vertices." << std::endl;
}

/**
 * Total induction step: collect the edges from the graph where, from each edge, both end vertices exist in
 * the sampled vertices set.
 * @param sampledVertices - set to hold the sampled vertices, collected from the Edge-based node sampling function.
 * @param sampledEdges - vector to hold the sampled edges, which are collected in this function.
 * @return
 */
void TIES::executeInductionStep(std::unordered_set<int> &sampledVertices, std::vector<Edge>& sampledEdges) {
    std::cout << "Performing total induction step." << std::endl;

    for (int i = 0; i < graph->getEdges().size(); i++) {
        Edge edge = graph->getEdges()[i];

        if (isVertexInSampledVertices(edge.getSource(), sampledVertices)
            && isVertexInSampledVertices(edge.getTarget(), sampledVertices)) {
            sampledEdges.push_back(edge);
        }
    }

    std::cout << "Finished performing total induction step: collected " << sampledEdges.size() << " edges." << std::endl;
}

/**
 * Checks if a given vertex exists in the sampled set
 * @param vertex
 * @param sampledVertices - set to hold the sampled vertices, collected from the Edge-based node sampling function.
 * @return
 */
bool TIES::isVertexInSampledVertices(int vertex, std::unordered_set<int> &sampledVertices) {
    return sampledVertices.find(vertex) != sampledVertices.end();
}

/**
 * Returns a random edge from the graph.
 * @return
 */
Edge TIES::getRandomEdge() {
    return graph->getEdges()[getRandomIntBetweenRange(0, graph->getEdges().size() - 1)];
}