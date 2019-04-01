//
// Created by Ahmed on 12-11-17.
//

#include "TIES.h"
#include <chrono>

/**
 * Sampling orchestrator function for TIES.
 * @param fraction: from the total amount of vertices.
 */
Graph* TIES::sample(float fraction) {
    if (fraction == 1.0) {
        return getFullGraphCopy();
    }

    std::unordered_set<long long> sampledVertices;
    edgeBasedNodeSamplingStep(sampledVertices, fraction);

    std::vector<Edge<long long>> sampledEdges;
    inductionStep(sampledVertices, sampledEdges);

    Graph* sampledGraph = new Graph(); // TODO: Delete this later
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
void TIES::edgeBasedNodeSamplingStep(std::unordered_set<long long> &sampledVertices, float fraction) {
    std::cout << "Performing Edge-based Node Sampling step." << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    long long requiredNumberOfVertices = getNumberOfVerticesFromFraction(fraction);

    std::mt19937 engine(seed());
    std::uniform_int_distribution<long long> dist(0, graph->getEdges().size() - 1);

    while (sampledVertices.size() < requiredNumberOfVertices) {
        Edge<long long> edge = graph->getEdges()[dist(engine)]; // Get random edge
        sampledVertices.insert(edge.getSource());
        sampledVertices.insert(edge.getTarget());
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed - edge based node sampling step: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" <<std::endl;
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
void TIES::inductionStep(std::unordered_set<long long> &sampledVertices, std::vector<Edge<long long>> &sampledEdges) {
    std::cout << "Performing total induction step." << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


    for (long long i = 0; i < graph->getEdges().size(); i++) {
        Edge<long long> edge = graph->getEdges()[i];

        if (isVertexSampled(edge.getSource(), sampledVertices)
            && isVertexSampled(edge.getTarget(), sampledVertices)) {
            sampledEdges.push_back(edge);
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed - induction step: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" <<std::endl;

    std::cout << "Finished performing total induction step: collected " << sampledEdges.size() << " edges." << std::endl;
}

/**
 * Checks if a given vertex exists in the sampled set
 * @param vertex
 * @param sampledVertices - set to hold the sampled vertices, collected from the Edge-based node sampling function.
 * @return
 */
bool TIES::isVertexSampled(long long vertex, std::unordered_set<long long> &sampledVertices) {
    return sampledVertices.find(vertex) != sampledVertices.end();
}

/**
 * Returns a random edge from the graph.
 * @return
 */
Edge<long long> TIES::getRandomEdge() {
    return graph->getEdges()[getRandomIntBetweenRange(0, graph->getEdges().size() - 1)];
}