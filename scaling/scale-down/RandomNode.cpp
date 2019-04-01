//
// Created by Ahmed on 2-9-18.
//

#include <chrono>
#include "RandomNode.h"

Graph* RandomNode::sample(float fraction) {
    if (fraction == 1.0) {
        return getFullGraphCopy();
    }

    std::unordered_set<long long> sampledVertices;
    samplingStep(sampledVertices, fraction);

    std::vector<Edge<long long>> sampledEdges;
    inductionStep(sampledVertices, sampledEdges);

    Graph* sampledGraph = new Graph(); // TODO: Delete this later.
    sampledGraph->setVertices(sampledVertices);
    sampledGraph->setEdges(sampledEdges);

    return sampledGraph;
}

void RandomNode::samplingStep(std::unordered_set<long long> &sampledVertices, float fraction) {
    std::cout << "Performing node sampling step." << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    long long requiredNumberOfVertices = getNumberOfVerticesFromFraction(fraction);

    while (sampledVertices.size() < requiredNumberOfVertices) {
        long long node = getRandomNode();
        sampledVertices.insert(node);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed - node sampling step: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" <<std::endl;
    std::cout << "Finished performing node sampling step: "
            "collected " << sampledVertices.size() << " sampled vertices." << std::endl;
}

void RandomNode::inductionStep(std::unordered_set<long long> &sampledVertices, std::vector<Edge<long long>> &sampledEdges) {
    std::cout << "Performing induction step." << std::endl;
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

    std::cout << "Finished performing induction step: collected " << sampledEdges.size() << " edges." << std::endl;
}

/**
 * Checks if a given vertex exists in the sampled set
 * @param vertex
 * @param sampledVertices - set to hold the sampled vertices, collected from the Edge-based node sampling function.
 * @return
 */
bool RandomNode::isVertexSampled(long long vertex, std::unordered_set<long long> &sampledVertices) {
    return sampledVertices.find(vertex) != sampledVertices.end();
}

/**
 * Returns a random edge from the graph.
 * @return
 */
long long RandomNode::getRandomNode() {
    return nodes[getRandomIntBetweenRange(0, nodes.size() - 1)];
}