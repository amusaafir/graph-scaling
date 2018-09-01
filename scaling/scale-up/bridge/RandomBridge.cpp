//
// Created by Ahmed on 12-12-17.
//

#include "RandomBridge.h"
#include <iostream>

RandomBridge::RandomBridge(long long numberOfInterconnections, bool addDirectedBridges)
        : Bridge(numberOfInterconnections, addDirectedBridges) {}

void RandomBridge::addBridgesBetweenGraphs(Graph *sourceGraph, Graph *targetGraph, std::vector<Edge<std::string>>& bridges) {
    for (long long i = 0; i < numberOfInterconnections; i++) {
        long long vertexSource = getRandomVertexFromGraph(sourceGraph);
        long long vertexTarget = getRandomVertexFromGraph(targetGraph);

        bridges.push_back(Edge<std::string>(std::to_string(vertexSource) + sourceGraph->getIdentifier(),
                                                std::to_string(vertexTarget) + targetGraph->getIdentifier()));

        if (!addDirectedBridges) {
            bridges.push_back(Edge<std::string>(std::to_string(vertexTarget) + targetGraph->getIdentifier(),
                                                    std::to_string(vertexSource) + sourceGraph->getIdentifier()                                                    ));
        }
    }
}

long long RandomBridge::getRandomVertexFromGraph(Graph *graph) {
    std::random_device seed;
    std::mt19937 engine(seed());

    // Select random edge
    std::uniform_int_distribution<long long> randomEdgeDist(0, graph->getEdges().size() - 1);
    Edge<long long> randomEdge = graph->getEdges()[randomEdgeDist(engine)];
    std::uniform_int_distribution<long long> randomVertexFromEdgeDist(0, 1);

    return (randomVertexFromEdgeDist(engine) > 0) ? randomEdge.getSource() : randomEdge.getTarget();
}

std::string RandomBridge::getName() {
    return "RandomBridge";
}