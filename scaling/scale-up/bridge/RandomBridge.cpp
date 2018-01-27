//
// Created by Ahmed on 12-12-17.
//

#include "RandomBridge.h"
#include <iostream>

RandomBridge::RandomBridge(int numberOfInterconnections, bool addDirectedBridges)
        : Bridge(numberOfInterconnections, addDirectedBridges) {}

void RandomBridge::addBridgesBetweenGraphs(Graph *sourceGraph, Graph *targetGraph, std::vector<Edge<std::string>>& bridges) {
    for (int i = 0; i < numberOfInterconnections; i++) {
        int vertexSource = getRandomVertexFromGraph(sourceGraph);
        int vertexTarget = getRandomVertexFromGraph(targetGraph);

        bridges.push_back(Edge<std::string>(std::to_string(vertexSource) + sourceGraph->getIdentifier(),
                                                std::to_string(vertexTarget) + targetGraph->getIdentifier()));

        if (!addDirectedBridges) {
            bridges.push_back(Edge<std::string>(std::to_string(vertexTarget) + targetGraph->getIdentifier(),
                                                    std::to_string(vertexSource) + sourceGraph->getIdentifier()                                                    ));
        }
    }
}

int RandomBridge::getRandomVertexFromGraph(Graph *graph) {
    std::random_device seed;
    std::mt19937 engine(seed());

    // Select random edge
    std::uniform_int_distribution<int> randomEdgeDist(0, graph->getEdges().size() - 1);
    Edge<int> randomEdge = graph->getEdges()[randomEdgeDist(engine)];
    std::uniform_int_distribution<int> randomVertexFromEdgeDist(0, 1);

    return (randomVertexFromEdgeDist(engine) > 0) ? randomEdge.getSource() : randomEdge.getTarget();
}

std::string RandomBridge::getName() {
    return "RandomBridge";
}