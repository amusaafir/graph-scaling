//
// Created by Ahmed on 12-12-17.
//

#include "RandomBridge.h"
#include <iostream>

RandomBridge::RandomBridge(int numberOfInterconnections, bool addDirectedBridges)
        : Bridge(numberOfInterconnections, addDirectedBridges) {}

void RandomBridge::addBridgesBetweenGraphs(Graph *left, Graph *right, std::vector<Edge<std::string>*>& bridges) {
    for (int i = 0; i < numberOfInterconnections; i++) {
        int vertexSource = getRandomVertexFromGraph(left);
        int vertexTarget = getRandomVertexFromGraph(right);

        bridges.push_back(new Edge<std::string>(std::to_string(vertexSource) + left->getIdentifier(),
                                                std::to_string(vertexTarget) + right->getIdentifier()));

        if (!addDirectedBridges) {
            bridges.push_back(new Edge<std::string>(std::to_string(vertexTarget) + right->getIdentifier(),
                                                    std::to_string(vertexSource) + left->getIdentifier()                                                    ));
        }
    }
}

int RandomBridge::getRandomVertexFromGraph(Graph *graph) {
    std::random_device seed;
    std::mt19937 engine(seed());

    // Select random edge
    std::uniform_int_distribution<int> randomEdgeDist(0, graph->getEdges().size() - 1);


    Edge<int>* randomEdge;

    try {
        *randomEdge = graph->getEdges()[randomEdgeDist(engine)];
    } catch(const std::out_of_range) {
        std::cerr << "Out of range error when selecting random edge." << std::endl;
    }

    std::uniform_int_distribution<int> randomVertexFromEdgeDist(0, 1);

    return (randomVertexFromEdgeDist(engine) > 0) ? randomEdge->getSource() : randomEdge->getTarget();
}

std::string RandomBridge::getName() {
    return "RandomBridge";
}