//
// Created by Ahmed on 12-12-17.
//

#include "HighDegreeBridge.h"

HighDegreeBridge::HighDegreeBridge(long long numberOfInterconnections, bool forceUndirectedEdges)  : Bridge(numberOfInterconnections, addDirectedBridges) {
}


long long HighDegreeBridge::getRandomHighDegreeVertex(Graph* graph) {
    // There already exists some high degree nodes here, so just select them for this set.
    if (graph->getHighDegreeVertices().size() > 0) {
        std::mt19937 engine(seed());

        // Select random high degree vertex
        std::uniform_int_distribution<long long> randomHighDegreeVertexDist(0, graph->getHighDegreeVertices().size() - 1);

        return graph->getHighDegreeVertices()[randomHighDegreeVertexDist(engine)];
    }

    // Since we didn't collect any high degree vertices for this graph, collect them and collect one from it.
    collectHighDegreeVertices(graph);

    return getRandomHighDegreeVertex(graph);
}

// Map all vertices onto a map along with their degree
void HighDegreeBridge::collectHighDegreeVertices(Graph *graph)  {
    std::unordered_map<long long, long long> nodeDegreeMap;

    for (auto &edge : graph->getEdges()) {
        ++nodeDegreeMap[edge.getSource()];
        ++nodeDegreeMap[edge.getTarget()];
    }

    // Convert the map to a vector
    std::vector<std::pair<long long, long long>> nodeDegreesVect(nodeDegreeMap.begin(), nodeDegreeMap.end());

    // Sort the vector (ascending, high degree nodes are on top)
    sort(nodeDegreesVect.begin(), nodeDegreesVect.end(), [](const std::pair<int, int> &left, const std::pair<long long, long long> &right) {
        return left.second > right.second;
    });

    // Collect only the nodes (half of the total nodes) that have a high degree
    for (long long i = 0; i < nodeDegreesVect.size() / 2; i++) {
        graph->getHighDegreeVertices().push_back(nodeDegreesVect[i].first);
    }
}

void HighDegreeBridge::addBridgesBetweenGraphs(Graph *sourceGraph, Graph *targetGraph, std::vector<Edge<std::string>>& bridges) {
    for (long long i = 0; i < numberOfInterconnections; i++) {
        long long vertexSource = getRandomHighDegreeVertex(sourceGraph);
        long long vertexTarget = getRandomHighDegreeVertex(targetGraph);

        bridges.push_back(Edge<std::string>(std::to_string(vertexSource) + sourceGraph->getIdentifier(),
                                            std::to_string(vertexTarget) + targetGraph->getIdentifier()));

        if (!addDirectedBridges) {
            bridges.push_back(Edge<std::string>(std::to_string(vertexTarget) + targetGraph->getIdentifier(),
                                                std::to_string(vertexSource) + sourceGraph->getIdentifier()                                                    ));
        }
    }
}

std::string HighDegreeBridge::getName() {
    return "HighDegreeBridge";
}