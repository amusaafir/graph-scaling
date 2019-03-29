//
// Created by Ahmed on 12-12-17.
//

#include "HighDegreeBridge.h"

HighDegreeBridge::HighDegreeBridge(long long numberOfInterconnections, bool forceUndirectedEdges)  : Bridge(numberOfInterconnections, addDirectedBridges) {
}

/**
 * Return a single random high degree vertex from a graph (note: this method assumes that a list of high degree
 * vertices were already collected using the collectHighDegreeVertices function.
 * @param graph
 * @return a random high degree vertex from a graph
 */
long long HighDegreeBridge::getRandomHighDegreeVertex(Graph* graph) {
    std::mt19937 engine(seed());

    // Select random high degree vertex
    std::uniform_int_distribution<long long> randomHighDegreeVertexDist(0, graph->getHighDegreeVertices().size() - 1);

    return graph->getHighDegreeVertices()[randomHighDegreeVertexDist(engine)];

}

/**
 * Maps all vertices onto a map along with their degree. Add the high degree vertices to the given graph.
 * @param graph - Graph in which the high degree vertices will be assigned to.
 */
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

/**
 * Adds bridges given two graphs.
 * @param sourceGraph
 * @param targetGraph
 * @param bridges
 */
void HighDegreeBridge::addBridgesBetweenGraphs(Graph *sourceGraph, Graph *targetGraph, std::vector<Edge<std::string>>& bridges) {
    if (sourceGraph->getHighDegreeVertices().empty()) {
        collectHighDegreeVertices(sourceGraph);
    }

    if (targetGraph->getHighDegreeVertices().empty()) {
        collectHighDegreeVertices(targetGraph);
    }

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