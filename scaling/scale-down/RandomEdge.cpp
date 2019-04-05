//
// Created by Ahmed on 3-4-19.
//

#include "RandomEdge.h"

Graph* RandomEdge::sample(float fraction) {
    if (fraction == 1.0) {
        return getFullGraphCopy();
    }

    std::unordered_set<long long> sampledVertices;
    std::vector<Edge<long long>> sampledEdges;
    edgeSamplingStep(sampledVertices, sampledEdges, fraction);

    Graph* sampledGraph = new Graph(); // TODO: Delete this later
    sampledGraph->setVertices(sampledVertices);
    sampledGraph->setEdges(sampledEdges); // convert to vector or use another vector
}

void RandomEdge::edgeSamplingStep(std::unordered_set<long long>& samplesVertices, std::vector<Edge<long long>>& sampledEdges, float fraction) {
    long long preferredEdgesSize = graph->getEdges().size() * fraction;

    std::unordered_map<long long, std::unordered_set<long long>> collectedEdges; // source, target


    while (sampledEdges.size() < preferredEdgesSize) {
        Edge<long long> sampledEdge = getRandomEdge();

        // Edge does not exist
        if (!(collectedEdges.count(sampledEdge.getSource()) &&
            collectedEdges[sampledEdge.getSource()].count(sampledEdge.getTarget()))) {


        }
    }
}

// TODO: Put in base class (as this function is also used in TIES)
Edge<long long> RandomEdge::getRandomEdge() {
    return graph->getEdges()[getRandomIntBetweenRange(0, graph->getEdges().size() - 1)];
}