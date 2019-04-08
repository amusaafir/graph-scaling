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

    return sampledGraph;
}

void RandomEdge::edgeSamplingStep(std::unordered_set<long long>& samplesVertices, std::vector<Edge<long long>>& sampledEdges, float fraction) {
    std::cout << "Performing edge sampling." << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    long long preferredEdgesSize = graph->getEdges().size() * fraction;
    long long edgeSizeOriginalGraph = graph->getEdges().size();
    std::unordered_set<long long> sampledEdgeIndices;

    while (sampledEdges.size() < preferredEdgesSize) {
        long long randomEdgeIndex = getRandomIntBetweenRange(0, edgeSizeOriginalGraph - 1);

        // Check if the edge (index) has already been sampled before. If it is isn't, collect the edge.
        if (!sampledEdgeIndices.count(randomEdgeIndex)) {
            Edge<long long int> edge = graph->getEdges()[randomEdgeIndex];
            sampledEdges.push_back(edge);
            sampledEdgeIndices.insert(randomEdgeIndex);
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed - edge sampling: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" <<std::endl;
    std::cout << "Finished performing edge sampling: "
                 "collected " << sampledEdges.size() << " sampled edges." << std::endl;
}