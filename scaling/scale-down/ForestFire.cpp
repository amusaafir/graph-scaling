//
// Created by Ahmed on 10-5-19.
//

#include <queue>
#include <chrono>
#include <set>
#include <assert.h>
#include "ForestFire.h"

// Fraction based on edges or nodes?
// If not connected graph: Jump without interconnection option
// TODO: what if stuck
Graph* ForestFire::sample(float fraction) {
    if (fraction == 1.0) {
        return getFullGraphCopy();
    }

    std::cout << "Performing forest fire." << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Create adjacency list (if not already done)
    if (graph->getAdjacencyList().size() == 0) {
        graph->createAdjacencyList();
    }

    const bool isFractionBasedOnEdges = true;
    const bool enableJumpWithInterconnection = true;
    const bool isUndirected = true;


    bool reachedDesiredNumberOfEdges = false;

    std::unordered_set<long long> sampledVertices;
    std::set<std::pair<long long, long long>> uniqueSampledEdgesSet;

    long long lastSourceVertex = -1;

    while (!reachedDesiredNumberOfEdges) {
        // If the sourceVertex is not given, pick a random one. This will hit any disconnected component eventually.
        if (sourceVertex == -1) {
            sourceVertex = graph->getEdges()[getRandomIntBetweenRange(0, graph->getEdges().size() - 1)].getSource();
        }
        std::cout<< "Selected source vertex: " << sourceVertex  << std::endl;

        // If option is enabled, connect the disconnected graphs after a jump
        if (enableJumpWithInterconnection && lastSourceVertex != -1) {
            uniqueSampledEdgesSet.insert(std::pair<long long, long long>(lastSourceVertex, sourceVertex));

            if (isUndirected) {
                uniqueSampledEdgesSet.insert(std::pair<long long, long long>(sourceVertex, lastSourceVertex));
            }
        }

        lastSourceVertex = sourceVertex;

        std::queue<long long> queue;

        queue.push(sourceVertex);

        long long desiredNumberOfEdges = (isUndirected) ? graph->getEdges().size() * fraction : graph->getEdges().size() * fraction;

        std::unordered_map<long long, std::vector<long long>> adjacencyList = graph->getAdjacencyList();

        assert(!queue.empty());

        while (!queue.empty() && !reachedDesiredNumberOfEdges) {
            long long sourceVertex = queue.front();
            queue.pop();

            if (sampledVertices.count(sourceVertex)) {
                continue;
            }

            sampledVertices.insert(sourceVertex);

            // Loop through the vertex' neighbors, add the target vertex and edge to the sampled graph
            for (int i = 0; i < adjacencyList[sourceVertex].size(); i++) {

                long long targetVertex = adjacencyList[sourceVertex][i];

                queue.push(targetVertex);

                if (uniqueSampledEdgesSet.size() >= desiredNumberOfEdges) {
                    reachedDesiredNumberOfEdges = true;
                    break;
                }


                //sampledVertices.insert(targetVertex);

                if (isUndirected) {
                    if (!uniqueSampledEdgesSet.count(std::pair<long long, long long>(targetVertex, sourceVertex))) {
                        uniqueSampledEdgesSet.insert(std::pair<long long, long long>(sourceVertex, targetVertex));
                    }
                } else {
                    uniqueSampledEdgesSet.insert(std::pair<long long, long long>(sourceVertex, targetVertex));
                }
            }
        }
/*
        if (!queue.empty()) {
            std::cout << "Adding " << sampledVertices.size() << " remaining vertices to the sampled set." << std::endl;

            while (!queue.empty()) {
                long long v = queue.front();
                sampledVertices.insert(v);
                queue.pop();
            }
        }
*/
        sourceVertex = graph->getEdges()[getRandomIntBetweenRange(0, graph->getEdges().size() - 1)].getSource();
    }

    sourceVertex = -1; // Reset for further samples

    sampledVertices.clear();

    // Convert edge set to vector
    std::vector<Edge<long long>> sampledEdges;
    for (std::pair<long long, long  long> e : uniqueSampledEdgesSet) {
        sampledEdges.push_back(Edge<long long> (e.first, e.second));
        sampledVertices.insert(e.first);
        sampledVertices.insert(e.second);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed - forest fire sampling: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" <<std::endl;
    std::cout << "Finished performing Forest Fire: "
            "collected " << sampledVertices.size() << " sampled vertices and " << uniqueSampledEdgesSet.size()
              << " edges." << std::endl;

    Graph* sampledGraph = new Graph(); // TODO: Delete this later
    sampledGraph->setVertices(sampledVertices);
    sampledGraph->setEdges(sampledEdges);

    return sampledGraph;
}