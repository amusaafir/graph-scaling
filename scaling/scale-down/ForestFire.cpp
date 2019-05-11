//
// Created by Ahmed on 10-5-19.
//

#include <queue>
#include <chrono>
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

    // If the sourceVertex is not given, pick a random one. This will hit any disconnected component eventually.
    if (sourceVertex == -1) {
        sourceVertex = graph->getEdges()[getRandomIntBetweenRange(0, graph->getEdges().size() - 1)].getSource();
    }

    assert(sourceVertex != -1);

    std::queue<long long> queue;

    std::unordered_set<long long> sampledVertices;
    std::vector<Edge<long long>> sampledEdges;

    queue.push(sourceVertex);

    long long desiredNumberOfEdges = graph->getEdges().size() * fraction;
    std::unordered_map<long long, std::vector<long long>> adjacencyList = graph->getAdjacencyList();

    assert(!queue.empty());

    bool reachedDesiredNumberOfEdges = false;

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


            //sampledVertices.insert(targetVertex);
            queue.push(targetVertex);


            std::cout<< sourceVertex << ", " << targetVertex<< std::endl;


            if (sampledEdges.size() >= 50) {
                reachedDesiredNumberOfEdges = true;
                break;
            }

            sampledEdges.push_back(Edge<long long>(sourceVertex, targetVertex));

            // if  (isUndirected) {
            //    sampledEdges.push_back(Edge<long long>(targetVertex, sourceVertex));
            //}
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Finished performing Forest Fire: "
            "collected " << sampledVertices.size() << " sampled vertices and " <<  sampledEdges.size()  << " edges." << std::endl;

    sourceVertex = -1; // Reset for further samples

    Graph* sampledGraph = new Graph(); // TODO: Delete this later
    sampledGraph->setVertices(sampledVertices);
    sampledGraph->setEdges(sampledEdges);

    return sampledGraph;
}