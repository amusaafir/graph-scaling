//
// Created by Ahmed on 12-11-17.
//

#include <iostream>
#include "Graph.h"
#include <assert.h>

void Graph::addVertex(long long vertex) {
    vertices.insert(vertex);
}

void Graph::addEdge(long long source, long long target) {
    Edge<long long> edge(source, target);
    edges.push_back(edge);
}

const std::unordered_set<long long> &Graph::getVertices() const {
    return vertices;
}

void Graph::setVertices(const std::unordered_set<long long> &vertices) {
    Graph::vertices = vertices;
}

const std::vector<Edge<long long>> &Graph::getEdges() const {
    return edges;
}

void Graph::setEdges(const std::vector<Edge<long long>> &edges) {
    Graph::edges = edges;
}

const std::string &Graph::getIdentifier() const {
    return identifier;
}

void Graph::setIdentifier(const std::string &identifier) {
    Graph::identifier = identifier;
}

std::vector<long long> &Graph::getHighDegreeVertices() {
    return highDegreeVertices;
}

// TODO: Check if it is an undirected graph
void Graph::createAdjacencyList() {
    const bool IS_UNDIRECTED = true;

    std::cout << "Converting graph to adjacency list." << std::endl;

    std::vector<Edge<long long>> edges = getEdges();

    assert(edges.size() != 0);

    for (long long i = 0; i < edges.size(); i++) {
        Edge<long long> edge = edges[i];

        long long source = edge.getSource();
        long long target = edge.getTarget();


        addEdgeToAdjacencyList(source, target);

        if (IS_UNDIRECTED) {
            addEdgeToAdjacencyList(target, source);
        }
    }

    std::cout << "Finished conversion to adjacency list." << std::endl;
}

void Graph::addEdgeToAdjacencyList(long long int source, long long int target) {
    if (adjacencyList.count(source)) {
        adjacencyList[source].push_back(target);
    } else {
        std::vector<long long> neighbors;
        neighbors.push_back(target);
        adjacencyList[source] = neighbors;
    }
}

const std::unordered_map<long long, std::vector<long long>> &Graph::getAdjacencyList() const {
    return adjacencyList;
}