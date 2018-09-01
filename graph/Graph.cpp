//
// Created by Ahmed on 12-11-17.
//

#include "Graph.h"

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

void Graph::setHighDegreeVertices(std::vector<long long> &highDegreeVertices) {
    Graph::highDegreeVertices = highDegreeVertices;
}
