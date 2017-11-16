//
// Created by Ahmed on 12-11-17.
//

#include "Graph.h"

void Graph::addVertex(int vertex) {
    vertices.insert(vertex);
}

void Graph::addEdge(int source, int target) {
    Edge edge(source, target);
    edges.push_back(edge);
}

const std::unordered_set<int> &Graph::getVertices() const {
    return vertices;
}

void Graph::setVertices(const std::unordered_set<int> &vertices) {
    Graph::vertices = vertices;
}

const std::vector<Edge> &Graph::getEdges() const {
    return edges;
}

void Graph::setEdges(const std::vector<Edge> &edges) {
    Graph::edges = edges;
}

const std::string &Graph::getIdentifier() const {
    return identifier;
}

void Graph::setIdentifier(const std::string &identifier) {
    Graph::identifier = identifier;
}
