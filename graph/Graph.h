// Created by Ahmed on 12-11-17.

#ifndef GRAPH_SCALING_TOOL_GRAPH_H
#define GRAPH_SCALING_TOOL_GRAPH_H

#include <vector>
#include <unordered_set>
#include "Edge.h"

class Graph {
private:
    std::string identifier;
    std::unordered_set<long long> vertices;
    std::vector<Edge<long long>> edges;
    std::vector<long long> highDegreeVertices;

public:
    void addVertex(long long vertex);

    void addEdge(long long source, long long target);

    const std::unordered_set<long long> &getVertices() const;

    void setVertices(const std::unordered_set<long long> &vertices);

    const std::vector<Edge<long long>> &getEdges() const;

    void setEdges(const std::vector<Edge<long long>> &edges);

    const std::string &getIdentifier() const;

    void setIdentifier(const std::string &identifier);

    std::vector<long long> &getHighDegreeVertices();

    void setHighDegreeVertices(std::vector<long long> &highDegreeVertices);
};


#endif //GRAPH_SCALING_TOOL_GRAPH_H
