// Created by Ahmed on 12-11-17.

#ifndef GRAPH_SCALING_TOOL_GRAPH_H
#define GRAPH_SCALING_TOOL_GRAPH_H

#include <vector>
#include <unordered_set>
#include "Edge.h"

class Graph {
private:
    std::string identifier;
    std::unordered_set<int> vertices;
    std::vector<Edge> edges;

public:
    void addVertex(int vertex);

    void addEdge(int source, int target);

    const std::unordered_set<int> &getVertices() const;

    void setVertices(const std::unordered_set<int> &vertices);

    const std::vector<Edge> &getEdges() const;

    void setEdges(const std::vector<Edge> &edges);

    const std::string &getIdentifier() const;

    void setIdentifier(const std::string &identifier);
};


#endif //GRAPH_SCALING_TOOL_GRAPH_H
