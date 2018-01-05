//
// Created by Ahmed on 12-11-17.
//

#include "GraphLoader.h"

Graph* GraphLoader::loadGraph(std::string path) {
    std::cout << "\nLoad graph: " << path << std::endl;

    Graph* graph = new Graph();
    std::ifstream infile(path);
    readLines(graph, infile);

    std::cout << "Finished loading the graph (" << graph->getVertices().size()
              << " vertices and " << graph->getEdges().size() << " edges)." << std::endl;

    return graph;
}

void GraphLoader::readLines(Graph *graph, std::ifstream &infile) const {
    std::string line;

    while (getline(infile, line)) {
        std::istringstream iss(line);
        int source, target;
        if (!(iss >> source >> target)) {
            continue;
        }

        graph->addVertex(source);
        graph->addVertex(target);
        graph->addEdge(source, target);
    }
}