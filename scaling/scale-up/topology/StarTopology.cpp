//
// Created by Ahmed on 3-1-18.
//

#include "StarTopology.h"

StarTopology::StarTopology(Bridge* bridge)
        : Topology(bridge) {}

std::vector<Edge<std::string>> StarTopology::getBridgeEdges(std::vector<Graph*> samples) {
    Graph* coreGraph = samples.front(); // Use the first graph sample as the core of the star

    std::vector<Edge<std::string>> bridges;

    for (int i = 1; i < samples.size(); i++) {
        bridge->addBridgesBetweenGraphs(samples[i], coreGraph, bridges);
    }

    return bridges;
}

std::string StarTopology::getName() {
    return "StarTopology";
}