//
// Created by Ahmed on 6-1-18.
//

#include "ChainTopology.h"

ChainTopology::ChainTopology(Bridge* bridge) : Topology(bridge) {}

std::vector<Edge<std::string>> ChainTopology::getBridgeEdges(std::vector<Graph*> samples) {
    std::vector<Edge<std::string>> bridges;

    for (int i = 0; i < samples.size() - 1; i++) {
        bridge->addBridgesBetweenGraphs(samples[i], samples[i + 1], bridges);
    }

    return bridges;
}

std::string ChainTopology::getName() {
    return "ChainTopology";
}