//
// Created by Ahmed on 6-1-18.
//

#include "FullyConnectedTopology.h"

FullyConnectedTopology::FullyConnectedTopology(Bridge* bridge) : Topology(bridge) {}

std::vector<Edge<std::string>> FullyConnectedTopology::getBridgeEdges(std::vector<Graph*> samples) {
    std::vector<Edge<std::string>> bridges;

    for (int i = 0; i < samples.size(); i++) {
        for (int p = 0; p < samples.size(); p++) {
            if (i == p) { // Skip, since we don't want to add bridges to the same graph
                continue;
            }

            bridge->addBridgesBetweenGraphs(samples[i], samples[p], bridges);
        }
    }

    return bridges;
}

std::string FullyConnectedTopology::getName() {
    return "FullyConnected";
}