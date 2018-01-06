//
// Created by Ahmed on 6-1-18.
//

#include "RingTopology.h"

RingTopology::RingTopology(Bridge* bridge) : Topology(bridge) {}

std::vector<Edge<std::string>> RingTopology::getBridgeEdges(std::vector<Graph*> samples) {
    std::vector<Edge<std::string>> bridges;

    for (int i = 0; i < samples.size() - 1; i++) {
        bridge->addBridgesBetweenGraphs(samples[i], samples[i+1], bridges);
    }

    // Connect the last sample with the first sample
    bridge->addBridgesBetweenGraphs(samples.back(), samples.front(), bridges);

    return bridges;
}

std::string RingTopology::getName() {
    return "RingTopology";
}