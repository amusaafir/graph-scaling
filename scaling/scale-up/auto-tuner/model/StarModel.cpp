//
// Created by Ahmed on 22-7-19.
//

#include "StarModel.h"
#include "../../topology/Topology.h"
#include "../../topology/StarTopology.h"

StarModel::StarModel(int originalDiameter, int numberOfSamples, float scalingFactor)
        : Model(originalDiameter, numberOfSamples, scalingFactor) {}

int StarModel::getMaxDiameter() {
    return 3 * originalDiameter + 2;
}

std::string StarModel::getName() {
    return "StarModel";
}

Topology* StarModel::createTopology(Bridge* bridge) {
    return new StarTopology(bridge);
}