//
// Created by Ahmed on 22-7-19.
//

#include "FullyConnectedModel.h"

FullyConnectedModel::FullyConnectedModel(int originalDiameter, int numberOfSamples, float scalingFactor)
        : Model(originalDiameter, numberOfSamples, scalingFactor) {}

int FullyConnectedModel::getMaxDiameter() {
    return originalDiameter * 2 + 1;
}

std::string FullyConnectedModel::getName() {
    return "FullyConnectedModel";
}