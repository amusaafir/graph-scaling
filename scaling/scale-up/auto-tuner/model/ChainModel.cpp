//
// Created by Ahmed on 22-7-19.
//

#include "ChainModel.h"

ChainModel::ChainModel(int originalDiameter, int numberOfSamples, float scalingFactor)
        : Model(originalDiameter, numberOfSamples, scalingFactor) {}

int ChainModel::getMaxDiameter() {
    float diameter = numberOfSamples - 1;

    for (int n = 0; n < numberOfSamples; n++) {
        diameter += originalDiameter;
    }

    return diameter;
}

std::string ChainModel::getName() {
    return "ChainModel";
}