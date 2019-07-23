//
// Created by Ahmed on 22-7-19.
//

#include "RingModel.h"

RingModel::RingModel(int originalDiameter, int numberOfSamples, float scalingFactor)
        : Model(originalDiameter, numberOfSamples, scalingFactor) {}

int RingModel::getMaxDiameter() {
    float diameter = numberOfSamples;

    for (int n = 0; n < numberOfSamples; n++) {
        diameter += originalDiameter;
    }

    return diameter;
}

std::string RingModel::getName() {
    return "RingModel";
}