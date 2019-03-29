//
// Created by Ahmed on 16-11-17.
//

#include "ScalingUpConfig.h"

ScalingUpConfig::ScalingUpConfig(float scalingFactor, float samplingFraction, Topology* topology) {
    this->scalingFactor = scalingFactor;
    this->samplingFraction = samplingFraction;
    this->topology = topology;

    setRemainder(fmod(scalingFactor, samplingFraction));
    setAmountOfSamples(scalingFactor / samplingFraction);
    determineAdditionalSampling();
}

bool ScalingUpConfig::hasSamplingRemainder() const {
    return getRemainder() > 0;
}

float ScalingUpConfig::getRemainder() const {
    return remainder;
}

void ScalingUpConfig::setRemainder(float remainder) {
    this->remainder = remainder;
}

int ScalingUpConfig::getAmountOfSamples() const {
    return amountOfSamples;
}

void ScalingUpConfig::setAmountOfSamples(int amountOfSamples) {
    this->amountOfSamples = amountOfSamples;
}

void ScalingUpConfig::determineAdditionalSampling() {
    if (hasSamplingRemainder()) {
        this->amountOfSamples++;
    }
}

float ScalingUpConfig::getScalingFactor() {
    return scalingFactor;
}

float ScalingUpConfig::getSamplingFraction() {
    return samplingFraction;
}

Topology* ScalingUpConfig::getTopology() {
    return topology;
}

void ScalingUpConfig::setTopology(Topology* topology) {
    this->topology = topology;
}