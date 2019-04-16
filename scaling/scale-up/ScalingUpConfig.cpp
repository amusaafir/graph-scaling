//
// Created by Ahmed on 16-11-17.
//

#include "ScalingUpConfig.h"

ScalingUpConfig::ScalingUpConfig(float scalingFactor, float samplingFraction, Topology* topology, Sampling* samplingAlgorithm) {
    this->scalingFactor = scalingFactor;
    this->samplingFraction = samplingFraction;
    this->topology = topology;
    this->samplingAlgorithm = samplingAlgorithm;

    float rem = (long long)round(scalingFactor * 100) % (long long)round(samplingFraction * 100) / 100.0;
    setRemainder(rem);
    int nrSamples = (scalingFactor*1000) / (samplingFraction*1000);
    setAmountOfSamples(nrSamples);
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

Sampling* ScalingUpConfig::getSamplingAlgorithm() {
    return samplingAlgorithm;
}