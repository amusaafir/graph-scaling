//
// Created by Ahmed on 16-11-17.
//

#include "ScaleUpSamplesInfo.h"

ScaleUpSamplesInfo::ScaleUpSamplesInfo(Topology* topology, float scalingFactor, float samplingFraction) {
    this->topology = topology;
    this->scalingFactor = scalingFactor;
    this->samplingFraction = samplingFraction;

    setRemainder(fmod(scalingFactor, samplingFraction));
    setAmountOfSamples(scalingFactor / samplingFraction);
    determineAdditionalSampling();
}

bool ScaleUpSamplesInfo::hasSamplingRemainder() const {
    return getRemainder() > 0;
}

float ScaleUpSamplesInfo::getRemainder() const {
    return remainder;
}

void ScaleUpSamplesInfo::setRemainder(float remainder) {
    this->remainder = remainder;
}

int ScaleUpSamplesInfo::getAmountOfSamples() const {
    return amountOfSamples;
}

void ScaleUpSamplesInfo::setAmountOfSamples(int amountOfSamples) {
    this->amountOfSamples = amountOfSamples;
}

void ScaleUpSamplesInfo::determineAdditionalSampling() {
    if (hasSamplingRemainder()) {
        this->amountOfSamples++;
    }
}

float ScaleUpSamplesInfo::getScalingFactor() {
    return scalingFactor;
}

float ScaleUpSamplesInfo::getSamplingFraction() {
    return samplingFraction;
}