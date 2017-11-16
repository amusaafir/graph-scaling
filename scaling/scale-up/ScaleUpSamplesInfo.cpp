//
// Created by Ahmed on 16-11-17.
//

#include "ScaleUpSamplesInfo.h"

ScaleUpSamplesInfo::ScaleUpSamplesInfo(float scalingFactor, float samplingFraction) {
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
