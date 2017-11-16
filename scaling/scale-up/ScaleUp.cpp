//
// Created by Ahmed on 15-11-17.
//

#include "ScaleUp.h"

ScaleUp::ScaleUp(Graph* graph, Sampling* sampling, float samplingFraction) {
    this->graph = graph;
    this->sampling = sampling;
    this->samplingFraction = samplingFraction;
}

void ScaleUp::executeScaleUp(float scalingFactor) {
    int amountOfSamples = scalingFactor / samplingFraction;
    float remainingFraction = fmod(scalingFactor, samplingFraction);
    bool shouldSampleRemainder = shouldSampleRemainingFraction(amountOfSamples, remainingFraction);

    std::cout << "Performing scale-up using a scaling factor of " << scalingFactor
              << " and sample size of " << samplingFraction
              << " (" << amountOfSamples << " different samples)" << "." << std::endl;

    createDifferentSamples(amountOfSamples, remainingFraction, shouldSampleRemainder);
}

void ScaleUp::createDifferentSamples(int amountOfSamples, float remainingFraction, bool shouldSampleRemainder) const {
    for (int i = 0; i < amountOfSamples; i++) {
        std::cout << "\n(" << i + 1 << "/" << amountOfSamples << ")" << std::endl;

        if (shouldSampleRemainder && i == (amountOfSamples-1)) {
            sampling->sample(remainingFraction);
            break;
        }

       sampling->sample(samplingFraction);
    }
}

bool ScaleUp::shouldSampleRemainingFraction(int &amountOfSamples, float remainingFraction) {
    if (remainingFraction > 0) {
        amountOfSamples++;
        return true;
    }

    return false;
}