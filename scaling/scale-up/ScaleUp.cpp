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
    ScaleUpSamplesInfo* scaleUpSamplesInfo = new ScaleUpSamplesInfo(scalingFactor, samplingFraction);

    printScaleUpSetup(scalingFactor, scaleUpSamplesInfo);

    createDifferentSamples(scaleUpSamplesInfo);

    delete(scaleUpSamplesInfo);
}

void ScaleUp::printScaleUpSetup(float scalingFactor, const ScaleUpSamplesInfo *scaleUpSamplesInfo) const {
    std::cout << "Performing scale-up using a scaling factor of " << scalingFactor
              << " and sample size of " << samplingFraction << ", resulting in "
              << scaleUpSamplesInfo->getAmountOfSamples() << " different samples ";

    if (scaleUpSamplesInfo->hasSamplingRemainder()) {
        std::cout << "(" << scaleUpSamplesInfo->getAmountOfSamples() - 1 << " x " << samplingFraction
                  << " and 1 x " << scaleUpSamplesInfo->getRemainder() << ")." << std::endl;
    } else {
        std::cout << "of size " << samplingFraction << " (for each sample)." << std::endl;
    }
}

void ScaleUp::createDifferentSamples(ScaleUpSamplesInfo* scaleUpSamplesInfo) const {
    for (int i = 0; i < scaleUpSamplesInfo->getAmountOfSamples(); i++) {
        std::cout << "\n(" << i + 1 << "/" << scaleUpSamplesInfo->getAmountOfSamples()  << ")" << std::endl;

        if (shouldApplyRemainderSampling(scaleUpSamplesInfo, i)) {
            sampling->sample(scaleUpSamplesInfo->getRemainder());
            break;
        }

        Graph* sampledGraph = sampling->sample(samplingFraction);
        sampledGraph->setIdentifier("aa");
        //delete(sampledGraph);
    }
}

bool ScaleUp::shouldApplyRemainderSampling(const ScaleUpSamplesInfo *scaleUpSampleRemainder, int currentLoopIteration) const {
    return currentLoopIteration == (scaleUpSampleRemainder->getAmountOfSamples() - 1)
           && scaleUpSampleRemainder->hasSamplingRemainder();
}