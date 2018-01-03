//
// Created by Ahmed on 15-11-17.
//

#include "ScaleUp.h"
#include "identifier/IdentifierTracker.h"

ScaleUp::ScaleUp(Graph* graph, Sampling* sampling, float samplingFraction) {
    this->graph = graph;
    this->sampling = sampling;
    this->samplingFraction = samplingFraction;
}

void ScaleUp::executeScaleUp(float scalingFactor) {
    ScaleUpSamplesInfo* scaleUpSamplesInfo = new ScaleUpSamplesInfo(scalingFactor, samplingFraction);

    printScaleUpSetup(scalingFactor, scaleUpSamplesInfo);

    std::vector<Graph*> samples = createDistinctSamples(scaleUpSamplesInfo);

    std::cout << "Length sample size: " << samples.size() << std::endl;

    delete(scaleUpSamplesInfo);
}

void ScaleUp::printScaleUpSetup(float scalingFactor, const ScaleUpSamplesInfo *scaleUpSamplesInfo) {
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

std::vector<Graph*> ScaleUp::createDistinctSamples(ScaleUpSamplesInfo *scaleUpSamplesInfo) {
    IdentifierTracker* identifierTracker = new IdentifierTracker();
    std::vector<Graph*> samples;

    for (int i = 0; i < scaleUpSamplesInfo->getAmountOfSamples(); i++) {
        std::cout << "\n(" << i + 1 << "/" << scaleUpSamplesInfo->getAmountOfSamples()  << ")" << std::endl;

        if (shouldSampleRemainder(scaleUpSamplesInfo, i)) {
            createSample(identifierTracker, samples, scaleUpSamplesInfo->getRemainder());
            break;
        }

        createSample(identifierTracker, samples, samplingFraction);
    }

    return samples;
}

void ScaleUp::createSample(IdentifierTracker *identifierTracker, std::vector<Graph*> &samples, float samplingFraction) {
    Graph* sampledGraph = sampling->sample(samplingFraction);
    sampledGraph->setIdentifier(identifierTracker->getNewIdentifier());

    samples.push_back(sampledGraph);
}

bool ScaleUp::shouldSampleRemainder(ScaleUpSamplesInfo *scaleUpSampleRemainder, int currentLoopIteration) {
    bool isLastSamplingIteration = currentLoopIteration == (scaleUpSampleRemainder->getAmountOfSamples() - 1);

    return isLastSamplingIteration && scaleUpSampleRemainder->hasSamplingRemainder();
}