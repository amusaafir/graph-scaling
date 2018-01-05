//
// Created by Ahmed on 15-11-17.
//

#include "ScaleUp.h"
#include "IdentifierTracker.h"

ScaleUp::ScaleUp(Graph* graph, Sampling* sampling, ScaleUpSamplesInfo* scaleUpSamplesInfo) {
    this->graph = graph;
    this->sampling = sampling;
    this->scaleUpSamplesInfo = scaleUpSamplesInfo;
}

void ScaleUp::executeScaleUp() {
    printScaleUpSetup();

    std::vector<Graph*> samples = createDistinctSamples();

    std::cout << "Length sample size: " << samples.size() << std::endl;

    // TODO: star implementation

    delete(scaleUpSamplesInfo);
}

void ScaleUp::printScaleUpSetup() {
    std::cout << "Performing scale-up using a scaling factor of " << scaleUpSamplesInfo->getScalingFactor()
              << " and sample size of " << scaleUpSamplesInfo->getSamplingFraction() << ", resulting in "
              << scaleUpSamplesInfo->getAmountOfSamples() << " different samples ";

    if (scaleUpSamplesInfo->hasSamplingRemainder()) {
        std::cout << "(" << scaleUpSamplesInfo->getAmountOfSamples() - 1 << " x " << scaleUpSamplesInfo->getSamplingFraction()
                  << " and 1 x " << scaleUpSamplesInfo->getRemainder() << ")." << std::endl;
    } else {
        std::cout << "of size " << scaleUpSamplesInfo->getSamplingFraction() << " (for each sample)." << std::endl;
    }
}

std::vector<Graph*> ScaleUp::createDistinctSamples() {
    IdentifierTracker* identifierTracker = new IdentifierTracker();
    std::vector<Graph*> samples;

    for (int i = 0; i < scaleUpSamplesInfo->getAmountOfSamples(); i++) {
        std::cout << "\n(" << i + 1 << "/" << scaleUpSamplesInfo->getAmountOfSamples()  << ")" << std::endl;

        if (shouldSampleRemainder(scaleUpSamplesInfo, i)) {
            createSample(identifierTracker, samples, scaleUpSamplesInfo->getRemainder());
            break;
        }

        createSample(identifierTracker, samples, scaleUpSamplesInfo->getSamplingFraction());
    }

    return samples;
}

void ScaleUp::createSample(IdentifierTracker *identifierTracker, std::vector<Graph*> &samples, float samplingFraction) {
    Graph* sampledGraph = sampling->sample(samplingFraction);
    sampledGraph->setIdentifier(identifierTracker->createNewIdentifier());
    samples.push_back(sampledGraph);
}

bool ScaleUp::shouldSampleRemainder(ScaleUpSamplesInfo *scaleUpSamplesInfo, int currentLoopIteration) {
    bool isLastSamplingIteration = currentLoopIteration == (scaleUpSamplesInfo->getAmountOfSamples() - 1);

    return isLastSamplingIteration && scaleUpSamplesInfo->hasSamplingRemainder();
}