//
// Created by Ahmed on 15-11-17.
//

#include "ScaleUp.h"

ScaleUp::ScaleUp(Graph* graph, Sampling* sampling, ScalingUpConfig* scaleUpSamplesInfo, std::string outputFolder) {
    this->graph = graph;
    this->sampling = sampling;
    this->scaleUpSamplesInfo = scaleUpSamplesInfo;
    this->outputFolder = outputFolder;
}

void ScaleUp::run() {
    printScaleUpSetup();

    std::vector<Graph*> samples = createDistinctSamples();

    std::vector<Edge<std::string>> bridges = scaleUpSamplesInfo->getTopology()->getBridgeEdges(samples);

    WriteScaledUpGraph* writeScaledUpGraph = new WriteScaledUpGraph(outputFolder, samples, bridges);
    writeScaledUpGraph->writeToFile(scaleUpSamplesInfo);

    delete(writeScaledUpGraph);
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
    std::vector<Graph*> samples;
    IdentifierTracker identifierTracker;

    for (int i = 0; i < scaleUpSamplesInfo->getAmountOfSamples(); i++) {
        std::cout << "\n(" << i + 1 << "/" << scaleUpSamplesInfo->getAmountOfSamples()  << ")" << std::endl;

        if (shouldSampleRemainder(scaleUpSamplesInfo, i)) {
            createSample(samples, scaleUpSamplesInfo->getRemainder(), identifierTracker.createNewIdentifier());
            break;
        }

        createSample(samples, scaleUpSamplesInfo->getSamplingFraction(), identifierTracker.createNewIdentifier());
    }

    return samples;
}

void ScaleUp::createSample(std::vector<Graph*> &samples, float samplingFraction, std::string identifier) {
    Graph* sampledGraph = sampling->sample(samplingFraction);
    sampledGraph->setIdentifier(identifier);
    samples.push_back(sampledGraph);
}

bool ScaleUp::shouldSampleRemainder(ScalingUpConfig *scaleUpSamplesInfo, int currentLoopIteration) {
    bool isLastSamplingIteration = currentLoopIteration == (scaleUpSamplesInfo->getAmountOfSamples() - 1);

    return isLastSamplingIteration && scaleUpSamplesInfo->hasSamplingRemainder();
}