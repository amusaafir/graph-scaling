//
// Created by Ahmed on 15-11-17.
//

#include "ScaleUp.h"
#include "auto-tuner/Autotuner.h"

ScaleUp::ScaleUp(Graph* graph, ScalingUpConfig* scaleUpSamplesInfo, std::string outputFolder) {
    this->graph = graph;
    this->scaleUpSamplesInfo = scaleUpSamplesInfo;
    this->outputFolder = outputFolder;
}

void ScaleUp::run() {
    printScaleUpSetup();

    std::vector<Graph*> samples = createDistinctSamples();

    bool isAutotunerEnabled = true; // TODO: Add to user input

    std::vector<Edge<std::string>> bridges;

    if (isAutotunerEnabled) {
        std::cout << "Starting auto tuner." << std::endl;

        //Autotuner* autotuner = new Autotuner(graph, samples);
        const int MAX_ITERATION = 5;
        int targetDiameter = 16; // Should be obtained from the original graph

        /*
          We'll assume here that the original graph is analysed and the diameter is known.
          The diameter of the test (facebook) graph is 8. Let's assume that the sampling algorithm preserves this
          number correctly for >=0.8 samples.
         */
        GraphAnalyser* graphAnalyser = new GraphAnalyser();

        graphAnalyser->loadGraph(samples, scaleUpSamplesInfo->getTopology()->getBridgeEdges(samples));

        int currentIteration = 0;

        while (currentIteration < MAX_ITERATION) {
            std::cout << "Current iteration: " << currentIteration + 1 << "/" << MAX_ITERATION << std::endl;
        }

        //delete(autotuner);
        delete(graphAnalyser);
    } else {
        bridges = scaleUpSamplesInfo->getTopology()->getBridgeEdges(samples);
    }

    WriteGraph* writeScaledUpGraph = new WriteScaledUpGraph(outputFolder, samples, bridges, scaleUpSamplesInfo);
    writeScaledUpGraph->write();

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
    Graph* sampledGraph = scaleUpSamplesInfo->getSamplingAlgorithm()->sampleBase(samplingFraction);
    sampledGraph->setIdentifier(identifier);
    samples.push_back(sampledGraph);
}

bool ScaleUp::shouldSampleRemainder(ScalingUpConfig *scaleUpSamplesInfo, int currentLoopIteration) {
    bool isLastSamplingIteration = currentLoopIteration == (scaleUpSamplesInfo->getAmountOfSamples() - 1);

    return isLastSamplingIteration && scaleUpSamplesInfo->hasSamplingRemainder();
}