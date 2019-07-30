//
// Created by Ahmed on 15-11-17.
//

#include "ScaleUp.h"
#include "auto-tuner/Autotuner.h"
#include "topology/StarTopology.h"
#include "bridge/RandomBridge.h"
#include "topology/FullyConnectedTopology.h"
#include "topology/ChainTopology.h"
#include "topology/RingTopology.h"
#include "auto-tuner/model/StarModel.h"
#include "auto-tuner/model/ChainModel.h"
#include "auto-tuner/model/FullyConnectedModel.h"
#include "auto-tuner/model/RingModel.h"

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

        const int ORIGINAL_DIAMETER = 8;
        Autotuner* autotuner = new Autotuner(ORIGINAL_DIAMETER, 6);
        const int MAX_ITERATION = 2;
        int targetDiameter = 16; // Should be obtained from the original graph

        /*
          We'll assume here that the original graph is analysed and the diameter is known.
          The diameter of the test (facebook) graph is 8. Let's assume that the sampling algorithm preserves this
          number correctly for >=0.5 samples, using a scaling factor of 3 (nr. of samples = 6).
         */
        GraphAnalyser* graphAnalyser = new GraphAnalyser();

        int currentIteration = 0;

        StarModel starModel(ORIGINAL_DIAMETER, 6, 3);
        ChainModel chainModel(ORIGINAL_DIAMETER, 6, 3);
        FullyConnectedModel fullyConnectedModel(ORIGINAL_DIAMETER, 6, 3);
        RingModel ringModel(ORIGINAL_DIAMETER, 6, 3);

        SuggestedParameters starDefault;
        starDefault.topology = starModel.createTopology(new RandomBridge(1, false));

        SuggestedParameters fullyConDefault;
        fullyConDefault.topology = fullyConnectedModel.createTopology(new RandomBridge(1, false));

        SuggestedParameters chainDefault;
        chainDefault.topology = chainModel.createTopology(new RandomBridge(1, false));

        SuggestedParameters ringDefault;
        ringDefault.topology = ringModel.createTopology(new RandomBridge(1, false));

        autotuner->addNodeToDiameterTree(ringModel.getMaxDiameter(), ringDefault, true);
        autotuner->addNodeToDiameterTree(fullyConnectedModel.getMaxDiameter(), fullyConDefault, true);
        autotuner->addNodeToDiameterTree(chainModel.getMaxDiameter(), chainDefault, true);
        autotuner->addNodeToDiameterTree(starModel.getMaxDiameter(), starDefault, true);

        while (currentIteration < MAX_ITERATION) {
            std::cout << "Current tuning iteration: " << currentIteration + 1 << "/" << MAX_ITERATION << std::endl;

            graphAnalyser->loadGraph(samples, scaleUpSamplesInfo->getTopology()->getBridgeEdges(samples));

            int diameter = graphAnalyser->calculateDiameter();
            std::cout << "Current diameter: " << diameter << std::endl;

            bool isGraphSuccessfullyDeleted = graphAnalyser->deleteGraph();

            if (!isGraphSuccessfullyDeleted) {
                std::cout << "Unsuccessful deletion of the graph." << std::endl;
            }

            SuggestedParameters suggestedParameters;
            suggestedParameters.topology = scaleUpSamplesInfo->getTopology();

            autotuner->addNodeToDiameterTree(diameter, suggestedParameters, false);

            currentIteration++;
        }

        delete(autotuner);
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