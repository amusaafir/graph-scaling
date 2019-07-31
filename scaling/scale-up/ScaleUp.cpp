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

        const int TARGET_DIAMETER = 30; // Should be obtained from the original graph
        const int ORIGINAL_DIAMETER = 8;
        Autotuner* autotuner = new Autotuner(ORIGINAL_DIAMETER, TARGET_DIAMETER, 6);
        const int MAX_ITERATION = 40;

        /*
          We'll assume here that the original graph is analysed and the diameter is known.
          The diameter of the test (facebook) graph is 8. Let's assume that the sampling algorithm preserves this
          number correctly for >=0.5 samples, using a scaling factor of 3 (nr. of samples = 6).
         */
        GraphAnalyser* graphAnalyser = new GraphAnalyser();

        int currentIteration = 0;

        std::unordered_set<std::string> usedParameters;

        StarModel starModel(ORIGINAL_DIAMETER, 6, 3);
        ChainModel chainModel(ORIGINAL_DIAMETER, 6, 3);
        FullyConnectedModel fullyConnectedModel(ORIGINAL_DIAMETER, 6, 3);
        RingModel ringModel(ORIGINAL_DIAMETER, 6, 3);

        SuggestedParameters starDefault;
        starDefault.topology = starModel.createTopology(new RandomBridge(1, false));
        //usedParameters.insert(starDefault.getParameterStringRepresentation());

        SuggestedParameters fullyConDefault;
        fullyConDefault.topology = fullyConnectedModel.createTopology(new RandomBridge(1, false));
       // usedParameters.insert(fullyConDefault.getParameterStringRepresentation());

        SuggestedParameters chainDefault;
        chainDefault.topology = chainModel.createTopology(new RandomBridge(1, false));
       //usedParameters.insert(chainDefault.getParameterStringRepresentation());

        SuggestedParameters ringDefault;
        ringDefault.topology = ringModel.createTopology(new RandomBridge(1, false));
        //usedParameters.insert(ringDefault.getParameterStringRepresentation());

        SuggestedParameters suggestedParametersArr[4];
        suggestedParametersArr[0] = starDefault;
        suggestedParametersArr[1] = chainDefault;
        suggestedParametersArr[2] = ringDefault;
        suggestedParametersArr[3] = fullyConDefault;

        //autotuner->addNodeToDiameterTree(ringModel.getMaxDiameter(), ringDefault, true);
        //autotuner->addNodeToDiameterTree(fullyConnectedModel.getMaxDiameter(), fullyConDefault, true);
        //autotuner->addNodeToDiameterTree(chainModel.getMaxDiameter(), chainDefault, true);
        //autotuner->addNodeToDiameterTree(starModel.getMaxDiameter(), starDefault, true);

        SuggestedParameters suggestedParameters;
        //suggestedParameters.topology = scaleUpSamplesInfo->getTopology();




        while (currentIteration < MAX_ITERATION + 4) {


            std::cout << "Current tuning iteration: " << currentIteration + 1 << "/" << MAX_ITERATION
                      << ", Target diameter: " << TARGET_DIAMETER << std::endl;

            std::cout<< "Loading graph.." << std::endl;

            std::vector<Edge<std::string>> bridges;
            if (currentIteration < 4) {
                bridges = suggestedParametersArr[currentIteration].topology->getBridgeEdges(samples);
                suggestedParameters = suggestedParametersArr[currentIteration];
            } else {
                bridges = suggestedParameters.topology->getBridgeEdges(samples);
            }
            graphAnalyser->loadGraph(samples, bridges);

            std::cout << "Analying diameter.." << std::endl;
            int diameter = graphAnalyser->calculateDiameter();
            std::cout << "Current diameter: " << diameter << std::endl;

            bool isGraphSuccessfullyDeleted = graphAnalyser->deleteGraph();

            if (isGraphSuccessfullyDeleted) {
                std::cout << "Graph successfully deleted." << std::endl;
            } else {
                std::cout << "Unsuccessful deletion of the graph." << std::endl;
            }

            usedParameters.insert(suggestedParameters.getParameterStringRepresentation());

            autotuner->addNodeToDiameterTree(diameter, suggestedParameters, false);

            std::cout << "Used parameters: " << std::endl;
            for (const auto& elem : usedParameters) {
                std::cout << elem << std::endl;
            }

            if (currentIteration > 3) {
                std::cout<<"New suggestion.." << std::endl;
                suggestedParameters = autotuner->getNewSuggestion();
            }
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