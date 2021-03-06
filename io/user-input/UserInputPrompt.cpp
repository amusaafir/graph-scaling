//
// Created by Ahmed on 25-1-18.
//

#include "UserInputPrompt.h"
#include "../../scaling/scale-up/bridge/HighDegreeBridge.h"
#include "../../scaling/scale-down/TIES.h"
#include "../../scaling/scale-down/RandomEdge.h"
#include "../../scaling/scale-down/RandomNode.h"
#include "../../scaling/scale-down/ForestFire.h"

UserInputPrompt::UserInputPrompt() {

}

int UserInputPrompt::getScalingType() {
    std::cout << "[1] - Scale-up" << std::endl;
    std::cout << "[2] - Scale-down" << std::endl;

    std::cout << "Select option:" << std::endl;
    int selection;
    std::cin >> selection;

    return selection;
}

std::string UserInputPrompt::getInputGraphPath() {
    std::cout << "Note: the input and output paths may not contain any white space (newlines/spaces)." << std::endl;

    std::cout << "Enter path input graph:" << std::endl;

    std::string specInputGraphPath;
    std::cin >> specInputGraphPath;

    return specInputGraphPath;
}

std::string UserInputPrompt::getOutputGraphPath() {
    std::string specOutputFolder;
    std::cout << "Enter path output folder:" << std::endl;

    std::cin >> specOutputFolder;

    return specOutputFolder;
}

Bridge* UserInputPrompt::getBridge() {
    int numberInterconnections = getNumberOfInterconnections();
    bool directedBridges = addDirectedBridges();

    std::cout << "Bridge type: [r]andom; [h]igh degree:" << std::endl;

    char bridgeType;
    std::cin >> bridgeType;

    if (bridgeType == 'h') {
        return new HighDegreeBridge(numberInterconnections, directedBridges);
    }

    return new RandomBridge(numberInterconnections, directedBridges);
}

bool UserInputPrompt::addDirectedBridges() {
    std::cout << "Add directed bridges (y/n)?" << std::endl;

    char addDirectedBridges;
    std::cin >> addDirectedBridges;

    return addDirectedBridges == 'y';
}

long long UserInputPrompt::getNumberOfInterconnections() {
    std::cout << "Number of interconnections between each graph:" << std::endl;

    int numberOfInterconnections;
    std::cin >> numberOfInterconnections;

    return numberOfInterconnections;
}

Topology* UserInputPrompt::getTopology() {
    bridge = getBridge();

    initTopologies();

    std::cout << "Topology: [s]tar; [c]hain; [r]ing; [f]ullyconnected:" << std::endl;
    char specTopology;
    std::cin >> specTopology;

    return topologies[specTopology];
}

float UserInputPrompt::getSamplingFraction() {
    std::cout << "Sample size:" << std::endl;

    float samplingFraction;
    std::cin >> samplingFraction;

    return samplingFraction;
}

float UserInputPrompt::getScalingFactor() {
    std::cout << "Scaling factor: " << std::endl;

    float scalingFactor;
    std::cin >> scalingFactor;

    return scalingFactor;
}

// TODO: Merge with input cmd and add random edge with both directions
Sampling* UserInputPrompt::getSamplingAlgorithm(Graph* graph) {
    std::cout << "Sampling algorithm: [t]ies; random [e]dge; random [n]ode; [f]orest fire:" << std::endl;

    char samplingAlgorithm;
    std::cin >> samplingAlgorithm;

    if (samplingAlgorithm == 't') {
        return new TIES(graph);
    } else if (samplingAlgorithm == 'n') {
        return new RandomNode(graph);
    } else if (samplingAlgorithm == 'f') {
        return new ForestFire(graph, 0);
    }

    return new RandomEdge(graph, false);
}