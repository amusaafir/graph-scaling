//
// Created by Ahmed on 25-1-18.
//

#include "UserInputCMD.h"
#include "../../scaling/scale-up/bridge/HighDegreeBridge.h"
#include "../../scaling/scale-down/Sampling.h"
#include "../../scaling/scale-down/RandomEdge.h"
#include "../../scaling/scale-down/TIES.h"
#include "../../scaling/scale-down/RandomNode.h"

UserInputCMD::UserInputCMD(int argc, char* argv[]) {
    insertArgumentValues(argc, argv);
}

int UserInputCMD::getScalingType() {
    const int ARGS_REQUIRED_FOR_SCALING_UP = 9;
    bool isScaleUp = inputArguments.size() == ARGS_REQUIRED_FOR_SCALING_UP;

    return isScaleUp;
}

std::string UserInputCMD::getInputGraphPath() {
    return inputArguments['i'];
}

std::string UserInputCMD::getOutputGraphPath() {
    return inputArguments['o'];
}

Bridge* UserInputCMD::getBridge() {
    if (inputArguments['b'] == "high") {
        return new HighDegreeBridge(getNumberOfInterconnections(), addDirectedBridges());
    }

    return new RandomBridge(getNumberOfInterconnections(), addDirectedBridges());

}

bool UserInputCMD::addDirectedBridges() {
    bool forceDirectedBridges = false;
    std::istringstream(inputArguments['d']) >> forceDirectedBridges;

    return forceDirectedBridges;
}

long long UserInputCMD::getNumberOfInterconnections() {
    return stoi(inputArguments['n']);
}

Topology* UserInputCMD::getTopology() {
    bridge = getBridge();

    initTopologies();

    std::string selectedTopology = inputArguments['t'];
    char topologyId = selectedTopology.at(0);

    if (!topologies.count(topologyId)) {
        std::cout << "Invalid topology specified: " << topologyId << std::endl;
        exit(1);
    }

    return topologies[topologyId];
}

float UserInputCMD::getSamplingFraction() {
    return stof(inputArguments['s']);
}

float UserInputCMD::getScalingFactor() {
    return stof(inputArguments['u']);
}

Sampling* UserInputCMD::getSamplingAlgorithm(Graph* graph) {
    if (inputArguments['a'] == "randomedge") {
        return new RandomEdge(graph);
    } else if (inputArguments['a'] == "randomnode") {
        return new RandomNode(graph);
    }

    return new TIES(graph);
}

void UserInputCMD::insertArgumentValues(int argc, char* argv[]) {
    int opt;

    while ((opt = getopt (argc, argv, "i:o:s:u:t:n:d:b:a:")) != -1) {
        if (!inputArguments.count(opt)) { // TODO: init map and check if it exists
            inputArguments[opt] = optarg;
        } else {
            char option = opt;
            std::cout << "Invalid option specified: " << option << "." << std::endl;
        }
    }
}