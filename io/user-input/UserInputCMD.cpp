//
// Created by Ahmed on 25-1-18.
//

#include "UserInputCMD.h"

UserInputCMD::UserInputCMD(int argc, char* argv[]) {
    insertArgumentValues(argc, argv);
}

int UserInputCMD::getScalingType() {
    bool isScaleUp = inputArguments.size() == 7;

    return isScaleUp;
}

std::string UserInputCMD::getInputGraphPath() {
    return inputArguments['i'];
}

std::string UserInputCMD::getOutputGraphPath() {
    return inputArguments['o'];
}

// TODO
Bridge* UserInputCMD::getBridge() {
    return new RandomBridge(getNumberOfInterconnections(), addDirectedBridges());
}

bool UserInputCMD::addDirectedBridges() {
    bool forceDirectedBridges = false;
    std::istringstream(inputArguments['d']) >> forceDirectedBridges;

    return forceDirectedBridges;
}

int UserInputCMD::getNumberOfInterconnections() {
    return stoi(inputArguments['b']);
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

void UserInputCMD::insertArgumentValues(int argc, char* argv[]) {
    int opt;

    while ((opt = getopt (argc, argv, "i:o:s:u:t:b:d:")) != -1) {
        if (!inputArguments.count(opt)) {
            inputArguments[opt] = optarg;
        } else {
            char option = opt;
            std::cout << "Invalid option specified: " << option << "." << std::endl;
        }
    }
}