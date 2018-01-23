//
// Created by Ahmed on 22-1-18.
//

#include "UserInput.h"
#include "../scaling/scale-up/bridge/RandomBridge.h"
#include "../scaling/scale-up/topology/StarTopology.h"
#include "../scaling/scale-up/topology/ChainTopology.h"
#include "../scaling/scale-up/topology/RingTopology.h"
#include "../scaling/scale-up/topology/FullyConnectedTopology.h"

UserInput::UserInput() {
}

UserInput::~UserInput() {
    delete(topologies['s']);
    delete(topologies['c']);
    delete(topologies['r']);
    delete(topologies['f']);

    delete(bridge);
}

void UserInput::initTopologies() {
    topologies['s'] = new StarTopology(bridge);
    topologies['c'] = new ChainTopology(bridge);
    topologies['r'] = new RingTopology(bridge);
    topologies['f'] = new FullyConnectedTopology(bridge);
}

int UserInput::getScalingType() {
    std::cout << "[1] - Scale-up" << std::endl;
    std::cout << "[2] - Scale-down" << std::endl;

    int selection;
    std::cout << "Select option:" << std::endl;
    std::cin >> selection;

    return selection;
}

std::string UserInput::getInputGraphPath() {
    std::cout << "Note: the input and output paths may not contain any white space (newlines/spaces)." << std::endl;

    std::string specInputGraphPath;
    std::cout << "Enter path input graph:" << std::endl;
    std::cin >> specInputGraphPath;

    return specInputGraphPath;
}

std::string UserInput::getOutputGraphPath() {
    std::string specOutputFolder;
    std::cout << "Enter path output folder:" << std::endl;
    std::cin >> specOutputFolder;

    return specOutputFolder;
}

// TODO
Bridge* UserInput::getBridge() {
    int numberInterconnections = getNumberOfInterconnections();
    bool directedBridges = addDirectedBridges();

    return new RandomBridge(numberInterconnections, directedBridges);
}

bool UserInput::addDirectedBridges() {
    char addDirectedBridges;
    std::cout << "Add directed bridges (y/n)?" << std::endl;
    std::cin >> addDirectedBridges;

    std::cout<<(addDirectedBridges == 'y' )<< std::endl;

    return addDirectedBridges == 'y';
}

int UserInput::getNumberOfInterconnections() {
    std::cout << "Number of interconnections between each graph:" << std::endl;

    int numberOfInterconnections;
    std::cin >> numberOfInterconnections;

    return numberOfInterconnections;
}

Topology* UserInput::getTopology() {
    bridge = getBridge();

    initTopologies();

    char specTopology;
    std::cout << "Topology: [s]tar; [c]hain; [r]ing; [f]ullyconnected:" << std::endl;
    std::cin >> specTopology;

    return topologies[specTopology];
}

float UserInput::getSamplingFraction() {
    float samplingFraction;
    std::cout << "Sample size per graph: " << std::endl;
    std::cin >> samplingFraction;

    return samplingFraction;
}

float UserInput::getScalingFactor() {
    float scalingFactor;
    std::cout << "Scaling factor: " << std::endl;
    std::cin >> scalingFactor;

    return scalingFactor;
}