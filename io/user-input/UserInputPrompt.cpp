//
// Created by Ahmed on 25-1-18.
//

#include "UserInputPrompt.h"

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

// TODO
Bridge* UserInputPrompt::getBridge() {
    int numberInterconnections = getNumberOfInterconnections();
    bool directedBridges = addDirectedBridges();

    return new RandomBridge(numberInterconnections, directedBridges);
}

bool UserInputPrompt::addDirectedBridges() {
    std::cout << "Add directed bridges (y/n)?" << std::endl;
    char addDirectedBridges;
    std::cin >> addDirectedBridges;

    std::cout<<(addDirectedBridges == 'y' )<< std::endl;

    return addDirectedBridges == 'y';
}

int UserInputPrompt::getNumberOfInterconnections() {
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
    std::cout << "Sample size per graph: " << std::endl;
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