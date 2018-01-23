//
// Created by Ahmed on 22-1-18.
//

#ifndef GRAPH_SCALING_TOOL_USERINPUT_H
#define GRAPH_SCALING_TOOL_USERINPUT_H

#include <iostream>
#include <map>
#include "../scaling/scale-up/bridge/Bridge.h"
#include "../scaling/scale-up/topology/Topology.h"

class UserInput {
private:
    std::map<char, Topology*> topologies;
    Bridge* bridge;

public:
    UserInput();

    ~UserInput();

    void initTopologies();

    int getScalingType();

    std::string getInputGraphPath();

    std::string getOutputGraphPath();

    int getNumberOfInterconnections();

    float getSamplingFraction();

    float getScalingFactor();

    bool addDirectedBridges();

    Bridge* getBridge();

    Topology* getTopology();
};


#endif //GRAPH_SCALING_TOOL_USERINPUT_H
