//
// Created by Ahmed on 22-1-18.
//

#ifndef GRAPH_SCALING_TOOL_USERINPUT_H
#define GRAPH_SCALING_TOOL_USERINPUT_H

#include <iostream>
#include <map>
#include <getopt.h>
#include "../../scaling/scale-up/bridge/Bridge.h"
#include "../../scaling/scale-up/topology/Topology.h"
#include "../../scaling/scale-up/topology/StarTopology.h"
#include "../../scaling/scale-up/topology/ChainTopology.h"
#include "../../scaling/scale-up/topology/RingTopology.h"
#include "../../scaling/scale-up/topology/FullyConnectedTopology.h"

class UserInput {
protected:
    std::map<char, Topology*> topologies;
    Bridge* bridge;

public:
    UserInput();

    ~UserInput();

    void initTopologies();

    virtual int getScalingType() = 0;

    virtual std::string getInputGraphPath() = 0;

    virtual std::string getOutputGraphPath() = 0;

    virtual int getNumberOfInterconnections() = 0;

    virtual float getSamplingFraction() = 0;

    virtual float getScalingFactor() = 0;

    virtual bool addDirectedBridges() = 0;

    virtual Bridge* getBridge() = 0;

    virtual Topology* getTopology() = 0;
};


#endif //GRAPH_SCALING_TOOL_USERINPUT_H
